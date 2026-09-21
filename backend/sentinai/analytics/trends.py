"""Module 8 — Temporal trend analysis, forecasting, spike detection and event correlation.

* Time series: daily (or hourly) counts of toxic posts, optionally per target demographic.
* Forecast: Prophet when installed, otherwise statsmodels ARIMA/Holt-Winters, otherwise a
  seasonal-naive baseline — all behind :func:`forecast`.
* Spike detection: rolling z-score with a robust (median / MAD) option.
* Event correlation: overlay events (NewsAPI / GDELT / manual) on the series and compute the
  lift in toxicity volume in the window after each event vs. the window before.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sentinai.config import get_settings
from sentinai.storage.models import ClassificationRow, EventRow, PostRow, TargetHitRow

log = logging.getLogger(__name__)


@dataclass
class TrendPoint:
    date: str
    total: int
    toxic: int
    toxic_ratio: float
    avg_severity: float
    zscore: float = 0.0
    spike: bool = False


@dataclass
class ForecastPoint:
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


@dataclass
class EventImpact:
    id: int
    date: str
    title: str
    category: str
    source: str
    before_mean: float
    after_mean: float
    lift: float
    related_targets: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# series
# --------------------------------------------------------------------------------------------


def toxic_series(session: Session, *, days: int = 90, target: str | None = None, freq: str = "D", threshold: float = 0.5, exclude_bots: bool = False, bot_threshold: float = 0.8) -> pd.DataFrame:
    """Return a DataFrame indexed by period with columns total/toxic/avg_severity."""
    since = datetime.utcnow() - timedelta(days=days)
    q = select(PostRow.created_at, ClassificationRow.final_toxicity, ClassificationRow.severity_level).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).where(PostRow.created_at >= since, PostRow.deleted_upstream.is_(False))
    if target:
        cat, _, label = target.partition(":")
        q = q.join(TargetHitRow, TargetHitRow.post_id == PostRow.id).where(TargetHitRow.category == cat)
        if label:
            q = q.where(TargetHitRow.label == label)
    if exclude_bots:
        from sentinai.storage.models import AuthorRow

        q = q.join(AuthorRow, AuthorRow.id == PostRow.author_id).where((AuthorRow.bot_probability.is_(None)) | (AuthorRow.bot_probability < bot_threshold))
    rows = session.execute(q).all()
    if not rows:
        idx = pd.date_range(end=datetime.utcnow().date(), periods=days, freq=freq)
        return pd.DataFrame({"total": 0, "toxic": 0, "avg_severity": 0.0}, index=idx)
    df = pd.DataFrame(rows, columns=["ts", "tox", "sev"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["toxic"] = (df["tox"] >= threshold).astype(int)
    g = df.set_index("ts").resample(freq)
    out = pd.DataFrame({"total": g["tox"].count(), "toxic": g["toxic"].sum(), "avg_severity": g["sev"].mean().fillna(0.0)})
    full_idx = pd.date_range(start=out.index.min(), end=max(out.index.max(), pd.Timestamp(datetime.utcnow().date())), freq=freq)
    return out.reindex(full_idx, fill_value=0).astype({"total": int, "toxic": int})


def zscores(values: list[float], window: int = 14, robust: bool = True) -> list[float]:
    """Rolling z-score of each point against the preceding ``window`` points."""
    out: list[float] = []
    for i, v in enumerate(values):
        hist = values[max(0, i - window) : i]
        if len(hist) < 3:
            out.append(0.0)
            continue
        if robust:
            med = sorted(hist)[len(hist) // 2]
            mad = sorted(abs(x - med) for x in hist)[len(hist) // 2]
            scale = 1.4826 * mad
            if scale == 0:
                scale = (sum((x - med) ** 2 for x in hist) / len(hist)) ** 0.5 or 1.0
            out.append((v - med) / scale)
        else:
            mean = sum(hist) / len(hist)
            sd = (sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5 or 1.0
            out.append((v - mean) / sd)
    return out


def trend_points(df: pd.DataFrame, z_threshold: float | None = None) -> list[TrendPoint]:
    z_threshold = z_threshold if z_threshold is not None else get_settings().spike_zscore_threshold
    zs = zscores([float(x) for x in df["toxic"].tolist()])
    pts = []
    for (ts, row), z in zip(df.iterrows(), zs, strict=True):
        total = int(row["total"])
        toxic = int(row["toxic"])
        pts.append(
            TrendPoint(
                date=ts.strftime("%Y-%m-%d") if len(df) < 400 else ts.isoformat(),
                total=total,
                toxic=toxic,
                toxic_ratio=round(toxic / total, 4) if total else 0.0,
                avg_severity=round(float(row["avg_severity"]), 3),
                zscore=round(float(z), 3),
                spike=bool(z >= z_threshold and toxic >= 3),
            )
        )
    return pts


# --------------------------------------------------------------------------------------------
# forecasting
# --------------------------------------------------------------------------------------------


def forecast(df: pd.DataFrame, horizon: int = 14) -> tuple[list[ForecastPoint], str]:
    y = df["toxic"].astype(float)
    if len(y) < 7 or y.sum() == 0:
        return _naive(y, horizon), "naive"
    try:  # Prophet (optional)
        from prophet import Prophet  # type: ignore

        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False, interval_width=0.8)
        m.fit(pd.DataFrame({"ds": y.index, "y": y.values}))
        fut = m.make_future_dataframe(periods=horizon)
        fc = m.predict(fut).tail(horizon)
        return [ForecastPoint(r.ds.strftime("%Y-%m-%d"), max(0.0, float(r.yhat)), max(0.0, float(r.yhat_lower)), max(0.0, float(r.yhat_upper))) for r in fc.itertuples()], "prophet"
    except Exception:
        pass
    try:  # statsmodels Holt-Winters / ARIMA
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore

        seasonal = "add" if len(y) >= 21 else None
        model = ExponentialSmoothing(y.values, trend="add", seasonal=seasonal, seasonal_periods=7 if seasonal else None, initialization_method="estimated").fit(optimized=True)
        pred = model.forecast(horizon)
        resid = y.values - model.fittedvalues
        sd = float(pd.Series(resid).std(ddof=1) or 1.0)
        dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return [ForecastPoint(d.strftime("%Y-%m-%d"), max(0.0, float(p)), max(0.0, float(p) - 1.28 * sd * math.sqrt(i + 1)), max(0.0, float(p) + 1.28 * sd * math.sqrt(i + 1))) for i, (d, p) in enumerate(zip(dates, pred, strict=True))], "holt-winters"
    except Exception as exc:
        log.debug("statsmodels forecast failed: %s", exc)
    return _naive(y, horizon), "naive"


def _naive(y: pd.Series, horizon: int) -> list[ForecastPoint]:
    last = y.tail(7)
    mean = float(last.mean()) if len(last) else 0.0
    sd = float(last.std(ddof=0)) if len(last) > 1 else max(1.0, mean**0.5)
    start = (y.index[-1] if len(y) else pd.Timestamp(datetime.utcnow().date())) + pd.Timedelta(days=1)
    return [ForecastPoint(d.strftime("%Y-%m-%d"), mean, max(0.0, mean - 1.28 * sd), mean + 1.28 * sd) for d in pd.date_range(start=start, periods=horizon, freq="D")]


# --------------------------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------------------------


def event_impacts(session: Session, df: pd.DataFrame, window_days: int = 3) -> list[EventImpact]:
    if df.empty:
        return []
    start, end = df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
    events = session.scalars(select(EventRow).where(EventRow.date >= start - timedelta(days=1), EventRow.date <= end + timedelta(days=1)).order_by(EventRow.date)).all()
    out = []
    series = df["toxic"].astype(float)
    for ev in events:
        d = pd.Timestamp(ev.date.date())
        before = series[(series.index >= d - pd.Timedelta(days=window_days)) & (series.index < d)]
        after = series[(series.index >= d) & (series.index < d + pd.Timedelta(days=window_days))]
        b = float(before.mean()) if len(before) else 0.0
        a = float(after.mean()) if len(after) else 0.0
        lift = (a - b) / b if b > 0 else (a if a > 0 else 0.0)
        out.append(EventImpact(ev.id, d.strftime("%Y-%m-%d"), ev.title, ev.category, ev.source, round(b, 2), round(a, 2), round(lift, 3), ev.related_targets or []))
    return out


def fetch_newsapi_events(query: str, days: int = 30, api_key: str | None = None) -> list[dict]:
    """Top headlines from NewsAPI (requires ``SENTINAI_NEWSAPI_KEY``)."""
    key = api_key or get_settings().newsapi_key
    if not key:
        return []
    since = (datetime.utcnow() - timedelta(days=min(days, 29))).strftime("%Y-%m-%d")
    r = httpx.get("https://newsapi.org/v2/everything", params={"q": query, "from": since, "sortBy": "popularity", "pageSize": 50, "apiKey": key}, timeout=20)
    r.raise_for_status()
    return [{"date": a["publishedAt"][:10], "title": a["title"], "url": a.get("url"), "source": "newsapi", "category": "news"} for a in r.json().get("articles", [])]


def fetch_gdelt_events(query: str, days: int = 30) -> list[dict]:
    """GDELT 2.0 DOC API — no key required."""
    if not get_settings().gdelt_enabled:
        return []
    r = httpx.get("https://api.gdeltproject.org/api/v2/doc/doc", params={"query": query, "mode": "artlist", "maxrecords": 75, "timespan": f"{days}d", "format": "json", "sort": "hybridrel"}, timeout=30)
    r.raise_for_status()
    arts = r.json().get("articles", []) if r.headers.get("content-type", "").startswith("application/json") else []
    out = []
    for a in arts:
        seen = a.get("seendate", "")
        date = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 else None
        if date:
            out.append({"date": date, "title": a.get("title", ""), "url": a.get("url"), "source": "gdelt", "category": "news"})
    return out


def store_events(session: Session, events: list[dict], related_targets: list[str] | None = None) -> int:
    n = 0
    for ev in events:
        date = datetime.fromisoformat(ev["date"])
        exists = session.scalar(select(func.count()).select_from(EventRow).where(EventRow.title == ev["title"], EventRow.date == date))
        if exists:
            continue
        session.add(EventRow(date=date, title=ev["title"][:512], category=ev.get("category", "news"), source=ev.get("source", "manual"), url=ev.get("url"), related_targets=related_targets or ev.get("related_targets"), magnitude=float(ev.get("magnitude", 1.0))))
        n += 1
    return n
