"""``sentinai`` command line: serve, ingest, classify, seed, maintenance jobs."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="SentinAI — Social Media Bias & Hate Speech Intelligence Platform", no_args_is_help=True)


@app.command()
def serve(host: str = typer.Option(None), port: int = typer.Option(None), reload: bool = False):
    """Run the API (and the built dashboard if frontend/dist exists)."""
    import uvicorn

    from sentinai.config import get_settings

    s = get_settings()
    uvicorn.run("sentinai.api.app:app", host=host or s.host, port=port or s.port, reload=reload)


@app.command()
def classify(text: str, explain: bool = True):
    """Classify a single text and print the JSON result."""
    from sentinai.classification.engine import ClassificationEngine

    c = ClassificationEngine(dedup=False).classify_text(text, explain=explain)
    typer.echo(json.dumps(c.model_dump(mode="json"), indent=2, ensure_ascii=False))


@app.command()
def seed(n_posts: int = 1600, seed_value: int = 42, reset: bool = typer.Option(False, help="Drop and recreate the database first")):
    """Populate the database with a synthetic multilingual demo corpus."""
    from sentinai.storage.db import get_engine, init_db
    from sentinai.storage.models import Base

    if reset:
        Base.metadata.drop_all(get_engine())
    init_db()
    from sentinai.demo import seed as _seed

    n = _seed(n_posts, seed_value)
    typer.echo(f"seeded {n} posts")


def _report_echo(report) -> None:
    d = report.to_dict()
    typer.echo(f"{d['kind']}: fetched {d['fetched']}, ingested {d['ingested']} (toxic {d['toxic']}) via {d['endpoint']} in {d['duration_seconds']}s")
    if d["by_label"]:
        typer.echo("  labels: " + ", ".join(f"{k}={v}" for k, v in sorted(d["by_label"].items())))
    for w in d["warnings"]:
        typer.echo(f"  warning: {w}")


def _x_client(max_wait: float | None = None):
    from sentinai.ingestion.credentials import get_bearer_token
    from sentinai.ingestion.x_client import XClient

    token = get_bearer_token()
    if not token:
        raise typer.BadParameter("No X API bearer token — set SENTINAI_X_BEARER_TOKEN (or save one in the dashboard under Ingest → Connect X)")
    return XClient(bearer_token=token, max_wait=max_wait)


@app.command()
def ingest(
    query: str,
    pages: int = typer.Option(5, help="Pages of 100 (recent) / 500 (full-archive) posts"),
    full_archive: bool = typer.Option(False, "--full-archive", help="Use /tweets/search/all (needs full-archive access; falls back to recent)"),
    recent: bool = typer.Option(False, help="Deprecated — recent search is the default now"),
    start_time: str | None = typer.Option(None, help="ISO-8601, e.g. 2026-09-01T00:00:00Z"),
    end_time: str | None = None,
    no_cursor: bool = typer.Option(False, "--no-cursor", help="Ignore the stored since_id and re-fetch the window"),
    media: bool = typer.Option(False, help="Download photo attachments for the vision pipeline"),
):
    """Search X (API v2) for QUERY and run every hit through the pipeline.

    Example: sentinai ingest '(muslims OR islam) lang:en -is:retweet' --pages 3
    """
    from sentinai.ingestion.worker import IngestionWorker, IngestReport
    from sentinai.storage.db import init_db, session_scope

    init_db()
    client = _x_client()
    report = IngestReport(kind="search", query=query)
    with session_scope() as session:
        IngestionWorker(session, client, download_media=media).run_query(query, full_archive=full_archive or None, max_pages=pages, start_time=start_time, end_time=end_time, use_cursor=not no_cursor, report=report)
    _report_echo(report)


@app.command("ingest-tweet")
def ingest_tweet(refs: list[str] = typer.Argument(..., help="Tweet URLs or ids"), conversation: bool = typer.Option(False, help="Also pull the replies"), media: bool = False):
    """Ingest one or more tweets by URL / id (plus their parent chain)."""
    from sentinai.ingestion.worker import IngestionWorker
    from sentinai.storage.db import init_db, session_scope

    init_db()
    client = _x_client()
    with session_scope() as session:
        worker = IngestionWorker(session, client, download_media=media)
        for ref in refs:
            _report_echo(worker.run_tweet(ref, include_conversation=conversation))


@app.command("ingest-user")
def ingest_user(username: str, pages: int = 3, exclude_replies: bool = False, exclude_retweets: bool = False, start_time: str | None = None, media: bool = False):
    """Ingest a user's timeline (@handle or profile URL)."""
    from sentinai.ingestion.worker import IngestionWorker
    from sentinai.storage.db import init_db, session_scope

    init_db()
    client = _x_client()
    with session_scope() as session:
        _report_echo(IngestionWorker(session, client, download_media=media).run_user(username, max_pages=pages, exclude_replies=exclude_replies, exclude_retweets=exclude_retweets, start_time=start_time))


@app.command("ingest-stream")
def ingest_stream(
    rule: list[str] = typer.Option([], "--rule", "-r", help="Filtered-stream rule (repeatable); replaces existing rules when given"),
    sample: bool = typer.Option(False, help="Use the 1% sampled stream"),
    max_posts: int | None = None,
    max_minutes: float | None = None,
    media: bool = False,
):
    """Connect to the filtered stream and classify posts as they arrive (Ctrl-C to stop)."""
    import signal
    import threading

    from sentinai.ingestion.worker import IngestionWorker
    from sentinai.storage.db import init_db, session_scope

    init_db()
    client = _x_client()
    if rule and not sample:
        rules = client.set_stream_rules([{"value": r} for r in rule])
        typer.echo("rules: " + "; ".join(r["value"] for r in rules))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with session_scope() as session:
        worker = IngestionWorker(session, client, download_media=media, progress=lambda stage, n: typer.echo(f"  {stage}: {n}", err=True))
        _report_echo(worker.run_stream(max_posts=max_posts, max_seconds=max_minutes * 60 if max_minutes else None, stop=stop, sample=sample))


@app.command("import")
def import_file(paths: list[Path] = typer.Argument(..., exists=True, readable=True), hydrate_parents: bool = typer.Option(False, help="Fetch missing parents from X (needs a token)")):
    """Import X API JSON/JSONL dumps, an X data archive (tweets.js / .zip), CSV or plain text."""
    from sentinai.ingestion.worker import IngestionWorker
    from sentinai.storage.db import init_db, session_scope

    init_db()
    client = None
    if hydrate_parents:
        try:
            client = _x_client()
        except typer.BadParameter as exc:
            typer.echo(f"warning: {exc} — importing without parent hydration", err=True)
    with session_scope() as session:
        worker = IngestionWorker(session, client)
        for path in paths:
            _report_echo(worker.run_import(path.read_bytes(), path.name, hydrate_parents=client is not None))


@app.command()
def compliance(retention_days: int | None = None, check_deleted: bool = typer.Option(False, help="Re-hydrate stored IDs and remove posts deleted upstream")):
    """Run retention purge and (optionally) the deleted-post sweep."""
    from sqlalchemy import select

    from sentinai.ingestion.compliance import enforce_retention, sync_deletions
    from sentinai.storage.db import init_db, session_scope
    from sentinai.storage.models import PostRow

    init_db()
    with session_scope() as session:
        result = enforce_retention(session, retention_days)
        if check_deleted:
            from sentinai.ingestion.x_client import XClient

            ids = session.scalars(select(PostRow.id).where(PostRow.deleted_upstream.is_(False))).all()
            result["deleted_upstream"] = sync_deletions(session, XClient(), list(ids))
    typer.echo(json.dumps(result))


@app.command()
def rescore():
    """Recompute bot probabilities and account-level scores."""
    from sentinai.analytics.accounts import score_all_accounts
    from sentinai.analytics.bots import score_all_authors
    from sentinai.storage.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        b = score_all_authors(session)
        a = score_all_accounts(session)
    typer.echo(f"bots scored: {b}, accounts scored: {a}")


@app.command()
def fairness(threshold: float = 0.5):
    """Evaluate the current classifier on the bundled dialect benchmarks."""
    from sentinai.analytics.fairness import bundled_benchmarks, evaluate, load_benchmark
    from sentinai.classification.engine import ClassificationEngine

    engine = ClassificationEngine(dedup=False)
    for path in bundled_benchmarks():
        rep = evaluate(engine.score_text, load_benchmark(path), threshold=threshold, benchmark=path.stem)
        typer.echo(f"== {rep.benchmark} (threshold {threshold}) FPR gap={rep.fpr_gap} FNR gap={rep.fnr_gap}")
        for g in rep.groups:
            typer.echo(f"  {g.group:9s} n={g.n:3d} FPR={g.fpr:.3f} FNR={g.fnr:.3f} calibrated_thr={rep.calibrated_thresholds[g.group]:.2f}")
        for note in rep.notes:
            typer.echo("  ! " + note)


@app.command("retrain-batch")
def retrain_batch(min_examples: int = 20, out_dir: Path | None = None):
    """Export accumulated human corrections as a fine-tuning batch (biweekly cron)."""
    from sentinai.hitl import build_retrain_batch
    from sentinai.storage.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        batch = build_retrain_batch(session, out_dir=out_dir, min_examples=min_examples)
        typer.echo("no batch created (not enough annotations)" if batch is None else f"batch {batch.id}: {batch.n_examples} examples ({batch.n_corrections} corrections) -> {batch.export_path}")


@app.command("weekly-audit")
def weekly_audit(days: int = 7):
    """Store a fairness audit snapshot computed on recent human annotations."""
    from sentinai.analytics.fairness import weekly_audit as _audit
    from sentinai.classification.engine import get_engine
    from sentinai.storage.db import init_db, session_scope

    init_db()
    with session_scope() as session:
        rep = _audit(session, get_engine().model_version, days=days)
    typer.echo(json.dumps(rep.to_dict(), indent=2))


@app.command("sync-events")
def sync_events(query: str, days: int = 30, source: str = "gdelt"):
    """Pull real-world events from GDELT / NewsAPI for trend correlation."""
    from sentinai.analytics.trends import fetch_gdelt_events, fetch_newsapi_events, store_events
    from sentinai.storage.db import init_db, session_scope

    init_db()
    evs = fetch_gdelt_events(query, days) if source == "gdelt" else fetch_newsapi_events(query, days)
    with session_scope() as session:
        n = store_events(session, evs)
    typer.echo(f"fetched {len(evs)} events, stored {n} new")


if __name__ == "__main__":
    app()
