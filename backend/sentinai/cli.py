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


@app.command()
def ingest(query: str, pages: int = 5, recent: bool = typer.Option(False, help="Use /search/recent instead of full-archive"), start_time: str | None = None):
    """Fetch posts from the X API v2 for QUERY, persist, classify and refresh analytics."""
    from sentinai.analytics.accounts import score_all_accounts
    from sentinai.analytics.bots import score_all_authors
    from sentinai.ingestion.worker import IngestionWorker
    from sentinai.ingestion.x_client import XClient
    from sentinai.storage.db import init_db, session_scope

    init_db()
    client = XClient()
    with session_scope() as session:
        n = IngestionWorker(session, client).run_query(query, full_archive=not recent, max_pages=pages, start_time=start_time)
        score_all_authors(session)
        score_all_accounts(session)
    typer.echo(f"ingested {n} posts for query {query!r}")


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
