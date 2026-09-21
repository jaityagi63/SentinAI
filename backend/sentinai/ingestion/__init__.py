"""Module 1 — Data ingestion engine (X API v2 client, rate-limit queue, compliance)."""

from sentinai.ingestion.x_client import XClient, XRateLimitError
from sentinai.ingestion.worker import IngestionWorker

__all__ = ["IngestionWorker", "XClient", "XRateLimitError"]
