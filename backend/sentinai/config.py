"""Central configuration (pydantic-settings; every value overridable via env / .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINAI_", env_file=".env", extra="ignore")

    # --- service --------------------------------------------------------------------------
    app_name: str = "SentinAI"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- storage --------------------------------------------------------------------------
    # PostgreSQL in production (`postgresql+psycopg://...`); SQLite for local dev / tests.
    database_url: str = f"sqlite:///{(REPO_DIR / 'data' / 'sentinai.db').as_posix()}"
    # Raw payload store: MongoDB (`mongodb://...`) or Elasticsearch (`http://...:9200`).
    # When unset, raw JSON payloads are kept in the `raw_payloads` relational table.
    raw_store_url: str | None = None
    data_dir: Path = REPO_DIR / "data"

    # --- ingestion (X / Twitter API v2) -----------------------------------------------------
    x_bearer_token: str | None = None
    x_api_base: str = "https://api.x.com/2"
    x_max_retries: int = 6
    x_backoff_base_seconds: float = 2.0
    x_backoff_max_seconds: float = 900.0
    # Compliance: retention window for raw payloads (GDPR/CCPA data minimisation).
    retention_days: int = 90
    # Compliance: how often the compliance stream / deleted-post sweep should run.
    compliance_sweep_hours: int = 24

    # --- models ---------------------------------------------------------------------------
    # "heuristic" = dependency-free lexicon/rule engine (always available, used in CI).
    # "transformer" = fine-tuned DeBERTa-v3 / XLM-R / mDeBERTa heads (requires the [ml] extra).
    classifier_backend: str = "heuristic"
    toxicity_model_name: str = "microsoft/deberta-v3-large"
    target_model_name: str = "xlm-roberta-large"
    multilingual_model_name: str = "microsoft/mdeberta-v3-base"
    model_cache_dir: Path = REPO_DIR / "models"
    fasttext_lid_path: Path | None = None

    # --- scoring knobs --------------------------------------------------------------------
    counterspeech_discount: float = 0.2  # Module 6: multiply toxicity when stance = Deny/Condemn
    visual_ensemble_weight: float = 0.4  # Module 5: weight of visual score in the text/visual ensemble
    account_min_posts: int = 50  # Module 12
    account_decay_half_life_days: float = 30.0  # Module 12
    active_learning_low: float = 0.4  # Module 14
    active_learning_high: float = 0.6  # Module 14
    spike_zscore_threshold: float = 2.5  # Module 8

    # --- auth -----------------------------------------------------------------------------
    jwt_secret: str = "sentinai-dev-only-secret-change-me-in-production-0123456789"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 8 * 60
    # Bootstrap accounts for local development. Override / remove in production.
    bootstrap_admin_password: str = "admin"
    bootstrap_researcher_password: str = "researcher"
    bootstrap_moderator_password: str = "moderator"

    # --- external event feeds (Module 8) ---------------------------------------------------
    newsapi_key: str | None = None
    gdelt_enabled: bool = True

    # --- demo -----------------------------------------------------------------------------
    seed_demo_data: bool = True
    demo_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
