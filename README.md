# SentinAI — Social Media Bias & Hate Speech Intelligence Platform

SentinAI ingests public social-media posts, detects and grades discriminatory content aimed at
ethnic, religious and national groups, explains every decision at the token level, tracks how
hate propagates through networks and over time, and keeps humans in the loop to correct and
retrain the models — all surfaced in a role-aware analytics dashboard.

```
 X API v2 ──► Ingestion ──► Preprocessing ──► Multi-model classification ──► Scoring / aggregation ──► Dashboard
  (M1)        queue+backoff   normalise, LID,     Task A toxicity                 accounts, bots,          React + Plotly/D3
              compliance      obfuscation, dedup  Task B targets (multi-label)    trends, networks,        RBAC, exports
                              (M2, M7)            Task C severity (ordinal 1–4)   fairness (M8–M12)        (M13)
                                                  + XAI (M4) + vision (M5)
                                                  + conversation context (M6)
                    ▲                                                                   │
                    └──────────────── HITL annotation / active learning / retrain (M14) ◄┘
```

> **Ethics notice.** Every score produced by SentinAI is a *model-estimated probability*, never a
> definitive judgment about a person. Account-level scores are only reported with ≥ 50 posts and
> confidence intervals, and the platform is designed for research and moderation triage — not
> for automated enforcement. See [Limitations & responsible use](#limitations--responsible-use).

---

## Quick start (2 minutes, no GPU, no external services)

```bash
# 1. backend (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e "backend[dev]"

# 2. dashboard (Node 20+)
(cd frontend && npm ci && npm run build)

# 3. run — seeds a realistic 2,800-post multilingual demo dataset on first start
sentinai serve            # → http://localhost:8000   (API docs at /api/docs)
```

Sign in with one of the bootstrap accounts (change them via `SENTINAI_BOOTSTRAP_*_PASSWORD`):

| user | password | role | can |
|---|---|---|---|
| `admin` | `admin` | Admin | everything (ingest, review, retrain, audit, export) |
| `researcher` | `researcher` | Researcher | all analytics views, fairness audit, export |
| `moderator` | `moderator` | Moderator | analytics, human review queue & annotation, export |

Or with Docker:

```bash
docker compose up --build                  # API + dashboard on :8000 (SQLite, demo data)
docker compose --profile db up --build     # + PostgreSQL (metadata) and MongoDB (raw payloads)
```

Run the test suites:

```bash
(cd backend && python -m pytest -q)                 # 71 tests: pipeline, analytics, API, RBAC, HITL
(cd frontend && npm run typecheck && npm test)      # strict TS + 14 render tests against a live API
```

---

## What you get out of the box

The default install is deliberately **dependency-light**: a lexicon/pattern-based classifier
with the exact same interfaces, taxonomy and outputs as the transformer stack, so the whole
platform (pipeline → storage → analytics → dashboard → HITL) is fully functional in CI, on a
laptop or in this repo's sandbox. Heavy model stacks are optional extras that plug in behind
the same interfaces (see [Optional model stacks](#optional-model-stacks)).

| Dashboard view | Module | What it shows |
|---|---|---|
| Overview | 13 | KPIs, toxic-volume sparkline with spikes, most-targeted groups, label/severity/language mix |
| Target heatmap | 3B, 3C | target demographic × severity level matrix (counts or severity mix), by category |
| Temporal trends | 8 | daily toxic volume, robust z-score spikes, Holt-Winters/ARIMA/Prophet forecast band, real-world event overlay with before/after lift |
| Propagation graph | 9 | D3 force graph of RT/quote/reply interactions; PageRank size, Louvain/Leiden communities, bot rings |
| Post explorer / explainability | 4, 6 | filters + token-level red/green attribution, Task A/B/C cards, stance & discount, thread context |
| Severity | 3C | ordinal taxonomy cards, severity × label, avg severity per target |
| Bots vs humans | 10 | P(bot) histogram, toxic ratio bot vs human, top suspicious accounts with feature signals; *exclude bots* toggle everywhere |
| Account scores | 12 | propensity score with bootstrap CI, component breakdown, ≥ 50-post gate, disclaimer |
| Classify text | 2–7 | playground: preprocessing report, context/media simulation, full classification + attribution |
| Human review | 14 | active-learning queue (0.4–0.6 band + high-severity), annotation form, Label Studio / Prodigy export, Cohen's/Fleiss' κ, retrain batches |
| Fairness audit | 11 | per-dialect FPR/FNR, gaps, calibrated group thresholds, weekly audit history |

Exports: **CSV / JSON** of classified posts and a **PDF** executive report (`/api/export/*`).

---

## Repository layout & module map

```
SentinAI/
├── backend/
│   ├── pyproject.toml                  package `sentinai`, extras: ml · vision · forecast · postgres · mongo · dev
│   ├── sentinai/
│   │   ├── config.py                   pydantic-settings (env prefix SENTINAI_), paths
│   │   ├── schemas.py                  Pydantic domain models & enums (labels, targets, severity, stance, roles)
│   │   ├── ingestion/                  M1  x_client.py (X API v2: search/tweet/user/stream), importers.py, jobs.py, media.py, worker.py, compliance.py
│   │   ├── preprocessing/              M2  normalize.py, obfuscation.py, dedup.py (MinHash/LSH)
│   │   │                               M7  language.py (FastText LID + fallback, code-switching, transliteration)
│   │   ├── classification/             M3  lexicon.py, heuristic.py (Tasks A/B/C), transformer.py (optional heads)
│   │   │                               M4  explain.py (SHAP / LIME / built-in attribution)
│   │   │                               M5  multimodal.py (OCR, CLIP, YOLOv8 symbols, ensemble)
│   │   │                               M6  context.py (originator/responder, stance, ×0.2 discount)
│   │   │                                   engine.py (orchestrates everything)
│   │   ├── analytics/                  M8  trends.py · M9 network.py · M10 bots.py · M11 fairness.py · M12 accounts.py
│   │   ├── hitl.py                     M14 review queue, annotations, κ, Label Studio/Prodigy I/O, retrain batches
│   │   ├── auth.py                     M13 JWT auth + RBAC (admin / researcher / moderator)
│   │   ├── api/                        M13 FastAPI app + routers (core, analytics, review/export), serves the SPA
│   │   ├── reporting.py                PDF executive report (reportlab)
│   │   ├── storage/                    SQLAlchemy models (SQLite/PostgreSQL), raw store (JSONL/MongoDB), repository
│   │   ├── demo.py                     deterministic multilingual demo corpus generator
│   │   ├── cli.py                      `sentinai` CLI
│   │   └── resources/                  lexicons (targets, hostility patterns, slurs) · fairness benchmarks
│   ├── scripts/                        training recipes: train_toxicity.py (A/C, adversarial debiasing),
│   │                                   train_targets.py (B), train_bot_detector.py (TwiBot-22), train_symbols.py (YOLOv8)
│   └── tests/                          pytest suite
├── frontend/                           React 18 + TypeScript + Vite; Plotly (charts) and D3 (force graph)
│   └── src/pages/                      one file per dashboard view; src/test/ render tests (vitest + jsdom)
├── Dockerfile · docker-compose.yml · .env.example
└── README.md
```

---

## How each module is implemented

### M1 — Data ingestion (real X posts)
`ingestion/x_client.py` wraps X API v2 with the expansions needed for author, engagement, reply
chains (`referenced_tweets`, `conversation_id`) and media (`attachments.media_keys`):

| what you want | dashboard (Ingest from X) | CLI | endpoint used |
|---|---|---|---|
| everything matching a query | **Search** tab | `sentinai ingest "(muslims OR islam) lang:en -is:retweet" --pages 3` | `/2/tweets/search/recent` (7-day window); `--full-archive` → `/search/all` with automatic fallback |
| one post + its thread | **Tweet URL** tab | `sentinai ingest-tweet https://x.com/<user>/status/<id> --conversation` | `/2/tweets`, `conversation_id:` search |
| an account's timeline | **User timeline** tab | `sentinai ingest-user @handle --pages 3` | `/2/users/by/username`, `/2/users/:id/tweets` |
| posts as they happen | **Live stream** tab | `sentinai ingest-stream -r "<rule>" --max-minutes 30` | `/2/tweets/search/stream` (+ rules) |
| files you already have | **Upload file** tab | `sentinai import dump.jsonl archive.zip posts.csv` | – (X API JSON/JSONL, X data archive `tweets.js`/`.zip`, CSV/TSV, plain text) |

Every path ends in the same `IngestionWorker`: parents of replies / quotes / retweets are
hydrated and processed first (so the stance detector can discount counter-speech), each post is
preprocessed and classified, bot / account scores are refreshed for the touched authors and
uncertain posts join the review queue. Per-query `since_id` cursors make re-runs incremental.
Rate limiting uses the `x-rate-limit-*` headers plus exponential backoff with jitter
(`SENTINAI_X_BACKOFF_*`); dashboard-triggered pulls stop waiting after
`SENTINAI_X_API_MAX_WAIT_SECONDS` and report what they got. Long pulls and the stream run as
background jobs (`GET /api/ingest/jobs`). Access-level errors are translated into actionable
messages (bad token vs. endpoint not included in the plan). Optional media download
(`SENTINAI_X_DOWNLOAD_MEDIA` / "download images") stores photos under `data/media` so Module 5 can
OCR them. Metadata goes to **PostgreSQL/SQLite** (SQLAlchemy), raw JSON to **MongoDB** or the
relational raw store. `ingestion/compliance.py` implements the X compliance contract (mark posts
deleted upstream, never re-serve them), GDPR/CCPA-style retention (`SENTINAI_RETENTION_DAYS`) and
author anonymisation past the window (`sentinai compliance --check-deleted`).

> **X API access (2026):** the official API is pay-per-use; recent search works on every paid
> plan, while full-archive search and the filtered stream need an upgraded access level. SentinAI
> defaults to recent search and falls back automatically, and the file importers work with no
> token at all. The bearer token can be set with `SENTINAI_X_BEARER_TOKEN` or pasted in the
> dashboard (stored in memory, or under `data/secrets/` with mode 600 when "remember" is ticked).

### M2 — Preprocessing
Lowercasing, URL removal, `@mention → @user` anonymisation, hashtag segmentation
(`#StopTheInvasion → stop the invasion`, wordninja), emoji → text, and obfuscation detection:
leetspeak (`k1ll`, `j3ws`), homoglyphs (Cyrillic/Greek look-alikes), zero-width characters,
spaced-out words (`n i g g e r`), repeated characters, plus fuzzy repair of misspelled slurs
guarded by a 126k-word English vocabulary so real words (`china`, `nation`) are never rewritten.
Near-duplicates are caught with MinHash + LSH (`datasketch`) and carry `duplicate_of`.

### M3 — Multi-model classification
Three heads share one interface (`ClassificationEngine.classify_post`):

* **Task A — toxicity**: `non_toxic / offensive / hate_speech / violent_extremism` with a
  calibrated probability `toxicity_score ∈ [0,1]`.
* **Task B — targets**: multi-label over `ethnicity / religion / nationality` with 40+ groups
  (`resources/lexicons/targets.yaml`, multilingual surface forms).
* **Task C — severity**: ordinal levels 1 microaggression → 2 dehumanization/exclusion →
  3 slurs/harassment → 4 incitement, produced CORAL-style from cumulative probabilities.

The default **heuristic backend** uses the weighted lexicon + 200+ hostility patterns; the
**transformer backend** (`SENTINAI_CLASSIFIER_BACKEND=transformer`, `[ml]` extra) loads
fine-tuned DeBERTa-v3-large / HateBERT (A), XLM-R-large (B) and a CORAL head (C) from
`SENTINAI_MODEL_CACHE_DIR`, ensembles them with the lexicon prior and falls back per-head when a
checkpoint is missing (`model_version` tells you which heads are live).

### M4 — Explainability
Every classification carries token-level attributions (`explanation.attributions`) rendered as a
red (drives toxicity) / green (mitigates) overlay. With the `[ml]` extra SHAP's partition
explainer runs over the transformer pipeline; LIME is available for any scorer; the heuristic
backend emits exact pattern/lexicon contributions, which are also listed under "matched rules".

### M5 — Multimodal
`classification/multimodal.py` runs Tesseract/EasyOCR on images, CLIP zero-shot prompts for
hateful iconography, and a custom YOLOv8 hate-symbol detector (19 classes, `train_symbols.py`).
Visual and textual scores are fused with a weighted ensemble (`SENTINAI_VISUAL_ENSEMBLE_WEIGHT`),
and a benign caption on top of a hateful image raises `juxtaposition_flag`. Without the
`[vision]` extra, alt-text/OCR text supplied with the post is still scored.

### M6 — Contextual analysis
`context.py` distinguishes originators from responders in reply chains, detects stance
(`support / deny / condemn / query / comment / amplify`), tells quote-tweet amplification from
criticism, and applies the ×0.2 discount (`SENTINAI_COUNTERSPEECH_DISCOUNT`) when a toxic-looking
reply *denies or condemns* the parent. The dashboard shows stance, discount and the thread.

### M7 — Multilingual
Language ID via FastText `lid.176` when present (`SENTINAI_FASTTEXT_LID_PATH`), otherwise a
script + stop-word model over en, es, ar, hi, fr, pt, de. Code-switching is detected per token
(Spanglish, Hinglish). Romanised Arabic (Arabizi, e.g. `o2tolo kol el yahood`) and Romanised
Hindi are transliterated before lexicon matching; the transformer backend routes non-English
text to the mDeBERTa/XLM-R heads.

### M8 — Temporal analysis
`analytics/trends.py` aggregates daily/hourly toxic volume per target, flags spikes with a
rolling median/MAD z-score (`SENTINAI_SPIKE_ZSCORE_THRESHOLD`, default 2.5), forecasts with
Prophet → statsmodels Holt-Winters/ARIMA → naive fallback, and overlays events from GDELT
(no key), NewsAPI (`SENTINAI_NEWSAPI_KEY`) or manual entry, computing the before/after lift.

### M9 — Network analysis
`analytics/network.py` builds a NetworkX digraph from retweets, quotes and replies around toxic
content, runs Leiden (if `leidenalg` is installed) or Louvain, computes PageRank & betweenness,
and returns the payload rendered by the D3 force graph (node size = PageRank, colour =
community, red ring = likely bot).

### M10 — Bot detection
Features: account age, posting frequency, follower/following ratio, profile completeness,
timing regularity, content repetitiveness, burstiness, night-time share, retweet/reply ratio,
username digits, default avatar. A RandomForest/XGBoost trained on TwiBot-22
(`scripts/train_bot_detector.py`) is used when `models/bots/random_forest.joblib` exists;
otherwise a calibrated logistic model over the same features. `P_bot` is stored per account,
flagged at ≥ 0.8 and every analytics endpoint accepts `exclude_bots=true`.

### M11 — Bias mitigation
`analytics/fairness.py` evaluates per-dialect FPR/FNR (AAVE, Chicano English, Hinglish,
Arabizi, British, SAE) on bundled BOLD/ToxiGen-style benchmarks
(`resources/benchmarks/*.jsonl` — add your own), derives **group-specific thresholds** that
equalise FPR against the reference group, records a **weekly FP audit** on human-annotated
posts, and `scripts/train_toxicity.py --adversarial` implements gradient-reversal adversarial
debiasing against the dialect tag.

### M12 — Account scoring
`Score = 0.4·AvgSeverity + 0.3·ToxicRatio + 0.2·TargetDiversity + 0.1·Recency`, with
exponential decay (`SENTINAI_ACCOUNT_DECAY_HALF_LIFE_DAYS`), a **bootstrap 95 % CI**, a hard
minimum of 50 posts (`SENTINAI_ACCOUNT_MIN_POSTS`) before a score is reported, and the
"Model-Estimated Probability" disclaimer attached to every payload and view.

### M13 — Dashboard & API
FastAPI (`/api/docs`) + React/TypeScript SPA (served by the API in production, Vite dev proxy
in development). JWT auth with three roles; the navigation and routes are gated by permission
(`read · review · annotate · export · ingest · retrain · audit · manage_users`). Views: overview,
target heatmap, temporal trends, propagation graph, post explorer + explainability viewer,
severity, bots vs humans, account scores, classify playground, human review, fairness audit and
— for admins — **Ingest from X** (connect a token, search / tweet URL / user timeline / live
stream / file upload, job monitor, saved cursors). Ingestion endpoints live under
`/api/ingest/*` (`status`, `x/token`, `x/test`, `x/search`, `x/tweet`, `x/user`,
`x/stream/{rules,start,stop}`, `upload`, `jobs`, `cursors`).

### M14 — Human-in-the-loop
Posts with model confidence in the **0.4–0.6 band** (plus high-severity items) are queued for
review; annotations record agreement with the model; **Cohen's κ** (pairwise) and Fleiss' κ are
tracked; tasks export/import in **Label Studio** and **Prodigy** formats; unused annotations are
bundled into JSONL **retraining batches** (`sentinai retrain-batch`, intended bi-weekly) that
feed `scripts/train_toxicity.py` directly.

---

## Configuration

All settings are environment variables prefixed `SENTINAI_` (see `.env.example`). Highlights:

| variable | default | purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/sentinai.db` | metadata DB (`postgresql+psycopg://…` with `[postgres]`) |
| `RAW_STORE_URL` | JSONL under `DATA_DIR` | `mongodb://…` with `[mongo]` |
| `X_BEARER_TOKEN` | – | enables live ingestion (or paste it in the dashboard) |
| `X_FULL_ARCHIVE` | `false` | prefer `/search/all`; falls back to recent search when not allowed |
| `X_DOWNLOAD_MEDIA` | `false` | fetch photo attachments into `data/media` for OCR / vision |
| `X_API_MAX_WAIT_SECONDS` | `45` | max rate-limit wait for dashboard-triggered pulls |
| `CLASSIFIER_BACKEND` | `heuristic` | `transformer` with `[ml]` + checkpoints |
| `MODEL_CACHE_DIR` | `models/` | fine-tuned heads, bot model, YOLO weights |
| `FASTTEXT_LID_PATH` | – | path to `lid.176.bin/ftz` |
| `COUNTERSPEECH_DISCOUNT` | `0.2` | M6 discount factor |
| `ACCOUNT_MIN_POSTS` | `50` | M12 reliability gate |
| `ACTIVE_LEARNING_LOW/HIGH` | `0.4 / 0.6` | M14 uncertainty band |
| `RETENTION_DAYS` | `90` | compliance retention window for raw payloads |
| `JWT_SECRET` | dev value | **set in production** |
| `SEED_DEMO_DATA` | `true` | seed the demo corpus when the DB is empty |

## CLI

```
sentinai serve                       run the API (+ dashboard if frontend/dist exists)
sentinai seed --n-posts 1600 --reset (re)generate the demo corpus
sentinai classify "text"             classify one string with explanation
sentinai ingest "query" --pages 5    search X (recent; --full-archive) into the pipeline
sentinai ingest-tweet <url|id>...    one or more tweets (+ parent chain, --conversation)
sentinai ingest-user @handle         a user's timeline
sentinai ingest-stream -r "<rule>"   filtered stream until Ctrl-C / --max-minutes
sentinai import <files>...           X API JSON/JSONL, X archive (.zip/tweets.js), CSV, text
sentinai compliance --check-deleted  retention sweep + upstream deletion sync
sentinai rescore                     recompute bot and account scores
sentinai fairness                    print the benchmark fairness report
sentinai weekly-audit                FP audit on the last 7 days of human labels
sentinai retrain-batch               bundle unused annotations for fine-tuning
sentinai sync-events "query"         pull GDELT/NewsAPI events for the trend overlay
```

## Optional model stacks

| extra | installs | unlocks |
|---|---|---|
| `pip install -e "backend[ml]"` | torch, transformers, shap, lime, fasttext | transformer heads (A/B/C), SHAP/LIME, FastText LID, adversarial training |
| `pip install -e "backend[vision]"` | pytesseract, easyocr, open_clip, ultralytics | OCR, CLIP zero-shot, YOLOv8 symbol detector |
| `pip install -e "backend[forecast]"` | prophet | Prophet forecasts (else Holt-Winters/ARIMA) |
| `pip install -e "backend[postgres]"` / `[mongo]` | psycopg / pymongo | production storage |

Training recipes live in `backend/scripts/` and write artefacts in the layout the runtime
expects under `SENTINAI_MODEL_CACHE_DIR`; datasets (TwiBot-22, LinCE, ToxiGen, BOLD, hate-symbol
imagery) must be obtained under their own licences and are never committed.

## Development

```bash
# API with auto-reload
(cd backend && sentinai serve --reload)
# dashboard with hot reload, proxying /api → :8000
(cd frontend && npm run dev)        # http://localhost:5173
```

## Limitations & responsible use

* The default heuristic classifier is precise on explicit hostility and robust to common
  obfuscation, but it is a **pattern model**: implicit or novel hate will be under-detected until
  the transformer heads are trained on your annotated data (that is exactly what the HITL loop
  and `scripts/train_*.py` are for). Treat its numbers as a demonstration of the platform, not
  as benchmark accuracy.
* Dialect and identity-term bias is measured, not solved. Review the fairness view before
  changing thresholds and keep the weekly audit running once real annotations exist.
* Account scores are aggregate, decayed, interval-estimated **model outputs** about observed
  posts. They must not be the sole basis for sanctions and should never be shown without the
  accompanying disclaimer and confidence interval.
* Respect platform terms and data-protection law: honour deletion/compliance events, set a
  retention window, and avoid storing more personal data than the analysis requires.

## License

See [LICENSE](LICENSE).
