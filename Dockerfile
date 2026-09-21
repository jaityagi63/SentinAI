# ---------- stage 1: build the React dashboard ----------
FROM node:22-alpine AS dashboard
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------- stage 2: API + dashboard in one image ----------
FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Build deps for psycopg / lxml-free wheels are not required for the core stack; keep the image small.
COPY README.md ./
COPY backend/pyproject.toml ./backend/
COPY backend/sentinai ./backend/sentinai
RUN pip install --upgrade pip && pip install -e "./backend[postgres,mongo]"

COPY --from=dashboard /app/frontend/dist ./frontend/dist

# Runtime data (SQLite DB, raw JSONL store, exports) lives in /app/data — mount a volume for persistence.
RUN mkdir -p /app/data
ENV SENTINAI_DATA_DIR=/app/data \
    SENTINAI_DATABASE_URL=sqlite:////app/data/sentinai.db \
    SENTINAI_HOST=0.0.0.0 \
    SENTINAI_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["uvicorn", "sentinai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
