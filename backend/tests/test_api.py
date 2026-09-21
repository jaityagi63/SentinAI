"""Module 13 — API + RBAC + exports; Module 1 — ingestion client (mocked transport)."""

import json
import time
from datetime import datetime

import httpx

from sentinai.ingestion.x_client import XClient, parse_search_response


def test_health_and_taxonomy(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["posts"] > 0
    t = client.get("/api/meta/taxonomy").json()
    assert set(t["targets"]) == {"ethnicity", "religion", "nationality"} and len(t["severity_levels"]) == 5


def test_rbac(client, admin, researcher, moderator):
    assert client.get("/api/analytics/overview").status_code == 401
    assert client.get("/api/analytics/overview", headers=researcher).status_code == 200
    assert client.get("/api/review/queue", headers=researcher).status_code == 403
    assert client.get("/api/review/queue", headers=moderator).status_code == 200
    assert client.get("/api/analytics/fairness", headers=moderator).status_code == 403
    assert client.post("/api/review/retrain-batch", headers=moderator).status_code == 403
    assert client.post("/api/review/retrain-batch", headers=admin).status_code == 200
    assert client.post("/api/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    assert client.get("/api/auth/me", headers=admin).json()["role"] == "admin"


def test_auth_token_transports(client):
    """The JWT is accepted as bearer header, custom header or cookie (proxies may drop `Authorization`)."""
    from fastapi.testclient import TestClient

    from sentinai.api.app import app

    with TestClient(app) as fresh:  # isolated cookie jar
        r = fresh.post("/api/auth/login", json={"username": "moderator", "password": "moderator"})
        assert r.status_code == 200
        token = r.json()["access_token"]
        assert "sentinai_token" in r.cookies  # login also sets the session cookie
        # 1. cookie only (no header at all)
        assert fresh.get("/api/auth/me").json()["username"] == "moderator"
        fresh.cookies.clear()
        assert fresh.get("/api/auth/me").status_code == 401
        # 2. custom header only
        assert fresh.get("/api/auth/me", headers={"X-SentinAI-Token": token}).status_code == 200
        # 3. a proxy replaced the bearer header with garbage but the custom header survives
        assert fresh.get("/api/auth/me", headers={"Authorization": "Bearer proxy-injected", "X-SentinAI-Token": token}).status_code == 200
        # 4. garbage everywhere → 401 (not a 500)
        assert fresh.get("/api/auth/me", headers={"Authorization": "Bearer nope", "X-SentinAI-Token": "nope"}).status_code == 401
        # 5. logout clears the cookie
        fresh.post("/api/auth/login", json={"username": "moderator", "password": "moderator"})
        assert fresh.get("/api/auth/me").status_code == 200
        fresh.post("/api/auth/logout")
        assert fresh.get("/api/auth/me").status_code == 401


def test_dashboard_endpoints(client, admin):
    o = client.get("/api/analytics/overview?days=90", headers=admin).json()
    assert o["posts"] > 0 and o["toxic_posts"] > 0 and set(o["by_severity"]) >= {0, 2, 3, 4} or set(map(str, o["by_severity"])) >= {"0", "2"}
    h = client.get("/api/analytics/heatmap", headers=admin).json()
    assert h["cells"] and set(h["by_category"]) <= {"ethnicity", "religion", "nationality"}
    t = client.get("/api/analytics/trends?days=90&horizon=7", headers=admin).json()
    assert len(t["series"]) >= 60 and len(t["forecast"]) == 7 and t["events"]
    tt = client.get("/api/analytics/trends/targets?top=3", headers=admin).json()
    assert len(tt["targets"]) == 3
    n = client.get("/api/analytics/network", headers=admin).json()
    assert n["stats"]["nodes"] > 0 and n["nodes"] and n["links"]
    s = client.get("/api/analytics/severity", headers=admin).json()
    assert "matrix" in s and s["per_target"]
    b = client.get("/api/analytics/bots", headers=admin).json()
    assert b["scored_accounts"] > 0 and "bot" in b["breakdown"] and "human" in b["breakdown"]
    a = client.get("/api/analytics/accounts?limit=5", headers=admin).json()
    assert a["items"] and a["disclaimer"].startswith("Model-Estimated")
    one = client.get(f"/api/analytics/accounts/{a['items'][0]['author_id']}", headers=admin).json()
    assert one["score"]["author_id"] == a["items"][0]["author_id"] and one["bot"] is not None
    ex = client.get("/api/analytics/bots?exclude_bots=true", headers=admin)
    assert ex.status_code == 200


def test_posts_and_detail(client, admin):
    p = client.get("/api/posts?min_toxicity=0.7&order=toxicity&limit=5", headers=admin).json()
    assert p["total"] > 0 and p["items"][0]["final_toxicity"] >= 0.7
    d = client.get(f"/api/posts/{p['items'][0]['id']}", headers=admin).json()
    assert d["classification"]["explanation"]["attributions"]
    assert client.get("/api/posts/does-not-exist", headers=admin).status_code == 404
    f = client.get("/api/posts?target=religion:islam&limit=3", headers=admin).json()
    assert all("religion:islam" in it["targets"] for it in f["items"])
    r = client.post(f"/api/posts/{p['items'][0]['id']}/reclassify", headers=admin)
    assert r.status_code == 200 and r.json()["post_id"] == p["items"][0]["id"]


def test_classify_endpoint(client, researcher):
    r = client.post("/api/classify", json={"text": "filthy muslims are a cancer on this country"}, headers=researcher)
    assert r.status_code == 200
    c = r.json()["classification"]
    assert c["toxicity"]["label"] == "hate_speech" and c["severity"]["level"] == 3
    assert any(t["label"] == "islam" for t in c["targets"])
    assert c["explanation"]["attributions"]
    r = client.post("/api/classify", json={"text": "this is disgusting racism, reported", "is_quote": True, "parent_toxicity": 0.9}, headers=researcher).json()["classification"]
    assert r["context"]["stance"] == "condemn"


def test_ingest_endpoint_and_thread_context(client, admin):
    now = datetime.utcnow().isoformat()
    posts = [
        {"id": "t-parent", "text": "kill all the muslims", "created_at": now, "author_id": "ing-1", "author_username": "hater"},
        {"id": "t-child", "text": "This is vile hate speech. Reported.", "created_at": now, "author_id": "ing-2", "author_username": "critic", "quoted_id": "t-parent"},
        {"id": "t-amp", "text": "RT @hater: kill all the muslims", "created_at": now, "author_id": "ing-3", "retweeted_id": "t-parent"},
    ]
    r = client.post("/api/ingest/posts", json={"posts": posts}, headers=admin)
    assert r.status_code == 200 and r.json()["ingested"] == 3
    child = client.get("/api/posts/t-child", headers=admin).json()
    assert child["stance"] == "condemn" and child["final_toxicity"] < 0.3 and child["parent"]["id"] == "t-parent"
    parent = client.get("/api/posts/t-parent", headers=admin).json()
    assert parent["toxicity_label"] == "violent_extremism" and len(parent["replies"]) >= 1


def test_review_flow(client, moderator, admin):
    q = client.get("/api/review/queue?limit=5", headers=moderator).json()
    assert q["items"] and q["stats"]["pending"] > 0
    item = q["items"][0]
    r = client.post("/api/review/annotate", json={"post_id": item["post_id"], "toxicity_label": "hate_speech", "severity": 2, "targets": ["religion:islam"]}, headers=moderator)
    assert r.status_code == 200 and "agrees_with_model" in r.json()
    q2 = client.get("/api/review/queue?limit=5", headers=moderator).json()
    assert all(it["post_id"] != item["post_id"] for it in q2["items"])
    if q2["items"]:
        assert client.post(f"/api/review/{q2['items'][0]['id']}/skip", headers=moderator).status_code == 200
    ls = client.get("/api/review/export/label-studio?limit=3", headers=moderator).json()
    assert "<View>" in ls["label_config"] and len(ls["tasks"]) <= 3
    if ls["tasks"]:
        task = ls["tasks"][0]
        imp = client.post("/api/review/import/label-studio", json=[{"data": task["data"], "annotations": [{"completed_by": 42, "result": [{"from_name": "toxicity", "to_name": "text", "type": "choices", "value": {"choices": ["offensive"]}}]}]}], headers=moderator)
        assert imp.status_code == 200 and imp.json()["imported"] == 1
    pr = client.get("/api/review/export/prodigy?limit=2", headers=moderator)
    assert pr.status_code == 200 and pr.headers["content-type"].startswith("application/x-ndjson")
    assert client.get("/api/review/agreement", headers=admin).status_code == 200
    assert client.get("/api/review/stats", headers=admin).json()["annotations"] >= 1


def test_exports(client, researcher):
    csv = client.get("/api/export/posts.csv?limit=20", headers=researcher)
    assert csv.status_code == 200 and csv.text.startswith("post_id,") and len(csv.text.splitlines()) > 2
    js = client.get("/api/export/posts.json?limit=5", headers=researcher).json()
    assert len(js["items"]) == 5 and "disclaimer" in js
    pdf = client.get("/api/export/report.pdf", headers=researcher)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_fairness_endpoint(client, researcher, admin):
    f = client.get("/api/analytics/fairness", headers=researcher).json()
    assert f["benchmarks"] and f["benchmarks"][0]["groups"]
    a = client.post("/api/analytics/fairness/audit?days=365", headers=admin)
    assert a.status_code == 200 and "groups" in a.json()


def test_events_crud(client, admin):
    r = client.post("/api/analytics/events", json={"date": "2026-08-01T00:00:00", "title": "Test event", "category": "test"}, headers=admin)
    assert r.status_code == 200 and r.json()["created"] == 1
    evs = client.get("/api/analytics/events", headers=admin).json()
    assert any(e["title"] == "Test event" for e in evs)


# --------------------------------------------------------------------------------------------
# Module 1 — X client with mocked transport
# --------------------------------------------------------------------------------------------

SAMPLE = {
    "data": [
        {"id": "1", "text": "kill all the muslims", "created_at": "2026-09-01T10:00:00.000Z", "author_id": "a1", "lang": "en", "public_metrics": {"like_count": 3, "retweet_count": 1, "reply_count": 0, "quote_count": 0}, "attachments": {"media_keys": ["3_1"]}},
        {"id": "2", "text": "this is vile, reported", "created_at": "2026-09-01T10:05:00.000Z", "author_id": "a2", "referenced_tweets": [{"type": "quoted", "id": "1"}], "public_metrics": {}},
    ],
    "includes": {"users": [{"id": "a1", "username": "hater", "name": "H", "created_at": "2026-08-01T00:00:00.000Z", "public_metrics": {"followers_count": 5, "following_count": 900, "tweet_count": 8000}, "profile_image_url": "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"}], "media": [{"media_key": "3_1", "type": "photo", "url": "https://pbs.twimg.com/x.jpg", "alt_text": "meme text 1488"}]},
    "meta": {"result_count": 2},
}


def test_parse_search_response():
    posts = parse_search_response(SAMPLE)
    assert len(posts) == 2
    assert posts[0].author.username == "hater" and posts[0].author.has_default_profile_image
    assert posts[0].media[0].alt_text == "meme text 1488"
    assert posts[1].quoted_id == "1" and posts[1].parent_id is None


def test_client_backoff_and_rate_limits(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"x-rate-limit-remaining": "0", "x-rate-limit-reset": str(int(time.time()) + 2)}, json={"title": "Too Many Requests"})
        if calls["n"] == 2:
            return httpx.Response(503, json={"title": "Service Unavailable"})
        page = dict(SAMPLE)
        page["meta"] = {"result_count": 2, "next_token": "abc"} if "next_token" not in request.url.params else {"result_count": 2}
        return httpx.Response(200, headers={"x-rate-limit-remaining": "299", "x-rate-limit-reset": str(int(time.time()) + 900)}, json=page)

    client = XClient(bearer_token="test", base_url="https://api.x.com/2", transport=httpx.MockTransport(handler), sleep=sleeps.append)
    posts = list(client.search("muslims", max_pages=3, full_archive=True))
    assert len(posts) == 4  # two pages
    assert calls["n"] == 4  # 429, 503, page1, page2
    assert len(sleeps) == 2 and sleeps[0] >= 1
    assert client.rate["/tweets/search/all"].remaining == 299
    # queue management
    seen = []
    client.enqueue("/tweets", {"ids": "1"}, callback=lambda d: seen.append(len(d["data"])))
    client.enqueue("/tweets", {"ids": "2"}, callback=lambda d: seen.append(len(d["data"])))
    assert client.queue_size == 2 and client.drain() == 2 and seen == [2, 2] and client.queue_size == 0


def test_compliance_sync_deletions(seeded_db):
    from sentinai.ingestion.compliance import enforce_retention, sync_deletions
    from sentinai.storage.db import session_scope
    from sentinai.storage.models import PostRow

    class FakeClient:
        def request(self, path, params):
            ids = params["ids"].split(",")
            return {"data": [{"id": i} for i in ids[1:]], "errors": [{"resource_id": ids[0], "title": "Not Found Error"}]}

    with session_scope() as s:
        ids = [r for r in s.query(PostRow.id).limit(3).all()]
        ids = [i[0] for i in ids]
        n = sync_deletions(s, FakeClient(), ids)
        assert n == 1
        assert s.get(PostRow, ids[0]).deleted_upstream and s.get(PostRow, ids[0]).text == "[deleted upstream]"
        res = enforce_retention(s, days=3650)
        assert set(res) == {"purged_raw_payloads", "anonymised_authors"}
    json.dumps(res)
