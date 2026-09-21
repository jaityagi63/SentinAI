"""Module 1 — real-X ingestion paths: single tweet / URL, user timeline, search with tier
fallback, filtered stream, bulk file imports and the /api/ingest/* endpoints (mocked X API)."""

from __future__ import annotations

import io
import json
import time
import zipfile
from datetime import datetime
from urllib.parse import parse_qs

import httpx
import pytest

from sentinai.ingestion import importers
from sentinai.ingestion.x_client import XAccessError, XClient, XNotFoundError, XRateLimitError, parse_search_response, parse_tweet_ref, parse_username

NOW = "2026-09-20T10:00:00.000Z"


def _user(uid: str, name: str) -> dict:
    return {"id": uid, "username": name, "name": name.title(), "created_at": "2020-01-01T00:00:00.000Z", "public_metrics": {"followers_count": 10, "following_count": 20, "tweet_count": 300}, "profile_image_url": "https://pbs.twimg.com/p.jpg"}


def _tweet(tid: str, text: str, uid: str = "u1", **extra) -> dict:
    return {"id": tid, "text": text, "created_at": NOW, "author_id": uid, "lang": "en", "conversation_id": extra.pop("conversation_id", tid), "public_metrics": {"like_count": 1, "retweet_count": 0, "reply_count": 0, "quote_count": 0}, **extra}


class FakeX:
    """Minimal X API v2 stand-in used through httpx.MockTransport."""

    def __init__(self, *, full_archive: bool = True, rate_limit_after: int | None = None):
        self.full_archive = full_archive
        self.calls: list[str] = []
        self.rules: list[dict] = []
        self.rate_limit_after = rate_limit_after
        self.tweets = {
            "9001": _tweet("9001", "kill all the muslims, every last one", "u1"),
            "9002": _tweet("9002", "This is vile — reported.", "u2", conversation_id="9001", referenced_tweets=[{"type": "quoted", "id": "9001"}]),
            "9003": _tweet("9003", "lol true", "u3", conversation_id="9001", referenced_tweets=[{"type": "replied_to", "id": "9001"}]),
            "9004": _tweet("9004", "the weather is nice today", "u2"),
        }
        self.users = {"u1": _user("u1", "hater"), "u2": _user("u2", "critic"), "u3": _user("u3", "amp")}

    def page(self, ids: list[str], next_token: str | None = None) -> dict:
        data = [self.tweets[i] for i in ids if i in self.tweets]
        uids = {t["author_id"] for t in data}
        meta = {"result_count": len(data)}
        if next_token:
            meta["next_token"] = next_token
        return {"data": data, "includes": {"users": [self.users[u] for u in uids]}, "meta": meta}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.replace("/2", "", 1)
        q = parse_qs(request.url.query.decode())
        self.calls.append(f"{request.method} {path}")
        if self.rate_limit_after is not None and len(self.calls) > self.rate_limit_after:
            return httpx.Response(429, headers={"x-rate-limit-remaining": "0", "x-rate-limit-reset": str(int(time.time()) + 900)}, json={"title": "Too Many Requests"})
        hdr = {"x-rate-limit-remaining": "59", "x-rate-limit-reset": str(int(time.time()) + 900)}
        if path == "/tweets/search/all":
            if not self.full_archive:
                return httpx.Response(403, json={"title": "Client Forbidden", "detail": "full-archive not available"})
            return httpx.Response(200, headers=hdr, json=self.page(["9001", "9002", "9003"]))
        if path == "/tweets/search/recent":
            query = q["query"][0]
            if query.startswith("conversation_id:"):
                return httpx.Response(200, headers=hdr, json=self.page(["9002", "9003"]))
            if "next_token" in q:
                return httpx.Response(200, headers=hdr, json=self.page(["9004"]))
            return httpx.Response(200, headers=hdr, json=self.page(["9001", "9002"], next_token="n2"))
        if path == "/tweets":
            ids = q["ids"][0].split(",")
            body = self.page(ids)
            missing = [i for i in ids if i not in self.tweets]
            if missing:
                body["errors"] = [{"resource_id": m, "title": "Not Found Error"} for m in missing]
            return httpx.Response(200, headers=hdr, json=body)
        if path.startswith("/users/by/username/"):
            name = path.rsplit("/", 1)[1]
            for u in self.users.values():
                if u["username"] == name:
                    return httpx.Response(200, headers=hdr, json={"data": u})
            return httpx.Response(200, headers=hdr, json={"errors": [{"detail": f"Could not find user with username: [{name}]", "title": "Not Found Error"}]})
        if path.startswith("/users/") and path.endswith("/tweets"):
            uid = path.split("/")[2]
            ids = [i for i, t in self.tweets.items() if t["author_id"] == uid]
            return httpx.Response(200, headers=hdr, json=self.page(ids))
        if path == "/tweets/search/stream/rules":
            if request.method == "GET":
                return httpx.Response(200, headers=hdr, json={"data": self.rules})
            body = json.loads(request.content)
            if "delete" in body:
                self.rules = [r for r in self.rules if r["id"] not in body["delete"]["ids"]]
            for i, r in enumerate(body.get("add", [])):
                self.rules.append({"id": f"r{i}", **r})
            return httpx.Response(201, headers=hdr, json={"data": self.rules})
        if path == "/tweets/search/stream":
            lines = []
            for tid in ("9001", "9002", "9003"):
                t = self.tweets[tid]
                lines.append(json.dumps({"data": t, "includes": {"users": [self.users[t["author_id"]]]}, "matching_rules": [{"id": "r0", "tag": "hate"}]}))
            payload = "\r\n".join(lines) + "\r\n\r\n"
            return httpx.Response(200, headers=hdr, content=payload.encode())
        return httpx.Response(404, json={"title": "Not Found"})

    def client(self, **kw) -> XClient:
        kw.setdefault("sleep", lambda s: None)
        return XClient(bearer_token="test-token-0123456789", base_url="https://api.x.com/2", transport=httpx.MockTransport(self.handler), **kw)


# --------------------------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------------------------


def test_parse_refs():
    assert parse_tweet_ref("https://x.com/jack/status/20") == "20"  # the very first tweet
    assert parse_tweet_ref("https://x.com/jack/status/1234567890123456789?s=20&t=abc") == "1234567890123456789"
    assert parse_tweet_ref("twitter.com/i/web/status/1234567890") == "1234567890"
    assert parse_tweet_ref("https://mobile.twitter.com/jack/statuses/1234567890/") == "1234567890"
    assert parse_tweet_ref("1234567890") == "1234567890" and parse_tweet_ref("hello world") is None
    assert parse_username("@Jack") == "Jack" and parse_username("https://x.com/jack/") == "jack" and parse_username("no spaces") is None


def test_get_tweet_and_conversation():
    fx = FakeX()
    c = fx.client()
    p = c.get_tweet("https://x.com/hater/status/9001")
    assert p.id == "9001" and p.author.username == "hater"
    with pytest.raises(XNotFoundError):
        c.get_tweet("123456")
    with pytest.raises(ValueError):
        c.get_tweet("not a tweet")
    replies = c.conversation("9001")
    assert {r.id for r in replies} == {"9002", "9003"}


def test_user_timeline_and_lookup():
    fx = FakeX()
    c = fx.client()
    posts = list(c.user_timeline("@critic", max_pages=1))
    assert {p.id for p in posts} == {"9002", "9004"} and all(p.author and p.author.username == "critic" for p in posts)
    with pytest.raises(XNotFoundError):
        list(c.user_timeline("nobody"))


def test_full_archive_fallback_to_recent():
    fx = FakeX(full_archive=False)
    c = fx.client(full_archive=True)
    posts = list(c.search("muslims", max_pages=5))
    assert [p.id for p in posts] == ["9001", "9002", "9004"]
    assert c.full_archive_available is False
    assert fx.calls[:2] == ["GET /tweets/search/all", "GET /tweets/search/recent"]
    # a second search goes straight to /recent
    list(c.search("jews", max_pages=1))
    assert fx.calls[-1] == "GET /tweets/search/recent" and "GET /tweets/search/all" not in fx.calls[2:]


def test_access_error_and_bounded_wait():
    fx = FakeX(rate_limit_after=1)
    c = fx.client(max_wait=10)
    first = list(c.search("x", max_pages=1, full_archive=False))
    assert len(first) == 2
    with pytest.raises(XRateLimitError) as exc:
        list(c.search("y", max_pages=1, full_archive=False))
    assert exc.value.reset_at > time.time() and "rate limit" in str(exc.value).lower()

    def forbidden(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"title": "Unauthorized", "detail": "bad token"})

    c2 = XClient(bearer_token="test-token-0123456789", transport=httpx.MockTransport(forbidden), sleep=lambda s: None)
    with pytest.raises(XAccessError) as exc2:
        c2.get_tweet("9001")
    assert exc2.value.status_code == 401 and "SENTINAI_X_BEARER_TOKEN" in str(exc2.value)


def test_stream_rules_and_consume():
    fx = FakeX()
    c = fx.client()
    rules = c.set_stream_rules([{"value": "muslims lang:en -is:retweet", "tag": "hate"}])
    assert rules and rules[0]["value"].startswith("muslims")
    posts = list(c.stream(max_posts=3))
    assert [p.id for p in posts] == ["9001", "9002", "9003"]
    assert all(p.source == "stream" for p in posts) and posts[0].raw["matching_rules"][0]["tag"] == "hate"


def test_parse_includes_referenced_tweets():
    page = {"data": [_tweet("1", "reply text", "u2", referenced_tweets=[{"type": "replied_to", "id": "0"}])], "includes": {"users": [_user("u2", "critic")], "tweets": [_tweet("0", "parent text", "u1")]}}
    posts = parse_search_response(page)
    assert [p.id for p in posts] == ["1", "0"] and posts[1].author is None
    assert len(parse_search_response(page, include_referenced=False)) == 1


# --------------------------------------------------------------------------------------------
# importers
# --------------------------------------------------------------------------------------------


def test_import_csv_and_jsonl_and_archive():
    csv_text = "tweet_id,text,created_at,username,likes,url\n" "111111,\"Those people are all criminals\",2026-09-01 10:00,angry_guy,4,\n" ",\"no id but a permalink\",2026-09-02,someone,0,https://x.com/someone/status/222222\n" ",\"no id at all\",,anon,,\n"
    res = importers.import_bytes(csv_text.encode(), "posts.csv")
    assert res.format == "csv" and [p.id for p in res.posts][:2] == ["111111", "222222"]
    assert res.posts[2].id.startswith("csv-") and res.posts[0].author.username == "angry_guy" and res.posts[0].engagement.like_count == 4
    assert res.posts[0].created_at == datetime(2026, 9, 1, 10, 0)

    lines = [json.dumps({"data": _tweet("5", "hello", "u1"), "includes": {"users": [_user("u1", "hater")]}}), json.dumps(_tweet("6", "world", "u1")), "not json"]
    res = importers.import_bytes("\n".join(lines).encode(), "dump.jsonl")
    assert res.format == "jsonl" and [p.id for p in res.posts] == ["5", "6"] and res.skipped == 1 and res.posts[0].author.username == "hater"

    legacy = {"id_str": "777", "full_text": "old tweet from the archive", "created_at": "Wed Oct 10 20:19:24 +0000 2018", "favorite_count": "3", "retweet_count": "1", "in_reply_to_status_id_str": "700", "lang": "en", "entities": {"media": [{"id_str": "m1", "type": "photo", "media_url_https": "https://pbs.twimg.com/m.jpg"}]}}
    js = "window.YTD.tweets.part0 = " + json.dumps([{"tweet": legacy}])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data/account.js", "window.YTD.account.part0 = " + json.dumps([{"account": {"accountId": "42", "username": "me", "accountDisplayName": "Me"}}]))
        zf.writestr("data/tweets.js", js)
    res = importers.import_bytes(buf.getvalue(), "twitter-archive.zip")
    assert res.format == "x-archive-zip" and len(res.posts) == 1
    p = res.posts[0]
    assert p.id == "777" and p.author.username == "me" and p.parent_id == "700" and p.created_at == datetime(2018, 10, 10, 20, 19, 24) and p.media[0].url.endswith("m.jpg") and p.source == "archive"

    res = importers.import_bytes(b"first line of text\n\nsecond line", "notes.txt")
    assert res.format == "text" and len(res.posts) == 2 and res.posts[0].id.startswith("txt-")

    res = importers.import_bytes(json.dumps({"data": [_tweet("8", "v2 response", "u1")], "includes": {"users": [_user("u1", "hater")]}}).encode(), "search.json")
    assert res.format == "x-api-v2-json" and res.posts[0].id == "8" and res.posts[0].source == "upload"


# --------------------------------------------------------------------------------------------
# worker + API
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def fake_x(monkeypatch):
    fx = FakeX()
    import sentinai.api.routes_ingest as ri
    import sentinai.ingestion.jobs as jobs
    from sentinai.ingestion import credentials

    monkeypatch.setattr(credentials, "_resolve", lambda: ("test-token-0123456789", "env"))

    def _client(*, full_archive=None):
        return fx.client(max_wait=5, full_archive=full_archive)

    monkeypatch.setattr(ri, "_client", _client)
    monkeypatch.setattr(jobs, "_make_client", lambda params: None if not params.get("needs_client", True) else fx.client(max_wait=5, full_archive=params.get("full_archive")))
    return fx


def _wait(client, admin, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/api/ingest/jobs/{job_id}", headers=admin).json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_ingest_status_and_rbac(client, admin, researcher, fake_x):
    assert client.get("/api/ingest/status", headers=researcher).status_code == 403
    s = client.get("/api/ingest/status", headers=admin).json()
    assert s["token"]["configured"] is True and "posts_by_source" in s and s["endpoints"]["search"].startswith("/tweets/search/")


def test_ingest_tweet_by_url(client, admin, fake_x):
    r = client.post("/api/ingest/x/tweet", json={"ref": "https://x.com/critic/status/9002", "include_conversation": True}, headers=admin)
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["kind"] == "tweet" and rep["ingested"] >= 3 and set(rep["post_ids"]) >= {"9001", "9002", "9003"}
    parent = client.get("/api/posts/9001", headers=admin).json()
    assert parent["toxicity_label"] == "violent_extremism"
    child = client.get("/api/posts/9002", headers=admin).json()
    assert child["stance"] == "condemn" and child["final_toxicity"] < 0.3 and child["parent"]["id"] == "9001"
    assert client.post("/api/ingest/x/tweet", json={"ref": "nonsense"}, headers=admin).status_code == 422
    assert client.post("/api/ingest/x/tweet", json={"ref": "123456789"}, headers=admin).status_code == 404


def test_ingest_search_job_and_cursor(client, admin, fake_x):
    r = client.post("/api/ingest/x/search", json={"query": "muslims lang:en", "max_pages": 3, "background": True}, headers=admin)
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    assert job["status"] in ("queued", "running", "done")
    j = _wait(client, admin, job["id"])
    assert j["status"] == "done", j
    rep = j["report"]
    assert rep["endpoint"] == "/tweets/search/recent" and rep["fetched"] == 3 and rep["newest_id"] == "9004"
    s = client.get("/api/ingest/status", headers=admin).json()
    cur = next(c for c in s["cursors"] if c["query"] == "muslims lang:en")
    assert cur["since_id"] == "9004" and cur["total_ingested"] == 3
    # inline run reuses the cursor
    r = client.post("/api/ingest/x/search", json={"query": "muslims lang:en", "max_pages": 1, "background": False}, headers=admin)
    assert r.status_code == 200 and r.json()["report"]["since_id"] == "9004"
    assert client.delete("/api/ingest/cursors/muslims lang:en", headers=admin).status_code == 200
    assert client.delete("/api/ingest/cursors/muslims lang:en", headers=admin).status_code == 404
    jobs = client.get("/api/ingest/jobs", headers=admin).json()["items"]
    assert any(x["id"] == job["id"] for x in jobs)


def test_ingest_user_timeline(client, admin, fake_x):
    r = client.post("/api/ingest/x/user", json={"username": "https://x.com/critic", "max_pages": 1, "background": False}, headers=admin)
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["kind"] == "user" and rep["fetched"] == 2 and rep["endpoint"] == "/users/:id/tweets"
    assert client.post("/api/ingest/x/user", json={"username": "nobody", "background": False}, headers=admin).status_code == 404


def test_ingest_stream_job(client, admin, fake_x):
    r = client.post("/api/ingest/x/stream/start", json={"rules": [{"value": "muslims lang:en", "tag": "hate"}], "max_posts": 3}, headers=admin)
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    j = _wait(client, admin, job["id"])
    assert j["status"] == "done" and j["report"]["fetched"] == 3 and j["report"]["endpoint"] == "/tweets/search/stream"
    assert fake_x.rules and fake_x.rules[0]["tag"] == "hate"
    assert client.get("/api/ingest/x/stream/rules", headers=admin).json()["rules"][0]["value"] == "muslims lang:en"
    assert client.post("/api/ingest/x/stream/stop", headers=admin).json()["stopped"] == []
    posts = client.get("/api/posts?q=vile&limit=5", headers=admin).json()
    assert posts["total"] >= 1


def test_ingest_upload(client, admin, fake_x):
    csv_text = "id,text,created_at,username\n" "550001,\"these dirty Asians need to go back to China\",2026-09-10T08:00:00Z,uploader1\n" "550002,\"lovely morning for a run\",2026-09-10T09:00:00Z,uploader2\n"
    r = client.post("/api/ingest/upload", files={"file": ("posts.csv", csv_text.encode(), "text/csv")}, headers=admin)
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    assert rep["endpoint"] == "csv" and rep["ingested"] == 2 and rep["toxic"] == 1
    p = client.get("/api/posts/550001", headers=admin).json()
    assert p["toxicity_label"] == "hate_speech" and p["author_username"] == "uploader1"
    s = client.get("/api/ingest/status", headers=admin).json()
    assert s["posts_by_source"].get("upload", 0) >= 2
    # background import of a JSONL file
    body = "\n".join(json.dumps(_tweet(str(660000 + i), f"upload line {i}", "u9")) for i in range(5))
    r = client.post("/api/ingest/upload", data={"background": "true"}, files={"file": ("dump.jsonl", body.encode(), "application/json")}, headers=admin)
    assert r.status_code == 200 and r.json()["job"]
    j = _wait(client, admin, r.json()["job"]["id"])
    assert j["status"] == "done" and j["report"]["ingested"] == 5
    assert client.post("/api/ingest/upload", files={"file": ("empty.csv", b"", "text/csv")}, headers=admin).status_code == 422


def test_token_endpoints(client, admin, tmp_path, monkeypatch):
    from sentinai.ingestion import credentials

    monkeypatch.setattr(credentials, "_token_file", lambda: tmp_path / "tok")
    credentials.clear_bearer_token()
    assert client.post("/api/ingest/x/token", json={"bearer_token": "short"}, headers=admin).status_code == 422
    r = client.post("/api/ingest/x/token", json={"bearer_token": "AAAAAAAAAAAAAAAAAAAAA%2Fexample-token-value", "persist": True}, headers=admin)
    assert r.status_code == 200 and r.json()["configured"] and r.json()["source"] == "file" and r.json()["hint"] == "…alue"
    assert (tmp_path / "tok").read_text().endswith("value")
    st = client.get("/api/ingest/status", headers=admin).json()["token"]
    assert st["configured"] and "bearer" not in json.dumps(st).lower().replace("bearer_token", "")
    assert client.delete("/api/ingest/x/token", headers=admin).json()["configured"] is False
    assert not (tmp_path / "tok").exists()
