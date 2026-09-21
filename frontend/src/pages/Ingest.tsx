import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, uploadFile, type IngestJob, type IngestReport, type IngestResponse, type IngestStatus, type XConnectionTest } from "../lib/api";
import { LabelBadge, Loading, fmtDate, useAsync } from "../components/ui";

type Tab = "search" | "tweet" | "user" | "stream" | "upload";

const TABS: { id: Tab; label: string; hint: string }[] = [
  { id: "search", label: "Search", hint: "keyword / operator query" },
  { id: "tweet", label: "Tweet URL", hint: "one post + its thread" },
  { id: "user", label: "User timeline", hint: "@handle" },
  { id: "stream", label: "Live stream", hint: "filtered stream rules" },
  { id: "upload", label: "Upload file", hint: "JSON · CSV · archive" },
];

const QUERY_EXAMPLES = [
  "(muslims OR islam) lang:en -is:retweet",
  "(immigrants OR migrants OR refugees) (invasion OR plague OR vermin) -is:retweet",
  "(jews OR zionists) lang:en has:images",
  "(inmigrantes OR extranjeros) lang:es -is:retweet",
  "(musulmans OR arabes) lang:fr -is:retweet",
];

const STREAM_EXAMPLES = ["(muslims OR islam OR jews OR immigrants) lang:en -is:retweet", "(inmigrantes OR musulmanes OR judíos) lang:es", "#StopIslam OR #BanIslam OR #SendThemBack"];

function ReportView({ report }: { report: IngestReport }) {
  return (
    <div className="list" style={{ marginTop: 8 }}>
      <div className="grid cols-4">
        <div>
          <div className="kpi">{report.fetched}</div>
          <div className="muted">fetched</div>
        </div>
        <div>
          <div className="kpi">{report.ingested}</div>
          <div className="muted">classified &amp; stored</div>
        </div>
        <div>
          <div className="kpi">{report.toxic}</div>
          <div className="muted">flagged toxic (≥ 0.5)</div>
        </div>
        <div>
          <div className="kpi">{report.duration_seconds}s</div>
          <div className="muted">{report.endpoint ?? "—"}</div>
        </div>
      </div>
      {Object.keys(report.by_label).length > 0 && (
        <div className="chips">
          {Object.entries(report.by_label)
            .sort((a, b) => b[1] - a[1])
            .map(([label, n]) => (
              <span key={label} className="chip">
                <LabelBadge label={label} /> {n}
              </span>
            ))}
        </div>
      )}
      {(report.hydrated_parents > 0 || report.media_downloaded > 0 || report.since_id) && (
        <div className="muted" style={{ fontSize: 12 }}>
          {report.hydrated_parents > 0 && <>{report.hydrated_parents} parent post(s) hydrated for reply-chain context · </>}
          {report.media_downloaded > 0 && <>{report.media_downloaded} image(s) downloaded · </>}
          {report.since_id && <>resumed after id {report.since_id} · </>}
          {report.newest_id && <>newest id {report.newest_id}</>}
        </div>
      )}
      {report.warnings.length > 0 && (
        <div className="alert">
          {report.warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
      {report.post_ids.length > 0 && (
        <div className="muted" style={{ fontSize: 12 }}>
          Open:{" "}
          {report.post_ids.slice(0, 8).map((id) => (
            <Link key={id} to={`/posts/${id}`} style={{ marginRight: 8 }}>
              {id}
            </Link>
          ))}
          {report.post_ids.length > 8 && <>… or browse them in the </>}
          {report.post_ids.length > 8 && <Link to="/explorer?order=recent">Post explorer</Link>}
        </div>
      )}
    </div>
  );
}

function JobRow({ job, onCancel }: { job: IngestJob; onCancel: (id: string) => void }) {
  const progress = Object.entries(job.progress)
    .map(([k, v]) => `${k} ${v}`)
    .join(" · ");
  return (
    <tr>
      <td>
        <span className={`badge ${job.status === "failed" ? "violent_extremism" : job.status === "done" ? "non_toxic" : job.status === "running" ? "stance" : "muted"}`}>{job.status}</span>
      </td>
      <td>
        <b>{job.kind}</b> <span className="mono">{job.label}</span>
      </td>
      <td className="muted">{fmtDate(job.created_at)}</td>
      <td>
        {job.report ? (
          <>
            {job.report.ingested} ingested{job.report.toxic > 0 && <>, {job.report.toxic} toxic</>}
          </>
        ) : job.error ? (
          <span className="error">{job.error}</span>
        ) : (
          <span className="muted">{progress || "—"}</span>
        )}
      </td>
      <td>
        {(job.status === "running" || job.status === "queued") && (
          <button onClick={() => onCancel(job.id)} style={{ padding: "2px 8px", fontSize: 12 }}>
            {job.kind === "stream" ? "Stop" : "Cancel"}
          </button>
        )}
      </td>
    </tr>
  );
}

export default function IngestPage() {
  const [tab, setTab] = useState<Tab>("search");
  const status = useAsync(() => api<IngestStatus>("/ingest/status"), []);
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // forms
  const [query, setQuery] = useState(QUERY_EXAMPLES[0]);
  const [pages, setPages] = useState(2);
  const [fullArchive, setFullArchive] = useState(false);
  const [startTime, setStartTime] = useState("");
  const [useCursor, setUseCursor] = useState(true);
  const [ref, setRef] = useState("");
  const [withReplies, setWithReplies] = useState(true);
  const [username, setUsername] = useState("");
  const [userPages, setUserPages] = useState(2);
  const [excludeRetweets, setExcludeRetweets] = useState(true);
  const [rules, setRules] = useState(STREAM_EXAMPLES[0]);
  const [maxMinutes, setMaxMinutes] = useState(30);
  const [downloadMedia, setDownloadMedia] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [hydrate, setHydrate] = useState(false);

  // token
  const [token, setToken] = useState("");
  const [persist, setPersist] = useState(true);
  const [test, setTest] = useState<XConnectionTest | null>(null);
  const [testing, setTesting] = useState(false);

  const refreshJobs = async () => {
    try {
      const r = await api<{ items: IngestJob[] }>("/ingest/jobs?limit=20");
      setJobs(r.items);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    void refreshJobs();
  }, []);

  const hasActive = jobs.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    if (!hasActive) return;
    const t = setInterval(() => {
      void refreshJobs();
      status.reload();
    }, 2000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasActive]);

  async function run(fn: () => Promise<IngestResponse>) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await fn();
      setResult(r);
      await refreshJobs();
      status.reload();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const post = (path: string, body: unknown) => api<IngestResponse>(path, { method: "POST", body: JSON.stringify(body) });

  function submitSearch(e: FormEvent) {
    e.preventDefault();
    void run(() => post("/ingest/x/search", { query, max_pages: pages, full_archive: fullArchive ? true : null, start_time: startTime ? new Date(startTime).toISOString() : null, use_cursor: useCursor, download_media: downloadMedia || null, background: true }));
  }
  function submitTweet(e: FormEvent) {
    e.preventDefault();
    void run(() => post("/ingest/x/tweet", { ref, include_conversation: withReplies, download_media: downloadMedia || null, background: false }));
  }
  function submitUser(e: FormEvent) {
    e.preventDefault();
    void run(() => post("/ingest/x/user", { username, max_pages: userPages, exclude_retweets: excludeRetweets, download_media: downloadMedia || null, background: true }));
  }
  function submitStream(e: FormEvent) {
    e.preventDefault();
    const ruleList = rules
      .split("\n")
      .map((r) => r.trim())
      .filter(Boolean)
      .map((value, i) => ({ value, tag: `rule-${i + 1}` }));
    void run(() => post("/ingest/x/stream/start", { rules: ruleList.length ? ruleList : null, max_minutes: maxMinutes || null, download_media: downloadMedia || null }));
  }
  function submitUpload(e: FormEvent) {
    e.preventDefault();
    const f = fileRef.current?.files?.[0];
    if (!f) {
      setError("Choose a file first");
      return;
    }
    void run(() => uploadFile<IngestResponse>("/ingest/upload", f, { hydrate_parents: String(hydrate), background: String(f.size > 512 * 1024) }));
  }
  async function stopStream() {
    await api("/ingest/x/stream/stop", { method: "POST" });
    await refreshJobs();
    status.reload();
  }
  async function cancelJob(id: string) {
    try {
      await api(`/ingest/jobs/${id}/cancel`, { method: "POST" });
    } catch {
      /* ignore */
    }
    await refreshJobs();
  }
  async function saveToken(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api("/ingest/x/token", { method: "POST", body: JSON.stringify({ bearer_token: token.trim(), persist }) });
      setToken("");
      status.reload();
    } catch (err) {
      setError((err as Error).message);
    }
  }
  async function clearToken() {
    await api("/ingest/x/token", { method: "DELETE" });
    setTest(null);
    status.reload();
  }
  async function testConnection() {
    setTesting(true);
    setTest(null);
    try {
      setTest(await api<XConnectionTest>("/ingest/x/test", { method: "POST" }));
    } catch (err) {
      setTest({ ok: false, error: (err as Error).message, rate_limits: {} });
    } finally {
      setTesting(false);
    }
  }
  async function resetCursor(q: string) {
    await api(`/ingest/cursors/${encodeURIComponent(q)}`, { method: "DELETE" });
    status.reload();
  }

  const s = status.data;
  const connected = !!s?.token.configured;
  const sources = s ? Object.entries(s.posts_by_source).sort((a, b) => b[1] - a[1]) : [];
  const realPosts = sources.filter(([k]) => k !== "demo").reduce((acc, [, v]) => acc + v, 0);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Ingest from X</h1>
          <p>Pull real posts from X (Twitter) — by search query, tweet URL, user timeline or the live filtered stream — or import files. Every post goes through preprocessing, the three classification heads, context discounting, bot and account scoring.</p>
        </div>
      </div>

      <div className="grid cols-3" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Connection</h3>
          {status.loading && !s ? (
            <Loading />
          ) : s ? (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <span className={`badge ${connected ? "non_toxic" : "offensive"}`}>{connected ? "token configured" : "no token"}</span>
                {connected && (
                  <span className="muted" style={{ fontSize: 12 }}>
                    from {s.token.source} · {s.token.hint}
                  </span>
                )}
              </div>
              {connected ? (
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button onClick={testConnection} disabled={testing}>
                    {testing ? "Testing…" : "Test connection"}
                  </button>
                  {s.token.source !== "env" && (
                    <button onClick={clearToken} className="danger">
                      Remove token
                    </button>
                  )}
                </div>
              ) : (
                <form onSubmit={saveToken} className="list">
                  <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="X API bearer token (App-only, from developer.x.com)" autoComplete="off" />
                  <label className="muted" style={{ fontSize: 12 }}>
                    <input type="checkbox" checked={persist} onChange={(e) => setPersist(e.target.checked)} /> remember on this server (data/secrets, mode 600)
                  </label>
                  <div>
                    <button className="primary" disabled={token.trim().length < 20}>
                      Connect X
                    </button>
                  </div>
                  <div className="muted" style={{ fontSize: 12 }}>
                    Or set <span className="mono">SENTINAI_X_BEARER_TOKEN</span> in <span className="mono">.env</span> and restart.
                  </div>
                </form>
              )}
              {test && (
                <div className={`alert ${test.ok ? "info" : ""}`} style={{ marginTop: 10 }}>
                  {test.ok ? (
                    <>
                      Connected — recent search OK ({test.latency_ms} ms){test.full_archive === true && <>, full-archive OK</>}
                      {test.full_archive === false && <>, full-archive not available on this plan (recent search will be used)</>}.
                    </>
                  ) : (
                    <>Connection failed: {test.error}</>
                  )}
                </div>
              )}
              <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                Recent search (7-day window) works on every paid / pay-per-use plan. Full-archive and the filtered stream need an upgraded access level — SentinAI falls back to recent search automatically.
              </div>
            </>
          ) : (
            <Loading error={status.error} />
          )}
        </div>

        <div className="card">
          <h3>Corpus by source</h3>
          {s ? (
            <>
              <div className="kpi">
                {realPosts.toLocaleString()} <small>real posts</small>
              </div>
              <dl className="kv" style={{ marginTop: 8 }}>
                {sources.map(([src, n]) => (
                  <div key={src} style={{ display: "contents" }}>
                    <dt>{src}</dt>
                    <dd>{n.toLocaleString()}</dd>
                  </div>
                ))}
                {sources.length === 0 && <dd className="muted">empty</dd>}
              </dl>
              <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                <span className="mono">x</span> = live API, <span className="mono">stream</span> = filtered stream, <span className="mono">upload</span> / <span className="mono">archive</span> = imported files, <span className="mono">demo</span> = synthetic seed corpus.
              </div>
            </>
          ) : (
            <Loading />
          )}
        </div>

        <div className="card">
          <h3>Saved cursors</h3>
          {s && s.cursors.length > 0 ? (
            <table>
              <tbody>
                {s.cursors.slice(0, 6).map((c) => (
                  <tr key={c.query}>
                    <td className="mono" style={{ fontSize: 12 }}>
                      {c.query.length > 34 ? c.query.slice(0, 34) + "…" : c.query}
                    </td>
                    <td className="muted" style={{ whiteSpace: "nowrap" }}>
                      {c.total_ingested} · {c.last_run ? fmtDate(c.last_run) : "—"}
                    </td>
                    <td>
                      <button onClick={() => resetCursor(c.query)} title="forget since_id and re-fetch the window" style={{ padding: "2px 8px", fontSize: 12 }}>
                        reset
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="muted">Re-running a query continues from its last seen post id; cursors appear here after the first run.</div>
          )}
        </div>
      </div>

      <div className="split">
        <div className="card">
          <div className="pill-group" style={{ marginBottom: 12 }}>
            {TABS.map((t) => (
              <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)} title={t.hint}>
                {t.label}
              </button>
            ))}
          </div>

          {tab === "search" && (
            <form onSubmit={submitSearch} className="list">
              <label className="muted">Search query (X operators: OR, -is:retweet, lang:, has:images, from:, conversation_id:)</label>
              <textarea rows={2} value={query} onChange={(e) => setQuery(e.target.value)} />
              <div className="chips">
                {QUERY_EXAMPLES.map((q) => (
                  <span key={q} className="chip" onClick={() => setQuery(q)}>
                    {q.length > 48 ? q.slice(0, 48) + "…" : q}
                  </span>
                ))}
              </div>
              <div className="toolbar" style={{ gap: 12, flexWrap: "wrap" }}>
                <label className="muted">
                  pages{" "}
                  <select value={pages} onChange={(e) => setPages(Number(e.target.value))}>
                    {[1, 2, 3, 5, 10, 20].map((n) => (
                      <option key={n} value={n}>
                        {n} × 100
                      </option>
                    ))}
                  </select>
                </label>
                <label className="muted">
                  from <input type="datetime-local" value={startTime} onChange={(e) => setStartTime(e.target.value)} />
                </label>
                <label className="muted">
                  <input type="checkbox" checked={useCursor} onChange={(e) => setUseCursor(e.target.checked)} /> only new posts since last run
                </label>
                <label className="muted">
                  <input type="checkbox" checked={fullArchive} onChange={(e) => setFullArchive(e.target.checked)} /> full archive
                </label>
                <label className="muted">
                  <input type="checkbox" checked={downloadMedia} onChange={(e) => setDownloadMedia(e.target.checked)} /> download images
                </label>
              </div>
              <div>
                <button className="primary" disabled={busy || !connected || !query.trim()}>
                  {busy ? "Starting…" : "Search & classify"}
                </button>
              </div>
            </form>
          )}

          {tab === "tweet" && (
            <form onSubmit={submitTweet} className="list">
              <label className="muted">Tweet URL or id — the parent chain is fetched so replies and quote-tweets get their context discount</label>
              <input value={ref} onChange={(e) => setRef(e.target.value)} placeholder="https://x.com/username/status/1234567890123456789" />
              <div className="toolbar" style={{ gap: 12, flexWrap: "wrap" }}>
                <label className="muted">
                  <input type="checkbox" checked={withReplies} onChange={(e) => setWithReplies(e.target.checked)} /> include replies (conversation)
                </label>
                <label className="muted">
                  <input type="checkbox" checked={downloadMedia} onChange={(e) => setDownloadMedia(e.target.checked)} /> download images
                </label>
              </div>
              <div>
                <button className="primary" disabled={busy || !connected || !ref.trim()}>
                  {busy ? "Fetching…" : "Fetch & classify"}
                </button>
              </div>
            </form>
          )}

          {tab === "user" && (
            <form onSubmit={submitUser} className="list">
              <label className="muted">Handle or profile URL — pulls the most recent tweets (100 per page, up to 3 200)</label>
              <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="@username" />
              <div className="toolbar" style={{ gap: 12, flexWrap: "wrap" }}>
                <label className="muted">
                  pages{" "}
                  <select value={userPages} onChange={(e) => setUserPages(Number(e.target.value))}>
                    {[1, 2, 3, 5, 10, 32].map((n) => (
                      <option key={n} value={n}>
                        {n} × 100
                      </option>
                    ))}
                  </select>
                </label>
                <label className="muted">
                  <input type="checkbox" checked={excludeRetweets} onChange={(e) => setExcludeRetweets(e.target.checked)} /> skip retweets
                </label>
                <label className="muted">
                  <input type="checkbox" checked={downloadMedia} onChange={(e) => setDownloadMedia(e.target.checked)} /> download images
                </label>
              </div>
              <div>
                <button className="primary" disabled={busy || !connected || !username.trim()}>
                  {busy ? "Starting…" : "Pull timeline"}
                </button>
              </div>
            </form>
          )}

          {tab === "stream" && (
            <form onSubmit={submitStream} className="list">
              <label className="muted">Filtered-stream rules (one per line; replaces the rules on your X app)</label>
              <textarea rows={3} value={rules} onChange={(e) => setRules(e.target.value)} />
              <div className="chips">
                {STREAM_EXAMPLES.map((q) => (
                  <span key={q} className="chip" onClick={() => setRules(q)}>
                    {q.length > 48 ? q.slice(0, 48) + "…" : q}
                  </span>
                ))}
              </div>
              <div className="toolbar" style={{ gap: 12, flexWrap: "wrap" }}>
                <label className="muted">
                  stop after{" "}
                  <select value={maxMinutes} onChange={(e) => setMaxMinutes(Number(e.target.value))}>
                    {[5, 15, 30, 60, 180, 720, 0].map((n) => (
                      <option key={n} value={n}>
                        {n === 0 ? "manual stop" : `${n} min`}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="muted">
                  <input type="checkbox" checked={downloadMedia} onChange={(e) => setDownloadMedia(e.target.checked)} /> download images
                </label>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="primary" disabled={busy || !connected || s?.stream_running}>
                  {busy ? "Connecting…" : "Start stream"}
                </button>
                {s?.stream_running && (
                  <button type="button" className="danger" onClick={stopStream}>
                    Stop stream
                  </button>
                )}
              </div>
              <div className="muted" style={{ fontSize: 12 }}>
                Posts are classified in batches of 25 as they arrive and show up in the dashboards immediately. The stream needs an access level that includes <span className="mono">/2/tweets/search/stream</span>.
              </div>
            </form>
          )}

          {tab === "upload" && (
            <form onSubmit={submitUpload} className="list">
              <label className="muted">
                X API JSON / JSONL (twarc dumps, saved search responses), your X data archive (<span className="mono">tweets.js</span> or the whole <span className="mono">.zip</span>), CSV/TSV with a text column, or plain text (one post per line). Up to {s?.upload_max_mb ?? 50} MB — no API token needed.
              </label>
              <input type="file" ref={fileRef} accept=".json,.jsonl,.ndjson,.js,.zip,.csv,.tsv,.txt,application/json,text/csv,text/plain,application/zip" />
              <label className="muted">
                <input type="checkbox" checked={hydrate} onChange={(e) => setHydrate(e.target.checked)} disabled={!connected} /> fetch missing parent posts from X for reply context {!connected && "(needs a token)"}
              </label>
              <div>
                <button className="primary" disabled={busy}>
                  {busy ? "Importing…" : "Import & classify"}
                </button>
              </div>
            </form>
          )}

          {error && (
            <div className="error" style={{ marginTop: 10 }}>
              {error}
            </div>
          )}
          {result?.report && (
            <div style={{ marginTop: 14 }}>
              <h3>Result</h3>
              <ReportView report={result.report} />
            </div>
          )}
          {result?.job && !result.report && (
            <div className="alert info" style={{ marginTop: 14 }}>
              Job <span className="mono">{result.job.id}</span> started in the background — progress appears in the job list; the dashboards update as batches are committed.
            </div>
          )}
        </div>

        <div className="card">
          <h3>How it works</h3>
          <ol style={{ paddingLeft: 18, margin: 0, fontSize: 13, lineHeight: 1.6 }}>
            <li>
              <b>Fetch</b> — X API v2 with the <span className="mono">author_id</span>, <span className="mono">referenced_tweets</span> and <span className="mono">attachments.media_keys</span> expansions; exponential backoff and rate-limit budgets are honoured.
            </li>
            <li>
              <b>Context</b> — parents of replies / quotes / retweets are hydrated and processed first so the stance detector can discount counter-speech (×0.2).
            </li>
            <li>
              <b>Classify</b> — normalisation, obfuscation repair, language routing, toxicity, target groups, severity, explanations, optional OCR on downloaded images.
            </li>
            <li>
              <b>Score</b> — bot probability and account-level propensity are recomputed for every touched author; uncertain posts (0.4–0.6) join the human review queue.
            </li>
          </ol>
          <div className="muted" style={{ fontSize: 12, marginTop: 12 }}>
            Compliance: deleted posts are removed by <span className="mono">sentinai compliance --check-deleted</span>; raw payloads expire after the retention window. Downloaded images stay under <span className="mono">data/media</span>.
          </div>
          <div className="muted" style={{ fontSize: 12, marginTop: 12 }}>
            CLI equivalents: <span className="mono">sentinai ingest "&lt;query&gt;"</span>, <span className="mono">sentinai ingest-tweet &lt;url&gt;</span>, <span className="mono">sentinai ingest-user @handle</span>, <span className="mono">sentinai ingest-stream -r "&lt;rule&gt;"</span>, <span className="mono">sentinai import file.jsonl</span>.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>
          Ingestion jobs{" "}
          <button onClick={refreshJobs} style={{ marginLeft: 8, padding: "2px 8px", fontSize: 12 }}>
            refresh
          </button>
        </h3>
        {jobs.length === 0 ? (
          <div className="muted">No jobs yet. Background searches, timelines, streams and large imports appear here.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>status</th>
                <th>job</th>
                <th>started</th>
                <th>result</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <JobRow key={j.id} job={j} onCancel={cancelJob} />
              ))}
            </tbody>
          </table>
        )}
        {jobs.some((j) => j.status === "done" && j.report) && (
          <details style={{ marginTop: 10 }}>
            <summary>Latest finished job report</summary>
            <ReportView report={jobs.find((j) => j.status === "done" && j.report)!.report!} />
          </details>
        )}
      </div>
    </>
  );
}
