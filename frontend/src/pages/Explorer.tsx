import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, downloadFile, type PostSummary } from "../lib/api";
import { Badge, LabelBadge, Loading, SeverityBadge, TargetBadges, fmtDate, useAsync } from "../components/ui";

const LABELS = ["", "non_toxic", "offensive", "hate_speech", "violent_extremism"];
const LANGS = ["", "en", "es", "ar", "hi", "fr", "pt", "de"];

export default function ExplorerPage() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const label = params.get("label") || "";
  const target = params.get("target") || "";
  const lang = params.get("lang") || "";
  const severity = params.get("severity") || "";
  const order = params.get("order") || "recent";
  const minTox = params.get("min_toxicity") || "0";
  const excludeBots = params.get("exclude_bots") === "true";
  const page = Number(params.get("page") || "0");
  const [q, setQ] = useState(params.get("q") || "");
  const [busy, setBusy] = useState<string | null>(null);
  const limit = 25;

  const set = (k: string, v: string) => {
    const p = new URLSearchParams(params);
    if (v) p.set(k, v);
    else p.delete(k);
    if (k !== "page") p.delete("page");
    setParams(p);
  };

  const query = new URLSearchParams({ limit: String(limit), offset: String(page * limit), order, min_toxicity: minTox });
  if (label) query.set("label", label);
  if (target) query.set("target", target);
  if (lang) query.set("lang", lang);
  if (severity) query.set("severity", severity);
  if (excludeBots) query.set("exclude_bots", "true");
  if (params.get("q")) query.set("q", params.get("q")!);
  const { data, error, loading } = useAsync(() => api<{ total: number; items: PostSummary[] }>(`/posts?${query.toString()}`), [query.toString()]);

  async function exportAs(kind: "csv" | "json" | "pdf") {
    setBusy(kind);
    try {
      const qs = `days=365&min_toxicity=${minTox}`;
      if (kind === "pdf") await downloadFile(`/export/report.pdf?days=90`, "sentinai-report.pdf");
      else await downloadFile(`/export/posts.${kind}?${qs}`, `sentinai-posts.${kind}`);
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const pages = data ? Math.ceil(data.total / limit) : 0;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Post explorer</h1>
          <p>Browse classified posts; open one for the token-level explanation and thread context.</p>
        </div>
        <div className="toolbar">
          <button onClick={() => exportAs("csv")} disabled={!!busy}>
            ⤓ CSV
          </button>
          <button onClick={() => exportAs("json")} disabled={!!busy}>
            ⤓ JSON
          </button>
          <button onClick={() => exportAs("pdf")} disabled={!!busy}>
            ⤓ PDF report
          </button>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 14 }}>
        <div className="toolbar">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              set("q", q);
            }}
            style={{ display: "flex", gap: 6 }}
          >
            <input placeholder="search text…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: 220 }} />
            <button>Search</button>
          </form>
          <select value={label} onChange={(e) => set("label", e.target.value)}>
            {LABELS.map((l) => (
              <option key={l} value={l}>
                {l ? l.replace("_", " ") : "any label"}
              </option>
            ))}
          </select>
          <select value={severity} onChange={(e) => set("severity", e.target.value)}>
            <option value="">any severity</option>
            {[0, 1, 2, 3, 4].map((s) => (
              <option key={s} value={s}>
                level {s}
              </option>
            ))}
          </select>
          <select value={lang} onChange={(e) => set("lang", e.target.value)}>
            {LANGS.map((l) => (
              <option key={l} value={l}>
                {l || "any language"}
              </option>
            ))}
          </select>
          <select value={minTox} onChange={(e) => set("min_toxicity", e.target.value)}>
            {["0", "0.3", "0.5", "0.7", "0.9"].map((v) => (
              <option key={v} value={v}>
                toxicity ≥ {v}
              </option>
            ))}
          </select>
          <select value={order} onChange={(e) => set("order", e.target.value)}>
            <option value="recent">newest first</option>
            <option value="toxicity">most toxic</option>
            <option value="severity">most severe</option>
          </select>
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={excludeBots} onChange={(e) => set("exclude_bots", e.target.checked ? "true" : "")} /> exclude bots
          </label>
          {target && (
            <span className="chip active" onClick={() => set("target", "")}>
              target: {target.replace(":", " · ").replace(/_/g, " ")} ✕
            </span>
          )}
        </div>
      </div>

      <div className="card">
        {loading && !data ? (
          <Loading error={error} />
        ) : (
          <>
            <div className="muted" style={{ marginBottom: 8 }}>
              {data?.total.toLocaleString()} posts match
            </div>
            <table>
              <thead>
                <tr>
                  <th>Post</th>
                  <th>Label</th>
                  <th>Severity</th>
                  <th>Targets</th>
                  <th>Score</th>
                  <th>Context</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                {data?.items.map((p) => (
                  <tr key={p.id} className="clickable" onClick={() => navigate(`/posts/${p.id}`)}>
                    <td>
                      <div className="post-text">{p.text}</div>
                      <div className="muted" style={{ fontSize: 11, marginTop: 3 }}>
                        @{p.author_username} {p.bot_probability !== null && p.bot_probability >= 0.8 && <span className="badge bot">bot</span>} · {p.language ?? p.lang}
                        {p.has_media && " · 🖼 media"}
                        {p.source && p.source !== "demo" && <span className="badge muted" style={{ marginLeft: 6 }} title="provenance">{p.source === "x" ? "live X" : p.source}</span>}
                        {p.needs_review && <span className="badge muted" style={{ marginLeft: 6 }}>needs review</span>}
                      </div>
                    </td>
                    <td>
                      <LabelBadge label={p.toxicity_label} />
                    </td>
                    <td>
                      <SeverityBadge level={p.severity_level} />
                    </td>
                    <td>
                      <TargetBadges targets={p.targets} />
                    </td>
                    <td className="mono">{p.final_toxicity?.toFixed(2)}</td>
                    <td>
                      {p.stance && p.stance !== "none" && <Badge kind="stance">{p.stance}</Badge>} {p.discount_factor !== undefined && p.discount_factor < 1 && <span className="muted">×{p.discount_factor}</span>}
                      {p.parent_id && <div className="muted" style={{ fontSize: 11 }}>reply</div>}
                      {p.quoted_id && <div className="muted" style={{ fontSize: 11 }}>quote</div>}
                    </td>
                    <td className="muted" style={{ whiteSpace: "nowrap" }}>
                      {fmtDate(p.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="toolbar" style={{ marginTop: 12, justifyContent: "space-between" }}>
              <span className="muted">
                page {page + 1} of {Math.max(pages, 1)}
              </span>
              <span>
                <button disabled={page === 0} onClick={() => set("page", String(page - 1))}>
                  ← prev
                </button>{" "}
                <button disabled={page + 1 >= pages} onClick={() => set("page", String(page + 1))}>
                  next →
                </button>
              </span>
            </div>
          </>
        )}
      </div>
      <p className="muted" style={{ fontSize: 12 }}>
        Want to test a sentence? Use the <Link to="/classify">classification playground</Link>.
      </p>
    </>
  );
}
