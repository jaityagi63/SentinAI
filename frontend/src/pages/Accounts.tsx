import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, loadSession, type AccountScore } from "../lib/api";
import { Loading, Plot, SeverityBadge, TargetBadges, fmtDate, fmtPct, useAsync } from "../components/ui";

interface AccountDetail {
  score: AccountScore;
  confidence: string;
  bot: { probability: number; features: Record<string, number>; top_signals: string[]; model: string } | null;
  recent_posts: { id: string; text: string; created_at: string; final_toxicity: number; severity_level: number; targets: string[] }[];
}

function ScoreBar({ s }: { s: AccountScore }) {
  if (s.score === null) return <span className="muted">n/a</span>;
  const lo = (s.ci_low ?? s.score) * 100;
  const hi = (s.ci_high ?? s.score) * 100;
  return (
    <div style={{ position: "relative", height: 14, background: "var(--bg-2)", borderRadius: 7 }} title={`${s.score.toFixed(2)} [${(lo / 100).toFixed(2)}, ${(hi / 100).toFixed(2)}]`}>
      <div style={{ position: "absolute", left: `${lo}%`, width: `${Math.max(1, hi - lo)}%`, top: 3, height: 8, background: "rgba(255,107,107,0.35)", borderRadius: 4 }} />
      <div style={{ position: "absolute", left: `calc(${s.score * 100}% - 2px)`, width: 4, top: 0, height: 14, background: "#ff6b6b", borderRadius: 2 }} />
    </div>
  );
}

function AccountDetailView({ id }: { id: string }) {
  const { data, error } = useAsync(() => api<AccountDetail>(`/analytics/accounts/${id}`), [id]);
  const weights = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const s = data.score;
    const parts = [
      { k: "0.4 × avg severity", v: 0.4 * (s.avg_severity / 4) },
      { k: "0.3 × toxic ratio", v: 0.3 * s.toxic_ratio },
      { k: "0.2 × target diversity", v: 0.2 * s.target_diversity },
      { k: "0.1 × recency", v: 0.1 * s.recency_weight },
    ];
    return [{ type: "bar", orientation: "h", y: parts.map((p) => p.k), x: parts.map((p) => p.v), marker: { color: ["#c8283a", "#f0663f", "#f5b74e", "#7c9cff"] }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>" }];
  }, [data]);
  if (!data) return <Loading error={error} />;
  const s = data.score;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>@{s.username ?? s.author_id}</h1>
          <p>
            <Link to="/accounts">← all accounts</Link> · {s.n_posts} posts analysed
          </p>
        </div>
      </div>
      <div className="grid cols-3">
        <div className="card">
          <h3>Model-estimated discrimination propensity</h3>
          {s.reliable && s.score !== null ? (
            <>
              <div className="kpi">
                {s.score.toFixed(2)}
                <small>
                  95% CI [{s.ci_low?.toFixed(2)}, {s.ci_high?.toFixed(2)}]
                </small>
              </div>
              <div style={{ marginTop: 8 }}>
                <ScoreBar s={s} />
              </div>
              <div className="sub" style={{ marginTop: 6 }}>
                {data.confidence}
              </div>
            </>
          ) : (
            <div className="alert">Insufficient sample: at least 50 posts are required before a score is reported ({s.n_posts} available).</div>
          )}
          <p className="disclaimer" style={{ marginTop: 10 }}>
            {s.disclaimer}
          </p>
        </div>
        <div className="card">
          <h3>Score components</h3>
          <Plot data={weights} layout={{ height: 200, margin: { l: 150, r: 20, t: 10, b: 30 }, xaxis: { range: [0, 0.42] } }} />
          <dl className="kv">
            <dt>Avg severity</dt>
            <dd className="mono">{s.avg_severity.toFixed(2)} / 4</dd>
            <dt>Toxic ratio</dt>
            <dd className="mono">{fmtPct(s.toxic_ratio)}</dd>
            <dt>Target diversity</dt>
            <dd className="mono">{s.target_diversity.toFixed(2)}</dd>
            <dt>Recency weight</dt>
            <dd className="mono">{s.recency_weight.toFixed(2)}</dd>
          </dl>
        </div>
        <div className="card">
          <h3>Automation assessment</h3>
          {data.bot ? (
            <>
              <div className="kpi">
                {data.bot.probability.toFixed(2)}
                <small>P(bot) · {data.bot.model}</small>
              </div>
              <div className="chips" style={{ marginTop: 8 }}>
                {data.bot.top_signals.map((t) => (
                  <span key={t} className="chip">
                    {t}
                  </span>
                ))}
              </div>
            </>
          ) : (
            <p className="muted">Not scored.</p>
          )}
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h3>Recent posts</h3>
        <table>
          <thead>
            <tr>
              <th>Post</th>
              <th>Toxicity</th>
              <th>Severity</th>
              <th>Targets</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {data.recent_posts.map((p) => (
              <tr key={p.id}>
                <td>
                  <Link to={`/posts/${p.id}`} style={{ color: "inherit" }}>
                    <div className="post-text">{p.text}</div>
                  </Link>
                </td>
                <td className="mono">{p.final_toxicity.toFixed(2)}</td>
                <td>
                  <SeverityBadge level={p.severity_level} />
                </td>
                <td>
                  <TargetBadges targets={p.targets} />
                </td>
                <td className="muted" style={{ whiteSpace: "nowrap" }}>
                  {fmtDate(p.created_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function AccountsPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [reliableOnly, setReliableOnly] = useState(true);
  const [busy, setBusy] = useState(false);
  const { data, error, reload } = useAsync(() => api<{ disclaimer: string; weights: Record<string, number>; items: AccountScore[] }>(`/analytics/accounts?limit=100&reliable_only=${reliableOnly}`), [reliableOnly]);
  const session = loadSession();
  if (id) return <AccountDetailView id={id} />;
  if (!data) return <Loading error={error} />;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Account discrimination propensity</h1>
          <p>
            Score = 0.4·AvgSeverity + 0.3·ToxicRatio + 0.2·TargetDiversity + 0.1·Recency, with bootstrap confidence intervals and exponential time decay.
          </p>
        </div>
        <div className="toolbar">
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={reliableOnly} onChange={(e) => setReliableOnly(e.target.checked)} /> only accounts with ≥ 50 posts
          </label>
          {session?.permissions.includes("ingest") && (
            <button
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await api("/analytics/accounts/rescore", { method: "POST" });
                  reload();
                } finally {
                  setBusy(false);
                }
              }}
            >
              ↻ Re-score all
            </button>
          )}
        </div>
      </div>
      <div className="alert" style={{ marginBottom: 14 }}>
        <b>Model-Estimated Probability.</b> {data.disclaimer} Scores describe patterns in observed posts and must never be used as the sole basis for enforcement.
      </div>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Account</th>
              <th>Posts</th>
              <th style={{ width: 220 }}>Score (95% CI)</th>
              <th>Avg sev</th>
              <th>Toxic</th>
              <th>Diversity</th>
              <th>Recency</th>
              <th>P(bot)</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((a) => (
              <tr key={a.author_id} className="clickable" onClick={() => navigate(`/accounts/${a.author_id}`)}>
                <td>@{a.username ?? a.author_id}</td>
                <td>{a.n_posts}</td>
                <td>
                  <ScoreBar s={a} />
                  <div className="mono muted" style={{ fontSize: 11 }}>
                    {a.score !== null ? `${a.score.toFixed(2)} [${a.ci_low?.toFixed(2)}, ${a.ci_high?.toFixed(2)}]` : "insufficient sample"}
                  </div>
                </td>
                <td className="mono">{a.avg_severity.toFixed(2)}</td>
                <td className="mono">{fmtPct(a.toxic_ratio, 0)}</td>
                <td className="mono">{a.target_diversity.toFixed(2)}</td>
                <td className="mono">{a.recency_weight.toFixed(2)}</td>
                <td>{a.bot_probability !== null && a.bot_probability >= 0.8 ? <span className="badge bot">{a.bot_probability.toFixed(2)}</span> : <span className="mono muted">{a.bot_probability?.toFixed(2) ?? "—"}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
