import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, loadSession, type BotReport } from "../lib/api";
import { DaysPicker, Disclaimer, Loading, Plot, fmtPct, useAsync } from "../components/ui";

const FEATURE_LABELS: Record<string, string> = {
  account_age_days: "Account age (days)",
  posts_per_day: "Posting frequency / day",
  follower_following_ratio: "Follower : following",
  profile_completeness: "Profile completeness",
  timing_regularity: "Timing regularity",
  content_repetitiveness: "Content repetitiveness",
  retweet_ratio: "Retweet ratio",
  mean_interval_seconds: "Mean interval (s)",
  burstiness: "Burstiness",
  night_share: "Night-time share",
  url_ratio: "URL ratio",
  hashtag_ratio: "Hashtag ratio",
  default_profile_image: "Default profile image",
  username_digits: "Digits in username",
};

export default function BotsPage() {
  const [days, setDays] = useState(90);
  const [threshold, setThreshold] = useState(0.8);
  const [busy, setBusy] = useState(false);
  const { data, error, reload } = useAsync(() => api<BotReport>(`/analytics/bots?days=${days}&threshold=${threshold}&limit=30`), [days, threshold]);
  const session = loadSession();

  const hist = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    return [{ type: "bar", x: data.histogram.map((_, i) => `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`), y: data.histogram, marker: { color: data.histogram.map((_, i) => ((i + 1) / 10 > threshold ? "#ff6b6b" : "#7c9cff")) } }];
  }, [data, threshold]);

  const compare = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const keys = ["human", "bot"];
    return [
      { type: "bar", name: "posts", x: keys, y: keys.map((k) => data.breakdown[k]?.posts || 0), marker: { color: "#7c9cff" }, yaxis: "y" },
      { type: "bar", name: "toxic ratio", x: keys, y: keys.map((k) => (data.breakdown[k]?.toxic_ratio || 0) * 100), marker: { color: "#f0663f" }, yaxis: "y2" },
    ];
  }, [data]);

  if (!data) return <Loading error={error} />;
  const bot = data.breakdown.bot;
  const human = data.breakdown.human;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Bot vs human activity</h1>
          <p>Automated-account probability from behavioural features (Module 10). Accounts above the threshold are flagged and can be excluded from every other view.</p>
        </div>
        <div className="toolbar">
          <label className="muted">
            threshold{" "}
            <select value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}>
              {[0.5, 0.6, 0.7, 0.8, 0.9].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          {session?.permissions.includes("ingest") && (
            <button
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await api("/analytics/bots/rescore", { method: "POST" });
                  reload();
                } finally {
                  setBusy(false);
                }
              }}
            >
              ↻ Re-score
            </button>
          )}
          <DaysPicker value={days} onChange={setDays} />
        </div>
      </div>

      <div className="grid cols-4">
        <div className="card">
          <h3>Scored accounts</h3>
          <div className="kpi">{data.scored_accounts.toLocaleString()}</div>
          <div className="sub">model: {data.model}</div>
        </div>
        <div className="card">
          <h3>Likely bots</h3>
          <div className="kpi">
            {data.likely_bots.toLocaleString()}
            <small>{fmtPct(data.scored_accounts ? data.likely_bots / data.scored_accounts : 0)}</small>
          </div>
          <div className="sub">P(bot) ≥ {threshold}</div>
        </div>
        <div className="card">
          <h3>Toxic ratio · bots</h3>
          <div className="kpi" style={{ color: "#ff9a7a" }}>
            {fmtPct(bot?.toxic_ratio)}
          </div>
          <div className="sub">{bot?.toxic_posts ?? 0} of {bot?.posts ?? 0} posts</div>
        </div>
        <div className="card">
          <h3>Toxic ratio · humans</h3>
          <div className="kpi">{fmtPct(human?.toxic_ratio)}</div>
          <div className="sub">{human?.toxic_posts ?? 0} of {human?.posts ?? 0} posts</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>P(bot) distribution</h3>
          <Plot data={hist} layout={{ height: 280, xaxis: { title: { text: "bot probability" } } }} />
        </div>
        <div className="card">
          <h3>Volume &amp; toxicity by account type</h3>
          <Plot data={compare} layout={{ height: 280, barmode: "group", showlegend: true, yaxis: { title: { text: "posts" } }, yaxis2: { title: { text: "toxic %" }, overlaying: "y", side: "right", gridcolor: "rgba(0,0,0,0)" } }} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Highest bot probability</h3>
        <table>
          <thead>
            <tr>
              <th>Account</th>
              <th>P(bot)</th>
              <th>Followers / following</th>
              <th>Account score</th>
              <th>Top signals</th>
            </tr>
          </thead>
          <tbody>
            {data.top_accounts.map((a) => {
              const feats = Object.entries(a.features || {})
                .filter(([k]) => ["posts_per_day", "timing_regularity", "content_repetitiveness", "profile_completeness", "account_age_days", "follower_following_ratio", "retweet_ratio"].includes(k))
                .slice(0, 5);
              return (
                <tr key={a.author_id}>
                  <td>
                    <Link to={`/accounts/${a.author_id}`}>@{a.username}</Link>
                  </td>
                  <td>
                    <span className={`badge ${a.bot_probability >= threshold ? "bot" : "muted"}`}>{a.bot_probability.toFixed(2)}</span>
                  </td>
                  <td className="mono">
                    {a.followers.toLocaleString()} / {a.following.toLocaleString()}
                  </td>
                  <td className="mono">{a.account_score !== null ? a.account_score.toFixed(2) : "—"}</td>
                  <td className="muted" style={{ fontSize: 12 }}>
                    {feats.map(([k, v]) => `${FEATURE_LABELS[k] || k}: ${typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : v}`).join(" · ")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 12 }}>
        <Disclaimer />
      </div>
    </>
  );
}
