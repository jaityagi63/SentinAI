import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type HeatCell, type Overview, type QueueStats, type Trends } from "../lib/api";
import { DaysPicker, Disclaimer, LABEL_COLORS, Loading, Plot, SEVERITY_COLORS, SEVERITY_NAMES, fmtPct, useAsync } from "../components/ui";

export default function OverviewPage() {
  const [days, setDays] = useState(90);
  const [excludeBots, setExcludeBots] = useState(false);
  const q = `days=${days}&exclude_bots=${excludeBots}`;
  const ov = useAsync(() => api<Overview>(`/analytics/overview?${q}`), [q]);
  const tr = useAsync(() => api<Trends>(`/analytics/trends?days=${Math.max(days, 7)}&horizon=0&exclude_bots=${excludeBots}`), [q]);
  const hm = useAsync(() => api<{ cells: HeatCell[] }>(`/analytics/heatmap?${q}`), [q]);
  const rs = useAsync(() => api<QueueStats>(`/review/stats`), []);

  const labelPie = useMemo<Plotly.Data[]>(() => {
    if (!ov.data) return [];
    const entries = Object.entries(ov.data.by_label);
    return [{ type: "pie", hole: 0.55, labels: entries.map(([k]) => k.replace("_", " ")), values: entries.map(([, v]) => v), marker: { colors: entries.map(([k]) => LABEL_COLORS[k]) }, textinfo: "percent", sort: false }];
  }, [ov.data]);

  const sevBar = useMemo<Plotly.Data[]>(() => {
    if (!ov.data) return [];
    const lv = [1, 2, 3, 4];
    return [{ type: "bar", x: lv.map((l) => `L${l} ${SEVERITY_NAMES[l].split(" / ")[0]}`), y: lv.map((l) => ov.data!.by_severity[String(l)] || 0), marker: { color: lv.map((l) => SEVERITY_COLORS[l]) } }];
  }, [ov.data]);

  const spark = useMemo<Plotly.Data[]>(() => {
    if (!tr.data) return [];
    const s = tr.data.series;
    return [
      { type: "scatter", mode: "lines", x: s.map((p) => p.date), y: s.map((p) => p.toxic), name: "toxic posts / day", line: { color: "#f0663f", width: 2 }, fill: "tozeroy", fillcolor: "rgba(240,102,63,0.15)" },
      { type: "scatter", mode: "markers", x: s.filter((p) => p.spike).map((p) => p.date), y: s.filter((p) => p.spike).map((p) => p.toxic), name: "spike (z ≥ 2.5)", marker: { color: "#ffd166", size: 9, symbol: "diamond" } },
    ];
  }, [tr.data]);

  const topTargets = useMemo(() => (hm.data ? hm.data.cells.slice(0, 8) : []), [hm.data]);
  const langBar = useMemo<Plotly.Data[]>(() => {
    if (!ov.data) return [];
    const e = Object.entries(ov.data.by_language);
    return [{ type: "bar", orientation: "h", y: e.map(([k]) => k), x: e.map(([, v]) => v), marker: { color: "#7c9cff" } }];
  }, [ov.data]);

  if (!ov.data) return <Loading error={ov.error} />;
  const o = ov.data;
  const realPosts = Object.entries(o.by_source ?? {}).reduce((acc, [k, v]) => (k === "demo" ? acc : acc + v), 0);
  const demoOnly = o.posts > 0 && realPosts === 0 && (o.by_source?.demo ?? 0) > 0;
  return (
    <>
      {demoOnly && (
        <div className="alert info" style={{ marginBottom: 16 }}>
          You are looking at the <b>synthetic demo corpus</b>. Connect an X API token under <Link to="/ingest">Ingest from X</Link> to analyse real posts — by search query, tweet URL, user timeline, live stream or file import.
        </div>
      )}
      <div className="page-head">
        <div>
          <h1>Overview</h1>
          <p>Global picture of discriminatory content detected in the last {days} days.</p>
        </div>
        <div className="toolbar">
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={excludeBots} onChange={(e) => setExcludeBots(e.target.checked)} /> exclude likely bots
          </label>
          <DaysPicker value={days} onChange={setDays} />
        </div>
      </div>

      <div className="grid cols-4">
        <div className="card">
          <h3>Posts analysed</h3>
          <div className="kpi">{o.posts.toLocaleString()}</div>
          <div className="sub">{o.authors.toLocaleString()} distinct authors</div>
        </div>
        <div className="card">
          <h3>Toxic posts</h3>
          <div className="kpi">
            {o.toxic_posts.toLocaleString()}
            <small>{fmtPct(o.toxic_ratio)}</small>
          </div>
          <div className="sub">final toxicity ≥ 0.5 (after context discount)</div>
        </div>
        <div className="card">
          <h3>Avg severity (toxic)</h3>
          <div className="kpi">{o.avg_severity_toxic.toFixed(2)}</div>
          <div className="sub">on the 1–4 harm taxonomy</div>
        </div>
        <div className="card">
          <h3>Awaiting human review</h3>
          <div className="kpi">{(rs.data?.pending ?? o.needs_review).toLocaleString()}</div>
          <div className="sub">
            <Link to="/review">active-learning queue →</Link> · {o.likely_bots} likely bots
          </div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Toxic volume over time</h3>
          {tr.data ? <Plot data={spark} layout={{ height: 280, showlegend: true }} /> : <Loading error={tr.error} />}
          <div className="sub">
            {tr.data?.events.length ? `${tr.data.events.length} real-world events overlaid in ` : ""}
            <Link to="/trends">trend explorer →</Link>
          </div>
        </div>
        <div className="card">
          <h3>Most targeted demographics</h3>
          {hm.data ? (
            <table>
              <thead>
                <tr>
                  <th>Target</th>
                  <th>Toxic posts</th>
                  <th>Avg severity</th>
                  <th>Share of levels</th>
                </tr>
              </thead>
              <tbody>
                {topTargets.map((c) => (
                  <tr key={`${c.category}:${c.label}`}>
                    <td>
                      <Link to={`/explorer?target=${c.category}:${c.label}`}>{c.label.replace(/_/g, " ")}</Link>
                      <div className="muted" style={{ fontSize: 11 }}>
                        {c.category}
                      </div>
                    </td>
                    <td>{c.total}</td>
                    <td>{c.avg_severity.toFixed(2)}</td>
                    <td style={{ minWidth: 140 }}>
                      <div className="prob-bar">
                        {[1, 2, 3, 4].map((l) => (
                          <span key={l} style={{ width: `${((c.by_severity[String(l)] || 0) / c.total) * 100}%`, background: SEVERITY_COLORS[l] }} title={`L${l}: ${c.by_severity[String(l)] || 0}`} />
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Loading error={hm.error} />
          )}
          <div className="sub" style={{ marginTop: 8 }}>
            <Link to="/heatmap">full heatmap →</Link>
          </div>
        </div>
      </div>

      <div className="grid cols-3" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Toxicity labels</h3>
          <Plot data={labelPie} layout={{ height: 260, showlegend: true, margin: { l: 10, r: 10, t: 10, b: 10 } }} />
        </div>
        <div className="card">
          <h3>Severity distribution</h3>
          <Plot data={sevBar} layout={{ height: 260 }} />
        </div>
        <div className="card">
          <h3>Languages</h3>
          <Plot data={langBar} layout={{ height: 260, margin: { l: 40, r: 10, t: 10, b: 30 } }} />
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <Disclaimer />
      </div>
    </>
  );
}
