import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { DaysPicker, Disclaimer, LABEL_COLORS, Loading, Plot, SEVERITY_COLORS, SEVERITY_NAMES, useAsync } from "../components/ui";

interface SeverityPayload {
  days: number;
  matrix: Record<string, Record<string, number>>;
  per_target: { target: string; avg_severity: number; n: number }[];
}

const TAXONOMY = [
  { level: 1, name: "Microaggressions & stereotyping", desc: "Generalisations, coded language, 'they all…' claims, backhanded compliments." },
  { level: 2, name: "Dehumanization & exclusion", desc: "Vermin/disease metaphors, calls for removal or exclusion ('send them back', 'not welcome')." },
  { level: 3, name: "Slurs & targeted harassment", desc: "Explicit slurs, degrading insults aimed at a protected group." },
  { level: 4, name: "Incitement to violence", desc: "Threats, calls for killing, attacking or 'cleansing' a group; extremist glorification." },
];

export default function SeverityPage() {
  const [days, setDays] = useState(90);
  const [excludeBots, setExcludeBots] = useState(false);
  const { data, error } = useAsync(() => api<SeverityPayload>(`/analytics/severity?days=${days}&exclude_bots=${excludeBots}`), [days, excludeBots]);

  const stacked = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const labels = ["non_toxic", "offensive", "hate_speech", "violent_extremism"];
    const levels = [0, 1, 2, 3, 4];
    return labels.map((lab) => ({ type: "bar", name: lab.replace("_", " "), x: levels.map((l) => `L${l}`), y: levels.map((l) => data.matrix[String(l)]?.[lab] || 0), marker: { color: LABEL_COLORS[lab] } }));
  }, [data]);

  const donut = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const levels = [1, 2, 3, 4];
    const vals = levels.map((l) => Object.values(data.matrix[String(l)] || {}).reduce((a, b) => a + b, 0));
    return [{ type: "pie", hole: 0.5, labels: levels.map((l) => `L${l} ${SEVERITY_NAMES[l].split(" / ")[0]}`), values: vals, marker: { colors: levels.map((l) => SEVERITY_COLORS[l]) }, sort: false, textinfo: "percent" }];
  }, [data]);

  const perTarget = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const rows = [...data.per_target].reverse();
    return [{ type: "bar", orientation: "h", y: rows.map((r) => r.target.split(":")[1].replace(/_/g, " ")), x: rows.map((r) => r.avg_severity), text: rows.map((r) => `n=${r.n}`), textposition: "outside", marker: { color: rows.map((r) => SEVERITY_COLORS[Math.min(4, Math.max(1, Math.round(r.avg_severity)))]) }, hovertemplate: "%{y}: avg severity %{x:.2f}<extra></extra>" }];
  }, [data]);

  if (!data) return <Loading error={error} />;
  const totals = [0, 1, 2, 3, 4].map((l) => Object.values(data.matrix[String(l)] || {}).reduce((a, b) => a + b, 0));
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Severity distribution</h1>
          <p>Ordinal harm taxonomy (Task C). Levels are cumulative: a level-4 post also exhibits lower-level harms.</p>
        </div>
        <div className="toolbar">
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={excludeBots} onChange={(e) => setExcludeBots(e.target.checked)} /> exclude bots
          </label>
          <DaysPicker value={days} onChange={setDays} />
        </div>
      </div>

      <div className="grid cols-4">
        {TAXONOMY.map((t) => (
          <Link key={t.level} to={`/explorer?severity=${t.level}&order=toxicity`} className="card" style={{ borderTop: `4px solid ${SEVERITY_COLORS[t.level]}`, color: "inherit" }}>
            <h3>
              Level {t.level} · {t.name}
            </h3>
            <div className="kpi">{totals[t.level].toLocaleString()}</div>
            <div className="sub">{t.desc}</div>
          </Link>
        ))}
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Severity × toxicity label</h3>
          <Plot data={stacked} layout={{ height: 320, barmode: "stack", showlegend: true }} />
          <div className="sub">Level 0 is non-toxic content (and offensive language that does not target a protected group).</div>
        </div>
        <div className="card">
          <h3>Share of harmful posts by level</h3>
          <Plot data={donut} layout={{ height: 320, showlegend: true, margin: { l: 10, r: 10, t: 10, b: 10 } }} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Average severity by target group (≥ 3 toxic posts)</h3>
        <Plot data={perTarget} layout={{ height: Math.max(260, 26 * data.per_target.length + 60), margin: { l: 150, r: 60, t: 10, b: 40 }, xaxis: { range: [0, 4.4], title: { text: "avg severity level" } } }} />
      </div>
      <div style={{ marginTop: 12 }}>
        <Disclaimer />
      </div>
    </>
  );
}
