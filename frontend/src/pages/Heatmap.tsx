import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, type HeatCell } from "../lib/api";
import { DaysPicker, Disclaimer, Loading, Plot, SEVERITY_NAMES, useAsync } from "../components/ui";

export default function HeatmapPage() {
  const [days, setDays] = useState(90);
  const [excludeBots, setExcludeBots] = useState(false);
  const [metric, setMetric] = useState<"count" | "severity">("count");
  const [category, setCategory] = useState<string>("all");
  const navigate = useNavigate();
  const { data, error } = useAsync(() => api<{ cells: HeatCell[]; by_category: Record<string, number> }>(`/analytics/heatmap?days=${days}&exclude_bots=${excludeBots}`), [days, excludeBots]);

  const cells = useMemo(() => (data ? data.cells.filter((c) => category === "all" || c.category === category).slice(0, 30) : []), [data, category]);

  const heat = useMemo<Plotly.Data[]>(() => {
    if (!cells.length) return [];
    const levels = [1, 2, 3, 4];
    const y = cells.map((c) => `${c.label.replace(/_/g, " ")}  (${c.category[0].toUpperCase()})`);
    const z = cells.map((c) => levels.map((l) => (metric === "count" ? c.by_severity[String(l)] || 0 : ((c.by_severity[String(l)] || 0) / c.total) * 100)));
    return [
      {
        type: "heatmap",
        x: levels.map((l) => `L${l} ${SEVERITY_NAMES[l].split(" / ")[0]}`),
        y,
        z,
        colorscale: [
          [0, "#141b3a"],
          [0.25, "#3b2f6b"],
          [0.5, "#a13a5e"],
          [0.75, "#e0603f"],
          [1, "#ffd166"],
        ],
        hovertemplate: metric === "count" ? "%{y}<br>%{x}: %{z} posts<extra></extra>" : "%{y}<br>%{x}: %{z:.1f}% of target's toxic posts<extra></extra>",
        showscale: true,
        colorbar: { title: { text: metric === "count" ? "posts" : "%" }, thickness: 12 },
      },
    ];
  }, [cells, metric]);

  const catBar = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const e = Object.entries(data.by_category);
    return [{ type: "bar", x: e.map(([k]) => k), y: e.map(([, v]) => v), marker: { color: ["#7c9cff", "#b48cff", "#3ecf8e"] } }];
  }, [data]);

  if (!data) return <Loading error={error} />;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Global toxicity heatmap</h1>
          <p>Target demographic × severity level. Click a row to open matching posts.</p>
        </div>
        <div className="toolbar">
          <div className="pill-group">
            {["all", "ethnicity", "religion", "nationality"].map((c) => (
              <button key={c} className={category === c ? "active" : ""} onClick={() => setCategory(c)}>
                {c}
              </button>
            ))}
          </div>
          <div className="pill-group">
            <button className={metric === "count" ? "active" : ""} onClick={() => setMetric("count")}>
              counts
            </button>
            <button className={metric === "severity" ? "active" : ""} onClick={() => setMetric("severity")}>
              severity mix %
            </button>
          </div>
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={excludeBots} onChange={(e) => setExcludeBots(e.target.checked)} /> exclude bots
          </label>
          <DaysPicker value={days} onChange={setDays} />
        </div>
      </div>
      <div className="split">
        <div className="card">
          <h3>Target × severity</h3>
          {cells.length ? (
            <Plot
              data={heat}
              layout={{ height: Math.max(360, 26 * cells.length + 80), margin: { l: 190, r: 20, t: 10, b: 60 }, yaxis: { autorange: "reversed" } }}
              onClick={(e) => {
                const idx = e.points?.[0]?.pointIndex as unknown as [number, number] | undefined;
                const row = Array.isArray(idx) ? idx[0] : (e.points?.[0] as unknown as { y: string })?.y ? cells.findIndex((c) => `${c.label.replace(/_/g, " ")}  (${c.category[0].toUpperCase()})` === (e.points[0] as unknown as { y: string }).y) : -1;
                const c = cells[row];
                if (c) navigate(`/explorer?target=${c.category}:${c.label}`);
              }}
            />
          ) : (
            <p className="muted">No toxic posts with an identified target in this window.</p>
          )}
        </div>
        <div className="list">
          <div className="card">
            <h3>By category</h3>
            <Plot data={catBar} layout={{ height: 220 }} />
          </div>
          <div className="card">
            <h3>Top targets</h3>
            <table>
              <thead>
                <tr>
                  <th>Target</th>
                  <th>Posts</th>
                  <th>Avg sev</th>
                </tr>
              </thead>
              <tbody>
                {cells.slice(0, 12).map((c) => (
                  <tr key={`${c.category}:${c.label}`} className="clickable" onClick={() => navigate(`/explorer?target=${c.category}:${c.label}`)}>
                    <td>{c.label.replace(/_/g, " ")}</td>
                    <td>{c.total}</td>
                    <td>{c.avg_severity.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Disclaimer />
        </div>
      </div>
    </>
  );
}
