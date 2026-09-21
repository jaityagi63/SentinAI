import { useMemo, useState } from "react";
import { api, type Trends } from "../lib/api";
import { DaysPicker, Loading, Plot, useAsync } from "../components/ui";

interface Taxonomy {
  targets: Record<string, string[]>;
}

export default function TrendsPage() {
  const [days, setDays] = useState(90);
  const [target, setTarget] = useState<string>("");
  const [excludeBots, setExcludeBots] = useState(false);
  const [horizon, setHorizon] = useState(14);
  const tax = useAsync(() => api<Taxonomy>("/meta/taxonomy"), []);
  const { data, error, loading } = useAsync(() => api<Trends>(`/analytics/trends?days=${days}&horizon=${horizon}&exclude_bots=${excludeBots}${target ? `&target=${target}` : ""}`), [days, target, excludeBots, horizon]);
  const multi = useAsync(() => api<{ targets: { target: string; series: { date: string; toxic: number }[] }[] }>(`/analytics/trends/targets?days=${days}&top=6`), [days]);

  const main = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    const s = data.series;
    const traces: Plotly.Data[] = [
      { type: "bar", x: s.map((p) => p.date), y: s.map((p) => p.total - p.toxic), name: "non-toxic", marker: { color: "rgba(124,156,255,0.35)" } },
      { type: "bar", x: s.map((p) => p.date), y: s.map((p) => p.toxic), name: "toxic", marker: { color: "#f0663f" } },
      { type: "scatter", mode: "markers", x: s.filter((p) => p.spike).map((p) => p.date), y: s.filter((p) => p.spike).map((p) => p.toxic), name: "spike (z-score ≥ 2.5)", marker: { color: "#ffd166", size: 11, symbol: "diamond", line: { color: "#0b1020", width: 1 } }, hovertemplate: "%{x}<br>%{y} toxic posts<extra>spike</extra>" },
    ];
    if (data.forecast.length) {
      traces.push(
        { type: "scatter", mode: "lines", x: data.forecast.map((f) => f.date), y: data.forecast.map((f) => f.yhat_upper), line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
        { type: "scatter", mode: "lines", x: data.forecast.map((f) => f.date), y: data.forecast.map((f) => f.yhat_lower), fill: "tonexty", fillcolor: "rgba(180,140,255,0.18)", line: { width: 0 }, name: "forecast 80% band" },
        { type: "scatter", mode: "lines", x: data.forecast.map((f) => f.date), y: data.forecast.map((f) => f.yhat), name: `forecast (${data.forecast_method})`, line: { color: "#b48cff", dash: "dot", width: 2 } },
      );
    }
    return traces;
  }, [data]);

  const eventShapes = useMemo(() => {
    if (!data) return { shapes: [], annotations: [] };
    const maxY = Math.max(1, ...data.series.map((p) => p.total));
    return {
      shapes: data.events.map((e) => ({ type: "line" as const, x0: e.date, x1: e.date, y0: 0, y1: maxY, line: { color: "rgba(255,209,102,0.7)", width: 1, dash: "dash" as const } })),
      annotations: data.events.map((e, i) => ({ x: e.date, y: maxY * (0.98 - (i % 3) * 0.08), text: `${e.title.slice(0, 28)}${e.title.length > 28 ? "…" : ""}`, showarrow: false, font: { size: 10, color: "#ffd166" }, xanchor: "left" as const, bgcolor: "rgba(11,16,32,0.7)" })),
    };
  }, [data]);

  const zplot = useMemo<Plotly.Data[]>(() => {
    if (!data) return [];
    return [
      { type: "scatter", mode: "lines", x: data.series.map((p) => p.date), y: data.series.map((p) => p.zscore), name: "robust z-score", line: { color: "#7c9cff" } },
      { type: "scatter", mode: "lines", x: [data.series[0]?.date, data.series[data.series.length - 1]?.date], y: [2.5, 2.5], name: "threshold", line: { color: "#ffd166", dash: "dash" } },
    ];
  }, [data]);

  const small = useMemo<Plotly.Data[]>(() => {
    if (!multi.data) return [];
    const palette = ["#7c9cff", "#b48cff", "#3ecf8e", "#f5b74e", "#f0663f", "#ff6b6b"];
    return multi.data.targets.map((t, i) => ({ type: "scatter", mode: "lines", stackgroup: "one", x: t.series.map((p) => p.date), y: t.series.map((p) => p.toxic), name: t.target.split(":")[1].replace(/_/g, " "), line: { color: palette[i % palette.length], width: 1 } }));
  }, [multi.data]);

  const targetOptions = useMemo(() => {
    if (!tax.data) return [];
    return Object.entries(tax.data.targets).flatMap(([cat, labels]) => labels.map((l) => `${cat}:${l}`));
  }, [tax.data]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Temporal trend explorer</h1>
          <p>Daily toxic volume with z-score spike detection, forecast and real-world event overlay.</p>
        </div>
        <div className="toolbar">
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="">all targets</option>
            <option value="ethnicity">— any ethnicity —</option>
            <option value="religion">— any religion —</option>
            <option value="nationality">— any nationality —</option>
            {targetOptions.map((t) => (
              <option key={t} value={t}>
                {t.replace(":", " · ").replace(/_/g, " ")}
              </option>
            ))}
          </select>
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            <option value={0}>no forecast</option>
            <option value={7}>forecast 7d</option>
            <option value={14}>forecast 14d</option>
            <option value={30}>forecast 30d</option>
          </select>
          <label className="muted" style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="checkbox" checked={excludeBots} onChange={(e) => setExcludeBots(e.target.checked)} /> exclude bots
          </label>
          <DaysPicker value={days} onChange={setDays} options={[30, 90, 180, 365]} />
        </div>
      </div>

      <div className="card">
        <h3>Toxic posts per day {target && `· ${target.replace(":", " · ").replace(/_/g, " ")}`}</h3>
        {loading || !data ? <Loading error={error} /> : <Plot data={main} layout={{ height: 400, barmode: "stack", shapes: eventShapes.shapes, annotations: eventShapes.annotations, showlegend: true }} />}
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h3>Anomaly score</h3>
          {data && <Plot data={zplot} layout={{ height: 240, showlegend: true }} />}
          <div className="sub">Rolling median/MAD z-score over the previous 14 days; spikes require z ≥ 2.5 and ≥ 3 toxic posts.</div>
        </div>
        <div className="card">
          <h3>Event correlation</h3>
          {data && data.events.length ? (
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Event</th>
                  <th>Before → after</th>
                  <th>Lift</th>
                </tr>
              </thead>
              <tbody>
                {data.events.map((e) => (
                  <tr key={e.id}>
                    <td className="mono">{e.date}</td>
                    <td>
                      {e.title}
                      <div className="muted" style={{ fontSize: 11 }}>
                        {e.category} · {e.source}
                        {e.related_targets.length ? ` · ${e.related_targets.map((t) => t.split(":")[1]).join(", ")}` : ""}
                      </div>
                    </td>
                    <td className="mono">
                      {e.before_mean.toFixed(1)} → {e.after_mean.toFixed(1)}
                    </td>
                    <td style={{ color: e.lift > 0.5 ? "#ff9a7a" : e.lift < 0 ? "#3ecf8e" : undefined, fontWeight: 600 }}>{e.lift >= 0 ? "+" : ""}{(e.lift * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No events in this window. Sync GDELT / NewsAPI via the CLI (`sentinai sync-events`) or POST /api/analytics/events.</p>
          )}
          <div className="sub">Lift = mean daily toxic volume in the 3 days after the event vs the 3 days before.</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Top targets over time (stacked)</h3>
        {multi.data ? <Plot data={small} layout={{ height: 300, showlegend: true }} /> : <Loading error={multi.error} />}
      </div>
    </>
  );
}
