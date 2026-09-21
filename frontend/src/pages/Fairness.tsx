import { useMemo, useState } from "react";
import { api, type FairnessReport } from "../lib/api";
import { Loading, Plot, fmtPct, useAsync } from "../components/ui";

interface FairnessPayload {
  benchmarks: FairnessReport[];
  audits: { id: number; created_at: string; model_version: string; report: FairnessReport }[];
}

function ReportView({ r }: { r: FairnessReport }) {
  const bars = useMemo<Plotly.Data[]>(() => {
    const g = r.groups;
    return [
      { type: "bar", name: "false-positive rate", x: g.map((x) => x.group), y: g.map((x) => x.fpr * 100), marker: { color: "#f0663f" } },
      { type: "bar", name: "false-negative rate", x: g.map((x) => x.group), y: g.map((x) => x.fnr * 100), marker: { color: "#7c9cff" } },
    ];
  }, [r]);
  return (
    <div className="card">
      <div className="toolbar" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }}>{r.benchmark}</h3>
        <span className="muted">
          threshold {r.threshold} · reference group <b>{r.reference_group}</b>
        </span>
      </div>
      <div className="grid cols-2" style={{ marginTop: 10 }}>
        <div>
          <div className="grid cols-2">
            <div>
              <div className="muted" style={{ fontSize: 12 }}>
                FPR gap (max − min)
              </div>
              <div className="kpi" style={{ color: r.fpr_gap > 0.1 ? "#ff9a7a" : "#3ecf8e" }}>
                {(r.fpr_gap * 100).toFixed(1)} pp
              </div>
            </div>
            <div>
              <div className="muted" style={{ fontSize: 12 }}>
                FNR gap (max − min)
              </div>
              <div className="kpi" style={{ color: r.fnr_gap > 0.1 ? "#ff9a7a" : "#3ecf8e" }}>
                {(r.fnr_gap * 100).toFixed(1)} pp
              </div>
            </div>
          </div>
          <Plot data={bars} layout={{ height: 260, barmode: "group", showlegend: true, yaxis: { title: { text: "%" } } }} />
        </div>
        <div>
          <table>
            <thead>
              <tr>
                <th>Group</th>
                <th>n</th>
                <th>FPR</th>
                <th>FNR</th>
                <th>Precision</th>
                <th>Mean score (benign)</th>
                <th>Calibrated θ</th>
              </tr>
            </thead>
            <tbody>
              {r.groups.map((g) => (
                <tr key={g.group}>
                  <td>{g.group}</td>
                  <td>{g.n}</td>
                  <td className="mono">{fmtPct(g.fpr)}</td>
                  <td className="mono">{fmtPct(g.fnr)}</td>
                  <td className="mono">{fmtPct(g.precision)}</td>
                  <td className="mono">{g.mean_score_negative.toFixed(3)}</td>
                  <td className="mono">{r.calibrated_thresholds[g.group]?.toFixed(2) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {r.notes.length > 0 && (
            <ul className="muted" style={{ fontSize: 12, paddingLeft: 18 }}>
              {r.notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export default function FairnessPage() {
  const { data, error, reload } = useAsync(() => api<FairnessPayload>("/analytics/fairness"), []);
  const [busy, setBusy] = useState(false);
  if (!data) return <Loading error={error} />;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Bias mitigation &amp; fairness audit</h1>
          <p>False-positive / false-negative parity across dialects and identity mentions (Module 11). Group-specific thresholds are derived to equalise FPR against the reference group.</p>
        </div>
        <div className="toolbar">
          <button
            className="primary"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                await api("/analytics/fairness/audit?days=7", { method: "POST" });
                reload();
              } catch (e) {
                alert((e as Error).message);
              } finally {
                setBusy(false);
              }
            }}
          >
            Run weekly audit on human labels
          </button>
        </div>
      </div>
      <div className="alert info" style={{ marginBottom: 14 }}>
        Benchmarks bundled in <span className="mono">resources/benchmarks/*.jsonl</span> (dialect smoke test in the AAVE / SAE / British style of BOLD &amp; ToxiGen). Drop additional JSONL files with <span className="mono">text / label / group</span> fields to extend the audit.
      </div>
      <div className="list">
        {data.benchmarks.map((b) => (
          <ReportView key={b.benchmark} r={b} />
        ))}
        <div className="card">
          <h3>Weekly audit history (annotated posts, grouped by dialect heuristics)</h3>
          {data.audits.length ? (
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Model</th>
                  <th>Groups</th>
                  <th>FPR gap</th>
                  <th>FNR gap</th>
                </tr>
              </thead>
              <tbody>
                {data.audits.map((a) => (
                  <tr key={a.id}>
                    <td className="muted">{new Date(a.created_at).toLocaleString()}</td>
                    <td className="mono">{a.model_version}</td>
                    <td>{a.report.groups.map((g) => `${g.group} (n=${g.n})`).join(", ") || "—"}</td>
                    <td className="mono">{(a.report.fpr_gap * 100).toFixed(1)} pp</td>
                    <td className="mono">{(a.report.fnr_gap * 100).toFixed(1)} pp</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">No audits yet — audits compare model output with human annotations gathered in the review queue.</p>
          )}
        </div>
      </div>
    </>
  );
}
