import { useEffect, useRef, useState, type ReactNode } from "react";
import Plotly from "plotly.js-cartesian-dist-min";
import type { Attribution, Explanation } from "../lib/api";

export const SEVERITY_NAMES: Record<number, string> = {
  0: "None",
  1: "Microaggression / Stereotyping",
  2: "Dehumanization / Exclusion",
  3: "Slurs / Targeted Harassment",
  4: "Incitement to Violence",
};

export const SEVERITY_COLORS = ["#2f3a68", "#f9d46b", "#f7a44a", "#f0663f", "#c8283a"];
export const LABEL_COLORS: Record<string, string> = {
  non_toxic: "#3ecf8e",
  offensive: "#f5b74e",
  hate_speech: "#f0663f",
  violent_extremism: "#c8283a",
};

export const PLOT_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c9d0ee", size: 12 },
  margin: { l: 48, r: 16, t: 24, b: 40 },
  xaxis: { gridcolor: "#26305c", zerolinecolor: "#26305c" },
  yaxis: { gridcolor: "#26305c", zerolinecolor: "#26305c" },
  legend: { orientation: "h", y: -0.2 },
  hoverlabel: { bgcolor: "#0b1020", bordercolor: "#26305c" },
};

export function Plot({ data, layout, style, onClick }: { data: Plotly.Data[]; layout?: Partial<Plotly.Layout>; style?: React.CSSProperties; onClick?: (e: Plotly.PlotMouseEvent) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const merged: Partial<Plotly.Layout> = { ...PLOT_LAYOUT, ...layout, xaxis: { ...PLOT_LAYOUT.xaxis, ...(layout?.xaxis || {}) }, yaxis: { ...PLOT_LAYOUT.yaxis, ...(layout?.yaxis || {}) } };
    Plotly.react(el, data, merged, { displayModeBar: false, responsive: true });
    if (onClick) {
      (el as unknown as { on: (ev: string, fn: (e: Plotly.PlotMouseEvent) => void) => void }).on("plotly_click", onClick);
    }
    return () => {
      Plotly.purge(el);
    };
  }, [data, layout, onClick]);
  return <div ref={ref} className="plot" style={style} />;
}

export function Badge({ kind, children, title }: { kind: string; children: ReactNode; title?: string }) {
  return (
    <span className={`badge ${kind}`} title={title}>
      {children}
    </span>
  );
}

export function LabelBadge({ label }: { label?: string }) {
  if (!label) return null;
  return <Badge kind={label}>{label.replace("_", " ")}</Badge>;
}

export function SeverityBadge({ level }: { level?: number | null }) {
  if (level === undefined || level === null) return null;
  return (
    <Badge kind={`sev sev-${level}`} title={SEVERITY_NAMES[level]}>
      L{level} · {SEVERITY_NAMES[level].split(" / ")[0]}
    </Badge>
  );
}

export function TargetBadges({ targets }: { targets?: string[] }) {
  if (!targets?.length) return <span className="muted">—</span>;
  return (
    <span className="chips" style={{ gap: 4 }}>
      {targets.map((t) => (
        <Badge key={t} kind="target">
          {t.replace(":", " · ").replace(/_/g, " ")}
        </Badge>
      ))}
    </span>
  );
}

export function Meter({ value }: { value: number }) {
  return (
    <div className="meter" title={value.toFixed(3)}>
      <div style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

export function ProbBar({ probs, colors }: { probs: Record<string, number>; colors: Record<string, string> | string[] }) {
  return (
    <div className="prob-bar">
      {Object.entries(probs).map(([k, v]) => (
        <span key={k} style={{ width: `${v * 100}%`, background: Array.isArray(colors) ? colors[Number(k)] : colors[k] }} title={`${k}: ${(v * 100).toFixed(1)}%`} />
      ))}
    </div>
  );
}

/** Module 4 — per-token heat overlay (red = pushes toward toxic, green = toward neutral). */
export function TokenHeatmap({ explanation, text }: { explanation: Explanation | null | undefined; text?: string }) {
  if (!explanation) return <p className="muted">No explanation available.</p>;
  const src = explanation.text || text || "";
  const parts: ReactNode[] = [];
  let cursor = 0;
  const sorted = [...explanation.attributions].sort((a, b) => a.start - b.start);
  sorted.forEach((a: Attribution, i) => {
    if (a.start > cursor) parts.push(<span key={`g${i}`}>{src.slice(cursor, a.start)}</span>);
    const w = Math.max(-1, Math.min(1, a.weight));
    const bg = w > 0 ? `rgba(255, 80, 80, ${0.12 + 0.75 * w})` : w < 0 ? `rgba(62, 207, 142, ${0.12 + 0.6 * -w})` : "transparent";
    parts.push(
      <span key={i} className="heat-token" style={{ background: bg }} title={`weight ${a.weight.toFixed(3)}`}>
        {src.slice(a.start, a.end)}
      </span>,
    );
    cursor = a.end;
  });
  if (cursor < src.length) parts.push(<span key="tail">{src.slice(cursor)}</span>);
  return (
    <div>
      <div style={{ fontSize: 15, lineHeight: 1.9 }}>{parts}</div>
      <div className="heat-legend">
        <span>neutral / counter-speech</span>
        <div className="bar" />
        <span>drives toxicity</span>
        <span className="mono muted" style={{ marginLeft: "auto" }}>
          {explanation.method}
        </span>
      </div>
    </div>
  );
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    fn()
      .then((d) => alive && setData(d))
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { data, error, loading, reload: () => setTick((t) => t + 1) };
}

export function Loading({ error }: { error?: string | null }) {
  if (error) return <div className="alert">Error: {error}</div>;
  return <div className="loading">Loading…</div>;
}

export function DaysPicker({ value, onChange, options = [7, 30, 90, 365] }: { value: number; onChange: (d: number) => void; options?: number[] }) {
  return (
    <div className="pill-group">
      {options.map((d) => (
        <button key={d} className={value === d ? "active" : ""} onClick={() => onChange(d)}>
          {d}d
        </button>
      ))}
    </div>
  );
}

export function fmtPct(x: number | null | undefined, digits = 1) {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(digits)}%`;
}

export function fmtDate(s: string) {
  const d = new Date(s);
  return isNaN(d.getTime()) ? s : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function Disclaimer() {
  return <p className="disclaimer">All scores are model-estimated probabilities — not definitive human judgments.</p>;
}
