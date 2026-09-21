import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, downloadFile, loadSession, type PostSummary, type QueueStats, type ReviewItem } from "../lib/api";
import { LabelBadge, Loading, SEVERITY_NAMES, SeverityBadge, TargetBadges, TokenHeatmap, fmtPct, useAsync } from "../components/ui";

const LABELS = ["non_toxic", "offensive", "hate_speech", "violent_extremism"];
const TARGET_OPTIONS = ["ethnicity:black", "ethnicity:asian", "ethnicity:latino", "ethnicity:white", "ethnicity:arab", "ethnicity:jewish", "ethnicity:south_asian", "ethnicity:indigenous", "ethnicity:roma", "religion:islam", "religion:judaism", "religion:christianity", "religion:hinduism", "religion:sikhism", "religion:buddhism", "religion:atheism", "nationality:mexican", "nationality:chinese", "nationality:indian", "nationality:pakistani", "nationality:nigerian", "nationality:polish", "nationality:romanian", "nationality:american", "nationality:immigrant"];

interface Agreement {
  annotators: string[];
  posts_with_multiple_annotations: number;
  pairwise_cohen_kappa: { a: string; b: string; n: number; kappa: number | null }[];
  fleiss_kappa: number | null;
  interpretation: string;
}

function AnnotationForm({ postId, initialLabel, initialSeverity, initialTargets, onDone }: { postId: string; initialLabel: string; initialSeverity: number; initialTargets: string[]; onDone: (message: string) => void }) {
  const [label, setLabel] = useState(initialLabel);
  const [severity, setSeverity] = useState(initialSeverity);
  const [targets, setTargets] = useState<string[]>(initialTargets);
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    setLabel(initialLabel);
    setSeverity(initialSeverity);
    setTargets(initialTargets);
    setMsg(null);
  }, [postId, initialLabel, initialSeverity, initialTargets]);

  async function submit() {
    setBusy(true);
    try {
      const r = await api<{ id: number; agrees_with_model: boolean | null }>("/review/annotate", {
        method: "POST",
        body: JSON.stringify({ post_id: postId, toxicity_label: label, severity: label === "non_toxic" ? 0 : severity, targets: label === "non_toxic" ? [] : targets, notes: notes || null }),
      });
      const m = r.agrees_with_model ? "Saved — agrees with the model." : "Saved — disagreement recorded for retraining.";
      setMsg(m);
      onDone(m);
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="list">
      <div>
        <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
          Toxicity label
        </div>
        <div className="pill-group">
          {LABELS.map((l) => (
            <button key={l} className={label === l ? "active" : ""} onClick={() => setLabel(l)}>
              {l.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>
      {label !== "non_toxic" && (
        <>
          <div>
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              Severity level
            </div>
            <div className="pill-group">
              {[1, 2, 3, 4].map((s) => (
                <button key={s} className={severity === s ? "active" : ""} onClick={() => setSeverity(s)} title={SEVERITY_NAMES[s]}>
                  L{s}
                </button>
              ))}
            </div>
            <div className="muted" style={{ fontSize: 11, marginTop: 3 }}>
              {SEVERITY_NAMES[severity]}
            </div>
          </div>
          <div>
            <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
              Targets
            </div>
            <div className="chips">
              {TARGET_OPTIONS.map((t) => (
                <span key={t} className={`chip ${targets.includes(t) ? "active" : ""}`} onClick={() => setTargets(targets.includes(t) ? targets.filter((x) => x !== t) : [...targets, t])}>
                  {t.split(":")[1].replace(/_/g, " ")}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
      <textarea rows={2} placeholder="notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} />
      <div className="toolbar">
        <button className="primary" disabled={busy} onClick={submit}>
          {busy ? "Saving…" : "Submit annotation"}
        </button>
        {msg && <span className="muted">{msg}</span>}
      </div>
    </div>
  );
}

export default function ReviewPage() {
  const [params] = useSearchParams();
  const focusPost = params.get("post");
  const [status, setStatus] = useState("pending");
  const [selected, setSelected] = useState<ReviewItem | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const session = loadSession();
  const queue = useAsync(() => api<{ items: ReviewItem[]; stats: QueueStats }>(`/review/queue?limit=40&status=${status}`), [status]);
  const agreement = useAsync(() => api<Agreement>("/review/agreement"), []);
  const batches = useAsync(() => api<{ id: number; created_at: string; n_examples: number; n_corrections: number; status: string }[]>("/review/retrain-batches"), []);
  const focused = useAsync(() => (focusPost ? api<PostSummary>(`/posts/${focusPost}`) : Promise.resolve(null)), [focusPost]);

  useEffect(() => {
    if (queue.data && !selected && queue.data.items.length && !focusPost) setSelected(queue.data.items[0]);
  }, [queue.data, selected, focusPost]);

  async function act(fn: () => Promise<unknown>, key: string) {
    setBusy(key);
    try {
      await fn();
      queue.reload();
      agreement.reload();
      batches.reload();
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const stats = queue.data?.stats;
  const canRetrain = session?.permissions.includes("retrain");
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Human-in-the-loop review</h1>
          <p>Active-learning queue: posts whose model confidence falls in the {stats ? `${stats.band[0]}–${stats.band[1]}` : "0.4–0.6"} band, plus flagged disagreements.</p>
        </div>
        <div className="toolbar">
          <button disabled={!!busy} onClick={() => act(() => api("/review/enqueue?limit=200", { method: "POST" }), "enqueue")}>
            ⟳ Refill queue
          </button>
          <button disabled={!!busy} onClick={() => act(() => downloadFile("/review/export/label-studio?limit=500", "label-studio-tasks.json"), "ls")}>
            ⤓ Label Studio
          </button>
          <button disabled={!!busy} onClick={() => act(() => downloadFile("/review/export/prodigy?limit=500", "prodigy.jsonl"), "prodigy")}>
            ⤓ Prodigy
          </button>
          {canRetrain && (
            <button className="primary" disabled={!!busy || !stats || stats.pending_for_retraining < 20} title="Bundles unused annotations into a fine-tuning batch (biweekly cadence)" onClick={() => act(() => api("/review/retrain-batch?min_examples=20", { method: "POST" }), "batch")}>
              Build retrain batch ({stats?.pending_for_retraining ?? 0})
            </button>
          )}
        </div>
      </div>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Pending</h3>
          <div className="kpi">{stats?.pending ?? "—"}</div>
          <div className="sub">{stats?.in_review ?? 0} in review · {stats?.skipped ?? 0} skipped</div>
        </div>
        <div className="card">
          <h3>Resolved</h3>
          <div className="kpi">{stats?.resolved ?? "—"}</div>
          <div className="sub">{stats?.annotations ?? 0} annotations total</div>
        </div>
        <div className="card">
          <h3>Model agreement</h3>
          <div className="kpi">{fmtPct(stats?.model_agreement_rate ?? null, 0)}</div>
          <div className="sub">human label = model label</div>
        </div>
        <div className="card">
          <h3>Inter-annotator κ</h3>
          <div className="kpi">{agreement.data?.fleiss_kappa !== null && agreement.data?.fleiss_kappa !== undefined ? agreement.data.fleiss_kappa.toFixed(2) : "—"}</div>
          <div className="sub">
            {agreement.data?.interpretation} · {agreement.data?.posts_with_multiple_annotations ?? 0} double-annotated
          </div>
        </div>
      </div>

      {focused.data && (
        <div className="card" style={{ marginBottom: 16, borderColor: "var(--accent)" }}>
          <h3>
            Annotate post <span className="mono">{focused.data.id}</span> · <Link to={`/posts/${focused.data.id}`}>details</Link>
          </h3>
          <TokenHeatmap explanation={focused.data.classification?.explanation} text={focused.data.text} />
          <div style={{ margin: "8px 0", display: "flex", gap: 6 }}>
            <LabelBadge label={focused.data.toxicity_label} /> <SeverityBadge level={focused.data.severity_level} /> <TargetBadges targets={focused.data.targets} />
          </div>
          <AnnotationForm postId={focused.data.id} initialLabel={focused.data.toxicity_label || "non_toxic"} initialSeverity={focused.data.severity_level || 1} initialTargets={focused.data.targets || []} onDone={() => queue.reload()} />
        </div>
      )}

      <div className="split">
        <div className="card">
          <div className="toolbar" style={{ justifyContent: "space-between", marginBottom: 10 }}>
            <h3 style={{ margin: 0 }}>Queue</h3>
            <div className="pill-group">
              {["pending", "resolved", "skipped"].map((s) => (
                <button key={s} className={status === s ? "active" : ""} onClick={() => setStatus(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
          {!queue.data ? (
            <Loading error={queue.error} />
          ) : queue.data.items.length === 0 ? (
            <p className="muted">Queue is empty. Use “Refill queue” to enqueue posts in the uncertainty band.</p>
          ) : (
            <div className="list">
              {queue.data.items.map((it) => (
                <div key={it.id} className="review-item" style={{ borderColor: selected?.id === it.id ? "var(--accent)" : undefined, cursor: "pointer" }} onClick={() => setSelected(it)}>
                  <div className="post-text" style={{ maxWidth: "none" }}>
                    {it.text}
                  </div>
                  <div style={{ marginTop: 6, display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                    <LabelBadge label={it.model_label} /> <SeverityBadge level={it.severity_level} /> <TargetBadges targets={it.targets} />
                    <span className="muted" style={{ fontSize: 11, marginLeft: "auto" }}>
                      conf {it.model_confidence.toFixed(2)} · tox {it.final_toxicity.toFixed(2)} · {it.reason} · priority {it.priority.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="list">
          {notice && (
            <div className="alert info" role="status">
              {notice}
            </div>
          )}
          {selected ? (
            <div className="card">
              <h3>
                Annotate · <Link to={`/posts/${selected.post_id}`}>open post</Link>
              </h3>
              <TokenHeatmap explanation={selected.explanation} text={selected.text} />
              <hr />
              <AnnotationForm
                postId={selected.post_id}
                initialLabel={selected.model_label}
                initialSeverity={selected.severity_level || 1}
                initialTargets={selected.targets}
                onDone={(m) => {
                  setNotice(`${m} Next item loaded.`);
                  queue.reload();
                  agreement.reload();
                  setSelected(null);
                }}
              />
              {status === "pending" && (
                <div style={{ marginTop: 10 }}>
                  <button disabled={!!busy} onClick={() => act(() => api(`/review/${selected.id}/skip`, { method: "POST" }), "skip").then(() => setSelected(null))}>
                    Skip item
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="card muted">Select a queue item to annotate.</div>
          )}
          <div className="card">
            <h3>Retraining batches</h3>
            {batches.data?.length ? (
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Created</th>
                    <th>Examples</th>
                    <th>Corrections</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {batches.data.map((b) => (
                    <tr key={b.id}>
                      <td>{b.id}</td>
                      <td className="muted">{new Date(b.created_at).toLocaleDateString()}</td>
                      <td>{b.n_examples}</td>
                      <td>{b.n_corrections}</td>
                      <td>
                        <span className="badge muted">{b.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">No fine-tuning batches yet. Annotations are bundled bi-weekly (or manually) once ≥ 20 are available.</p>
            )}
          </div>
          {agreement.data && agreement.data.pairwise_cohen_kappa.length > 0 && (
            <div className="card">
              <h3>Pairwise Cohen's κ</h3>
              <table>
                <tbody>
                  {agreement.data.pairwise_cohen_kappa.map((p) => (
                    <tr key={`${p.a}-${p.b}`}>
                      <td>
                        {p.a} ↔ {p.b}
                      </td>
                      <td className="mono">{p.kappa?.toFixed(2) ?? "—"}</td>
                      <td className="muted">n={p.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
