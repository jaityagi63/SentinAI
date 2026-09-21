import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, loadSession, type Classification, type PostSummary } from "../lib/api";
import { Badge, Disclaimer, LABEL_COLORS, LabelBadge, Loading, ProbBar, SEVERITY_COLORS, SEVERITY_NAMES, SeverityBadge, TargetBadges, TokenHeatmap, fmtDate, useAsync } from "../components/ui";

export function ClassificationDetail({ c }: { c: Classification }) {
  return (
    <div className="grid cols-2">
      <div className="card">
        <h3>Task A · toxicity</h3>
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
          <LabelBadge label={c.toxicity.label} />
          <span className="muted">confidence {(c.toxicity.confidence * 100).toFixed(0)}%</span>
        </div>
        <ProbBar probs={c.toxicity.probabilities} colors={LABEL_COLORS} />
        <dl className="kv" style={{ marginTop: 10 }}>
          <dt>Toxicity score</dt>
          <dd className="mono">{c.toxicity.toxicity_score.toFixed(3)}</dd>
          <dt>Final (after context)</dt>
          <dd className="mono">{c.final_toxicity.toFixed(3)}</dd>
          <dt>Language</dt>
          <dd>{c.language}</dd>
          <dt>Obfuscation score</dt>
          <dd className="mono">{c.obfuscation_score.toFixed(2)}</dd>
          <dt>Model</dt>
          <dd className="mono">{c.model_version}</dd>
        </dl>
      </div>
      <div className="card">
        <h3>Task C · severity (ordinal)</h3>
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 8 }}>
          <SeverityBadge level={c.severity.level} />
          <span className="muted">{SEVERITY_NAMES[c.severity.level]}</span>
        </div>
        <ProbBar probs={c.severity.probabilities} colors={SEVERITY_COLORS} />
        <dl className="kv" style={{ marginTop: 10 }}>
          <dt>Expected level</dt>
          <dd className="mono">{c.severity.expected_level.toFixed(2)}</dd>
          <dt>Confidence</dt>
          <dd className="mono">{c.severity.confidence.toFixed(2)}</dd>
          <dt>Needs review</dt>
          <dd>{c.needs_review ? <Badge kind="muted">yes — confidence in the 0.4–0.6 band</Badge> : "no"}</dd>
        </dl>
      </div>
      <div className="card">
        <h3>Task B · target groups</h3>
        {c.targets.length ? (
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Group</th>
                <th>Confidence</th>
                <th>Evidence</th>
              </tr>
            </thead>
            <tbody>
              {c.targets.map((t) => (
                <tr key={`${t.category}:${t.label}`}>
                  <td>{t.category}</td>
                  <td>{t.label.replace(/_/g, " ")}</td>
                  <td className="mono">{t.confidence.toFixed(2)}</td>
                  <td className="mono muted">{t.evidence.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No protected-group target identified.</p>
        )}
      </div>
      <div className="card">
        <h3>Context &amp; multimodal</h3>
        {c.context ? (
          <dl className="kv">
            <dt>Role</dt>
            <dd>{c.context.is_originator ? "originator" : c.context.is_quote ? "quote-tweet" : c.context.is_reply ? "responder" : "retweet"}</dd>
            <dt>Stance</dt>
            <dd>
              <Badge kind="stance">{c.context.stance}</Badge> <span className="muted">({(c.context.stance_confidence * 100).toFixed(0)}%)</span>
            </dd>
            <dt>Discount</dt>
            <dd className="mono">×{c.context.discount_factor}</dd>
            {c.context.reasoning && (
              <>
                <dt>Reasoning</dt>
                <dd className="muted">{c.context.reasoning}</dd>
              </>
            )}
          </dl>
        ) : (
          <p className="muted">Original post (no conversation context).</p>
        )}
        {c.visual && (
          <>
            <hr />
            <dl className="kv">
              <dt>Visual toxicity</dt>
              <dd className="mono">{c.visual.visual_toxicity.toFixed(2)}</dd>
              <dt>OCR text</dt>
              <dd>{c.visual.ocr_text || <span className="muted">—</span>}</dd>
              <dt>Symbols</dt>
              <dd>{c.visual.symbols.length ? JSON.stringify(c.visual.symbols) : <span className="muted">none</span>}</dd>
              <dt>Juxtaposition</dt>
              <dd>{c.visual.juxtaposition_flag ? "⚠ benign text + hateful image" : "no"}</dd>
            </dl>
          </>
        )}
        {c.duplicate_of && (
          <p className="muted">
            Near-duplicate of <Link to={`/posts/${c.duplicate_of}`}>{c.duplicate_of}</Link>
          </p>
        )}
      </div>
    </div>
  );
}

export default function PostPage() {
  const { id } = useParams();
  const { data, error, reload } = useAsync(() => api<PostSummary>(`/posts/${id}`), [id]);
  const [busy, setBusy] = useState(false);
  const session = loadSession();
  const canReview = session?.permissions.includes("review");

  if (!data) return <Loading error={error} />;
  const c = data.classification;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Post explainability</h1>
          <p>
            <Link to="/explorer">← back to explorer</Link> · id <span className="mono">{data.id}</span>
          </p>
        </div>
        <div className="toolbar">
          {canReview && (
            <button
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await api(`/posts/${id}/reclassify`, { method: "POST" });
                  reload();
                } finally {
                  setBusy(false);
                }
              }}
            >
              ↻ Re-classify
            </button>
          )}
          {canReview && <Link to={`/review?post=${data.id}`}>annotate →</Link>}
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Token attribution (SHAP / LIME style)</h3>
        <TokenHeatmap explanation={c?.explanation} text={data.text} />
        <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>
          @{data.author_username} · {fmtDate(data.created_at)} · ❤ {data.engagement.likes} · ↻ {data.engagement.retweets} · 💬 {data.engagement.replies}
          {data.bot_probability !== null && data.bot_probability >= 0.8 && (
            <>
              {" "}
              · <span className="badge bot">likely bot ({data.bot_probability.toFixed(2)})</span>
            </>
          )}
          {" · "}
          <Link to={`/accounts/${data.author_id}`}>account score</Link>
          {data.source && data.source !== "demo" && /^\d+$/.test(data.id) && (
            <>
              {" · "}
              <a href={`https://x.com/i/web/status/${data.id}`} target="_blank" rel="noreferrer">
                open on X ↗
              </a>
            </>
          )}
          {data.source === "demo" && <> · synthetic demo post</>}
        </div>
        {c?.explanation && (
          <div className="grid cols-2" style={{ marginTop: 12 }}>
            <div>
              <div className="muted" style={{ fontSize: 12 }}>
                Top toxic drivers
              </div>
              <div className="chips">
                {c.explanation.top_positive.map((t) => (
                  <span key={t} className="chip" style={{ borderColor: "rgba(255,80,80,0.6)" }}>
                    {t}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <div className="muted" style={{ fontSize: 12 }}>
                Mitigating tokens
              </div>
              <div className="chips">
                {c.explanation.top_negative.length ? (
                  c.explanation.top_negative.map((t) => (
                    <span key={t} className="chip" style={{ borderColor: "rgba(62,207,142,0.6)" }}>
                      {t}
                    </span>
                  ))
                ) : (
                  <span className="muted">—</span>
                )}
              </div>
            </div>
          </div>
        )}
        {c?.explanation?.matched_patterns?.length ? (
          <details style={{ marginTop: 10 }}>
            <summary>{c.explanation.matched_patterns.length} matched lexicon / pattern rules</summary>
            <table style={{ marginTop: 6 }}>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Span</th>
                  <th>Weight</th>
                  <th>Pattern</th>
                </tr>
              </thead>
              <tbody>
                {c.explanation.matched_patterns.map((m, i) => (
                  <tr key={i}>
                    <td>{m.category}</td>
                    <td className="mono">{m.span}</td>
                    <td className="mono">{m.weight.toFixed(2)}</td>
                    <td className="mono muted" style={{ fontSize: 11 }}>
                      {m.pattern}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ) : null}
      </div>

      {c && <ClassificationDetail c={c} />}

      {(data.parent || (data.replies && data.replies.length > 0)) && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Conversation thread</h3>
          {data.parent && (
            <div className="review-item" style={{ marginBottom: 10 }}>
              <div className="muted" style={{ fontSize: 11 }}>
                ↑ parent · @{data.parent.author_username}
              </div>
              <Link to={`/posts/${data.parent.id}`}>{data.parent.text}</Link>
              <div style={{ marginTop: 6, display: "flex", gap: 6 }}>
                <LabelBadge label={data.parent.toxicity_label} /> <SeverityBadge level={data.parent.severity_level} /> <TargetBadges targets={data.parent.targets} />
              </div>
            </div>
          )}
          {data.replies?.map((r) => (
            <div className="review-item" key={r.id} style={{ marginLeft: 24, marginBottom: 8 }}>
              <div className="muted" style={{ fontSize: 11 }}>
                ↳ {r.quoted_id === data.id ? "quote" : "reply"} · @{r.author_username} · stance <Badge kind="stance">{r.stance}</Badge> {r.discount_factor !== undefined && r.discount_factor < 1 && <span className="muted">discount ×{r.discount_factor}</span>}
              </div>
              <Link to={`/posts/${r.id}`}>{r.text}</Link>
              <div style={{ marginTop: 6, display: "flex", gap: 6 }}>
                <LabelBadge label={r.toxicity_label} /> <SeverityBadge level={r.severity_level} />
              </div>
            </div>
          ))}
        </div>
      )}
      <div style={{ marginTop: 12 }}>
        <Disclaimer />
      </div>
    </>
  );
}
