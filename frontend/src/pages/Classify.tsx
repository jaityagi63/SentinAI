import { useState, type FormEvent } from "react";
import { api, type Classification } from "../lib/api";
import { Disclaimer, TokenHeatmap } from "../components/ui";
import { ClassificationDetail } from "./Post";

const EXAMPLES = [
  "Those people are all criminals and should be sent back where they came from.",
  "k1ll all the j3ws before they take over",
  "Can't believe someone said 'all Muslims are terrorists' — that's disgusting, report them.",
  "my nigga just got into college, so proud",
  "The immigrants are taking over the whole town",
  "Los inmigrantes son una plaga que hay que eliminar",
  "o2tolo kol el yahood",
];

export default function ClassifyPage() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [parentText, setParentText] = useState("");
  const [isReply, setIsReply] = useState(false);
  const [isQuote, setIsQuote] = useState(false);
  const [altText, setAltText] = useState("");
  const [result, setResult] = useState<{ classification: Classification; preprocessed: Record<string, unknown> } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(e?: FormEvent) {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ classification: Classification; preprocessed: Record<string, unknown> }>("/classify", {
        method: "POST",
        body: JSON.stringify({ text, parent_text: parentText || null, is_reply: isReply || !!parentText, is_quote: isQuote, media_alt_text: altText ? [altText] : [], explain: true }),
      });
      setResult(r);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const pre = result?.preprocessed as
    | {
        normalized?: string;
        tokens?: string[];
        language?: string;
        language_confidence?: number;
        script?: string | null;
        transliterated?: boolean;
        hashtags?: string[];
        emojis?: string[];
        mentions_count?: number;
        urls_count?: number;
        obfuscation?: { leetspeak: string[]; homoglyphs: string[]; zero_width_chars: number; spaced_out_words: string[]; repeated_chars: string[]; score: number };
      }
    | undefined;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Classification playground</h1>
          <p>Run the full pipeline on arbitrary text — preprocessing, three classification heads, context discounting and token attribution.</p>
        </div>
      </div>
      <div className="split">
        <div className="card">
          <form onSubmit={run} className="list">
            <textarea rows={4} value={text} onChange={(e) => setText(e.target.value)} placeholder="Post text…" />
            <div className="chips">
              {EXAMPLES.map((ex) => (
                <span key={ex} className="chip" onClick={() => setText(ex)}>
                  {ex.slice(0, 40)}
                  {ex.length > 40 ? "…" : ""}
                </span>
              ))}
            </div>
            <details>
              <summary>Conversation context &amp; media (Modules 5 &amp; 6)</summary>
              <div className="list" style={{ marginTop: 8 }}>
                <textarea rows={2} value={parentText} onChange={(e) => setParentText(e.target.value)} placeholder="Parent post text (makes this a reply / quote)" />
                <label className="muted">
                  <input type="checkbox" checked={isReply} onChange={(e) => setIsReply(e.target.checked)} /> reply
                </label>
                <label className="muted">
                  <input type="checkbox" checked={isQuote} onChange={(e) => setIsQuote(e.target.checked)} /> quote-tweet
                </label>
                <input value={altText} onChange={(e) => setAltText(e.target.value)} placeholder="Image alt text / OCR text (simulates the vision pipeline)" />
              </div>
            </details>
            <div>
              <button className="primary" disabled={busy || !text.trim()}>
                {busy ? "Classifying…" : "Classify"}
              </button>
            </div>
            {error && <div className="error">{error}</div>}
          </form>
        </div>
        <div className="card">
          <h3>Preprocessing (Module 2)</h3>
          {pre ? (
            <dl className="kv">
              <dt>Normalized</dt>
              <dd className="mono">{pre.normalized}</dd>
              <dt>Language</dt>
              <dd>
                {pre.language} <span className="muted">({((pre.language_confidence ?? 0) * 100).toFixed(0)}%{pre.script ? `, ${pre.script} script` : ""})</span> {pre.transliterated && <span className="badge muted">transliterated</span>}
              </dd>
              <dt>Obfuscation</dt>
              <dd>
                <span className="mono">{pre.obfuscation?.score.toFixed(2)}</span>
                {pre.obfuscation && pre.obfuscation.score > 0 && (
                  <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>
                    {pre.obfuscation.leetspeak.length > 0 && `leetspeak: ${pre.obfuscation.leetspeak.join(", ")} `}
                    {pre.obfuscation.homoglyphs.length > 0 && `homoglyphs: ${pre.obfuscation.homoglyphs.join(", ")} `}
                    {pre.obfuscation.spaced_out_words.length > 0 && `spaced: ${pre.obfuscation.spaced_out_words.join(", ")} `}
                    {pre.obfuscation.repeated_chars.length > 0 && `repeats: ${pre.obfuscation.repeated_chars.join(", ")} `}
                    {pre.obfuscation.zero_width_chars > 0 && `${pre.obfuscation.zero_width_chars} zero-width chars`}
                  </span>
                )}
              </dd>
              <dt>Removed</dt>
              <dd className="muted">
                {pre.urls_count ?? 0} URLs · {pre.mentions_count ?? 0} mentions anonymised
              </dd>
              {pre.hashtags && pre.hashtags.length > 0 && (
                <>
                  <dt>Hashtags</dt>
                  <dd>{pre.hashtags.join(", ")}</dd>
                </>
              )}
              {pre.emojis && pre.emojis.length > 0 && (
                <>
                  <dt>Emojis</dt>
                  <dd>{pre.emojis.join(" ")}</dd>
                </>
              )}
              <dt>Tokens</dt>
              <dd className="mono" style={{ fontSize: 11 }}>
                {pre.tokens?.join(" · ")}
              </dd>
            </dl>
          ) : (
            <p className="muted">Run a classification to see normalisation, language routing and obfuscation detection.</p>
          )}
        </div>
      </div>

      {result && (
        <>
          <div className="card" style={{ marginTop: 16 }}>
            <h3>Token attribution (Module 4)</h3>
            <TokenHeatmap explanation={result.classification.explanation} text={text} />
          </div>
          <div style={{ marginTop: 16 }}>
            <ClassificationDetail c={result.classification} />
          </div>
          <div style={{ marginTop: 12 }}>
            <Disclaimer />
          </div>
        </>
      )}
    </>
  );
}
