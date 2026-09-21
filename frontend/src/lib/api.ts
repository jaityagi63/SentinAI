/** Typed API client. All calls go through the relative `/api` prefix (proxied in dev). */

export type Role = "admin" | "researcher" | "moderator";

export interface Session {
  token: string;
  username: string;
  role: Role;
  permissions: string[];
}

const KEY = "sentinai.session";

export function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function saveSession(s: Session | null) {
  if (s) localStorage.setItem(KEY, JSON.stringify(s));
  else localStorage.removeItem(KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(fn: () => void) {
  onUnauthorized = fn;
}

/** Auth headers. The token is sent twice on purpose: some reverse proxies / preview tunnels
 *  rewrite or drop `Authorization`, so the API also accepts `X-SentinAI-Token` (and a cookie). */
export function authHeaders(): Record<string, string> {
  const session = loadSession();
  if (!session) return {};
  return { Authorization: `Bearer ${session.token}`, "X-SentinAI-Token": session.token };
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders(), ...(init.headers as Record<string, string>) };
  if (init.body && !(init.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, { credentials: "include", ...init, headers });
  if (res.status === 401 && !path.startsWith("/auth/login")) {
    onUnauthorized?.();
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = j.detail ?? JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, msg);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

export async function login(username: string, password: string): Promise<Session> {
  saveSession(null);
  const r = await api<{ access_token: string; role: Role; username: string; permissions: string[] }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  const s: Session = { token: r.access_token, role: r.role, username: r.username, permissions: r.permissions };
  saveSession(s);
  return s;
}

export async function logout(): Promise<void> {
  try {
    await api("/auth/logout", { method: "POST" });
  } catch {
    /* best effort — the local session is cleared regardless */
  }
  saveSession(null);
}

export function downloadUrl(path: string): string {
  return `/api${path}`;
}

export async function downloadFile(path: string, filename: string) {
  const res = await fetch(`/api${path}`, { headers: authHeaders(), credentials: "include" });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------------------------
// response types
// ---------------------------------------------------------------------------------------------

export interface Overview {
  days: number;
  posts: number;
  by_source?: Record<string, number>;
  toxic_posts: number;
  toxic_ratio: number;
  avg_severity_toxic: number;
  by_label: Record<string, number>;
  by_severity: Record<string, number>;
  by_language: Record<string, number>;
  needs_review: number;
  authors: number;
  likely_bots: number;
}

export interface HeatCell {
  category: string;
  label: string;
  total: number;
  by_severity: Record<string, number>;
  avg_toxicity: number;
  avg_severity: number;
}

export interface TrendPoint {
  date: string;
  total: number;
  toxic: number;
  toxic_ratio: number;
  avg_severity: number;
  zscore: number;
  spike: boolean;
}

export interface ForecastPoint {
  date: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

export interface EventImpact {
  id: number;
  date: string;
  title: string;
  category: string;
  source: string;
  before_mean: number;
  after_mean: number;
  lift: number;
  related_targets: string[];
}

export interface Trends {
  series: TrendPoint[];
  spikes: TrendPoint[];
  forecast: ForecastPoint[];
  forecast_method: string;
  events: EventImpact[];
}

export interface NetNode {
  id: string;
  username: string;
  pagerank: number;
  betweenness: number;
  community: number;
  in_degree: number;
  out_degree: number;
  avg_toxicity: number;
  n_posts: number;
  bot_probability: number | null;
  followers: number;
}

export interface NetLink {
  source: string;
  target: string;
  weight: number;
  toxicity: number;
  kinds: Record<string, number>;
}

export interface Network {
  nodes: NetNode[];
  links: NetLink[];
  communities: { id: number; size: number; avg_toxicity: number; top_members: string[] }[];
  stats: Record<string, unknown> & { nodes: number; edges: number; communities?: number; algorithm?: string; top_amplifiers?: string[]; modularity_intra_ratio?: number };
}

export interface Attribution {
  token: string;
  start: number;
  end: number;
  weight: number;
}

export interface Explanation {
  method: string;
  text: string;
  attributions: Attribution[];
  base_value: number;
  top_positive: string[];
  top_negative: string[];
  matched_patterns: { category: string; weight: number; span: string; pattern: string }[];
}

export interface Classification {
  post_id: string;
  model_version: string;
  language: string;
  toxicity: { label: string; confidence: number; probabilities: Record<string, number>; toxicity_score: number };
  targets: { category: string; label: string; confidence: number; evidence: string[] }[];
  severity: { level: number; level_name: string; confidence: number; probabilities: Record<string, number>; expected_level: number };
  explanation: Explanation | null;
  visual: { ocr_text: string; ocr_engine: string | null; visual_toxicity: number; symbols: unknown[]; vlm_labels: Record<string, number>; juxtaposition_flag: boolean } | null;
  context: { is_reply: boolean; is_quote: boolean; is_retweet: boolean; is_originator: boolean; parent_id: string | null; stance: string; stance_confidence: number; discount_factor: number; reasoning: string | null } | null;
  obfuscation_score: number;
  duplicate_of: string | null;
  final_toxicity: number;
  needs_review: boolean;
}

export interface PostSummary {
  id: string;
  text: string;
  created_at: string;
  author_id: string;
  author_username: string | null;
  bot_probability: number | null;
  lang: string | null;
  parent_id: string | null;
  quoted_id: string | null;
  retweeted_id: string | null;
  engagement: { likes: number; retweets: number; replies: number; quotes: number };
  has_media: boolean;
  media: { media_key: string; type: string; alt_text?: string | null }[] | null;
  source?: string;
  toxicity_label?: string;
  toxicity_confidence?: number;
  final_toxicity?: number;
  severity_level?: number;
  targets?: string[];
  stance?: string | null;
  discount_factor?: number;
  visual_toxicity?: number | null;
  needs_review?: boolean;
  duplicate_of?: string | null;
  language?: string;
  model_version?: string;
  classification?: Classification;
  parent?: PostSummary;
  replies?: PostSummary[];
}

export interface ReviewItem {
  id: number;
  post_id: string;
  text: string;
  created_at: string;
  model_label: string;
  model_confidence: number;
  final_toxicity: number;
  severity_level: number;
  targets: string[];
  reason: string;
  priority: number;
  status: string;
  assigned_to: string | null;
  explanation: Explanation | null;
}

export interface QueueStats {
  pending: number;
  in_review: number;
  resolved: number;
  skipped: number;
  annotations: number;
  model_agreement_rate: number | null;
  pending_for_retraining: number;
  band: [number, number];
}

export interface BotReport {
  days: number;
  threshold: number;
  scored_accounts: number;
  likely_bots: number;
  breakdown: Record<string, { posts: number; toxic_posts: number; toxic_ratio: number }>;
  histogram: number[];
  top_accounts: { author_id: string; username: string; bot_probability: number; features: Record<string, number> | null; followers: number; following: number; account_score: number | null }[];
  model: string;
}

export interface AccountScore {
  author_id: string;
  username: string | null;
  n_posts: number;
  reliable: boolean;
  score: number | null;
  ci_low: number | null;
  ci_high: number | null;
  avg_severity: number;
  toxic_ratio: number;
  target_diversity: number;
  recency_weight: number;
  bot_probability: number | null;
  disclaimer: string;
}

export interface FairnessGroup {
  group: string;
  n: number;
  n_positive: number;
  n_negative: number;
  fpr: number;
  fnr: number;
  tpr: number;
  precision: number;
  mean_score_negative: number;
}

export interface FairnessReport {
  benchmark: string;
  threshold: number;
  reference_group: string;
  fpr_gap: number;
  fnr_gap: number;
  groups: FairnessGroup[];
  calibrated_thresholds: Record<string, number>;
  notes: string[];
}

// ---------------------------------------------------------------------------------------------
// ingestion (Module 1 — live X + imports)
// ---------------------------------------------------------------------------------------------

export interface IngestReport {
  kind: string;
  query: string | null;
  fetched: number;
  ingested: number;
  hydrated_parents: number;
  media_downloaded: number;
  endpoint: string | null;
  since_id: string | null;
  newest_id: string | null;
  toxic: number;
  by_label: Record<string, number>;
  post_ids: string[];
  warnings: string[];
  rate_limits: Record<string, { remaining: number | null; limit: number | null; reset_at: string | null }>;
  duration_seconds: number;
}

export interface IngestJob {
  id: string;
  kind: string;
  label: string;
  owner: string;
  params: Record<string, unknown>;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: Record<string, number>;
  report: IngestReport | null;
  error: string | null;
}

export interface IngestResponse {
  job: IngestJob | null;
  report: IngestReport | null;
}

export interface IngestStatus {
  token: { configured: boolean; source: string | null; hint: string | null };
  api_base: string;
  full_archive_default: boolean;
  download_media_default: boolean;
  max_wait_seconds: number;
  upload_max_mb: number;
  posts_by_source: Record<string, number>;
  cursors: { query: string; since_id: string | null; last_run: string | null; total_ingested: number }[];
  active_jobs: IngestJob[];
  stream_running: boolean;
  endpoints: Record<string, string>;
}

export interface XConnectionTest {
  ok: boolean;
  recent_search?: boolean;
  full_archive?: boolean | null;
  sample_posts?: number;
  latency_ms?: number;
  status?: number;
  error?: string;
  rate_limits: Record<string, { remaining: number | null; limit: number | null; reset_at: string | null }>;
}

export async function uploadFile<T>(path: string, file: File, fields: Record<string, string> = {}): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  for (const [k, v] of Object.entries(fields)) form.append(k, v);
  return api<T>(path, { method: "POST", body: form });
}
