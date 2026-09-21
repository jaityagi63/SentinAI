import { useState, type FormEvent } from "react";
import { login, type Session } from "../lib/api";

export default function LoginPage({ onLogin, notice }: { onLogin: (s: Session) => void; notice?: string | null }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLogin(await login(username.trim(), password));
    } catch (err) {
      const e = err as { status?: number; message?: string };
      setError(e.status === 401 ? "Invalid username or password." : e.message || "Login failed — is the API reachable?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <div className="card">
        <h1>🛡️ SentinAI</h1>
        <p className="muted">Social Media Bias &amp; Hate Speech Intelligence Platform</p>
        {notice && <div className="alert">{notice}</div>}
        <form onSubmit={submit}>
          <label>
            <div className="muted">Username</div>
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus autoComplete="username" autoCapitalize="none" spellCheck={false} style={{ width: "100%" }} />
          </label>
          <label>
            <div className="muted">Password</div>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" style={{ width: "100%" }} />
          </label>
          {error && <div className="error">{error}</div>}
          <button className="primary" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <hr />
        <div className="muted" style={{ fontSize: 12 }}>
          Development accounts: <span className="mono">admin / admin</span>, <span className="mono">researcher / researcher</span>, <span className="mono">moderator / moderator</span>
        </div>
      </div>
    </div>
  );
}
