import { useState, type FormEvent } from "react";
import { login, type Session } from "../lib/api";

export default function LoginPage({ onLogin }: { onLogin: (s: Session) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onLogin(await login(username, password));
    } catch (err) {
      setError((err as Error).message || "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <div className="card">
        <h1>🛡️ SentinAI</h1>
        <p className="muted">Social Media Bias &amp; Hate Speech Intelligence Platform</p>
        <form onSubmit={submit}>
          <label>
            <div className="muted">Username</div>
            <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus style={{ width: "100%" }} />
          </label>
          <label>
            <div className="muted">Password</div>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} style={{ width: "100%" }} />
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
