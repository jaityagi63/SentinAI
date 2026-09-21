import { useEffect, useState } from "react";
import { BrowserRouter, Navigate, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { loadSession, saveSession, setUnauthorizedHandler, type Session } from "./lib/api";
import LoginPage from "./pages/Login";
import OverviewPage from "./pages/Overview";
import HeatmapPage from "./pages/Heatmap";
import TrendsPage from "./pages/Trends";
import NetworkPage from "./pages/Network";
import ExplorerPage from "./pages/Explorer";
import PostPage from "./pages/Post";
import SeverityPage from "./pages/Severity";
import BotsPage from "./pages/Bots";
import AccountsPage from "./pages/Accounts";
import ReviewPage from "./pages/Review";
import FairnessPage from "./pages/Fairness";
import ClassifyPage from "./pages/Classify";

const NAV = [
  { to: "/", icon: "◎", label: "Overview", perm: "read" },
  { to: "/heatmap", icon: "▦", label: "Target heatmap", perm: "read" },
  { to: "/trends", icon: "↗", label: "Temporal trends", perm: "read" },
  { to: "/network", icon: "⦿", label: "Propagation graph", perm: "read" },
  { to: "/explorer", icon: "☰", label: "Post explorer", perm: "read" },
  { to: "/severity", icon: "▤", label: "Severity", perm: "read" },
  { to: "/bots", icon: "⚙", label: "Bots vs humans", perm: "read" },
  { to: "/accounts", icon: "◉", label: "Account scores", perm: "read" },
  { to: "/classify", icon: "⌕", label: "Classify text", perm: "read" },
  { to: "/review", icon: "✎", label: "Human review", perm: "review" },
  { to: "/fairness", icon: "⚖", label: "Fairness audit", perm: "audit" },
];

function Shell({ session, onLogout }: { session: Session; onLogout: () => void }) {
  const can = (p: string) => session.permissions.includes(p);
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span>🛡️</span>
          <div>
            SentinAI
            <small>Bias &amp; hate speech intelligence</small>
          </div>
        </div>
        <nav className="nav">
          {NAV.filter((n) => can(n.perm)).map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}>
              <span className="icon">{n.icon}</span>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        <div className="userbox">
          <div>
            <b>{session.username}</b> <span className="role">{session.role}</span>
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
            <a href="/api/docs" target="_blank" rel="noreferrer">
              API docs
            </a>
            <button onClick={onLogout} style={{ padding: "3px 8px", fontSize: 12 }}>
              Sign out
            </button>
          </div>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/heatmap" element={<HeatmapPage />} />
          <Route path="/trends" element={<TrendsPage />} />
          <Route path="/network" element={<NetworkPage />} />
          <Route path="/explorer" element={<ExplorerPage />} />
          <Route path="/posts/:id" element={<PostPage />} />
          <Route path="/severity" element={<SeverityPage />} />
          <Route path="/bots" element={<BotsPage />} />
          <Route path="/accounts" element={<AccountsPage />} />
          <Route path="/accounts/:id" element={<AccountsPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="/review" element={can("review") ? <ReviewPage /> : <Navigate to="/" />} />
          <Route path="/fairness" element={can("audit") ? <FairnessPage /> : <Navigate to="/" />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}

function Root() {
  const [session, setSession] = useState<Session | null>(() => loadSession());
  const navigate = useNavigate();
  useEffect(() => {
    setUnauthorizedHandler(() => {
      saveSession(null);
      setSession(null);
      navigate("/login");
    });
  }, [navigate]);
  if (!session) {
    return (
      <Routes>
        <Route path="*" element={<LoginPage onLogin={(s) => setSession(s)} />} />
      </Routes>
    );
  }
  return (
    <Shell
      session={session}
      onLogout={() => {
        saveSession(null);
        setSession(null);
      }}
    />
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Root />
    </BrowserRouter>
  );
}
