/**
 * Render smoke tests for every dashboard view against a live backend (see setup.ts).
 * Each test asserts that the page renders real data (not the loading/error state) and that no
 * React runtime error is thrown, which catches schema drift between the API and the UI.
 */
import { cleanup, render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { login, loadSession, saveSession, api, type PostSummary } from "../lib/api";
import { backendAvailable } from "./setup";
import App from "../App";
import OverviewPage from "../pages/Overview";
import HeatmapPage from "../pages/Heatmap";
import TrendsPage from "../pages/Trends";
import NetworkPage from "../pages/Network";
import ExplorerPage from "../pages/Explorer";
import PostPage from "../pages/Post";
import SeverityPage from "../pages/Severity";
import BotsPage from "../pages/Bots";
import AccountsPage from "../pages/Accounts";
import ReviewPage from "../pages/Review";
import FairnessPage from "../pages/Fairness";
import ClassifyPage from "../pages/Classify";
import IngestPage from "../pages/Ingest";

const live = await backendAvailable();
const d = live ? describe : describe.skip;

function renderAt(path: string, routes: JSX.Element) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>{routes}</Routes>
    </MemoryRouter>,
  );
}

const noError = () => expect(screen.queryByText(/^Error:/)).toBeNull();

d("dashboard pages (live backend)", () => {
  beforeAll(async () => {
    await login("admin", "admin");
    expect(loadSession()?.role).toBe("admin");
  });
  beforeEach(async () => {
    if (loadSession()?.role !== "admin") await login("admin", "admin");
  });
  afterEach(cleanup);

  it("login flow + role-gated navigation", async () => {
    saveSession(null);
    render(<App />);
    expect(await screen.findByText(/Development accounts/)).toBeTruthy();
    const inputs = document.querySelectorAll("input");
    fireEvent.change(inputs[0], { target: { value: "researcher" } });
    fireEvent.change(inputs[1], { target: { value: "researcher" } });
    fireEvent.click(screen.getByText("Sign in"));
    await waitFor(() => expect(loadSession()?.role).toBe("researcher"), { timeout: 15000 });
    expect(await screen.findByText("Fairness audit")).toBeTruthy();
    expect(screen.queryByText("Human review")).toBeNull(); // researcher cannot review
    cleanup();
    saveSession(null);
    render(<App />);
    await screen.findByText(/Development accounts/);
    const inputs2 = document.querySelectorAll("input");
    fireEvent.change(inputs2[0], { target: { value: "moderator" } });
    fireEvent.change(inputs2[1], { target: { value: "moderator" } });
    fireEvent.click(screen.getByText("Sign in"));
    await waitFor(() => expect(loadSession()?.role).toBe("moderator"), { timeout: 15000 });
    expect(await screen.findByText("Human review")).toBeTruthy();
    expect(screen.queryByText("Fairness audit")).toBeNull(); // moderator cannot audit
  });

  it("overview renders KPIs and charts", async () => {
    renderAt("/", <Route path="/" element={<OverviewPage />} />);
    expect(await screen.findByText("Posts analysed", {}, { timeout: 20000 })).toBeTruthy();
    await waitFor(() => expect(screen.getAllByTestId("plot").length).toBeGreaterThanOrEqual(3), { timeout: 20000 });
    expect(await screen.findByText(/full heatmap/)).toBeTruthy();
    noError();
  });

  it("heatmap renders target × severity matrix", async () => {
    renderAt("/heatmap", <Route path="/heatmap" element={<HeatmapPage />} />);
    expect(await screen.findByText("Target × severity", {}, { timeout: 20000 })).toBeTruthy();
    await waitFor(() => expect(screen.getAllByTestId("plot").length).toBeGreaterThanOrEqual(2));
    fireEvent.click(screen.getByText("religion"));
    await waitFor(() => expect(screen.getAllByTestId("plot").length).toBeGreaterThanOrEqual(2));
    noError();
  });

  it("trends renders series, forecast band and events", async () => {
    renderAt("/trends", <Route path="/trends" element={<TrendsPage />} />);
    expect(await screen.findByText(/Toxic posts per day/, {}, { timeout: 20000 })).toBeTruthy();
    await waitFor(() => expect(screen.getAllByTestId("plot").length).toBeGreaterThanOrEqual(3), { timeout: 20000 });
    expect(await screen.findByText("Event correlation")).toBeTruthy();
    const plots = screen.getAllByTestId("plot");
    // main plot has non-toxic + toxic + spikes + 3 forecast traces
    expect(Number(plots[0].getAttribute("data-traces"))).toBeGreaterThanOrEqual(3);
    noError();
  });

  it("network renders stats and amplifier table", async () => {
    renderAt("/network", <Route path="/network" element={<NetworkPage />} />);
    expect(await screen.findByText("Nodes / edges", {}, { timeout: 20000 })).toBeTruthy();
    await waitFor(() => expect(document.querySelectorAll("svg circle").length).toBeGreaterThan(5), { timeout: 20000 });
    expect(screen.getByText(/Key amplifiers/)).toBeTruthy();
    noError();
  });

  it("explorer lists posts and filters by label", async () => {
    renderAt("/explorer?label=violent_extremism&order=toxicity", <Route path="/explorer" element={<ExplorerPage />} />);
    expect(await screen.findByText(/posts match/, {}, { timeout: 20000 })).toBeTruthy();
    const badges = await screen.findAllByText("violent extremism");
    expect(badges.length).toBeGreaterThan(1);
    noError();
  });

  it("post page renders token heatmap + all three task cards", async () => {
    const list = await api<{ items: PostSummary[] }>("/posts?label=hate_speech&limit=1&order=toxicity");
    const id = list.items[0].id;
    renderAt(`/posts/${id}`, <Route path="/posts/:id" element={<PostPage />} />);
    expect(await screen.findByText("Task A · toxicity", {}, { timeout: 20000 })).toBeTruthy();
    expect(screen.getByText("Task B · target groups")).toBeTruthy();
    expect(screen.getByText("Task C · severity (ordinal)")).toBeTruthy();
    expect(document.querySelectorAll(".heat-token").length).toBeGreaterThan(0);
    noError();
  });

  it("post page shows conversation thread for a reply", async () => {
    const list = await api<{ items: PostSummary[] }>("/posts?limit=200");
    const reply = list.items.find((p) => p.parent_id);
    if (!reply) return;
    renderAt(`/posts/${reply.id}`, <Route path="/posts/:id" element={<PostPage />} />);
    expect(await screen.findByText("Conversation thread", {}, { timeout: 20000 })).toBeTruthy();
    expect(screen.getByText(/↑ parent/)).toBeTruthy();
    noError();
  });

  it("severity page renders taxonomy cards", async () => {
    renderAt("/severity", <Route path="/severity" element={<SeverityPage />} />);
    expect(await screen.findByText(/Level 4 · Incitement to violence/, {}, { timeout: 20000 })).toBeTruthy();
    await waitFor(() => expect(screen.getAllByTestId("plot").length).toBe(3));
    noError();
  });

  it("bots page renders breakdown", async () => {
    renderAt("/bots", <Route path="/bots" element={<BotsPage />} />);
    expect(await screen.findByText("Likely bots", {}, { timeout: 20000 })).toBeTruthy();
    expect(screen.getByText("Toxic ratio · bots")).toBeTruthy();
    expect(await screen.findByText("Highest bot probability")).toBeTruthy();
    noError();
  });

  it("accounts list + detail with disclaimer", async () => {
    renderAt("/accounts", <Route path="/accounts" element={<AccountsPage />} />);
    expect((await screen.findAllByText(/Model-Estimated Probability/, {}, { timeout: 20000 })).length).toBeGreaterThan(0);
    const rows = await waitFor(() => {
      const r = document.querySelectorAll("tbody tr.clickable");
      expect(r.length).toBeGreaterThan(0);
      return r;
    });
    const id = rows[0].textContent || "";
    expect(id.startsWith("@")).toBe(true);
    cleanup();
    const list = await api<{ items: { author_id: string }[] }>("/analytics/accounts?limit=1&reliable_only=true");
    renderAt(`/accounts/${list.items[0].author_id}`, <Route path="/accounts/:id" element={<AccountsPage />} />);
    expect(await screen.findByText("Model-estimated discrimination propensity", {}, { timeout: 20000 })).toBeTruthy();
    expect(screen.getByText(/95% CI/)).toBeTruthy();
    expect(screen.getByText("Recent posts")).toBeTruthy();
    noError();
  });

  it("review queue renders items and submits an annotation", async () => {
    renderAt("/review", <Route path="/review" element={<ReviewPage />} />);
    expect(await screen.findByText("Pending", {}, { timeout: 20000 })).toBeTruthy();
    await waitFor(() => expect(document.querySelectorAll(".review-item").length).toBeGreaterThan(0), { timeout: 20000 });
    const submit = await screen.findByText("Submit annotation");
    fireEvent.click(submit);
    await waitFor(() => expect(screen.getAllByText(/Saved —/).length).toBeGreaterThan(0), { timeout: 20000 });
    noError();
  });

  it("fairness page renders benchmark report", async () => {
    renderAt("/fairness", <Route path="/fairness" element={<FairnessPage />} />);
    expect(await screen.findByText("dialect_smoke", {}, { timeout: 30000 })).toBeTruthy();
    expect(screen.getByText(/FPR gap/)).toBeTruthy();
    expect(screen.getAllByText(/Calibrated/).length).toBeGreaterThan(0);
    noError();
  });

  it("classify playground runs the pipeline and shows attribution", async () => {
    renderAt("/classify", <Route path="/classify" element={<ClassifyPage />} />);
    const ta = await screen.findByPlaceholderText("Post text…");
    fireEvent.change(ta, { target: { value: "k1ll all the j3ws before they take over" } });
    fireEvent.click(screen.getByText("Classify"));
    expect(await screen.findByText("Task A · toxicity", {}, { timeout: 20000 })).toBeTruthy();
    expect(screen.getByText("violent extremism")).toBeTruthy();
    expect(screen.getByText(/L4 · Incitement/)).toBeTruthy();
    expect(document.querySelectorAll(".heat-token").length).toBeGreaterThan(0);
    expect(screen.getByText(/leetspeak/)).toBeTruthy();
    noError();
  });

  it("ingest page shows connection state, imports a CSV and reports the result", async () => {
    renderAt("/ingest", <Route path="/ingest" element={<IngestPage />} />);
    expect(await screen.findByText("Corpus by source", {}, { timeout: 15000 })).toBeTruthy();
    expect(await screen.findByText(/real posts/)).toBeTruthy();
    fireEvent.click(screen.getByText("Upload file"));
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const csv = new File(["id,text,created_at,username\n880001,\"these dirty Asians need to go back to China\",2026-09-10T08:00:00Z,vitest_user\n"], "posts.csv", { type: "text/csv" });
    Object.defineProperty(input, "files", { value: [csv] });
    fireEvent.click(screen.getByText("Import & classify"));
    expect(await screen.findByText("classified & stored", {}, { timeout: 30000 })).toBeTruthy();
    expect(screen.getByText("hate speech")).toBeTruthy();
    noError();
  });
});
