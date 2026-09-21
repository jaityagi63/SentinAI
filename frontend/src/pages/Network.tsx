import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { Link } from "react-router-dom";
import { api, type NetLink, type NetNode, type Network } from "../lib/api";
import { DaysPicker, Loading, useAsync } from "../components/ui";

type SimNode = NetNode & d3.SimulationNodeDatum;
type SimLink = d3.SimulationLinkDatum<SimNode> & { weight: number; toxicity: number; kinds: Record<string, number> };

const PALETTE = d3.schemeTableau10;

export default function NetworkPage() {
  const [days, setDays] = useState(90);
  const [minTox, setMinTox] = useState(0.5);
  const [maxNodes, setMaxNodes] = useState(150);
  const [selected, setSelected] = useState<NetNode | null>(null);
  const { data, error, loading } = useAsync(() => api<Network>(`/analytics/network?days=${days}&min_toxicity=${minTox}&max_nodes=${maxNodes}`), [days, minTox, maxNodes]);
  const svgRef = useRef<SVGSVGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const svg = d3.select<SVGSVGElement, unknown>(svgEl);
    svg.selectAll("*").remove();
    if (!data || !data.nodes.length) return;
    const width = svgEl.clientWidth || 900;
    const height = svgEl.clientHeight || 620;
    const nodes: SimNode[] = data.nodes.map((n) => ({ ...n }));
    const ids = new Set(nodes.map((n) => n.id));
    const links: SimLink[] = data.links.filter((l: NetLink) => ids.has(l.source) && ids.has(l.target)).map((l) => ({ source: l.source, target: l.target, weight: l.weight, toxicity: l.toxicity, kinds: l.kinds }));
    const prMax = d3.max(nodes, (n) => n.pagerank) || 1;
    const r = (n: SimNode) => 4 + 22 * Math.sqrt(n.pagerank / prMax);

    const g = svg.append("g");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 6]).on("zoom", (ev) => g.attr("transform", ev.transform)),
    );
    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 14)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#5b6aa6");

    const link = g
      .append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (l) => d3.interpolateRgb("#4b5a99", "#ff6b6b")(l.toxicity))
      .attr("stroke-opacity", 0.55)
      .attr("stroke-width", (l) => Math.min(6, 0.8 + Math.log2(1 + l.weight)))
      .attr("marker-end", "url(#arrow)");

    const node = g
      .append("g")
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(nodes)
      .join("circle")
      .attr("r", r)
      .attr("fill", (n) => PALETTE[(n.community + 10) % 10])
      .attr("stroke", (n) => ((n.bot_probability ?? 0) >= 0.8 ? "#ff6b6b" : "#0b1020"))
      .attr("stroke-width", (n) => ((n.bot_probability ?? 0) >= 0.8 ? 2.5 : 1))
      .style("cursor", "pointer")
      .on("click", (_, n) => setSelected(n))
      .on("mousemove", (ev: MouseEvent, n) => {
        const t = tipRef.current;
        if (!t) return;
        t.style.display = "block";
        t.style.left = `${ev.clientX + 12}px`;
        t.style.top = `${ev.clientY + 12}px`;
        t.innerHTML = `<b>@${n.username}</b><br/>community ${n.community} · PageRank ${n.pagerank.toFixed(4)}<br/>avg toxicity ${n.avg_toxicity.toFixed(2)} · ${n.n_posts} posts<br/>in ${n.in_degree} / out ${n.out_degree}${(n.bot_probability ?? 0) >= 0.8 ? "<br/><span style='color:#ff6b6b'>likely bot</span>" : ""}`;
      })
      .on("mouseleave", () => {
        if (tipRef.current) tipRef.current.style.display = "none";
      })
      .call(
        d3
          .drag<SVGCircleElement, SimNode>()
          .on("start", (ev, n) => {
            if (!ev.active) sim.alphaTarget(0.3).restart();
            n.fx = n.x;
            n.fy = n.y;
          })
          .on("drag", (ev, n) => {
            n.fx = ev.x;
            n.fy = ev.y;
          })
          .on("end", (ev, n) => {
            if (!ev.active) sim.alphaTarget(0);
            n.fx = null;
            n.fy = null;
          }),
      );

    const label = g
      .append("g")
      .selectAll("text")
      .data(nodes.filter((n) => n.pagerank / prMax > 0.35))
      .join("text")
      .text((n) => `@${n.username}`)
      .attr("font-size", 10)
      .attr("fill", "#e6e9f5")
      .attr("dx", (n) => r(n) + 3)
      .attr("dy", 3);

    const sim = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink<SimNode, SimLink>(links).id((d) => d.id).distance(60).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-160))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide<SimNode>().radius((n) => r(n) + 2))
      .on("tick", () => {
        link
          .attr("x1", (l) => (l.source as SimNode).x!)
          .attr("y1", (l) => (l.source as SimNode).y!)
          .attr("x2", (l) => (l.target as SimNode).x!)
          .attr("y2", (l) => (l.target as SimNode).y!);
        node.attr("cx", (n) => n.x!).attr("cy", (n) => n.y!);
        label.attr("x", (n) => n.x!).attr("y", (n) => n.y!);
      });
    return () => {
      sim.stop();
    };
  }, [data]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Propagation graph</h1>
          <p>Retweet / quote / reply interactions around toxic content. Node size = PageRank influence, colour = community, red ring = likely bot.</p>
        </div>
        <div className="toolbar">
          <label className="muted">
            min toxicity{" "}
            <select value={minTox} onChange={(e) => setMinTox(Number(e.target.value))}>
              {[0, 0.3, 0.5, 0.7, 0.9].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="muted">
            nodes{" "}
            <select value={maxNodes} onChange={(e) => setMaxNodes(Number(e.target.value))}>
              {[50, 150, 300, 600].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <DaysPicker value={days} onChange={setDays} />
        </div>
      </div>
      <div className="split">
        <div className="card" style={{ padding: 8 }}>
          {loading && <Loading error={error} />}
          <svg ref={svgRef} className="network-svg" />
          <div ref={tipRef} className="tooltip" style={{ display: "none" }} />
        </div>
        <div className="list">
          <div className="card">
            <h3>Graph statistics</h3>
            {data && (
              <dl className="kv">
                <dt>Nodes / edges</dt>
                <dd>
                  {data.stats.nodes} / {data.stats.edges}
                </dd>
                <dt>Communities</dt>
                <dd>
                  {data.stats.communities} ({data.stats.algorithm})
                </dd>
                <dt>Intra-community edge ratio</dt>
                <dd>{data.stats.modularity_intra_ratio ?? "—"}</dd>
              </dl>
            )}
          </div>
          <div className="card">
            <h3>Key amplifiers (PageRank)</h3>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Account</th>
                  <th>PR</th>
                  <th>Tox</th>
                </tr>
              </thead>
              <tbody>
                {data?.nodes.slice(0, 12).map((n, i) => (
                  <tr key={n.id} className="clickable" onClick={() => setSelected(n)}>
                    <td className="muted">{i + 1}</td>
                    <td>
                      @{n.username} {(n.bot_probability ?? 0) >= 0.8 && <span className="badge bot">bot</span>}
                    </td>
                    <td className="mono">{n.pagerank.toFixed(3)}</td>
                    <td className="mono">{n.avg_toxicity.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card">
            <h3>Communities</h3>
            <table>
              <thead>
                <tr>
                  <th>Id</th>
                  <th>Size</th>
                  <th>Avg tox</th>
                  <th>Top members</th>
                </tr>
              </thead>
              <tbody>
                {data?.communities.slice(0, 8).map((c) => (
                  <tr key={c.id}>
                    <td>
                      <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 5, background: PALETTE[(c.id + 10) % 10], marginRight: 6 }} />
                      {c.id}
                    </td>
                    <td>{c.size}</td>
                    <td className="mono">{c.avg_toxicity.toFixed(2)}</td>
                    <td className="muted" style={{ fontSize: 12 }}>
                      {c.top_members.map((m) => `@${m}`).join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {selected && (
            <div className="card">
              <h3>Selected account</h3>
              <dl className="kv">
                <dt>Username</dt>
                <dd>@{selected.username}</dd>
                <dt>Community</dt>
                <dd>{selected.community}</dd>
                <dt>PageRank / betweenness</dt>
                <dd>
                  {selected.pagerank.toFixed(4)} / {selected.betweenness.toFixed(4)}
                </dd>
                <dt>Avg toxicity</dt>
                <dd>{selected.avg_toxicity.toFixed(2)} over {selected.n_posts} posts</dd>
                <dt>Bot probability</dt>
                <dd>{selected.bot_probability?.toFixed(2) ?? "—"}</dd>
                <dt>Followers</dt>
                <dd>{selected.followers.toLocaleString()}</dd>
              </dl>
              <div style={{ marginTop: 8 }}>
                <Link to={`/accounts/${selected.id}`}>account score →</Link>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
