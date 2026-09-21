"""Module 9 — Social network analysis & propagation mapping.

* Graph: directed author→author edges from retweets, quotes and replies (weighted by count and
  by the toxicity of the propagated content).
* Community detection: Leiden (``leidenalg`` + ``igraph``) when available, otherwise Louvain
  (``networkx.community.louvain_communities``).
* Influence: PageRank + betweenness centrality → "amplifier" ranking.
* Output is a D3-friendly ``{nodes, links, communities}`` payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from sentinai.storage.models import AuthorRow, ClassificationRow, PostRow


@dataclass
class NetworkPayload:
    nodes: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def build_graph(session: Session, *, days: int = 90, min_toxicity: float = 0.0, target: str | None = None) -> nx.DiGraph:
    since = datetime.utcnow() - timedelta(days=days)
    q = (
        select(PostRow.id, PostRow.author_id, PostRow.parent_id, PostRow.quoted_id, PostRow.retweeted_id, ClassificationRow.final_toxicity, ClassificationRow.target_labels)
        .join(ClassificationRow, ClassificationRow.post_id == PostRow.id)
        .where(PostRow.created_at >= since, PostRow.deleted_upstream.is_(False))
    )
    rows = session.execute(q).all()
    post_author = {r.id: r.author_id for r in rows}
    post_tox = {r.id: r.final_toxicity for r in rows}
    G = nx.DiGraph()
    for r in rows:
        if target and not any(t == target or t.startswith(target + ":") for t in (r.target_labels or [])):
            continue
        for ref, kind in ((r.retweeted_id, "retweet"), (r.quoted_id, "quote"), (r.parent_id, "reply")):
            if not ref or ref not in post_author:
                continue
            src, dst = r.author_id, post_author[ref]
            if src == dst:
                continue
            tox = post_tox.get(ref, 0.0) if kind != "reply" else r.final_toxicity
            if tox < min_toxicity:
                continue
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += 1
                G[src][dst]["toxicity"] = max(G[src][dst]["toxicity"], tox)
                G[src][dst]["kinds"][kind] = G[src][dst]["kinds"].get(kind, 0) + 1
            else:
                G.add_edge(src, dst, weight=1, toxicity=tox, kinds={kind: 1})
    return G


def detect_communities(G: nx.Graph) -> tuple[dict[str, int], str]:
    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        return {}, "none"
    try:  # Leiden
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        nodes = list(U.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in U.edges()])
        g.es["weight"] = [U[a][b].get("weight", 1) for a, b in U.edges()]
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=42)
        return {nodes[i]: cid for cid, comm in enumerate(part) for i in comm}, "leiden"
    except Exception:
        pass
    comms = nx.community.louvain_communities(U, weight="weight", seed=42)
    return {n: cid for cid, comm in enumerate(comms) for n in comm}, "louvain"


def analyze_network(session: Session, *, days: int = 90, min_toxicity: float = 0.5, max_nodes: int = 300, target: str | None = None) -> NetworkPayload:
    G = build_graph(session, days=days, min_toxicity=min_toxicity, target=target)
    if G.number_of_nodes() == 0:
        return NetworkPayload(stats={"nodes": 0, "edges": 0})

    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight=None, normalized=True, k=min(200, G.number_of_nodes()), seed=42) if G.number_of_nodes() > 2 else dict.fromkeys(G.nodes, 0.0)
    communities, algo = detect_communities(G)

    # keep the most influential nodes
    keep = sorted(G.nodes, key=lambda n: pagerank.get(n, 0), reverse=True)[:max_nodes]
    keep_set = set(keep)
    H = G.subgraph(keep_set)

    authors = {a.id: a for a in session.scalars(select(AuthorRow).where(AuthorRow.id.in_(list(keep_set)))).all()}
    # per-author toxicity (mean over their posts in window)
    since = datetime.utcnow() - timedelta(days=days)
    tox_rows = session.execute(
        select(PostRow.author_id, ClassificationRow.final_toxicity).join(ClassificationRow, ClassificationRow.post_id == PostRow.id).where(PostRow.author_id.in_(list(keep_set)), PostRow.created_at >= since)
    ).all()
    tox_acc: dict[str, list[float]] = {}
    for aid, t in tox_rows:
        tox_acc.setdefault(aid, []).append(t)

    nodes = []
    for n in keep:
        a = authors.get(n)
        toxs = tox_acc.get(n, [])
        nodes.append(
            {
                "id": n,
                "username": a.username if a else n,
                "pagerank": round(pagerank.get(n, 0.0), 6),
                "betweenness": round(betweenness.get(n, 0.0), 6),
                "community": communities.get(n, -1),
                "in_degree": G.in_degree(n, weight="weight"),
                "out_degree": G.out_degree(n, weight="weight"),
                "avg_toxicity": round(sum(toxs) / len(toxs), 3) if toxs else 0.0,
                "n_posts": len(toxs),
                "bot_probability": a.bot_probability if a else None,
                "followers": a.followers_count if a else 0,
            }
        )
    links = [{"source": u, "target": v, "weight": d["weight"], "toxicity": round(d["toxicity"], 3), "kinds": d["kinds"]} for u, v, d in H.edges(data=True)]

    comm_stats: dict[int, dict] = {}
    for nd in nodes:
        c = comm_stats.setdefault(nd["community"], {"id": nd["community"], "size": 0, "avg_toxicity": 0.0, "top_members": []})
        c["size"] += 1
        c["avg_toxicity"] += nd["avg_toxicity"]
        c["top_members"].append((nd["pagerank"], nd["username"]))
    comms = []
    for c in comm_stats.values():
        c["avg_toxicity"] = round(c["avg_toxicity"] / c["size"], 3)
        c["top_members"] = [u for _, u in sorted(c["top_members"], reverse=True)[:5]]
        comms.append(c)
    comms.sort(key=lambda c: c["size"], reverse=True)

    # Echo-chamber score: fraction of edges inside communities (higher → more insular).
    intra = sum(1 for u, v in G.edges if communities.get(u) == communities.get(v))
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "shown_nodes": len(nodes),
        "communities": len(comms),
        "algorithm": algo,
        "modularity_intra_ratio": round(intra / G.number_of_edges(), 3) if G.number_of_edges() else 0.0,
        "top_amplifiers": [nd["username"] for nd in sorted(nodes, key=lambda n: n["pagerank"], reverse=True)[:10]],
    }
    return NetworkPayload(nodes=nodes, links=links, communities=comms, stats=stats)
