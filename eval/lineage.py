from __future__ import annotations

import pandas as pd
import networkx as nx


def build_lineage_graph(strategy_memory: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in strategy_memory.iterrows():
        cid = row["campaign_id"]
        g.add_node(cid, generation=int(row["generation"]), family=row.get("attack_family", "unknown"),
                   fitness=float(row.get("fitness", 0)), detection_probability=float(row.get("detection_probability", row.get("risk_score", 0))),
                   detected=bool(row.get("detected", False)), mutation_summary=row.get("mutation_summary", ""))
        parent = row.get("parent_campaign_id")
        if pd.notna(parent) and parent:
            g.add_edge(parent, cid, mutation_summary=row.get("mutation_summary", ""))
    return g


def lineage_paths(g: nx.DiGraph):
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    paths = []
    for leaf in leaves:
        candidates = [r for r in roots if nx.has_path(g, r, leaf)]
        if candidates:
            paths.append(nx.shortest_path(g, candidates[0], leaf))
    return paths


def plot_lineage_tree(g: nx.DiGraph, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    by_gen = {}
    for node, data in g.nodes(data=True):
        by_gen.setdefault(data["generation"], []).append(node)
    pos = {}
    for gen, nodes in sorted(by_gen.items()):
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = ((i - (n - 1) / 2) / max(1, n), -gen)
    for u, v in g.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], alpha=.35, linewidth=1)
    for node, data in g.nodes(data=True):
        x, y = pos[node]
        ax.scatter(x, y, s=60 + 300 * float(data["fitness"]), alpha=.85)
    ax.set_xticks([])
    ax.set_yticks([-g for g in sorted(by_gen)])
    ax.set_yticklabels([f"Gen {g}" for g in sorted(by_gen)])
    ax.set_title("Attack Lineage — node size = fitness")
    return ax
