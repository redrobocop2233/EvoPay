# EVO-PAY — Lineage Tree + Time-to-Adapt KPI

> Read the real files first — `red_and_blue_team/red_team.py` (MutationEngine,
> EvolutionEngine, MemoryRecord, StrategyMemory), `integration/closed_loop.py`,
> and both `dashboard/app.py` / `ui/app.py` — before editing. Field names below
> (`parent_id`, `family`, etc.) are best-guess based on the context doc; confirm
> against the actual `MemoryRecord`/`AttackStrategy` definitions and adapt.
>
> These two features share data (both need generation + family + fitness/
> detection history per campaign), so do Section 1 first — it adds the parent
> tracking that Section 2's per-family analysis also benefits from, even though
> Section 2 doesn't strictly require it.

---

## Section 1 — Genome Lineage Tree

### Problem
The mutation history across generations already exists implicitly (each
generation's population is derived from the previous one via
`EvolutionEngine`/`MutationEngine`), but nothing currently visualizes it as a
tree. This is the single most "instantly legible from across a room" addition
available — a live demo is being sold on the idea of evolution, and right now
the only view of that is a line chart.

### Step 1 — Track parentage explicitly
Check whether `MemoryRecord` (or wherever a campaign's evaluation result is
stored) already has a parent reference. If not, add one:

```python
# red_and_blue_team/red_team.py
@dataclass
class MemoryRecord:
    campaign_id: str
    generation: int
    genome: "AttackGenome"
    family: str
    fitness: float
    detection_probability: float
    attack_success: float
    parent_campaign_id: str | None = None   # NEW — None for generation-0 seeds
    mutation_summary: str | None = None      # NEW — e.g. "velocity +0.18, geographic -0.09"
```

Populate `parent_campaign_id` and `mutation_summary` wherever `MutationEngine`
produces a child genome from a parent — this is likely in a method like
`mutate(parent_genome) -> AttackGenome` inside `EvolutionEngine`/
`MutationEngine`. Wrap that call site so the resulting child `MemoryRecord`
records which parent it came from and a short human-readable diff:

```python
def _summarize_mutation(parent_genome: "AttackGenome", child_genome: "AttackGenome",
                          dim_names=("amount","temporal","device","geographic",
                                     "merchant","velocity","coordination")) -> str:
    diffs = []
    for name, p, c in zip(dim_names, parent_genome.values, child_genome.values):
        if abs(c - p) > 0.03:
            sign = "+" if c > p else "-"
            diffs.append(f"{name} {sign}{abs(c - p):.2f}")
    return ", ".join(diffs) if diffs else "no significant change"
```

Ensure `parent_campaign_id` and `mutation_summary` get written into
`strategy_memory.csv` alongside the existing columns (generation, fitness,
detection_probability, family, etc.) — this file becomes the single source for
both the lineage tree and, if useful, richer autopsy displays.

### Step 2 — Build the lineage graph post-run
Use `networkx`, already in `requirements.txt`:

```python
# eval/lineage.py
import networkx as nx
import pandas as pd

def build_lineage_graph(strategy_memory: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in strategy_memory.iterrows():
        g.add_node(
            row["campaign_id"],
            generation=row["generation"],
            family=row["family"],
            fitness=row["fitness"],
            detection_probability=row["detection_probability"],
            detected=row["detection_probability"] >= 0.5,  # adapt threshold as needed
        )
        if pd.notna(row.get("parent_campaign_id")):
            g.add_edge(row["parent_campaign_id"], row["campaign_id"],
                       mutation_summary=row.get("mutation_summary", ""))
    return g
```

### Step 3 — Render it
Simplest reliable option given current dependencies (matplotlib is already
there, no need for graphviz/plotly as a new dependency): a manual generation-
banded layout — x-position spreads nodes within a generation, y-position is
generation number (so it reads top-to-bottom or left-to-right as a dendrogram).

```python
# eval/lineage.py (continued)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_lineage_tree(g: nx.DiGraph, ax=None):
    ax = ax or plt.gca()

    # group nodes by generation, assign x positions within each generation band
    by_gen: dict[int, list[str]] = {}
    for node, data in g.nodes(data=True):
        by_gen.setdefault(data["generation"], []).append(node)

    pos = {}
    for gen, nodes in sorted(by_gen.items()):
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - n / 2) / max(n, 1)
            pos[node] = (x, -gen)  # negative so gen 0 is at top

    # edges first (so nodes draw on top)
    for u, v in g.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#3A4250", linewidth=1, zorder=1, alpha=0.6)

    # nodes: color = detected (blue) vs bypassed (red), size = fitness
    for node, data in g.nodes(data=True):
        x, y = pos[node]
        color = "#E5484D" if not data["detected"] else "#38BDF8"
        size = 60 + 300 * data["fitness"]
        ax.scatter(x, y, s=size, c=color, edgecolors="#0D1117", linewidths=0.8, zorder=2)

    ax.set_yticks(sorted(-g for g in by_gen.keys()))
    ax.set_yticklabels([f"Gen {g}" for g in sorted(by_gen.keys())])
    ax.set_xticks([])
    ax.set_facecolor("#0D1117")
    ax.set_title("Attack Lineage — bypassed (red) vs detected (blue), size = fitness")
    return ax
```

Wire into the dashboard as a new tab (both `dashboard/app.py` and `ui/app.py`
already have a tabbed layout per the Section 5/6 work — add "Lineage" as a tab
alongside the existing Core Thesis / Evolution / Data tabs):

```python
# dashboard/app.py (or ui/app.py) — inside the existing tab structure
with tab_lineage:
    strategy_memory = pd.read_csv(results_dir / "strategy_memory.csv")
    graph = build_lineage_graph(strategy_memory)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0D1117")
    plot_lineage_tree(graph, ax=ax)
    st.pyplot(fig)

    # optional: click-free hover substitute — a selectbox to inspect one lineage path
    leaf_options = [n for n in graph.nodes if graph.out_degree(n) == 0]
    selected = st.selectbox("Trace a lineage", leaf_options)
    if selected:
        path = nx.shortest_path(graph, source=[n for n in graph.nodes
                                  if graph.in_degree(n) == 0 and nx.has_path(graph, n, selected)][0],
                                 target=selected)
        for node in path:
            data = graph.nodes[node]
            st.markdown(f"**Gen {data['generation']}** — {data['family']} — "
                        f"fitness={data['fitness']:.2f} — "
                        f"{'bypassed' if not data['detected'] else 'detected'}")
```

### Acceptance criteria
- `strategy_memory.csv` has `parent_campaign_id` and `mutation_summary` columns
  populated for every non-generation-0 row.
- The lineage tab renders without errors on the frozen seed-sweep-winning run
  and visually shows branching/mutation across generations.
- At least one selectable lineage path shows a coherent mutation story (e.g. a
  campaign that started detected and, through 2-3 mutations, ends up bypassing
  Blue) — worth checking this exists in your frozen run before demo day; if
  every lineage in your winning seed gets caught immediately, the tree will
  look inert rather than dynamic.

---

## Section 2 — Time-to-Adapt KPI

### Problem
Detection rate, F1, and the two-curve chart are all correct but require a
judge to interpret a trend. A single number — "the system needed N generations
to bring a new attack family's detection above 90%" — is a one-sentence,
memorable proof of the adaptive-defense thesis that doesn't require reading a
chart.

### Step 1 — Compute per-family, per-generation detection rate
This needs `strategy_memory.csv` grouped by `(family, generation)`:

```python
# eval/time_to_adapt.py
import pandas as pd

DETECTION_TARGET = 0.90

def compute_time_to_adapt(strategy_memory: pd.DataFrame,
                            target: float = DETECTION_TARGET) -> pd.DataFrame:
    """
    Returns one row per family:
      family, first_generation, first_gen_detection_rate,
      generation_reached_target (None if never reached within the run),
      time_to_adapt (generation_reached_target - first_generation, or None)
    """
    strategy_memory = strategy_memory.copy()
    strategy_memory["detected"] = strategy_memory["detection_probability"] >= 0.5

    per_family_gen = (
        strategy_memory.groupby(["family", "generation"])["detected"]
        .mean()
        .reset_index(name="detection_rate")
    )

    rows = []
    for family, group in per_family_gen.groupby("family"):
        group = group.sort_values("generation")
        first_gen = int(group["generation"].iloc[0])
        first_rate = float(group["detection_rate"].iloc[0])

        reached = group[group["detection_rate"] >= target]
        if not reached.empty:
            reached_gen = int(reached["generation"].iloc[0])
            time_to_adapt = reached_gen - first_gen
        else:
            reached_gen = None
            time_to_adapt = None

        rows.append({
            "family": family,
            "first_generation": first_gen,
            "first_gen_detection_rate": round(first_rate, 3),
            "generation_reached_target": reached_gen,
            "time_to_adapt": time_to_adapt,
        })

    return pd.DataFrame(rows)


def summarize_time_to_adapt(df: pd.DataFrame) -> dict:
    adapted = df.dropna(subset=["time_to_adapt"])
    return {
        "families_evaluated": len(df),
        "families_adapted_within_run": len(adapted),
        "avg_time_to_adapt_generations": (
            round(float(adapted["time_to_adapt"].mean()), 2) if not adapted.empty else None
        ),
        "median_time_to_adapt_generations": (
            round(float(adapted["time_to_adapt"].median()), 2) if not adapted.empty else None
        ),
        "fastest_adapt": (
            int(adapted["time_to_adapt"].min()) if not adapted.empty else None
        ),
        "families_never_adapted": len(df) - len(adapted),
    }
```

Note: a family that only ever appears in one generation (never survives
selection into a later generation) can't have a "time to adapt" computed —
that's expected and fine, it just means that family got wiped out immediately,
which is itself a valid (if less interesting) outcome to report.

### Step 2 — Wire into `closed_loop.py` and save to summary
```python
# integration/closed_loop.py, in the final summary-building step
strategy_memory_df = pd.read_csv(output_dir / "strategy_memory.csv")
ttа_df = compute_time_to_adapt(strategy_memory_df)
ttа_summary = summarize_time_to_adapt(ttа_df)

ttа_df.to_csv(output_dir / "time_to_adapt_by_family.csv", index=False)
summary["time_to_adapt"] = ttа_summary
```

### Step 3 — Surface it as a headline metric in the dashboard
Add it as a prominent stat card — this should sit near the top of the
dashboard, not buried in a data tab, since it's meant to be the number judges
repeat back later:

```python
# dashboard/app.py / ui/app.py — near the existing metrics strip
tta = summary.get("time_to_adapt", {})
if tta.get("avg_time_to_adapt_generations") is not None:
    st.metric(
        "⚡ Avg Time-to-Adapt",
        f"{tta['avg_time_to_adapt_generations']} generations",
        help=f"{tta['families_adapted_within_run']}/{tta['families_evaluated']} "
             f"families reached 90%+ detection within the run"
    )
```

Also add a small expandable table below it (`st.dataframe(tta_df)`) so a
curious judge can see the per-family breakdown if they ask.

### Acceptance criteria
- `time_to_adapt_by_family.csv` and `summary["time_to_adapt"]` exist after a
  run.
- The headline metric renders on the dashboard's main view (not buried in a
  secondary tab).
- Sanity-check the number on your frozen seed-sweep-winning run before demo
  day — if `avg_time_to_adapt_generations` comes out as 0 or 1 for every
  family (because Blue saturates almost immediately, per the saturation issue
  flagged earlier), it undercuts the "adaptive" story by making it look trivial
  rather than impressive. This is another reason to fix Blue's early
  saturation (Section B, Step 1 of the previous plan) before finalizing — a
  time-to-adapt of "essentially instant" isn't as compelling a story as
  "2-3 generations," even though it sounds good in isolation.

---

## Suggested demo integration
Sequence these two into the same "wow" moment: open on the headline
Time-to-Adapt metric (one number, easy to say out loud), then click into the
Lineage tab and trace one specific family's path — showing visually *why* it
took that many generations, with the mutation-summary annotations from
Section 1 narrating each step. That combination — a number + a visual proof of
that number — is stronger than either alone.
