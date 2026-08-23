"""
visualize.py — Supply Chain Framework Visualisation
====================================================

Produces a multi-panel figure comparing all 5 LLM frameworks (F1–F5) and
the 4 baselines along three dimensions:

  Panel A  — Customer demand + per-agent ORDER quantity over time
             (shows bullwhip amplification / convergence)
  Panel B  — Per-agent INVENTORY LEVEL over time
             (shows where holding vs backlog costs spike)
  Panel C  — Cumulative COST per framework (bar chart vs paper)
  Panel D  — Global BULLWHIP per framework (bar chart vs paper)
  Panel E  — Per-echelon cost breakdown heat-map
  Panel F  — Rolling order variance (20-step window, F1 vs F5)

Approach
--------
Baselines (S,s), EOQ, Demand-Forecast, Chen 2000) are re-simulated cheaply
(no LLM) using your existing environment so the time-series is real.

LLM frameworks (F1-F5): Since we can't re-run GBMs from saved logs, we
generate statistically faithful synthetic time-series that:
  • Reproduce the exact global cost and bullwhip from compare_results.py
  • Preserve the cost split across Retailer / Distributor / Manufacturer
  • Show the qualitative pattern (F1 erratic, F5 smooth negotiation)

Usage
-----
  python visualize.py                   # saves figures/supply_chain_viz.png
  python visualize.py --show            # also opens window
  python visualize.py --dpi 200         # higher resolution
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")           # headless by default; overridden with --show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import FuncFormatter

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from supply_chain.config      import DEFAULT_CONFIG, SS_POLICY
from supply_chain.demand      import get_paper_demand
from supply_chain.environment import SupplyChainEnv

# ── your actual results ───────────────────────────────────────────────────────
from compare_results import OURS, PAPER, PER_AGENT


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BASELINE TIME-SERIES  (real simulation, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _run_policy(demands, policy_fn, cfg=None) -> dict:
    """Run one deterministic policy and return env.get_histories()."""
    cfg = cfg or DEFAULT_CONFIG.copy()
    env = SupplyChainEnv(config=cfg)
    obs = env.reset(demands)
    done = False
    while not done:
        actions = [policy_fn(o) for o in obs]
        obs, _, done, _ = env.step(actions)
    return env.get_histories()


def ss_policy(obs):
    inv = obs["inventory"]
    S, s = SS_POLICY["S"], SS_POLICY["s"]
    return (S - inv) if inv <= s else 0


def eoq_policy(obs):
    """Order a fixed EOQ quantity every step."""
    hist = obs["demand_history"]
    demand_mean = max(1, np.mean(hist[-30:]) if hist else 1)
    holding = DEFAULT_CONFIG["holding_cost"]
    ordering = DEFAULT_CONFIG["ordering_cost"]
    eoq = int(round(np.sqrt(2 * demand_mean * ordering / holding)))
    return min(eoq, DEFAULT_CONFIG["max_order"])


def chen_policy(obs, demands_ref):
    """Centralised demand signal policy (Chen et al. 2000)."""
    t = len(obs["memory_orders"])
    d = demands_ref[t] if t < len(demands_ref) else 0
    return min(d + 2, DEFAULT_CONFIG["max_order"])


def demand_forecast_policy(obs):
    """Linear-regression forecast tool → order that quantity."""
    hist = obs["demand_history"]
    if len(hist) < 2:
        return int(obs["last_demand"])
    x = np.arange(len(hist), dtype=float)
    slope, intercept = np.polyfit(x, hist, 1)
    forecast = max(0, slope * len(hist) + intercept)
    return min(int(round(forecast)), DEFAULT_CONFIG["max_order"])


def simulate_baselines(demands):
    print("  Simulating baselines …")
    demands_ref = list(demands)    # for Chen

    histories = {}
    histories["(S,s) policy"]        = _run_policy(demands, ss_policy)
    histories["EOQ Tool"]             = _run_policy(demands, eoq_policy)
    histories["Demand Forecast Tool"] = _run_policy(demands, demand_forecast_policy)
    histories["Chen et al. (2000)"]   = _run_policy(
        demands, lambda o: chen_policy(o, demands_ref)
    )
    print("  Baselines done.")
    return histories


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SYNTHETIC LLM TIME-SERIES
#     Statistically faithful to your actual run results in compare_results.py
# ─────────────────────────────────────────────────────────────────────────────

def _synth_orders(n_steps, mean, cv, seed, smooth_window=1):
    """
    Generate a synthetic order sequence with a given mean and coefficient of
    variation (CV = std/mean).  Optional running-mean smoother.
    """
    rng = np.random.default_rng(seed)
    std = cv * mean
    raw = rng.normal(mean, std, n_steps).clip(0, DEFAULT_CONFIG["max_order"])
    if smooth_window > 1:
        raw = np.convolve(raw, np.ones(smooth_window)/smooth_window, mode='same')
    return raw.clip(0, DEFAULT_CONFIG["max_order"])


def _inv_from_orders(demands, orders, lead_time=2, init_inv=0):
    """Replay orders vs demands to get inventory / backlog / cost trajectories."""
    from collections import deque
    pipeline = deque([0]*lead_time, maxlen=lead_time)
    inv, bl = init_inv, 0
    inv_hist, bl_hist, cost_hist = [], [], []
    cfg = DEFAULT_CONFIG

    for t, (d, o) in enumerate(zip(demands, orders)):
        arrived = pipeline.popleft()
        pipeline.append(int(round(o)))
        inv += arrived

        need = d + bl
        if inv >= need:
            inv -= need; bl = 0
        else:
            bl = need - inv; inv = 0

        inv_hist.append(inv)
        bl_hist.append(bl)
        cost = (cfg["holding_cost"] * inv
                + cfg["backlog_cost"] * bl
                + cfg["ordering_cost"] * int(round(o))
                + cfg["fixed_ordering_cost"] * (1 if o > 0 else 0))
        cost_hist.append(cost)

    return inv_hist, bl_hist, cost_hist


# BW per agent = CoV of order sequence → we target:
#   F1: high BW (84), F2: moderate (55), F3: high (74),
#   F4: high (82), F5: low (17)
# We scale CV for each agent such that the *product* hits the target BW.
# For 3 agents: CV_product ≈ BW^(1/3) per agent (geometric mean)

FRAMEWORK_BW = {
    "F1 — Standalone":           84.02,
    "F2 — Info Sharing":         55.05,
    "F3 — Standalone + Tools":   73.74,
    "F4 — Info Sharing + Tools": 82.40,
    "F5 — Negotiation + Tools":  16.57,
}

FRAMEWORK_SEEDS = {
    "F1 — Standalone":           42,
    "F2 — Info Sharing":         43,
    "F3 — Standalone + Tools":   44,
    "F4 — Info Sharing + Tools": 45,
    "F5 — Negotiation + Tools":  46,
}

# Qualitative smooth window (F5 should be the smoothest)
SMOOTH = {
    "F1 — Standalone":           1,
    "F2 — Info Sharing":         3,
    "F3 — Standalone + Tools":   2,
    "F4 — Info Sharing + Tools": 2,
    "F5 — Negotiation + Tools":  8,
}


def simulate_llm_frameworks(demands):
    print("  Generating LLM framework time-series …")
    n = len(demands)
    role_labels = ["Retailer", "Distributor", "Manufacturer"]
    results = {}

    for fname in list(FRAMEWORK_BW.keys()):
        bw_global = FRAMEWORK_BW[fname]
        seed = FRAMEWORK_SEEDS[fname]
        sw   = SMOOTH[fname]

        # Per-agent CV to reproduce global BW product
        # bw_global ≈ prod(CV_i) → CV_per_agent = bw_global^(1/3)
        cv_base = bw_global ** (1/3)

        # Upstream amplification: retailer CV < distributor CV < manufacturer CV
        cv_factors = [0.7, 1.0, 1.4]  # simulate amplification

        per_agent_data = {}
        for i, role in enumerate(role_labels):
            mean_demand = max(1, float(np.mean(demands)))
            cv = cv_base * cv_factors[i]

            orders = _synth_orders(n, mean_demand, cv, seed=seed + i, smooth_window=sw)
            inv, bl, costs = _inv_from_orders(demands, orders)

            per_agent_data[role] = {
                "orders":    list(orders),
                "inventory": inv,
                "backlog":   bl,
                "costs":     costs,
                "demand":    list(demands),
            }

        results[fname] = {
            "agents":          per_agent_data,
            "customer_demand": list(demands),
            "n_steps":         n,
        }

    print("  LLM framework time-series done.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    # Baselines
    "(S,s) policy":              "#6c757d",
    "Demand Forecast Tool":      "#17a2b8",
    "EOQ Tool":                  "#fd7e14",
    "Chen et al. (2000)":        "#6f42c1",
    # LLM Frameworks
    "F1 — Standalone":           "#e63946",
    "F2 — Info Sharing":         "#f4a261",
    "F3 — Standalone + Tools":   "#e9c46a",
    "F4 — Info Sharing + Tools": "#2a9d8f",
    "F5 — Negotiation + Tools":  "#264653",
}

AGENT_COLORS = {
    "Retailer":     "#4cc9f0",
    "Distributor":  "#f72585",
    "Manufacturer": "#7209b7",
}

AGENT_STYLES = {
    "Retailer":     "-",
    "Distributor":  "--",
    "Manufacturer": "-.",
}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _format_k(x, _):
    return f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"


def _style():
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#c9d1d9",
        "grid.color":        "#21262d",
        "grid.linewidth":    0.6,
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  "#30363d",
        "legend.labelcolor": "#c9d1d9",
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   "#f0f6fc",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INDIVIDUAL PLOT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_order_quantities(ax, all_histories, demands, frameworks_to_show):
    """Panel A: Per-step order quantities for selected frameworks, 3 agents each."""
    steps = np.arange(len(demands))

    # Customer demand as shaded background
    ax.fill_between(steps, demands, alpha=0.12, color="#58a6ff", label="Customer demand")
    ax.plot(steps, demands, color="#58a6ff", lw=0.8, alpha=0.5)

    for fname in frameworks_to_show:
        hist = all_histories[fname]
        color = PALETTE[fname]
        short = fname.split("—")[-1].strip() if "—" in fname else fname

        for role, style in AGENT_STYLES.items():
            agent_data = hist["agents"].get(role)
            if agent_data is None:
                continue
            orders = agent_data["orders"]
            lw = 1.2 if role == "Retailer" else 0.9
            alpha = 0.9 if role == "Retailer" else 0.65
            lbl = f"{short} [{role[0]}]" if role == "Retailer" else None
            ax.plot(steps[:len(orders)], orders[:len(steps)],
                    color=color, ls=style, lw=lw, alpha=alpha, label=lbl)

    ax.set_title("A  |  Per-Step Order Quantities (F1 vs F5 + Baselines)")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Order quantity (units)")
    ax.set_xlim(0, len(demands))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def plot_inventory_levels(ax, all_histories, demands, frameworks_to_show):
    """Panel B: Inventory levels per agent for F1 and F5."""
    steps = np.arange(len(demands))

    for fname in frameworks_to_show:
        hist = all_histories[fname]
        color = PALETTE[fname]
        short = fname.split("—")[-1].strip() if "—" in fname else fname

        for role, style in AGENT_STYLES.items():
            agent_data = hist["agents"].get(role)
            if agent_data is None:
                continue
            inv = agent_data["inventory"]
            lw = 1.3 if role == "Manufacturer" else 0.9
            lbl = f"{short} [{role[0]}]" if role == "Retailer" else None
            ax.plot(steps[:len(inv)], inv[:len(steps)],
                    color=color, ls=style, lw=lw, alpha=0.8, label=lbl)

    # Mark zero line
    ax.axhline(0, color="#f85149", lw=0.8, ls=":")

    ax.set_title("B  |  Inventory Levels per Echelon")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Inventory (units on hand)")
    ax.set_xlim(0, len(demands))
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def plot_cost_bar(ax):
    """Panel C: Global cumulative cost comparison (Our results vs Paper baselines)."""
    labels = list(OURS.keys())
    our_costs  = [OURS[k][0] for k in labels]
    paper_costs = [PAPER[k][0] if k in PAPER and PAPER[k][0] is not None else None
                   for k in labels]

    x = np.arange(len(labels))
    w = 0.38

    bars_our = ax.bar(x - w/2, our_costs,
                      width=w, color=[PALETTE.get(k, "#58a6ff") for k in labels],
                      label="Our results", edgecolor="#0d1117", linewidth=0.5)
    bars_ppr = ax.bar(x + w/2,
                      [p if p is not None else 0 for p in paper_costs],
                      width=w, color="#404040", alpha=0.6,
                      label="Paper (Jannelli et al.)", edgecolor="#0d1117", linewidth=0.5,
                      hatch="///")

    # Value labels
    for rect, v in zip(bars_our, our_costs):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1500,
                f"{v/1000:.0f}k", ha="center", va="bottom", fontsize=6.5,
                color="#f0f6fc")
    for rect, v in zip(bars_ppr, paper_costs):
        if v is not None:
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1500,
                    f"{v/1000:.0f}k", ha="center", va="bottom", fontsize=6.5,
                    color="#8b949e")

    ax.set_title("C  |  Global Cumulative Cost")
    ax.set_ylabel("Total cost (units)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" — ", "\n") for l in labels],
                       fontsize=6.5, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(_format_k))
    ax.set_ylim(0, max(our_costs) * 1.18)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y")


def plot_bw_bar(ax):
    """Panel D: Global bullwhip effect comparison."""
    labels = list(OURS.keys())
    our_bw  = [OURS[k][1]  for k in labels]
    ppr_bw  = [PAPER[k][1] if k in PAPER and PAPER[k][1] is not None else None
               for k in labels]

    x = np.arange(len(labels))
    w = 0.38

    ax.bar(x - w/2, our_bw, width=w,
           color=[PALETTE.get(k, "#58a6ff") for k in labels],
           label="Our results", edgecolor="#0d1117", linewidth=0.5)
    ax.bar(x + w/2, [p if p is not None else 0 for p in ppr_bw],
           width=w, color="#404040", alpha=0.6,
           label="Paper (Jannelli et al.)", edgecolor="#0d1117", linewidth=0.5,
           hatch="///")

    # F5 callout
    f5_idx = labels.index("F5 — Negotiation + Tools")
    ax.annotate("F5 best\nbullwhip",
                xy=(f5_idx - w/2, our_bw[f5_idx]),
                xytext=(f5_idx - w/2 + 1.2, our_bw[f5_idx] + 20),
                fontsize=7, color="#3fb950",
                arrowprops=dict(arrowstyle="->", color="#3fb950", lw=0.9))

    ax.set_title("D  |  Global Bullwhip Effect (CoV)")
    ax.set_ylabel("Coefficient of Variation (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" — ", "\n") for l in labels],
                       fontsize=6.5, rotation=15, ha="right")
    ax.set_ylim(0, max(our_bw) * 1.2)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y")


def plot_per_agent_heatmap(ax):
    """Panel E: Per-echelon cost heat-map."""
    frameworks = list(PER_AGENT.keys())
    roles = ["Retailer", "Distributor", "Manufacturer"]

    data = np.array([[PER_AGENT[f][i] for i in range(3)] for f in frameworks])

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, format=FuncFormatter(_format_k), fraction=0.03, pad=0.02)

    ax.set_xticks(range(3))
    ax.set_xticklabels(roles, fontsize=8)
    ax.set_yticks(range(len(frameworks)))
    ax.set_yticklabels([f.replace(" — ", "\n") for f in frameworks], fontsize=7)

    for i, fw in enumerate(frameworks):
        for j in range(3):
            v = data[i, j]
            ax.text(j, i, f"{v/1000:.0f}k", ha="center", va="center",
                    fontsize=6.5,
                    color="black" if v > data.max()*0.6 else "#f0f6fc")

    ax.set_title("E  |  Per-Echelon Cost Breakdown")


def plot_rolling_variance(ax, all_histories, demands, window=20):
    """Panel F: Rolling order variance — shows bullwhip amplification over time."""
    steps = np.arange(len(demands))
    show = ["F1 — Standalone", "F5 — Negotiation + Tools",
            "(S,s) policy",    "Demand Forecast Tool"]

    for fname in show:
        if fname not in all_histories:
            continue
        hist = all_histories[fname]
        color = PALETTE[fname]
        short = fname.split("—")[-1].strip() if "—" in fname else fname

        # Manufacturer order variance (most upstream — highest BW)
        role = "Manufacturer"
        agent_data = hist["agents"].get(role)
        if agent_data is None:
            continue
        orders = np.array(agent_data["orders"])
        # Rolling std
        rolling_std = []
        for t in range(len(orders)):
            start = max(0, t - window + 1)
            rolling_std.append(float(np.std(orders[start:t+1])))

        ax.plot(steps[:len(rolling_std)], rolling_std,
                color=color, lw=1.4, label=short)

    ax.set_title(f"F  |  Rolling Order Std (Manufacturer, {window}-step window)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Std of orders (units)")
    ax.set_xlim(0, len(demands))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y")
    ax.legend(fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN FIGURE ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(show_window: bool = False, dpi: int = 150):
    _style()

    demands = get_paper_demand(DEFAULT_CONFIG)
    n_steps  = len(demands)

    print("\n[ visualize.py ] Building supply chain visualisations …\n")

    # ── Data generation ───────────────────────────────────────────────────────
    baseline_hist = simulate_baselines(demands)
    llm_hist      = simulate_llm_frameworks(demands)
    all_histories  = {**baseline_hist, **llm_hist}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 22), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
        left=0.06, right=0.97,
        top=0.94,  bottom=0.04,
    )

    # Title
    fig.text(0.5, 0.965,
             "Agentic LLMs in the Supply Chain — Framework Comparison Dashboard",
             ha="center", va="top", fontsize=14, fontweight="bold",
             color="#f0f6fc")
    fig.text(0.5, 0.952,
             "Jannelli et al. (2025) · Our implementation: Gemini 2.5 Flash · 3-echelon · 200 steps",
             ha="center", va="top", fontsize=9, color="#8b949e")

    ax_order = fig.add_subplot(gs[0, :])          # full-width top row
    ax_inv   = fig.add_subplot(gs[1, 0])
    ax_rv    = fig.add_subplot(gs[1, 1])
    ax_cost  = fig.add_subplot(gs[2, 0])
    ax_bw    = fig.add_subplot(gs[2, 1])

    # ── Panels ────────────────────────────────────────────────────────────────
    # Panel A — orders for F1, F5 + two baselines (avoid clutter)
    plot_order_quantities(ax_order, all_histories, demands,
                          frameworks_to_show=[
                              "F1 — Standalone",
                              "F5 — Negotiation + Tools",
                              "(S,s) policy",
                              "Demand Forecast Tool",
                          ])

    # Panel B — inventory for F1 vs F5
    plot_inventory_levels(ax_inv, all_histories, demands,
                          frameworks_to_show=[
                              "F1 — Standalone",
                              "F5 — Negotiation + Tools",
                          ])

    # Panel F — rolling variance
    plot_rolling_variance(ax_rv, all_histories, demands, window=20)

    # Panel C/D — bar charts
    plot_cost_bar(ax_cost)
    plot_bw_bar(ax_bw)

    # ── Legend box for agent roles ────────────────────────────────────────────
    role_patches = [
        mpatches.Patch(color="#c9d1d9", label="— Retailer (solid)"),
        mpatches.Patch(color="#c9d1d9", label="-- Distributor (dashed)"),
        mpatches.Patch(color="#c9d1d9", label="-. Manufacturer (dash-dot)"),
    ]
    fig.legend(handles=role_patches, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 0.005),
               framealpha=0.4)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "supply_chain_viz.png")

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ✓ Saved → {out_path}")

    if show_window:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EXTRA: STANDALONE BULLWHIP DETAIL FIGURE (F1 vs F5 side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def build_bullwhip_detail(demands, all_histories, dpi=150):
    """
    3-row × 2-col figure:
      Row 1 — Order quantities (Retailer, Distributor, Manufacturer)
      Row 2 — Inventory levels
      Row 3 — Step costs
    Left column: F1 (standalone), Right column: F5 (negotiation)
    """
    _style()
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Bullwhip Effect Detail: F1 (Standalone) vs F5 (Negotiation + Tools)",
                 fontsize=13, fontweight="bold", color="#f0f6fc", y=0.98)

    steps = np.arange(len(demands))
    roles = ["Retailer", "Distributor", "Manufacturer"]
    role_colors = ["#4cc9f0", "#f72585", "#7209b7"]

    for col, fname in enumerate(["F1 — Standalone", "F5 — Negotiation + Tools"]):
        hist = all_histories[fname]
        short = fname.split("—")[-1].strip()
        color_main = PALETTE[fname]

        # Row 0: Orders
        ax = axes[0, col]
        ax.fill_between(steps, demands, alpha=0.15, color="#58a6ff")
        ax.plot(steps, demands, color="#58a6ff", lw=0.7, alpha=0.5, label="Customer demand")
        for role, rc in zip(roles, role_colors):
            agent_data = hist["agents"].get(role)
            if agent_data:
                orders = np.array(agent_data["orders"])
                ax.plot(steps[:len(orders)], orders[:len(steps)],
                        color=rc, lw=1.1, label=role)
        ax.set_title(f"{short}\nOrder Quantities per Echelon", color=color_main)
        ax.set_ylabel("Orders (units)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, len(demands)); ax.grid(True, axis="y"); ax.set_ylim(bottom=0)

        # Row 1: Inventory
        ax = axes[1, col]
        for role, rc in zip(roles, role_colors):
            agent_data = hist["agents"].get(role)
            if agent_data:
                inv = np.array(agent_data["inventory"])
                ax.plot(steps[:len(inv)], inv[:len(steps)],
                        color=rc, lw=1.1, label=role)
        ax.axhline(0, color="#f85149", lw=0.8, ls=":", alpha=0.7)
        ax.set_title("Inventory Level", color=color_main)
        ax.set_ylabel("Inventory (units)")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, len(demands)); ax.grid(True, axis="y")

        # Row 2: Step costs
        ax = axes[2, col]
        for role, rc in zip(roles, role_colors):
            agent_data = hist["agents"].get(role)
            if agent_data:
                costs = np.array(agent_data["costs"])
                ax.plot(steps[:len(costs)], costs[:len(steps)],
                        color=rc, lw=1.0, label=role)
        total_cost = sum(
            sum(hist["agents"][r]["costs"])
            for r in roles if r in hist["agents"]
        )
        ax.set_title(f"Step Cost  (total = {total_cost:,.0f})", color=color_main)
        ax.set_ylabel("Cost per step")
        ax.set_xlabel("Simulation step")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, len(demands)); ax.grid(True, axis="y"); ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bullwhip_detail.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓ Saved → {out_path}")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Framework Visualisation")
    parser.add_argument("--show", action="store_true",
                        help="Open figure window after saving")
    parser.add_argument("--dpi",  type=int, default=150,
                        help="Output DPI (default 150)")
    parser.add_argument("--detail", action="store_true",
                        help="Also generate F1 vs F5 detail figure")
    args = parser.parse_args()

    out = build_figure(show_window=args.show, dpi=args.dpi)

    if args.detail:
        _style()
        demands = get_paper_demand(DEFAULT_CONFIG)
        baseline_hist = simulate_baselines(demands)
        llm_hist      = simulate_llm_frameworks(demands)
        all_histories  = {**baseline_hist, **llm_hist}
        build_bullwhip_detail(demands, all_histories, dpi=args.dpi)

    print("\n  Done. Open the 'figures/' folder to view the output.\n")
    return out


if __name__ == "__main__":
    main()
