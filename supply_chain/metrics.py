"""
Metrics — Bullwhip Effect and Cost calculations.

Paper formulas (§3.1 and §3.2):

  Per-agent bullwhip:
    BW_i = σ(d_i) / μ(d_i)
    where d_i is the sequence of ORDERS placed by Agent i (its "demand" signal
    sent upstream). A BW_i < 1 means negligible bullwhip at that echelon.

  Global bullwhip (Fransoo & Wouters 2000):
    BW_global = Π BW_i   (product across all echelons)

  Per-agent cumulative cost:
    cost_i = Σ_t C_i(t)

  Global cost:
    cost_global = Σ_i cost_i
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from .environment import SupplyChainEnv


# ─────────────────────────────────────────────────────────────────────────────
# Bullwhip Effect
# ─────────────────────────────────────────────────────────────────────────────

def bullwhip_per_agent(orders: list[int] | np.ndarray) -> float:
    """
    Coefficient of Variation of an agent's order sequence.

    BW_i = σ(orders_i) / μ(orders_i)

    Returns 0.0 if there are fewer than 2 observations or mean is 0
    (avoids division-by-zero in early steps).
    """
    arr = np.asarray(orders, dtype=float)
    if len(arr) < 2 or np.mean(arr) == 0:
        return 0.0
    return float(np.std(arr, ddof=1) / np.mean(arr))


def compute_bullwhip(env: "SupplyChainEnv") -> Dict:
    """
    Compute per-agent and global bullwhip effect after a full run.

    Parameters
    ----------
    env : completed SupplyChainEnv (env.t == len(env.demands))

    Returns
    -------
    dict with keys:
      per_agent  : list[float]   BW_i for each agent
      global     : float         Π BW_i
    """
    per_agent = [
        bullwhip_per_agent(agent.order_history)
        for agent in env.agents
    ]

    # Paper: global bullwhip = product of all per-agent values
    global_bw = float(np.prod([bw for bw in per_agent if bw > 0]))

    return {
        "per_agent": per_agent,
        "global":    global_bw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Costs
# ─────────────────────────────────────────────────────────────────────────────

def compute_costs(env: "SupplyChainEnv") -> Dict:
    """
    Aggregate cost metrics after a full run.

    Returns
    -------
    dict with keys:
      per_agent  : list[float]   cumulative cost per agent
      global     : float         sum of all agent costs
      per_step   : list[float]   global cost at each step (for plotting)
    """
    per_agent = [sum(agent.cost_history) for agent in env.agents]

    # Global cost time-series: sum across agents at each step
    n_steps   = env.t
    per_step  = [
        sum(env.agents[i].cost_history[t] for i in range(env.n))
        for t in range(n_steps)
    ]

    return {
        "per_agent": per_agent,
        "global":    float(sum(per_agent)),
        "per_step":  per_step,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────────────────────

def print_results(label: str, results: Dict) -> None:
    """Print a formatted summary matching the paper's Tables 1 & 2."""
    costs    = results["costs"]
    bullwhip = results["bullwhip"]

    print(f"\n{'-'*55}")
    print(f"  {label}")
    print('-'*55)
    print(f"  Global cost      : {costs['global']:>12,.0f}")
    print(f"  Global bullwhip  : {bullwhip['global']:>12.5f}")
    print()
    for i, (c, bw) in enumerate(zip(costs["per_agent"], bullwhip["per_agent"])):
        role = {0: "Retailer", 1: "Distributor", 2: "Manufacturer"}.get(i, f"Agent {i}")
        print(f"  {role:<14} cost={c:>10,.0f}   BW={bw:.5f}")
    print('-'*55)
