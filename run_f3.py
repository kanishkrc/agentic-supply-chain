"""
Framework 3 — Standalone + Tools.

Each agent independently calls two decision-support tools before placing
its order:

  1. Demand Forecast Tool   — linear regression on last 30 demand observations
                              + inventory-position adjustment (P5 in prompt)
  2. EOQ Tool               — sqrt(2 * mean_demand * c_o / c_h)
                              selfish-optimal order quantity

The tool outputs are included in the prompt as grounding signals.  The LLM
then decides its final order, informed (but not bound) by the tool values.

Paper §4.3-4.4 / Framework 3.

Usage
-----
  python run_f3.py
  python run_f3.py --steps 20 --verbose
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LinearRegression

from supply_chain.config       import DEFAULT_CONFIG, ROLE_LABELS
from supply_chain.demand       import get_paper_demand
from supply_chain.environment  import SupplyChainEnv
from supply_chain.llm_client   import LLMClient, parse_order
from supply_chain.metrics      import print_results
from supply_chain.prompt_builder import build_prompt_f3


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations (reused from test_simulation.py, cleaned up)
# ─────────────────────────────────────────────────────────────────────────────

def run_demand_forecast_tool(obs: dict, cfg: dict, n_prev: int = 30) -> int:
    """
    Predict next-period demand via linear regression, then compute an
    inventory-position-adjusted order quantity.

    Returns the suggested order quantity.
    """
    history = obs["demand_history"][-n_prev:]

    if len(history) < 2:
        forecasted = float(history[-1]) if history else float(cfg["demand_S0"])
    else:
        X = np.arange(len(history)).reshape(-1, 1)
        y = np.array(history, dtype=float)
        model = LinearRegression().fit(X, y)
        forecasted = max(0.0, float(model.predict([[len(history)]])[0]))

    lead_time     = cfg["lead_time"]
    target        = round(forecasted * (lead_time + 1))
    inv_position  = obs["inventory"] + sum(obs["pipeline"]) - obs["backlog"]
    order         = max(0, target - inv_position)
    return int(np.clip(order, 0, cfg["max_order"]))


def run_eoq_tool(obs: dict, cfg: dict) -> int:
    """
    Compute the EOQ — the selfish-optimal order quantity ignoring chain effects.

    EOQ = sqrt(2 * mean_demand * ordering_cost / holding_cost)
    """
    history = obs["demand_history"][-cfg["n_obs_forecast"]:]
    if not history:
        eoq0 = np.sqrt(
            2 * cfg["demand_S0"] * cfg["ordering_cost"] / cfg["holding_cost"]
        )
        return int(np.clip(round(eoq0), 0, cfg["max_order"]))

    mean_demand = max(1, np.mean(history))
    eoq = np.sqrt(
        2 * mean_demand * cfg["ordering_cost"] / cfg["holding_cost"]
    )
    return int(np.clip(round(eoq), 0, cfg["max_order"]))


def compute_tools(obs: dict, cfg: dict) -> Dict[str, int]:
    """Run both tools and return a dict of results."""
    return {
        "demand_forecast": run_demand_forecast_tool(obs, cfg),
        "eoq":             run_eoq_tool(obs, cfg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent decision (F3)
# ─────────────────────────────────────────────────────────────────────────────

def llm_decide_f3(
    client: LLMClient,
    obs: dict,
    cfg: dict,
    verbose: bool = False,
) -> int:
    tool_results = compute_tools(obs, cfg)
    prompt       = build_prompt_f3(obs, tool_results)
    response     = client.call(prompt)
    order        = parse_order(response, max_order=cfg["max_order"])

    if verbose:
        role = ROLE_LABELS.get(obs["agent_id"], f"Agent {obs['agent_id']}")
        print(f"    [{role}] inv={obs['inventory']:3d}  bl={obs['backlog']:3d}  "
              f"df={tool_results['demand_forecast']:3d}  "
              f"eoq={tool_results['eoq']:3d}  "
              f"order={order:3d}  LLM={response.strip()!r}")

    return order


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation run
# ─────────────────────────────────────────────────────────────────────────────

def run_f3(
    cfg: dict,
    demands: list,
    verbose: bool = False,
    log_interval: int = 10,
    model: str | None = None,
) -> dict:
    client = LLMClient(model=model) if model else LLMClient()

    env  = SupplyChainEnv(config=cfg)
    obs  = env.reset(demands)
    done = False
    t    = 0

    print(f"\n  Model : {client.model}")
    print(f"  Steps : {cfg['n_steps']}, agents: {cfg['n_agents']}, "
          f"lead_time: {cfg['lead_time']}\n")

    step_times: List[float] = []

    while not done:
        t_start = time.time()

        if verbose or (t % log_interval == 0):
            print(f"  Step {t:3d}/{cfg['n_steps']} -- customer demand: {demands[t]}")

        with ThreadPoolExecutor(max_workers=cfg["n_agents"]) as pool:
            futures = {
                pool.submit(llm_decide_f3, client, obs[i], cfg, verbose): i
                for i in range(cfg["n_agents"])
            }
            actions_map = {}
            for fut in as_completed(futures):
                actions_map[futures[fut]] = fut.result()
        actions = [actions_map[i] for i in range(cfg["n_agents"])]

        obs, costs, done, info = env.step(actions)

        elapsed = time.time() - t_start
        step_times.append(elapsed)

        if verbose or (t % log_interval == 0):
            print(f"    Global cost this step: {info['global_cost']:,.0f}  "
                  f"({elapsed:.1f}s)")

        t += 1

    avg_step = sum(step_times) / len(step_times) if step_times else 0
    print(f"\n  Completed {t} steps in {sum(step_times):.1f}s "
          f"(avg {avg_step:.2f}s/step)")

    return env.get_results()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Framework 3 -- Standalone + Tools")
    parser.add_argument("--steps",   type=int,  default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log",     type=int,  default=10)
    parser.add_argument("--model",   type=str,  default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.steps:
        cfg["n_steps"] = args.steps

    demands = list(get_paper_demand(cfg))

    print("=" * 55)
    print("  Framework 3 -- Standalone + Tools")
    print("=" * 55)

    results = run_f3(
        cfg=cfg,
        demands=demands,
        verbose=args.verbose,
        log_interval=args.log,
        model=args.model,
    )

    print_results("Framework 3 -- Standalone + Tools", results)

    print("\n  Reference (our baselines + prior frameworks):")
    print(f"  F1 Standalone       cost= 109,758   BW= 84.02")
    print(f"  F2 Info Sharing     cost= 119,955   BW=105.74")
    print(f"  Demand Forecast     cost=   7,938   BW= 41.66")
    print(f"  Chen et al. 2000    cost=  41,206   BW=  3.53")
    print(f"  (S,s) policy        cost=  81,755   BW=102.98")

    return results


if __name__ == "__main__":
    main()
