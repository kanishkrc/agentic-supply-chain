"""
Framework 4 — Information Sharing + Tools.

Combines F2 (neighbour state broadcast) with F3 (demand-forecast and EOQ
tools).  Each agent sees its neighbours' inventory/backlog AND gets tool
outputs before making its decision.

Paper §4.5 / Framework 4.

Usage
-----
  python run_f4.py
  python run_f4.py --steps 20 --verbose
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from supply_chain.config       import DEFAULT_CONFIG, ROLE_LABELS
from supply_chain.demand       import get_paper_demand
from supply_chain.environment  import SupplyChainEnv
from supply_chain.llm_client   import LLMClient, parse_order
from supply_chain.metrics      import print_results
from supply_chain.prompt_builder import build_prompt_f4, format_neighbour_message
from run_f2 import build_neighbour_messages
from run_f3 import compute_tools


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent decision (F4)
# ─────────────────────────────────────────────────────────────────────────────

def llm_decide_f4(
    client: LLMClient,
    obs: dict,
    neighbour_msgs: List[str],
    cfg: dict,
    verbose: bool = False,
) -> int:
    tool_results = compute_tools(obs, cfg)
    prompt       = build_prompt_f4(obs, neighbour_msgs, tool_results)
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

def run_f4(
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

        neighbour_msgs = build_neighbour_messages(obs)

        with ThreadPoolExecutor(max_workers=cfg["n_agents"]) as pool:
            futures = {
                pool.submit(
                    llm_decide_f4, client, obs[i], neighbour_msgs[i], cfg, verbose
                ): i
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
    parser = argparse.ArgumentParser(
        description="Framework 4 -- Information Sharing + Tools"
    )
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
    print("  Framework 4 -- Information Sharing + Tools")
    print("=" * 55)

    results = run_f4(
        cfg=cfg,
        demands=demands,
        verbose=args.verbose,
        log_interval=args.log,
        model=args.model,
    )

    print_results("Framework 4 -- Information Sharing + Tools", results)

    print("\n  Reference (our baselines + prior frameworks):")
    print(f"  F1 Standalone       cost= 109,758   BW= 84.02")
    print(f"  F2 Info Sharing     cost= 119,955   BW=105.74")
    print(f"  F3 Standalone+Tools cost=  97,608   BW= 73.74")
    print(f"  Demand Forecast     cost=   7,938   BW= 41.66")
    print(f"  Chen et al. 2000    cost=  41,206   BW=  3.53")
    print(f"  (S,s) policy        cost=  81,755   BW=102.98")

    return results


if __name__ == "__main__":
    main()
