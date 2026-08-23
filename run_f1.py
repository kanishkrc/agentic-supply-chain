"""
Framework 1 — Standalone LLM Agent.

Each of the 3 supply chain agents is an independent LLM that:
  1. Receives its local observation (inventory, backlog, pipeline, history).
  2. Builds a P1-P4-P7 prompt (no tools, no inter-agent communication).
  3. Calls Gemini 2.5 to decide its order quantity.
  4. The environment executes the action and advances the simulation.

Paper §4.1 / Framework 1 description.

Usage
-----
  python run_f1.py
  python run_f1.py --steps 20 --verbose        # short debug run
  python run_f1.py --model gemini-2.5-flash-preview-04-17
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from supply_chain.config      import DEFAULT_CONFIG, ROLE_LABELS
from supply_chain.demand      import get_paper_demand
from supply_chain.environment import SupplyChainEnv
from supply_chain.llm_client  import LLMClient, parse_order
from supply_chain.metrics     import print_results
from supply_chain.prompt_builder import build_prompt_f1


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent decision
# ─────────────────────────────────────────────────────────────────────────────

def llm_decide(
    client: LLMClient,
    obs: dict,
    cfg: dict,
    verbose: bool = False,
) -> int:
    """
    Build a F1 prompt for one agent and call the LLM.

    Returns the parsed integer order quantity, clipped to [0, max_order].
    Falls back to (S,s) policy if the LLM response cannot be parsed.
    """
    prompt   = build_prompt_f1(obs)
    response = client.call(prompt)

    order = parse_order(response, max_order=cfg["max_order"])

    if verbose:
        role = ROLE_LABELS.get(obs["agent_id"], f"Agent {obs['agent_id']}")
        print(f"    [{role}] inv={obs['inventory']:3d}  bl={obs['backlog']:3d}  "
              f"order={order:3d}  LLM={response.strip()!r}")

    return order


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation run
# ─────────────────────────────────────────────────────────────────────────────

def run_f1(
    cfg: dict,
    demands: list,
    verbose: bool = False,
    log_interval: int = 10,
    model: str | None = None,
) -> dict:
    """
    Run the full supply chain simulation with Framework 1 LLM agents.

    Parameters
    ----------
    cfg          : simulation config dict
    demands      : pre-generated demand sequence
    verbose      : print per-agent decisions each step
    log_interval : print a progress line every N steps
    model        : override the LLM model (defaults to .env MODEL_ID)

    Returns
    -------
    results dict from env.get_results()
    """
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
            print(f"  Step {t:3d}/{cfg['n_steps']} — customer demand: {demands[t]}")

        # All agents decide in parallel (F1: no inter-agent dependency this step)
        with ThreadPoolExecutor(max_workers=cfg["n_agents"]) as pool:
            futures = {
                pool.submit(llm_decide, client, o, cfg, verbose): i
                for i, o in enumerate(obs)
            }
            actions_map = {}
            for fut in as_completed(futures):
                i = futures[fut]
                actions_map[i] = fut.result()
        actions = [actions_map[i] for i in range(cfg["n_agents"])]

        obs, costs, done, info = env.step(actions)

        elapsed = time.time() - t_start
        step_times.append(elapsed)

        if verbose or (t % log_interval == 0):
            print(f"    Global cost this step: {info['global_cost']:,.0f}  "
                  f"(step took {elapsed:.1f}s)")

        t += 1

    avg_step = sum(step_times) / len(step_times) if step_times else 0
    print(f"\n  Completed {t} steps in {sum(step_times):.1f}s "
          f"(avg {avg_step:.2f}s/step)")

    return env.get_results()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Framework 1 — Standalone LLM Agent")
    parser.add_argument("--steps",    type=int,  default=None,
                        help="Override n_steps (default: full 200)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-agent decisions each step")
    parser.add_argument("--log",      type=int,  default=10,
                        help="Progress log interval (default: every 10 steps)")
    parser.add_argument("--model",    type=str,  default=None,
                        help="Override the Gemini model ID")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.steps:
        cfg["n_steps"] = args.steps

    demands = list(get_paper_demand(cfg))

    print("=" * 55)
    print("  Framework 1 — Standalone LLM Agent")
    print("=" * 55)

    results = run_f1(
        cfg=cfg,
        demands=demands,
        verbose=args.verbose,
        log_interval=args.log,
        model=args.model,
    )

    print_results("Framework 1 — Standalone LLM Agent", results)

    # Paper reference (Table 1 / Table 2) for comparison
    paper_ref = {
        "costs":    {"global": None},   # paper didn't report F1 separately
        "bullwhip": {"global": None},
    }
    print("\n  Reference baselines for comparison:")
    print(f"  Demand Forecast cost ~  16,645   BW ~  1.01")
    print(f"  Chen et al. 2000    cost ~  39,276   BW ~  0.94")
    print(f"  (S,s) policy        cost ~ 159,094   BW ~ 65.77")

    return results


if __name__ == "__main__":
    main()
