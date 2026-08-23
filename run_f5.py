"""
Framework 5 — Negotiation + Tools.

Each simulation step runs `num_iter` (default 3) rounds of negotiation
between neighbouring agents before the final order is placed.

Negotiation protocol (paper §4.5, Fig 7):
  Round 1  — each agent proposes its EOQ (selfish initial anchor) using
              only its own state + tool outputs.  No neighbour proposals yet.
  Rounds 2+ — agents exchange their previous-round proposals with immediate
              neighbours.  Each agent revises its proposal informed by what
              its neighbours need.
  After the final round each agent's last proposal becomes its order.

Neighbour topology (same as F2 / F4):
  Retailer  <-->  Distributor  <-->  Manufacturer

All agents within a round propose in parallel; rounds execute sequentially.

Paper §4.5 / Framework 5.

Usage
-----
  python run_f5.py
  python run_f5.py --steps 20 --verbose
  python run_f5.py --rounds 3           # override num_iter
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from supply_chain.config       import DEFAULT_CONFIG, ROLE_LABELS
from supply_chain.demand       import get_paper_demand
from supply_chain.environment  import SupplyChainEnv
from supply_chain.llm_client   import LLMClient, parse_order
from supply_chain.metrics      import print_results
from supply_chain.prompt_builder import build_prompt_f5_propose
from run_f3 import compute_tools, run_demand_forecast_tool


# ─────────────────────────────────────────────────────────────────────────────
# Proposal message formatting
# ─────────────────────────────────────────────────────────────────────────────

def _format_proposal(agent_id: int, proposal: int, relationship: str) -> str:
    """Format a neighbour's previous-round proposal for the prompt."""
    role = ROLE_LABELS.get(agent_id, f"Agent {agent_id}")
    return f"Your {relationship} neighbour ({role}) proposes: {proposal} units"


def build_neighbour_proposals(
    prev_proposals: List[Optional[int]],
    n: int,
) -> List[List[str]]:
    """
    For each agent, collect the formatted proposal strings from its immediate
    neighbours (based on the PREVIOUS round's proposals).

    Returns a list of length n; each element is a list of message strings.
    If prev_proposals is all None (round 1), returns empty lists.
    """
    msgs: List[List[str]] = [[] for _ in range(n)]

    if all(p is None for p in prev_proposals):
        return msgs   # Round 1: no prior proposals

    for i in range(n):
        if i > 0 and prev_proposals[i - 1] is not None:
            msgs[i].append(
                _format_proposal(i - 1, prev_proposals[i - 1], "downstream")
            )
        if i < n - 1 and prev_proposals[i + 1] is not None:
            msgs[i].append(
                _format_proposal(i + 1, prev_proposals[i + 1], "upstream")
            )

    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Single-agent proposal (one round of negotiation)
# ─────────────────────────────────────────────────────────────────────────────

def llm_propose(
    client: LLMClient,
    obs: dict,
    tool_results: Dict,
    negotiation_round: int,
    num_rounds: int,
    neighbour_proposals: List[str],
    cfg: dict,
    verbose: bool = False,
) -> int:
    prompt   = build_prompt_f5_propose(
        obs, tool_results, negotiation_round, neighbour_proposals or None
    )
    response = client.call(prompt)
    proposal = parse_order(response, max_order=cfg["max_order"])

    if verbose:
        role = ROLE_LABELS.get(obs["agent_id"], f"Agent {obs['agent_id']}")
        print(f"      [{role}] R{negotiation_round}  "
              f"df={tool_results['demand_forecast']:3d}  "
              f"eoq={tool_results['eoq']:3d}  "
              f"proposal={proposal:3d}  ({response.strip()!r})")

    return proposal


# ─────────────────────────────────────────────────────────────────────────────
# Full negotiation for one simulation step
# ─────────────────────────────────────────────────────────────────────────────

def negotiate_step(
    client: LLMClient,
    obs_list: List[dict],
    cfg: dict,
    num_rounds: int,
    verbose: bool = False,
) -> List[int]:
    """
    Run `num_rounds` negotiation rounds for all agents at one simulation step.

    Returns the final proposals (list of ints, one per agent).
    """
    n = cfg["n_agents"]

    # Compute tools once per agent (tools don't change within a step)
    tool_results_list = [compute_tools(obs_list[i], cfg) for i in range(n)]

    proposals: List[Optional[int]] = [None] * n

    for r in range(1, num_rounds + 1):
        neighbour_msgs = build_neighbour_proposals(proposals, n)

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(
                    llm_propose,
                    client,
                    obs_list[i],
                    tool_results_list[i],
                    r,
                    num_rounds,
                    neighbour_msgs[i],
                    cfg,
                    verbose,
                ): i
                for i in range(n)
            }
            new_proposals: Dict[int, int] = {}
            for fut in as_completed(futures):
                new_proposals[futures[fut]] = fut.result()

        proposals = [new_proposals[i] for i in range(n)]

        if verbose:
            print(f"      -- Round {r} proposals: {proposals}")

    # Programmatic safety net: use EOQ as the floor (lower than demand_forecast).
    # The improved prompts carry the main backlog-clearing logic; this net only
    # catches true zero-collapse when EOQ itself is positive.
    from run_f3 import run_eoq_tool
    eoq_floors = [run_eoq_tool(obs_list[i], cfg) for i in range(n)]
    final = [max(proposals[i], eoq_floors[i]) for i in range(n)]  # type: ignore[index]

    if verbose and any(final[i] > proposals[i] for i in range(n)):
        print(f"      -- EOQ floor applied: proposals={proposals} -> final={final}")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation run
# ─────────────────────────────────────────────────────────────────────────────

def run_f5(
    cfg: dict,
    demands: list,
    verbose: bool = False,
    log_interval: int = 10,
    model: str | None = None,
    num_rounds: int | None = None,
) -> dict:
    client     = LLMClient(model=model) if model else LLMClient()
    n_rounds   = num_rounds or cfg.get("num_iter", 3)

    env  = SupplyChainEnv(config=cfg)
    obs  = env.reset(demands)
    done = False
    t    = 0

    print(f"\n  Model       : {client.model}")
    print(f"  Steps       : {cfg['n_steps']}, agents: {cfg['n_agents']}, "
          f"lead_time: {cfg['lead_time']}")
    print(f"  Neg. rounds : {n_rounds}\n")

    step_times: List[float] = []

    while not done:
        t_start = time.time()

        if verbose or (t % log_interval == 0):
            print(f"  Step {t:3d}/{cfg['n_steps']} -- customer demand: {demands[t]}")

        actions = negotiate_step(client, obs, cfg, n_rounds, verbose=verbose)

        obs, costs, done, info = env.step(actions)

        elapsed = time.time() - t_start
        step_times.append(elapsed)

        if verbose or (t % log_interval == 0):
            print(f"    actions={actions}  "
                  f"global_cost={info['global_cost']:,.0f}  ({elapsed:.1f}s)")

        t += 1

    avg_step = sum(step_times) / len(step_times) if step_times else 0
    print(f"\n  Completed {t} steps in {sum(step_times):.1f}s "
          f"(avg {avg_step:.2f}s/step, {n_rounds} rounds each)")

    return env.get_results()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Framework 5 -- Negotiation + Tools"
    )
    parser.add_argument("--steps",   type=int,  default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log",     type=int,  default=10)
    parser.add_argument("--model",   type=str,  default=None)
    parser.add_argument("--rounds",  type=int,  default=None,
                        help="Number of negotiation rounds per step (default: cfg num_iter=3)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.steps:
        cfg["n_steps"] = args.steps

    demands = list(get_paper_demand(cfg))

    print("=" * 55)
    print("  Framework 5 -- Negotiation + Tools")
    print("=" * 55)

    results = run_f5(
        cfg=cfg,
        demands=demands,
        verbose=args.verbose,
        log_interval=args.log,
        model=args.model,
        num_rounds=args.rounds,
    )

    print_results("Framework 5 -- Negotiation + Tools", results)

    print("\n  Reference (our baselines + prior frameworks):")
    print(f"  F1 Standalone       cost= 109,758   BW= 84.02")
    print(f"  F2 Info Sharing     cost= 119,955   BW=105.74")
    print(f"  F3 Standalone+Tools cost=  97,608   BW= 73.74")
    print(f"  F4 InfoShare+Tools  cost=  84,341   BW= 82.40")
    print(f"  F5 prev (old prompt) cost= 133,468   BW= 24.69")
    print(f"  Demand Forecast     cost=   7,938   BW= 41.66")
    print(f"  Chen et al. 2000    cost=  41,206   BW=  3.53")
    print(f"  (S,s) policy        cost=  81,755   BW=102.98")

    return results


if __name__ == "__main__":
    main()
