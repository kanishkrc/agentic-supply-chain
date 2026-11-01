"""
Equivalence test for the LangGraph F5 negotiation.

Checks that the LangGraph negotiation graph (run_f5.build_negotiation_graph)
makes exactly the same decisions as the original round-loop implementation
(copied below verbatim from the initial commit as `reference_negotiate_step`).

No API key or network is needed: Gemini is replaced by a deterministic fake
client whose answer depends on the full prompt text. So identical orders mean
the graph sends identical prompts, in the same rounds, with the same
neighbour proposals, and applies the same EOQ floor.

Run:
  python test_f5_graph.py
  pytest test_f5_graph.py
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from supply_chain.config      import DEFAULT_CONFIG
from supply_chain.demand      import get_paper_demand
from supply_chain.environment import SupplyChainEnv
from run_f3 import compute_tools, run_eoq_tool
from run_f5 import (
    build_negotiation_graph,
    build_neighbour_proposals,
    llm_propose,
    negotiate_step,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake LLM: deterministic, thread-safe, depends on the whole prompt
# ─────────────────────────────────────────────────────────────────────────────

class FakeLLMClient:
    """Stands in for LLMClient. Same prompt -> same integer reply."""

    model = "fake-deterministic"

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def call(self, prompt: str, **_: object) -> str:
        with self._lock:
            self.calls += 1
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        return str(int(digest, 16) % 25)          # an order between 0 and 24


# ─────────────────────────────────────────────────────────────────────────────
# Reference: the original for-loop negotiation (initial commit 9d34f53)
# ─────────────────────────────────────────────────────────────────────────────

def reference_negotiate_step(client, obs_list, cfg, num_rounds, verbose=False):
    n = cfg["n_agents"]
    tool_results_list = [compute_tools(obs_list[i], cfg) for i in range(n)]
    proposals: List[Optional[int]] = [None] * n

    for r in range(1, num_rounds + 1):
        neighbour_msgs = build_neighbour_proposals(proposals, n)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(llm_propose, client, obs_list[i], tool_results_list[i],
                            r, num_rounds, neighbour_msgs[i], cfg, verbose): i
                for i in range(n)
            }
            new_proposals: Dict[int, int] = {}
            for fut in as_completed(futures):
                new_proposals[futures[fut]] = fut.result()
        proposals = [new_proposals[i] for i in range(n)]

    eoq_floors = [run_eoq_tool(obs_list[i], cfg) for i in range(n)]
    return [max(proposals[i], eoq_floors[i]) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simulate(step_fn, n_steps: int):
    """Run the full environment with a given negotiation function."""
    cfg     = DEFAULT_CONFIG.copy()
    cfg["n_steps"] = n_steps
    demands = list(get_paper_demand(cfg))[:n_steps]
    client  = FakeLLMClient()
    env     = SupplyChainEnv(config=cfg)
    obs     = env.reset(demands)

    actions_log, done = [], False
    while not done:
        actions = step_fn(client, obs, cfg)
        actions_log.append(actions)
        obs, _, done, _ = env.step(actions)

    return actions_log, env.get_results(), client.calls


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_structure():
    """Graph has the expected nodes and a self-loop on propose_round."""
    graph = build_negotiation_graph(FakeLLMClient(), DEFAULT_CONFIG.copy())
    g     = graph.get_graph()
    nodes = set(g.nodes)
    assert {"compute_tools", "propose_round", "finalize"} <= nodes, nodes

    edges = {(e.source, e.target) for e in g.edges}
    assert ("propose_round", "propose_round") in edges, edges   # negotiation loop
    assert ("propose_round", "finalize") in edges, edges


def test_round_count():
    """One step runs exactly num_rounds rounds -> n_agents * num_rounds LLM calls."""
    cfg    = DEFAULT_CONFIG.copy()
    env    = SupplyChainEnv(config=cfg)
    obs    = env.reset(list(get_paper_demand(cfg)))
    for rounds in (1, 3, 5):
        client = FakeLLMClient()
        negotiate_step(client, obs, cfg, rounds)
        assert client.calls == cfg["n_agents"] * rounds, (rounds, client.calls)


def test_graph_matches_reference_full_run():
    """200-step run: LangGraph orders == original loop orders at every step."""
    n_steps = DEFAULT_CONFIG["n_steps"]
    rounds  = DEFAULT_CONFIG["num_iter"]

    graph_holder = {}

    def graph_step(client, obs, cfg):
        # compile once per client (mirrors run_f5, which compiles once per run)
        if graph_holder.get("client") is not client:
            graph_holder["client"] = client
            graph_holder["graph"]  = build_negotiation_graph(client, cfg)
        return negotiate_step(client, obs, cfg, rounds, graph=graph_holder["graph"])

    def loop_step(client, obs, cfg):
        return reference_negotiate_step(client, obs, cfg, rounds)

    ref_actions,   ref_results,   ref_calls   = _simulate(loop_step,  n_steps)
    graph_actions, graph_results, graph_calls = _simulate(graph_step, n_steps)

    for t, (a, b) in enumerate(zip(ref_actions, graph_actions)):
        assert a == b, f"step {t}: loop={a} graph={b}"
    assert ref_results["costs"]["global"]    == graph_results["costs"]["global"]
    assert ref_results["bullwhip"]["global"] == graph_results["bullwhip"]["global"]
    assert ref_calls == graph_calls == n_steps * DEFAULT_CONFIG["n_agents"] * rounds

    return ref_results, graph_results, graph_calls


if __name__ == "__main__":
    test_graph_structure()
    print("  [pass] graph structure: compute_tools -> propose_round (loop) -> finalize")
    test_round_count()
    print("  [pass] rounds: 1/3/5 rounds -> 3/9/15 LLM calls per step")
    ref, new, calls = test_graph_matches_reference_full_run()
    print(f"  [pass] 200-step run identical to original loop "
          f"({calls} fake LLM calls each)")
    print(f"         global cost {new['costs']['global']:,.0f} | "
          f"global BW {new['bullwhip']['global']:.5f}  (fake model, not real results)")
    print("\n  Mermaid diagram of the negotiation graph:\n")
    print(build_negotiation_graph(FakeLLMClient(), DEFAULT_CONFIG.copy())
          .get_graph().draw_mermaid())
