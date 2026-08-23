"""
Prompt Builder — constructs structured prompts for each Framework.

Paper reference: Jannelli et al. (2025), Appendix 2 (prompt sections P1-P7).

Section map
-----------
P1  Role identity        — who the agent is in the supply chain
P2  Task description     — what it must decide and optimise
P3  Current state        — inventory, backlog, pipeline, last order, last demand
P4  Memory               — rolling window of past n_obs_memory steps
P5  Tool context         — demand history for tool use (F3-F5 only)
P6  Shared information   — upstream/downstream messages (F2, F4, F5 only)
P7  Output format        — strict instruction to reply with a single integer
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .config import ROLE_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _role_label(agent_id: int) -> str:
    return ROLE_LABELS.get(agent_id, f"Agent {agent_id}")


def _p1_role(agent_id: int) -> str:
    role = _role_label(agent_id)
    upstream = {
        0: "a Distributor",
        1: "a Manufacturer",
        2: "an unlimited external supplier",
    }.get(agent_id, "an upstream supplier")
    downstream = {
        0: "end customers",
        1: "a Retailer",
        2: "a Distributor",
    }.get(agent_id, "downstream agents")

    return (
        f"You are the {role} in a 3-echelon supply chain.\n"
        f"You receive goods from {upstream} and fulfil orders from {downstream}.\n"
        f"Your goal is to minimise total cost over the simulation.\n"
    )


def _p2_task() -> str:
    return (
        "Each period you must decide how many units to ORDER from your upstream supplier.\n"
        "Cost per period = holding_cost * max(inventory, 0)\n"
        "                + backlog_cost  * backlog\n"
        "                + ordering_cost * order_qty\n"
        "                + fixed_cost    * (1 if order_qty > 0 else 0)\n"
        "All cost weights are 1. Ordered goods arrive after a lead time of 2 periods.\n"
        "You can only see your own state — you do NOT know other agents' inventories.\n"
    )


def _p3_state(obs: Dict) -> str:
    pipeline_str = ", ".join(str(x) for x in obs["pipeline"])
    return (
        "=== Current State ===\n"
        f"Inventory        : {obs['inventory']} units\n"
        f"Backlog          : {obs['backlog']} units\n"
        f"Last order placed: {obs['last_order']} units\n"
        f"Last demand recv : {obs['last_demand']} units\n"
        f"Pipeline (arrive in 1, 2 steps): [{pipeline_str}]\n"
        f"Cumulative cost  : {obs['cumulative_cost']:,.0f}\n"
    )


def _p4_memory(obs: Dict) -> str:
    inv_str   = ", ".join(str(x) for x in obs["memory_inventory"])
    bl_str    = ", ".join(str(x) for x in obs["memory_backlog"])
    ord_str   = ", ".join(str(x) for x in obs["memory_orders"])
    return (
        "=== Recent History (newest last) ===\n"
        f"Inventory : [{inv_str}]\n"
        f"Backlog   : [{bl_str}]\n"
        f"Orders    : [{ord_str}]\n"
    )


def _p5_demand_history(obs: Dict) -> str:
    hist = obs["demand_history"]
    hist_str = ", ".join(str(x) for x in hist)
    return (
        "=== Demand History (last 30 periods, newest last) ===\n"
        f"[{hist_str}]\n"
    )


def _p7_output_format() -> str:
    return (
        "=== Your Decision ===\n"
        "Output ONLY a single integer — your order quantity (0 to 100).\n"
        "No words, no explanation, no punctuation. Just the integer.\n"
        "Example valid responses: 0    5    42    100\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Framework 1 — Standalone Agent (no tools, no communication)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_f1(obs: Dict) -> str:
    """
    Construct the Framework 1 prompt for a single agent.

    Sections: P1 (role) + P2 (task) + P3 (state) + P4 (memory) + P7 (format)
    """
    return "\n".join([
        _p1_role(obs["agent_id"]),
        _p2_task(),
        _p3_state(obs),
        _p4_memory(obs),
        _p7_output_format(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Framework 2 — Information Sharing (neighbours broadcast state summaries)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_f2(obs: Dict, neighbour_messages: List[str]) -> str:
    """
    Framework 2 prompt: own state + messages from neighbouring agents.

    neighbour_messages : list of pre-formatted strings from each neighbour
    """
    # P1 reframed as chain-level objective (same as F5)
    role = _role_label(obs["agent_id"])
    upstream_label = {
        0: "a Distributor",
        1: "a Manufacturer",
        2: "an unlimited external supplier",
    }.get(obs["agent_id"], "an upstream supplier")
    downstream_label = {
        0: "end customers",
        1: "a Retailer",
        2: "a Distributor",
    }.get(obs["agent_id"], "downstream agents")

    p1 = (
        f"You are the {role} in a 3-echelon supply chain.\n"
        f"You receive goods from {upstream_label} and fulfil orders from "
        f"{downstream_label}.\n"
        f"GOAL: minimise TOTAL SUPPLY CHAIN cost — not just your own. "
        f"Smooth, proportional ordering reduces cost for everyone.\n"
    )

    parts = [
        p1,
        _p2_task(),
        _p3_state(obs),
        _p4_memory(obs),
    ]

    # Inline inventory-position heuristic so agent has a concrete anchor
    pipeline = obs.get("pipeline", [])
    inventory    = obs.get("inventory", 0)
    backlog      = obs.get("backlog", 0)
    last_demand  = obs.get("last_demand", 0)
    lead_time    = 2
    inv_position = inventory + sum(pipeline) - backlog
    # Conservative target: cover (lead_time+1) periods of last observed demand
    target       = last_demand * (lead_time + 1)
    anchor       = max(0, target - inv_position)

    parts.append(
        f"=== Inventory Position Heuristic ===\n"
        f"Inventory position = inventory + pipeline - backlog\n"
        f"                   = {inventory} + {sum(pipeline)} - {backlog} = {inv_position}\n"
        f"Target coverage    = last_demand x (lead_time+1) = {last_demand} x {lead_time+1} = {target}\n"
        f"Suggested order    = max(0, target - inv_position) = {anchor} units\n"
        f"\n"
        f"Use {anchor} units as your BASE ORDER. "
        f"Adjust up or down by a few units only — do not ignore this anchor."
    )

    if neighbour_messages:
        parts.append("=== Information from Neighbours ===")
        for msg in neighbour_messages:
            parts.append(msg)

        # Anti-panic guidance: small adjustments from anchor, not panic reactions
        parts.append(
            "=== How to Use Neighbour Information ===\n"
            "Your anchor is already computed above. Neighbour info allows SMALL adjustments:\n"
            "\n"
            f"  + Downstream neighbour has HIGH backlog  -> order anchor + 1 to +3 units\n"
            f"  + Downstream neighbour has HIGH inventory -> order anchor - 1 to -2 units\n"
            f"  + Upstream neighbour has LOW inventory    -> order anchor + 1 (build buffer)\n"
            f"  + Upstream neighbour has HIGH inventory   -> no change needed\n"
            f"\n"
            f"NEVER exceed anchor + 5 in reaction to a neighbour's state.\n"
            f"Large spikes create bullwhip that raises costs for the whole chain."
        )

    parts.append(_p7_output_format())
    return "\n".join(parts)


def format_neighbour_message(neighbour_obs: Dict, relationship: str) -> str:
    """
    Format a neighbour's observation into a human-readable message.

    relationship : "upstream" or "downstream"
    """
    role = _role_label(neighbour_obs["agent_id"])
    return (
        f"Your {relationship} neighbour ({role}) reports:\n"
        f"  Inventory={neighbour_obs['inventory']}, "
        f"Backlog={neighbour_obs['backlog']}, "
        f"Last order={neighbour_obs['last_order']}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Framework 3 — Standalone + Tools
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_f3(obs: Dict, tool_results: Optional[Dict] = None) -> str:
    """
    Framework 3 prompt: own state + demand history + tool outputs.

    tool_results : dict with optional keys 'demand_forecast' and 'eoq'
    """
    parts = [
        _p1_role(obs["agent_id"]),
        _p2_task(),
        _p3_state(obs),
        _p4_memory(obs),
        _p5_demand_history(obs),
    ]

    if tool_results:
        parts.append("=== Tool Outputs ===")
        if "demand_forecast" in tool_results:
            parts.append(
                f"Demand Forecast (linear regression, next period): "
                f"{tool_results['demand_forecast']} units"
            )
        if "eoq" in tool_results:
            parts.append(
                f"EOQ (economic order quantity): {tool_results['eoq']} units"
            )

    parts.append(_p7_output_format())
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Framework 4 — Information Sharing + Tools
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_f4(
    obs: Dict,
    neighbour_messages: List[str],
    tool_results: Optional[Dict] = None,
) -> str:
    """Framework 4: F2 (info sharing) + F3 (tools)."""
    parts = [
        _p1_role(obs["agent_id"]),
        _p2_task(),
        _p3_state(obs),
        _p4_memory(obs),
        _p5_demand_history(obs),
    ]

    if neighbour_messages:
        parts.append("=== Information from Neighbours ===")
        for msg in neighbour_messages:
            parts.append(msg)

    if tool_results:
        parts.append("=== Tool Outputs ===")
        if "demand_forecast" in tool_results:
            parts.append(
                f"Demand Forecast (linear regression, next period): "
                f"{tool_results['demand_forecast']} units"
            )
        if "eoq" in tool_results:
            parts.append(f"EOQ: {tool_results['eoq']} units")

    parts.append(_p7_output_format())
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Framework 5 — Negotiation + Tools
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_f5_propose(
    obs: Dict,
    tool_results: Dict,
    negotiation_round: int,
    neighbour_proposals: Optional[List[str]] = None,
) -> str:
    """
    Framework 5 negotiation prompt — propose an order quantity.

    Called at each negotiation round (paper: 3 rounds between neighbours).
    neighbour_proposals : proposals received from neighbours in this round.
    """
    num_rounds = 3
    df_val  = tool_results.get("demand_forecast", 0) if tool_results else 0
    eoq_val = tool_results.get("eoq", 0) if tool_results else 0
    backlog = obs.get("backlog", 0)
    lead_time = 2   # matches DEFAULT_CONFIG

    # ---- P1 role — reframed as CHAIN-level objective -------------------------
    role = _role_label(obs["agent_id"])
    upstream_label = {
        0: "a Distributor",
        1: "a Manufacturer",
        2: "an unlimited external supplier",
    }.get(obs["agent_id"], "an upstream supplier")
    downstream_label = {
        0: "end customers",
        1: "a Retailer",
        2: "a Distributor",
    }.get(obs["agent_id"], "downstream agents")

    p1 = (
        f"You are the {role} in a 3-echelon supply chain.\n"
        f"You receive goods from {upstream_label} and fulfil orders from "
        f"{downstream_label}.\n"
        f"GOAL: minimise TOTAL SUPPLY CHAIN cost — not just your own. "
        f"Your neighbour's backlog directly feeds your future costs. "
        f"Coordinate so the whole chain clears its obligations efficiently.\n"
    )

    parts = [
        p1,
        _p2_task(),
        _p3_state(obs),
        _p4_memory(obs),
        _p5_demand_history(obs),
    ]

    # ---- Tool anchor — treat demand forecast as a HARD minimum ---------------
    if tool_results:
        # Compute backlog contribution: orders to clear debt over remaining horizon
        min_for_backlog = max(1, backlog // (lead_time + 1)) if backlog > 0 else 0
        tool_min = max(df_val, min_for_backlog)

        parts.append("=== Tool Outputs (Anchors) ===")
        parts.append(
            f"Demand Forecast (next period, inventory-position adjusted): "
            f"{df_val} units\n"
            f"EOQ (economically optimal batch size): {eoq_val} units\n"
            f"\n"
            f"MINIMUM ORDER FLOOR: {tool_min} units\n"
            f"Your proposal MUST be at least {tool_min} units.\n"
            f"  - Ordering below {df_val} leaves you short next period — "
            f"backlog costs exceed holding costs.\n"
            f"  - Ordering below {min_for_backlog} ignores your existing backlog "
            f"obligation (you owe {backlog} units downstream).\n"
            f"You MAY propose more than {tool_min} to help a neighbour in crisis."
        )

    # ---- Round-specific instruction ------------------------------------------
    parts.append(f"=== Negotiation Round {negotiation_round}/{num_rounds} ===")

    if negotiation_round == 1:
        parts.append(
            "Round 1 — Anchor proposal.\n"
            "There are no neighbour proposals yet.\n"
            f"Start with your demand forecast ({df_val}) adjusted upward by any "
            f"backlog you carry ({backlog} units outstanding).\n"
            f"Initial proposal = max(demand_forecast, eoq) + adjustment for backlog.\n"
            f"Do NOT propose 0 — a zero order propagates shortages upstream and "
            f"creates backlog costs for the whole chain."
        )
    elif neighbour_proposals:
        # Parse numeric proposals from the message strings for midpoint guidance
        import re as _re
        neighbour_nums = []
        for prop_str in neighbour_proposals:
            m = _re.search(r"proposes:\s*(\d+)", prop_str)
            if m:
                neighbour_nums.append(int(m.group(1)))

        if negotiation_round == 2:
            parts.append(
                f"Round 2 — Incorporate neighbour context.\n"
                "Neighbour proposals received:"
            )
            for prop in neighbour_proposals:
                parts.append(f"  {prop}")
            if neighbour_nums:
                avg_neighbour = sum(neighbour_nums) / len(neighbour_nums)
                guidance_val  = max(tool_min, round((df_val + avg_neighbour) / 2))
                parts.append(
                    f"\nYour tools recommend {df_val} units. "
                    f"Neighbour average is {avg_neighbour:.0f} units.\n"
                    f"Suggested midpoint: {guidance_val} units.\n"
                    f"If a neighbour proposes 0, that signals they CANNOT fill "
                    f"your upstream orders — raise your own proposal to build buffer.\n"
                    f"Move toward the midpoint but stay above your floor ({tool_min})."
                )
        else:  # round 3 — converge
            parts.append(
                f"Round 3 — Final convergence.\n"
                "Neighbour proposals received:"
            )
            for prop in neighbour_proposals:
                parts.append(f"  {prop}")
            if neighbour_nums:
                avg_neighbour = sum(neighbour_nums) / len(neighbour_nums)
                converge_val  = max(tool_min, round((df_val + avg_neighbour) / 2))
                parts.append(
                    f"\nThis is the final round — your proposal becomes your order.\n"
                    f"Converge toward {converge_val} units "
                    f"(midpoint of your forecast and neighbour average).\n"
                    f"Ensure your proposal is >= {tool_min} (your hard floor).\n"
                    f"Small consistent orders beat erratic large ones — "
                    f"smooth ordering reduces bullwhip for the whole chain."
                )
    else:
        # rounds 2-3 but no neighbour proposals (shouldn't happen)
        parts.append(
            f"Round {negotiation_round} — No neighbour proposals available.\n"
            f"Maintain your anchor from Round 1 (>= {tool_min} units)."
        )

    parts.append(
        "=== Your Proposal ===\n"
        "Output ONLY a single integer — your proposed order quantity (0 to 100).\n"
        "No words, no explanation. Just the integer.\n"
    )
    return "\n".join(parts)
