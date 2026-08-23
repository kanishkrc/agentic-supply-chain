"""
Configuration — all parameters from the paper (Appendix 6) and experiment setup.

Paper reference:
  Jannelli et al. (2025) "Agentic LLMs in the supply chain:
  towards autonomous multi-agent consensus-seeking"
  Table 2 (Appendix 6), Table 1 (experiment setup)
"""

from enum import IntEnum


class AgentRole(IntEnum):
    """Agent positions in the sequential supply chain.
    Agent 0 is closest to the customer (Retailer).
    Agent n-1 is furthest upstream (Manufacturer).
    """
    RETAILER     = 0
    DISTRIBUTOR  = 1
    MANUFACTURER = 2


# ── Core simulation parameters (Appendix 6) ─────────────────────────────────
DEFAULT_CONFIG = {
    # Supply chain structure
    "n_agents":         3,       # Retailer, Distributor, Manufacturer
    "lead_time":        2,       # steps until ordered goods arrive
    "max_order":        100,     # hard upper bound on any order quantity
    "initial_inventory": 0,      # starting inventory for every agent

    # Cost weights — all equal to 1 (Appendix 6)
    "holding_cost":       1,     # cost per unit of positive inventory per step
    "backlog_cost":       1,     # cost per unit of unmet demand per step
    "ordering_cost":      1,     # variable cost per unit ordered
    "fixed_ordering_cost": 1,    # fixed cost if any order is placed (>0)

    # Simulation length
    "n_steps": 200,

    # Agent memory — how many past observations to keep in prompt
    "n_obs_memory":   10,        # observation window for LLM memory
    "n_obs_forecast": 30,        # look-back window for demand-forecast tool

    # LLM generation
    "temperature":   0.1,        # near-deterministic (paper §5.2)
    "max_tokens":    90,         # max output tokens (Appendix 6)

    # Negotiation
    "num_iter": 3,               # negotiation rounds between neighbours (Fig 7)

    # Demand model (Merton Jump Diffusion, run 13 from Liu et al. 2022)
    "demand_seed": 13,
    "demand_S0":   8.0,          # initial demand level
    "demand_mu":  -0.015,        # drift (mild downward trend)
    "demand_sigma": 0.25,        # diffusion volatility
    "demand_lam":   0.03,        # Poisson jump intensity (jumps per step)
    "demand_mu_j": -4.0,         # mean log-jump size (large negative spike)
    "demand_sigma_j": 1.0,       # jump-size std dev
    "demand_clip": (0, 20),      # clip demand to [0, 20] (paper §5)
}

# ── Baseline policy parameters ───────────────────────────────────────────────
SS_POLICY = {
    "S": 100,   # order-up-to level  (paper §5.1)
    "s": 60,    # reorder point
}

# ── Vertex AI model IDs ──────────────────────────────────────────────────────
MODELS = {
    "flash": "gemini-2.5-flash-preview-04-17",
    "pro":   "gemini-2.5-pro-preview-05-06",
}

# Agent role labels used in prompts and outputs
ROLE_LABELS = {
    0: "Retailer",
    1: "Distributor",
    2: "Manufacturer",
}
