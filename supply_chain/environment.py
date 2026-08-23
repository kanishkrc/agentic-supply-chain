"""
Supply Chain Simulation Environment.

Implements the multi-echelon sequential supply chain from Liu et al. (2022),
extended by Jannelli et al. (2025) for LLM-based consensus-seeking.

Chain topology (paper Fig 2):
  Customer → Retailer (Agent 0) → Distributor (Agent 1) → Manufacturer (Agent 2)
  ← replenishment ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
  → demand (orders) →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→

Step mechanics (executed each period):
  1. Each agent receives the shipment that was ordered `lead_time` steps ago.
  2. Each agent fills downstream demand from current inventory.
     Unmet demand accumulates as backlog.
  3. State is observed (inventory, backlog, pipeline, demand history).
  4. Each agent places a new order to its upstream neighbour.

Cost per agent per step (paper §3.1):
  C_i(t) = holding_cost  * max(inventory, 0)
          + backlog_cost  * backlog
          + ordering_cost * order_qty
          + fixed_cost    * (1 if order_qty > 0 else 0)

Partial observability:
  Each agent only sees its own state (inventory, backlog, pipeline, last order,
  demand from its immediate downstream neighbour). It has NO visibility of
  other agents' inventories or orders beyond its direct neighbour.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Single Agent
# ─────────────────────────────────────────────────────────────────────────────

class SupplyChainAgent:
    """
    One node in the sequential supply chain.

    Attributes
    ----------
    agent_id    : int position (0 = Retailer, n-1 = Manufacturer)
    inventory   : units currently on hand
    backlog     : units of demand that could not be filled
    pipeline    : deque of length `lead_time`; pipeline[0] arrives next step,
                  pipeline[-1] arrives in `lead_time` steps
    last_order  : order placed in the previous step
    last_demand : downstream demand received in the previous step
    """

    def __init__(self, agent_id: int, config: dict) -> None:
        self.agent_id = agent_id
        self.cfg      = config
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.inventory   : int   = self.cfg["initial_inventory"]
        self.backlog     : int   = 0
        self.last_order  : int   = 0
        self.last_demand : int   = 0

        # Pipeline: slot 0 arrives next step (FIFO queue)
        self.pipeline: deque[int] = deque(
            [0] * self.cfg["lead_time"],
            maxlen=self.cfg["lead_time"],
        )

        # Rolling history (used for prompt memory and tools)
        self.inventory_history : List[int]   = []
        self.backlog_history   : List[int]   = []
        self.order_history     : List[int]   = []
        self.demand_history    : List[int]   = []
        self.cost_history      : List[float] = []

    # ── Per-step mechanics ────────────────────────────────────────────────────

    def receive_shipment(self) -> int:
        """
        Advance the pipeline: the oldest shipment arrives.
        Returns the number of units received.
        """
        arrived = self.pipeline.popleft()   # oldest entry
        self.inventory += arrived
        self.pipeline.append(0)             # placeholder for the new slot
        return arrived

    def fill_demand(self, demand: int) -> Tuple[int, int]:
        """
        Attempt to fill `demand` plus any outstanding backlog from inventory.

        Returns
        -------
        shipped   : units actually shipped downstream
        new_backlog : remaining unfulfilled demand
        """
        self.last_demand = demand
        total_needed = demand + self.backlog

        if self.inventory >= total_needed:
            shipped          = total_needed
            self.inventory  -= total_needed
            self.backlog     = 0
        else:
            shipped          = max(0, self.inventory)
            self.backlog     = total_needed - shipped
            self.inventory   = 0

        return shipped, self.backlog

    def place_order(self, qty: int) -> int:
        """
        Clip order to [0, max_order], record it, and insert into pipeline.
        The pipeline slot at index [-1] is the order placed this step;
        it will shift to index [0] after `lead_time - 1` more steps.

        Returns the actual (clipped) order quantity.
        """
        qty = int(np.clip(qty, 0, self.cfg["max_order"]))

        # Overwrite the placeholder appended in receive_shipment
        self.pipeline[-1] = qty
        self.last_order   = qty
        self.order_history.append(qty)
        return qty

    def record_state(self) -> None:
        """Snapshot inventory, backlog, and demand into rolling histories."""
        self.inventory_history.append(self.inventory)
        self.backlog_history.append(self.backlog)
        self.demand_history.append(self.last_demand)

    def compute_step_cost(self, order_qty: int) -> float:
        """
        Evaluate the cost function for this step (paper §3.1).
        Called AFTER fill_demand so inventory/backlog reflect the new state.
        """
        holding  = self.cfg["holding_cost"]       * max(0, self.inventory)
        backcost = self.cfg["backlog_cost"]        * self.backlog
        variable = self.cfg["ordering_cost"]       * order_qty
        fixed    = self.cfg["fixed_ordering_cost"] * (1 if order_qty > 0 else 0)

        step_cost = holding + backcost + variable + fixed
        self.cost_history.append(step_cost)
        return step_cost

    # ── Observation (returned to agents / LLM) ────────────────────────────────

    def get_observation(self) -> Dict:
        """
        Return a dict of everything this agent can observe — its LOCAL view.

        Key fields used in the LLM prompt (Appendix 2, P3–P5):
          inventory       → P3
          backlog         → P3
          last_order      → P3
          pipeline        → P3  (incoming / transfer orders)
          last_demand     → P3  (demand from downstream neighbour)
          memory_*        → P4  (last n_obs_memory steps)
          demand_history  → P5  (last n_obs_forecast steps for tool)
        """
        mem = self.cfg["n_obs_memory"]
        fct = self.cfg["n_obs_forecast"]

        return {
            "agent_id":        self.agent_id,
            "inventory":       self.inventory,
            "backlog":         self.backlog,
            "last_order":      self.last_order,
            "last_demand":     self.last_demand,
            "pipeline":        list(self.pipeline),   # [arrives_t+1, arrives_t+2]
            # Memory for LLM context (P4)
            "memory_inventory": self.inventory_history[-mem:],
            "memory_backlog":   self.backlog_history[-mem:],
            "memory_orders":    self.order_history[-mem:],
            # Demand history for forecasting tool (P5)
            "demand_history":   self.demand_history[-fct:],
            # Running costs
            "cumulative_cost":  sum(self.cost_history),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Echelon Environment
# ─────────────────────────────────────────────────────────────────────────────

class SupplyChainEnv:
    """
    Sequential multi-echelon supply chain environment.

    Usage
    -----
    env = SupplyChainEnv(config=DEFAULT_CONFIG)
    obs = env.reset(demands)          # list of customer demands, length n_steps

    while not done:
        actions = policy(obs)         # list of n_agents order quantities
        obs, costs, done, info = env.step(actions)

    results = env.get_results()       # full cost / bullwhip summary
    """

    def __init__(self, config: dict | None = None) -> None:
        self.cfg     = config or DEFAULT_CONFIG
        self.n       = self.cfg["n_agents"]
        self.agents  = [SupplyChainAgent(i, self.cfg) for i in range(self.n)]
        self.demands : List[int] = []
        self.t       : int       = 0

    # ── Setup ─────────────────────────────────────────────────────────────────

    def reset(self, demands: List[int]) -> List[Dict]:
        """
        Reset all agents and load the demand sequence.

        Parameters
        ----------
        demands : pre-generated customer demand for every step (length n_steps)

        Returns
        -------
        Initial observations for all agents.
        """
        for agent in self.agents:
            agent.reset()

        self.demands = list(demands)
        self.t       = 0
        return [a.get_observation() for a in self.agents]

    # ── Core step ─────────────────────────────────────────────────────────────

    def step(
        self,
        actions: List[int],
    ) -> Tuple[List[Dict], List[float], bool, Dict]:
        """
        Advance the simulation by one step.

        Parameters
        ----------
        actions : order quantities [a_0, a_1, ..., a_{n-1}]
                  a_i is the order placed by Agent i to its upstream.

        Returns
        -------
        observations : updated local observations for each agent
        costs        : step cost for each agent
        done         : True when all demand steps are exhausted
        info         : extra diagnostics (global cost, customer demand)
        """
        assert len(actions) == self.n, \
            f"Expected {self.n} actions, got {len(actions)}"
        assert self.t < len(self.demands), "Environment is done. Call reset()."

        customer_demand = self.demands[self.t]

        # ── 1. All agents receive incoming shipments ──────────────────────────
        for agent in self.agents:
            agent.receive_shipment()

        # ── 2. Demands flow downstream→upstream ──────────────────────────────
        # Agent 0 faces customer demand.
        # Agent i (i>0) faces the order placed by Agent i-1 in the PREVIOUS
        # step — this models partial observability: Agent i cannot see Agent
        # i-1's order for the current step until after its own decision.
        demands_this_step = [customer_demand] + [
            self.agents[i - 1].last_order for i in range(1, self.n)
        ]

        for i, agent in enumerate(self.agents):
            agent.fill_demand(demands_this_step[i])

        # ── 3. Record state snapshot ──────────────────────────────────────────
        for agent in self.agents:
            agent.record_state()

        # ── 4. Apply actions (place orders) and compute costs ─────────────────
        costs = []
        for i, agent in enumerate(self.agents):
            actual_qty = agent.place_order(actions[i])
            cost       = agent.compute_step_cost(actual_qty)
            costs.append(cost)

        # ── 5. Advance time ───────────────────────────────────────────────────
        self.t += 1
        done = (self.t >= len(self.demands))

        observations = [a.get_observation() for a in self.agents]
        info = {
            "step":            self.t,
            "customer_demand": customer_demand,
            "global_cost":     sum(costs),
        }

        return observations, costs, done, info

    # ── Results ───────────────────────────────────────────────────────────────

    def get_results(self) -> Dict:
        """
        Aggregate results after a full run.

        Returns cost and bullwhip metrics matching the paper's Tables 1 & 2.
        """
        from .metrics import compute_bullwhip, compute_costs
        return {
            "costs":    compute_costs(self),
            "bullwhip": compute_bullwhip(self),
        }

    def get_histories(self) -> Dict:
        """
        Return full per-step time-series for every agent — used for plotting.

        Keys per agent (list of length n_steps):
          orders      : order placed each step
          inventory   : inventory on hand after filling demand
          backlog     : unfilled backlog after filling demand
          costs       : step cost
          demand      : demand received from downstream neighbour
        Also includes:
          customer_demand : demand sequence seen by the Retailer
          n_steps         : number of completed steps
        """
        role_labels = {0: "Retailer", 1: "Distributor", 2: "Manufacturer"}
        agents_data = {}
        for agent in self.agents:
            label = role_labels.get(agent.agent_id, f"Agent {agent.agent_id}")
            agents_data[label] = {
                "orders":    list(agent.order_history),
                "inventory": list(agent.inventory_history),
                "backlog":   list(agent.backlog_history),
                "costs":     list(agent.cost_history),
                "demand":    list(agent.demand_history),
            }
        return {
            "agents":          agents_data,
            "customer_demand": list(self.demands[: self.t]),
            "n_steps":         self.t,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_observations(self) -> List[Dict]:
        """Return current observations for all agents without stepping."""
        return [a.get_observation() for a in self.agents]

    def __repr__(self) -> str:
        agents = ", ".join(
            f"A{a.agent_id}(inv={a.inventory}, bl={a.backlog})"
            for a in self.agents
        )
        return f"SupplyChainEnv(t={self.t}/{len(self.demands)}, [{agents}])"
