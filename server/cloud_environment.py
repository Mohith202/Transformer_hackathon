"""
Cloud Resource Management Environment Implementation.

A real-world OpenEnv environment simulating cloud GPU/CPU resource management.
Uses data patterns from the Google Open Cloud dataset to create realistic
workload scenarios. An AI agent must make scaling decisions to optimize
resource utilization, avoid overloads, and manage costs.

3 Tasks (easy → medium → hard):
  1. single_server_scaling   – 1 server,  5 steps
  2. multi_server_balancing  – 3 servers, 8 steps
  3. cost_optimized_planning – 5 servers, 12 steps + budget constraint

MCP Tools:
  - get_cluster_state()           → current metrics for all servers
  - get_task_info()               → task description & objectives
  - take_action(decisions: str)   → apply scaling, advance timestep, return reward
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = {
    "single_server_scaling": {
        "description": (
            "Manage a single cloud server to keep CPU utilization near 70% of capacity. "
            "The workload changes each timestep. Decide whether to scale_up, scale_down, or maintain."
        ),
        "difficulty": "easy",
        "num_servers": 1,
        "max_steps": 5,
        "target_cpu_pct": 0.70,
        "budget_per_step": None,
    },
    "multi_server_balancing": {
        "description": (
            "Balance workload across 3 servers while avoiding overloads. "
            "Each server has different capacity. Keep all servers near 70% CPU utilization."
        ),
        "difficulty": "medium",
        "num_servers": 3,
        "max_steps": 8,
        "target_cpu_pct": 0.70,
        "budget_per_step": None,
    },
    "cost_optimized_planning": {
        "description": (
            "Manage 5 servers under a budget constraint of 150 cost-units per step. "
            "Scaling up increases cost; scaling down saves money. "
            "Balance cost efficiency with SLA compliance (no overloads, target 70% CPU)."
        ),
        "difficulty": "hard",
        "num_servers": 5,
        "max_steps": 12,
        "target_cpu_pct": 0.70,
        "budget_per_step": 150.0,
    },
}

# Server flavor templates (inspired by Google Cloud dataset)
FLAVOR_TEMPLATES = [
    {"name": "general.small", "cpu_capacity": 2.0, "memory_capacity": 4096.0, "cost": 10.0},
    {"name": "general.medium", "cpu_capacity": 4.0, "memory_capacity": 8192.0, "cost": 20.0},
    {"name": "general.large", "cpu_capacity": 8.0, "memory_capacity": 16384.0, "cost": 40.0},
    {"name": "compute.medium", "cpu_capacity": 8.0, "memory_capacity": 8192.0, "cost": 35.0},
    {"name": "memory.medium", "cpu_capacity": 4.0, "memory_capacity": 32768.0, "cost": 30.0},
]


def _generate_workload_trace(num_steps: int, base: float, rng: random.Random) -> list[float]:
    """Generate a realistic workload utilization trace in [0.1, 0.95]."""
    trend = rng.uniform(-0.03, 0.03)
    trace = []
    for t in range(num_steps):
        val = base + trend * t + rng.gauss(0, 0.06)
        if rng.random() < 0.12:
            val += rng.uniform(0.15, 0.35)  # workload spike
        trace.append(max(0.08, min(0.95, val)))
    return trace


class CloudResourceEnvironment(MCPEnvironment):
    """Cloud GPU/CPU resource management environment with MCP tools."""

    def __init__(self):
        mcp = FastMCP("cloud_resource_env")

        # ---- MCP Tools ----
        @mcp.tool
        def get_cluster_state() -> dict:
            """
            Get the current state of all servers in the cluster.
            Returns CPU/memory usage, capacity, utilization percentages, and cost.
            """
            return self._build_cluster_state()

        @mcp.tool
        def get_task_info() -> dict:
            """
            Get information about the current task including objectives and constraints.
            """
            return self._build_task_info()

        @mcp.tool
        def take_action(decisions: str) -> dict:
            """
            Apply scaling decisions and advance the simulation by one timestep.

            Args:
                decisions: JSON string mapping server_id to action.
                    Example: {"server_0": "scale_up", "server_1": "maintain"}
                    Valid actions: "scale_up", "scale_down", "maintain"

            Returns:
                Dictionary with reward, done, feedback, updated cluster_state, and score.
            """
            return self._process_action(decisions)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = "single_server_scaling"
        self._task_cfg: dict = TASKS[self._task_name]
        self._servers: list[dict] = []
        self._workloads: dict[str, list[float]] = {}
        self._mem_workloads: dict[str, list[float]] = {}
        self._timestep: int = 0
        self._step_rewards: list[float] = []
        self._rng = random.Random(42)
        self._episode_done = False

    # ------------------------------------------------------------------
    # OpenEnv lifecycle
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        task = kwargs.get("task", "single_server_scaling")
        if task not in TASKS:
            task = "single_server_scaling"

        self._task_name = task
        self._task_cfg = TASKS[task]
        self._rng = random.Random(seed if seed is not None else 42)
        self._timestep = 0
        self._step_rewards = []
        self._episode_done = False

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Initialize servers
        num = self._task_cfg["num_servers"]
        self._servers = []
        for i in range(num):
            tmpl = FLAVOR_TEMPLATES[i % len(FLAVOR_TEMPLATES)]
            self._servers.append({
                "server_id": f"server_{i}",
                "flavor": tmpl["name"],
                "cpu_capacity": tmpl["cpu_capacity"],
                "memory_capacity": tmpl["memory_capacity"],
                "cost_per_step": tmpl["cost"],
            })

        # Generate workload traces
        max_steps = self._task_cfg["max_steps"]
        self._workloads = {}
        self._mem_workloads = {}
        for srv in self._servers:
            sid = srv["server_id"]
            base_cpu = self._rng.uniform(0.35, 0.75)
            base_mem = self._rng.uniform(0.30, 0.70)
            self._workloads[sid] = _generate_workload_trace(max_steps + 1, base_cpu, self._rng)
            self._mem_workloads[sid] = _generate_workload_trace(max_steps + 1, base_mem, self._rng)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task": self._task_name,
                "difficulty": self._task_cfg["difficulty"],
                "message": f"Cloud resource environment ready. Task: {self._task_name}",
                "cluster_state": json.dumps(self._build_cluster_state()),
                "task_info": json.dumps(self._build_task_info()),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP direct step() calls with CloudAction."""
        if hasattr(action, "decisions"):
            result = self._process_action(action.decisions)
            return Observation(
                done=result["done"],
                reward=result["reward"],
                metadata=result,
            )
        return Observation(
            done=False,
            reward=0.0,
            metadata={"error": f"Unknown action type: {type(action).__name__}. Use MCP tools or CloudAction."},
        )

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _current_server_metrics(self, srv: dict) -> dict:
        sid = srv["server_id"]
        t = min(self._timestep, len(self._workloads.get(sid, [0.5])) - 1)
        cpu_util_pct = self._workloads[sid][t]
        mem_util_pct = self._mem_workloads[sid][t]
        cpu_usage = cpu_util_pct * srv["cpu_capacity"]
        mem_usage = mem_util_pct * srv["memory_capacity"]
        return {
            "server_id": sid,
            "flavor": srv["flavor"],
            "cpu_capacity": round(srv["cpu_capacity"], 2),
            "memory_capacity": round(srv["memory_capacity"], 2),
            "cpu_usage": round(cpu_usage, 2),
            "memory_usage": round(mem_usage, 2),
            "cpu_utilization_pct": round(cpu_util_pct * 100, 1),
            "memory_utilization_pct": round(mem_util_pct * 100, 1),
            "cost_per_step": round(srv["cost_per_step"], 2),
        }

    def _build_cluster_state(self) -> dict:
        servers = [self._current_server_metrics(s) for s in self._servers]
        total_cost = sum(s["cost_per_step"] for s in servers)
        budget = self._task_cfg.get("budget_per_step")
        return {
            "timestep": self._timestep,
            "max_timesteps": self._task_cfg["max_steps"],
            "servers": servers,
            "total_cost_per_step": round(total_cost, 2),
            "budget_per_step": budget,
            "budget_remaining": round(budget - total_cost, 2) if budget else None,
        }

    def _build_task_info(self) -> dict:
        return {
            "task_name": self._task_name,
            "difficulty": self._task_cfg["difficulty"],
            "description": self._task_cfg["description"],
            "num_servers": self._task_cfg["num_servers"],
            "max_steps": self._task_cfg["max_steps"],
            "target_cpu_utilization_pct": self._task_cfg["target_cpu_pct"] * 100,
            "budget_per_step": self._task_cfg.get("budget_per_step"),
            "objectives": [
                "Keep CPU utilization near target (70%)",
                "Avoid overloads (utilization > 100%)",
                "Minimize unnecessary scaling actions",
            ]
            + (["Stay within budget constraint"] if self._task_cfg.get("budget_per_step") else []),
        }

    def _process_action(self, decisions_str: str) -> dict:
        if self._episode_done:
            return {
                "reward": 0.0,
                "done": True,
                "feedback": "Episode already finished.",
                "cluster_state": self._build_cluster_state(),
                "score": self._compute_score(),
            }

        # Parse decisions
        try:
            decisions = json.loads(decisions_str) if isinstance(decisions_str, str) else decisions_str
        except (json.JSONDecodeError, TypeError):
            decisions = {}

        # Apply scaling
        feedback_lines = []
        for srv in self._servers:
            sid = srv["server_id"]
            action = decisions.get(sid, "maintain")
            if action == "scale_up":
                srv["cpu_capacity"] = round(srv["cpu_capacity"] * 1.5, 2)
                srv["memory_capacity"] = round(srv["memory_capacity"] * 1.5, 2)
                srv["cost_per_step"] = round(srv["cost_per_step"] * 1.5, 2)
                feedback_lines.append(f"{sid}: scaled UP (capacity +50%, cost +50%)")
            elif action == "scale_down":
                srv["cpu_capacity"] = round(max(1.0, srv["cpu_capacity"] / 1.5), 2)
                srv["memory_capacity"] = round(max(1024.0, srv["memory_capacity"] / 1.5), 2)
                srv["cost_per_step"] = round(max(5.0, srv["cost_per_step"] / 1.5), 2)
                feedback_lines.append(f"{sid}: scaled DOWN (capacity -33%, cost -33%)")
            else:
                feedback_lines.append(f"{sid}: maintained")

        # Advance timestep
        self._timestep += 1

        # Compute reward
        target = self._task_cfg["target_cpu_pct"]
        step_reward = 0.0
        overloaded = False
        for srv in self._servers:
            metrics = self._current_server_metrics(srv)
            cpu_pct = metrics["cpu_utilization_pct"] / 100.0
            mem_pct = metrics["memory_utilization_pct"] / 100.0

            if cpu_pct > 1.0 or mem_pct > 1.0:
                overloaded = True
                step_reward += -0.5
                feedback_lines.append(f"⚠️ {srv['server_id']} OVERLOADED!")
            else:
                eff = max(0.0, 1.0 - 2.0 * abs(cpu_pct - target))
                step_reward += eff

        step_reward /= len(self._servers)
        step_reward = max(0.0, min(1.0, step_reward))

        # Budget penalty (hard task)
        budget = self._task_cfg.get("budget_per_step")
        if budget:
            total_cost = sum(s["cost_per_step"] for s in self._servers)
            if total_cost > budget:
                step_reward *= 0.5
                feedback_lines.append(f"⚠️ Over budget! Cost {total_cost:.0f} > Budget {budget:.0f}")

        self._step_rewards.append(step_reward)
        done = self._timestep >= self._task_cfg["max_steps"]
        if done:
            self._episode_done = True

        score = self._compute_score()

        return {
            "reward": round(step_reward, 4),
            "done": done,
            "feedback": " | ".join(feedback_lines),
            "cluster_state": self._build_cluster_state(),
            "score": round(score, 4),
            "overloaded": overloaded,
            "timestep": self._timestep,
            "max_timesteps": self._task_cfg["max_steps"],
        }

    def _compute_score(self) -> float:
        if not self._step_rewards:
            return 0.0
        return max(0.0, min(1.0, sum(self._step_rewards) / len(self._step_rewards)))
