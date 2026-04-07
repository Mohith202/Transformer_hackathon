"""
Cloud GPU+CPU Resource Management Environment Implementation.

A real-world OpenEnv environment simulating cloud GPU **and** CPU resource
management.  Three progressively harder tasks:

  1. gpu_cpu_allocation        – combined GPU+CPU allocation with cost optimisation
  2. thermal_management        – threshold-based thermal monitoring & cooling
  3. heuristic_fragmentation   – GPU allocation via heuristic fragmentation strategies

MCP Tools:
  - get_cluster_state()             → current metrics for all nodes
  - get_task_info()                 → task description & objectives
  - take_action(decisions: str)     → apply action, advance timestep, return reward
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
# GPU node templates (models commonly found in cloud)
# ---------------------------------------------------------------------------
GPU_NODE_TEMPLATES = [
    {
        "name": "T4-node",
        "gpu_type": "T4",
        "gpu_count": 1,
        "gpu_vram_gb": 16.0,
        "cpu_capacity": 4.0,
        "memory_capacity_gb": 16.0,
        "cost_per_step": 8.0,
        "tdp_watts": 70.0,
        "max_temp_c": 83.0,
    },
    {
        "name": "A100-node",
        "gpu_type": "A100",
        "gpu_count": 1,
        "gpu_vram_gb": 40.0,
        "cpu_capacity": 8.0,
        "memory_capacity_gb": 64.0,
        "cost_per_step": 30.0,
        "tdp_watts": 250.0,
        "max_temp_c": 85.0,
    },
    {
        "name": "H100-node",
        "gpu_type": "H100",
        "gpu_count": 1,
        "gpu_vram_gb": 80.0,
        "cpu_capacity": 16.0,
        "memory_capacity_gb": 128.0,
        "cost_per_step": 55.0,
        "tdp_watts": 350.0,
        "max_temp_c": 83.0,
    },
    {
        "name": "V100-node",
        "gpu_type": "V100",
        "gpu_count": 1,
        "gpu_vram_gb": 32.0,
        "cpu_capacity": 8.0,
        "memory_capacity_gb": 32.0,
        "cost_per_step": 18.0,
        "tdp_watts": 300.0,
        "max_temp_c": 84.0,
    },
    {
        "name": "L4-node",
        "gpu_type": "L4",
        "gpu_count": 1,
        "gpu_vram_gb": 24.0,
        "cpu_capacity": 4.0,
        "memory_capacity_gb": 32.0,
        "cost_per_step": 12.0,
        "tdp_watts": 72.0,
        "max_temp_c": 82.0,
    },
]

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = {
    "gpu_cpu_allocation": {
        "description": (
            "Manage a cluster of GPU+CPU nodes to maximise compute throughput "
            "while minimising cost.  Each node has GPU (VRAM, compute) and CPU "
            "resources.  Incoming workloads vary in demand.  Choose how to "
            "allocate resources across nodes and GPU types."
        ),
        "difficulty": "easy",
        "num_nodes": 3,
        "max_steps": 8,
        "target_gpu_util_pct": 0.70,
        "target_cpu_util_pct": 0.70,
        "budget_per_step": 120.0,
        "valid_actions": ["allocate_high", "allocate_low", "maintain", "migrate"],
    },
    "thermal_management": {
        "description": (
            "Monitor GPU and ambient temperatures across the cluster.  "
            "If any GPU exceeds its thermal threshold, redistribute load "
            "to cooler nodes or increase cooling.  Balance thermal safety "
            "with performance and energy cost."
        ),
        "difficulty": "medium",
        "num_nodes": 4,
        "max_steps": 10,
        "target_gpu_util_pct": 0.70,
        "target_cpu_util_pct": 0.70,
        "budget_per_step": None,
        "temp_safe_low": 55.0,
        "temp_safe_high": 75.0,
        "valid_actions": ["increase_cooling", "decrease_cooling", "migrate_load", "maintain"],
    },
    "heuristic_fragmentation": {
        "description": (
            "Allocate GPU resources in a fragmented cluster using heuristic "
            "strategies.  Nodes have 8 GPU slots each; workloads need "
            "contiguous blocks of varying sizes (1,2,4,8).  Choose "
            "placement and defragmentation strategies to minimise waste."
        ),
        "difficulty": "hard",
        "num_nodes": 5,
        "max_steps": 12,
        "target_gpu_util_pct": 0.80,
        "target_cpu_util_pct": 0.70,
        "budget_per_step": 200.0,
        "slots_per_node": 8,
        "valid_actions": ["best_fit", "first_fit", "compact", "split_workload"],
    },
}


# ---------------------------------------------------------------------------
# Trace generators
# ---------------------------------------------------------------------------
def _generate_workload_trace(num_steps: int, base: float, rng: random.Random) -> list[float]:
    """Generate a realistic workload utilisation trace in [0.08, 0.95]."""
    trend = rng.uniform(-0.03, 0.03)
    trace: list[float] = []
    for t in range(num_steps):
        val = base + trend * t + rng.gauss(0, 0.06)
        if rng.random() < 0.12:
            val += rng.uniform(0.15, 0.35)
        trace.append(max(0.08, min(0.95, val)))
    return trace


def _generate_ambient_trace(num_steps: int, rng: random.Random) -> list[float]:
    """Simulate ambient temperature with day/night cycle + heat spikes."""
    trace: list[float] = []
    for t in range(num_steps):
        # Day-night sinusoidal: 22-34 °C
        day_night = 28.0 + 6.0 * math.sin(2.0 * math.pi * t / max(num_steps, 1))
        noise = rng.gauss(0, 1.5)
        spike = rng.uniform(5.0, 12.0) if rng.random() < 0.10 else 0.0
        trace.append(round(max(18.0, min(45.0, day_night + noise + spike)), 1))
    return trace


def _generate_pending_workloads(num_steps: int, rng: random.Random) -> list[list[int]]:
    """Generate pending workload queues (GPU slot requirements) per step."""
    workloads_per_step: list[list[int]] = []
    for _ in range(num_steps):
        count = rng.randint(1, 4)
        sizes = [rng.choice([1, 1, 2, 2, 4, 8]) for _ in range(count)]
        workloads_per_step.append(sizes)
    return workloads_per_step


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------
class CloudResourceEnvironment(MCPEnvironment):
    """Cloud GPU+CPU resource management environment with MCP tools."""

    def __init__(self):
        mcp = FastMCP("cloud_resource_env")

        # ---- MCP Tools ----
        @mcp.tool
        def get_cluster_state() -> dict:
            """
            Get the current state of all GPU+CPU nodes in the cluster.
            Returns GPU/CPU usage, capacity, temperature, fragmentation, and cost.
            """
            return self._build_cluster_state()

        @mcp.tool
        def get_task_info() -> dict:
            """
            Get information about the current task including objectives,
            constraints, and valid actions.
            """
            return self._build_task_info()

        @mcp.tool
        def take_action(decisions: str) -> dict:
            """
            Apply resource management decisions and advance by one timestep.

            Args:
                decisions: JSON string mapping node_id to action.
                    Task 1 (gpu_cpu_allocation):
                        {"node_0": "allocate_high", "node_1": "maintain"}
                        Valid: allocate_high, allocate_low, maintain, migrate
                    Task 2 (thermal_management):
                        {"node_0": "increase_cooling", "node_1": "migrate_load"}
                        Valid: increase_cooling, decrease_cooling, migrate_load, maintain
                    Task 3 (heuristic_fragmentation):
                        {"node_0": "best_fit", "node_1": "compact"}
                        Valid: best_fit, first_fit, compact, split_workload

            Returns:
                Dictionary with reward, done, feedback, updated cluster_state, and score.
            """
            return self._process_action(decisions)

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = "gpu_cpu_allocation"
        self._task_cfg: dict = TASKS[self._task_name]
        self._nodes: list[dict] = []

        # Workload traces
        self._gpu_workloads: dict[str, list[float]] = {}
        self._cpu_workloads: dict[str, list[float]] = {}
        self._mem_workloads: dict[str, list[float]] = {}

        # Thermal traces
        self._ambient_trace: list[float] = []
        self._cooling_levels: dict[str, int] = {}  # 0-3

        # Fragmentation state
        self._slot_maps: dict[str, list[int]] = {}  # 0=free, workload_id otherwise
        self._pending_workloads: list[list[int]] = []
        self._next_workload_id: int = 1

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
        task = kwargs.get("task", "gpu_cpu_allocation")
        if task not in TASKS:
            task = "gpu_cpu_allocation"

        self._task_name = task
        self._task_cfg = TASKS[task]
        self._rng = random.Random(seed if seed is not None else 42)
        self._timestep = 0
        self._step_rewards = []
        self._episode_done = False
        self._next_workload_id = 1

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        num = self._task_cfg["num_nodes"]
        max_steps = self._task_cfg["max_steps"]

        # --- Initialise nodes ---
        self._nodes = []
        for i in range(num):
            tmpl = GPU_NODE_TEMPLATES[i % len(GPU_NODE_TEMPLATES)]
            self._nodes.append({
                "node_id": f"node_{i}",
                "gpu_type": tmpl["gpu_type"],
                "gpu_count": tmpl["gpu_count"],
                "gpu_vram_gb": tmpl["gpu_vram_gb"],
                "cpu_capacity": tmpl["cpu_capacity"],
                "memory_capacity_gb": tmpl["memory_capacity_gb"],
                "cost_per_step": tmpl["cost_per_step"],
                "tdp_watts": tmpl["tdp_watts"],
                "max_temp_c": tmpl["max_temp_c"],
                "name": tmpl["name"],
            })

        # --- Workload traces ---
        self._gpu_workloads = {}
        self._cpu_workloads = {}
        self._mem_workloads = {}
        for node in self._nodes:
            nid = node["node_id"]
            self._gpu_workloads[nid] = _generate_workload_trace(
                max_steps + 1, self._rng.uniform(0.35, 0.80), self._rng
            )
            self._cpu_workloads[nid] = _generate_workload_trace(
                max_steps + 1, self._rng.uniform(0.30, 0.75), self._rng
            )
            self._mem_workloads[nid] = _generate_workload_trace(
                max_steps + 1, self._rng.uniform(0.25, 0.70), self._rng
            )

        # --- Thermal traces ---
        self._ambient_trace = _generate_ambient_trace(max_steps + 1, self._rng)
        self._cooling_levels = {n["node_id"]: 1 for n in self._nodes}

        # --- Fragmentation state ---
        slots = self._task_cfg.get("slots_per_node", 8)
        self._slot_maps = {n["node_id"]: [0] * slots for n in self._nodes}
        self._pending_workloads = _generate_pending_workloads(max_steps + 1, self._rng)

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task": self._task_name,
                "difficulty": self._task_cfg["difficulty"],
                "message": f"Cloud GPU+CPU environment ready. Task: {self._task_name}",
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
    # Metrics helpers
    # ------------------------------------------------------------------
    def _gpu_temp_for_node(self, node: dict) -> float:
        """Compute GPU temperature from utilisation, ambient temp, and cooling."""
        nid = node["node_id"]
        t = min(self._timestep, len(self._gpu_workloads.get(nid, [0.5])) - 1)
        util = self._gpu_workloads[nid][t]
        ambient = self._ambient_trace[min(t, len(self._ambient_trace) - 1)]
        cooling = self._cooling_levels.get(nid, 1)

        # Base temp from utilisation: idle ~35°C, full load ~TDP-mapped
        base_temp = 35.0 + util * 50.0  # 35-85°C range at full util
        # Ambient contribution
        ambient_factor = (ambient - 25.0) * 0.3  # deviation from 25°C baseline
        # Cooling reduction: each level reduces ~5°C
        cooling_reduction = cooling * 5.0
        # Random jitter
        jitter = self._rng.gauss(0, 1.0)

        temp = base_temp + ambient_factor - cooling_reduction + jitter
        return round(max(30.0, min(100.0, temp)), 1)

    def _current_node_metrics(self, node: dict) -> dict:
        nid = node["node_id"]
        t = min(self._timestep, len(self._gpu_workloads.get(nid, [0.5])) - 1)

        gpu_util = self._gpu_workloads[nid][t]
        cpu_util = self._cpu_workloads[nid][t]
        mem_util = self._mem_workloads[nid][t]

        gpu_vram_used = round(gpu_util * node["gpu_vram_gb"], 2)
        cpu_usage = round(cpu_util * node["cpu_capacity"], 2)
        mem_usage = round(mem_util * node["memory_capacity_gb"], 2)
        power_draw = round(node["tdp_watts"] * (0.3 + 0.7 * gpu_util), 1)

        gpu_temp = self._gpu_temp_for_node(node)
        ambient = self._ambient_trace[min(t, len(self._ambient_trace) - 1)]
        cooling = self._cooling_levels.get(nid, 1)
        thermal_throttle = gpu_temp > node["max_temp_c"]

        # Fragmentation info
        slots = self._slot_maps.get(nid, [])
        total_slots = len(slots)
        free_slots = slots.count(0)
        frag_score = self._fragmentation_score(nid) if total_slots > 0 else 0.0

        metrics = {
            "node_id": nid,
            "gpu_type": node["gpu_type"],
            "node_name": node["name"],
            # GPU
            "gpu_count": node["gpu_count"],
            "gpu_utilization_pct": round(gpu_util * 100, 1),
            "gpu_vram_used_gb": gpu_vram_used,
            "gpu_vram_capacity_gb": node["gpu_vram_gb"],
            # CPU
            "cpu_usage": cpu_usage,
            "cpu_capacity": node["cpu_capacity"],
            "cpu_utilization_pct": round(cpu_util * 100, 1),
            # Memory
            "memory_usage_gb": mem_usage,
            "memory_capacity_gb": node["memory_capacity_gb"],
            "memory_utilization_pct": round(mem_util * 100, 1),
            # Thermal
            "gpu_temp_celsius": gpu_temp,
            "ambient_temp_celsius": ambient,
            "cooling_level": cooling,
            "max_temp_threshold": node["max_temp_c"],
            "thermal_throttle": thermal_throttle,
            # Power & cost
            "power_draw_watts": power_draw,
            "cost_per_step": round(node["cost_per_step"], 2),
            # Fragmentation
            "gpu_slots_total": total_slots,
            "gpu_slots_free": free_slots,
            "gpu_slots_used": total_slots - free_slots,
            "fragmentation_score": round(frag_score, 3),
        }
        return metrics

    def _fragmentation_score(self, nid: str) -> float:
        """
        Compute fragmentation score for a node.
        0.0 = all free slots are contiguous (ideal)
        1.0 = maximally fragmented
        """
        slots = self._slot_maps.get(nid, [])
        if not slots:
            return 0.0
        free_count = slots.count(0)
        if free_count == 0 or free_count == len(slots):
            return 0.0

        # Count number of contiguous free blocks
        blocks = 0
        in_block = False
        for s in slots:
            if s == 0 and not in_block:
                blocks += 1
                in_block = True
            elif s != 0:
                in_block = False

        if blocks <= 1:
            return 0.0
        # Normalise: 1 block = 0, max blocks = free_count
        return round(min(1.0, (blocks - 1) / max(1, free_count - 1)), 3)

    def _build_cluster_state(self) -> dict:
        nodes = [self._current_node_metrics(n) for n in self._nodes]
        total_cost = sum(n["cost_per_step"] for n in nodes)
        budget = self._task_cfg.get("budget_per_step")

        state: dict[str, Any] = {
            "timestep": self._timestep,
            "max_timesteps": self._task_cfg["max_steps"],
            "task": self._task_name,
            "nodes": nodes,
            "total_cost_per_step": round(total_cost, 2),
            "budget_per_step": budget,
            "budget_remaining": round(budget - total_cost, 2) if budget else None,
        }

        # Task-specific extras
        if self._task_name == "thermal_management":
            t = min(self._timestep, len(self._ambient_trace) - 1)
            state["ambient_temp_celsius"] = self._ambient_trace[t]
            state["any_throttling"] = any(
                self._current_node_metrics(n)["thermal_throttle"] for n in self._nodes
            )

        if self._task_name == "heuristic_fragmentation":
            pw_idx = min(self._timestep, len(self._pending_workloads) - 1)
            state["pending_workloads"] = self._pending_workloads[pw_idx]
            state["cluster_fragmentation"] = round(
                sum(self._fragmentation_score(n["node_id"]) for n in self._nodes) / len(self._nodes), 3
            )

        return state

    def _build_task_info(self) -> dict:
        cfg = self._task_cfg
        objectives = [
            f"Keep GPU utilisation near {cfg['target_gpu_util_pct'] * 100:.0f}%",
            f"Keep CPU utilisation near {cfg['target_cpu_util_pct'] * 100:.0f}%",
            "Avoid GPU overloads (utilisation > 100%)",
        ]

        if self._task_name == "gpu_cpu_allocation":
            objectives.append("Minimise cost while meeting demand")
            objectives.append("Migrate workloads to cheaper GPUs when possible")

        elif self._task_name == "thermal_management":
            objectives.append("Keep GPU temperatures below threshold")
            objectives.append("Redistribute load when GPU overheats")
            objectives.append("Minimise cooling energy cost")

        elif self._task_name == "heuristic_fragmentation":
            objectives.append("Place pending workloads efficiently")
            objectives.append("Minimise fragmentation")
            objectives.append("Use heuristic strategies (best-fit, first-fit)")
            if cfg.get("budget_per_step"):
                objectives.append("Stay within budget constraint")

        return {
            "task_name": self._task_name,
            "difficulty": cfg["difficulty"],
            "description": cfg["description"],
            "num_nodes": cfg["num_nodes"],
            "max_steps": cfg["max_steps"],
            "target_gpu_utilization_pct": cfg["target_gpu_util_pct"] * 100,
            "target_cpu_utilization_pct": cfg["target_cpu_util_pct"] * 100,
            "budget_per_step": cfg.get("budget_per_step"),
            "valid_actions": cfg["valid_actions"],
            "objectives": objectives,
        }

    # ------------------------------------------------------------------
    # Action processing (per-task)
    # ------------------------------------------------------------------
    def _process_action(self, decisions_str: str) -> dict:
        if self._episode_done:
            return {
                "reward": 0.0,
                "done": True,
                "feedback": "Episode already finished.",
                "cluster_state": self._build_cluster_state(),
                "score": self._compute_score(),
            }

        try:
            decisions = json.loads(decisions_str) if isinstance(decisions_str, str) else decisions_str
        except (json.JSONDecodeError, TypeError):
            decisions = {}

        if self._task_name == "gpu_cpu_allocation":
            result = self._process_gpu_cpu_allocation(decisions)
        elif self._task_name == "thermal_management":
            result = self._process_thermal_management(decisions)
        elif self._task_name == "heuristic_fragmentation":
            result = self._process_heuristic_fragmentation(decisions)
        else:
            result = {"feedback_lines": [], "step_reward": 0.0}

        # Advance timestep
        self._timestep += 1

        # Recompute reward after timestep advance (observe new workload)
        step_reward = result["step_reward"]
        self._step_rewards.append(step_reward)

        done = self._timestep >= self._task_cfg["max_steps"]
        if done:
            self._episode_done = True

        score = self._compute_score()

        return {
            "reward": round(step_reward, 4),
            "done": done,
            "feedback": " | ".join(result["feedback_lines"]),
            "cluster_state": self._build_cluster_state(),
            "score": round(score, 4),
            "timestep": self._timestep,
            "max_timesteps": self._task_cfg["max_steps"],
        }

    # --- Task 1: GPU+CPU Allocation with Cost Optimisation ---
    def _process_gpu_cpu_allocation(self, decisions: dict) -> dict:
        feedback: list[str] = []
        valid = self._task_cfg["valid_actions"]

        for node in self._nodes:
            nid = node["node_id"]
            action = decisions.get(nid, "maintain")
            if action not in valid:
                action = "maintain"

            if action == "allocate_high":
                # Scale up GPU+CPU capacity by 50%
                node["gpu_vram_gb"] = round(node["gpu_vram_gb"] * 1.5, 2)
                node["cpu_capacity"] = round(node["cpu_capacity"] * 1.5, 2)
                node["memory_capacity_gb"] = round(node["memory_capacity_gb"] * 1.5, 2)
                node["cost_per_step"] = round(node["cost_per_step"] * 1.5, 2)
                feedback.append(f"{nid}: allocate_high (+50% capacity, +50% cost)")

            elif action == "allocate_low":
                # Scale down by 33%
                node["gpu_vram_gb"] = round(max(8.0, node["gpu_vram_gb"] / 1.5), 2)
                node["cpu_capacity"] = round(max(1.0, node["cpu_capacity"] / 1.5), 2)
                node["memory_capacity_gb"] = round(max(4.0, node["memory_capacity_gb"] / 1.5), 2)
                node["cost_per_step"] = round(max(3.0, node["cost_per_step"] / 1.5), 2)
                feedback.append(f"{nid}: allocate_low (-33% capacity, -33% cost)")

            elif action == "migrate":
                # Reduce this node's workload, slightly increase others
                nid_idx = [n["node_id"] for n in self._nodes].index(nid)
                t = min(self._timestep, len(self._gpu_workloads[nid]) - 1)
                migrated = self._gpu_workloads[nid][t] * 0.3
                self._gpu_workloads[nid][t] *= 0.7
                self._cpu_workloads[nid][t] *= 0.7
                # Spread to other nodes
                others = [n for n in self._nodes if n["node_id"] != nid]
                if others:
                    share = migrated / len(others)
                    for other in others:
                        oid = other["node_id"]
                        ot = min(self._timestep, len(self._gpu_workloads[oid]) - 1)
                        self._gpu_workloads[oid][ot] = min(0.95, self._gpu_workloads[oid][ot] + share)
                        self._cpu_workloads[oid][ot] = min(0.95, self._cpu_workloads[oid][ot] + share * 0.5)
                feedback.append(f"{nid}: migrate (30% load moved to other nodes)")

            else:
                feedback.append(f"{nid}: maintained")

        # Compute reward
        target_gpu = self._task_cfg["target_gpu_util_pct"]
        target_cpu = self._task_cfg["target_cpu_util_pct"]
        reward = 0.0
        for node in self._nodes:
            m = self._current_node_metrics(node)
            gpu_pct = m["gpu_utilization_pct"] / 100.0
            cpu_pct = m["cpu_utilization_pct"] / 100.0

            if gpu_pct > 1.0 or cpu_pct > 1.0:
                reward -= 0.5
                feedback.append(f"⚠️ {node['node_id']} OVERLOADED!")
            else:
                gpu_eff = max(0.0, 1.0 - 2.0 * abs(gpu_pct - target_gpu))
                cpu_eff = max(0.0, 1.0 - 2.0 * abs(cpu_pct - target_cpu))
                reward += (gpu_eff * 0.6 + cpu_eff * 0.4)  # GPU weighted more

        reward /= len(self._nodes)

        # Budget penalty
        budget = self._task_cfg.get("budget_per_step")
        if budget:
            total_cost = sum(n["cost_per_step"] for n in self._nodes)
            if total_cost > budget:
                reward *= 0.5
                feedback.append(f"⚠️ Over budget! Cost {total_cost:.0f} > Budget {budget:.0f}")

        reward = max(0.0, min(1.0, reward))
        return {"feedback_lines": feedback, "step_reward": reward}

    # --- Task 2: Thermal Management ---
    def _process_thermal_management(self, decisions: dict) -> dict:
        feedback: list[str] = []
        valid = self._task_cfg["valid_actions"]
        cooling_energy_cost = 0.0

        for node in self._nodes:
            nid = node["node_id"]
            action = decisions.get(nid, "maintain")
            if action not in valid:
                action = "maintain"

            current_cooling = self._cooling_levels.get(nid, 1)

            if action == "increase_cooling":
                new_cooling = min(3, current_cooling + 1)
                self._cooling_levels[nid] = new_cooling
                cooling_energy_cost += 5.0 * new_cooling
                feedback.append(f"{nid}: cooling ↑ (level {current_cooling}→{new_cooling})")

            elif action == "decrease_cooling":
                new_cooling = max(0, current_cooling - 1)
                self._cooling_levels[nid] = new_cooling
                feedback.append(f"{nid}: cooling ↓ (level {current_cooling}→{new_cooling})")

            elif action == "migrate_load":
                # Move 40% of load to coolest node
                t = min(self._timestep, len(self._gpu_workloads[nid]) - 1)
                migrated = self._gpu_workloads[nid][t] * 0.4
                self._gpu_workloads[nid][t] *= 0.6
                self._cpu_workloads[nid][t] *= 0.6

                # Find coolest other node
                others = [n for n in self._nodes if n["node_id"] != nid]
                if others:
                    coolest = min(others, key=lambda n: self._gpu_temp_for_node(n))
                    cid = coolest["node_id"]
                    ct = min(self._timestep, len(self._gpu_workloads[cid]) - 1)
                    self._gpu_workloads[cid][ct] = min(0.95, self._gpu_workloads[cid][ct] + migrated)
                    self._cpu_workloads[cid][ct] = min(0.95, self._cpu_workloads[cid][ct] + migrated * 0.5)
                    feedback.append(f"{nid}: migrated 40% load → {cid} (coolest)")
                else:
                    feedback.append(f"{nid}: migrate_load (no other nodes)")

            else:
                feedback.append(f"{nid}: maintained")

        # Compute reward
        safe_low = self._task_cfg["temp_safe_low"]
        safe_high = self._task_cfg["temp_safe_high"]
        target_gpu = self._task_cfg["target_gpu_util_pct"]
        reward = 0.0
        any_throttle = False

        for node in self._nodes:
            m = self._current_node_metrics(node)
            temp = m["gpu_temp_celsius"]
            gpu_pct = m["gpu_utilization_pct"] / 100.0

            # Temperature reward
            if temp <= safe_high and temp >= safe_low:
                temp_reward = 1.0  # In safe zone
            elif temp > node["max_temp_c"]:
                temp_reward = -1.0  # Critical — throttling
                any_throttle = True
                feedback.append(f"🔥 {node['node_id']} THERMAL THROTTLE! {temp:.1f}°C > {node['max_temp_c']}°C")
            elif temp > safe_high:
                # Warning zone
                overshoot = (temp - safe_high) / (node["max_temp_c"] - safe_high)
                temp_reward = max(-0.5, 0.5 - overshoot)
            else:
                # Below safe_low — overcooled, wasting energy
                temp_reward = 0.5

            # Utilisation efficiency
            util_eff = max(0.0, 1.0 - 2.0 * abs(gpu_pct - target_gpu))

            # Weighted: 50% thermal, 40% utilisation, 10% cooling cost penalty
            node_reward = temp_reward * 0.5 + util_eff * 0.4
            reward += node_reward

        reward /= len(self._nodes)

        # Cooling energy penalty
        cooling_penalty = cooling_energy_cost / (len(self._nodes) * 15.0)  # normalise
        reward -= cooling_penalty * 0.1

        reward = max(0.0, min(1.0, reward))
        return {"feedback_lines": feedback, "step_reward": reward}

    # --- Task 3: Heuristic Fragmentation GPU Allocation ---
    def _process_heuristic_fragmentation(self, decisions: dict) -> dict:
        feedback: list[str] = []
        valid = self._task_cfg["valid_actions"]
        slots_per = self._task_cfg["slots_per_node"]

        # Get pending workloads for this step
        pw_idx = min(self._timestep, len(self._pending_workloads) - 1)
        pending = list(self._pending_workloads[pw_idx])

        # Randomly free some old slots to create fragmentation
        for node in self._nodes:
            nid = node["node_id"]
            for i in range(len(self._slot_maps[nid])):
                if self._slot_maps[nid][i] != 0 and self._rng.random() < 0.15:
                    self._slot_maps[nid][i] = 0

        # Determine global strategy from decisions (majority vote or per-node)
        strategy_votes: dict[str, int] = {}
        for node in self._nodes:
            nid = node["node_id"]
            action = decisions.get(nid, "best_fit")
            if action not in valid:
                action = "best_fit"
            strategy_votes[action] = strategy_votes.get(action, 0) + 1

        primary_strategy = max(strategy_votes, key=strategy_votes.get)  # type: ignore
        feedback.append(f"Strategy: {primary_strategy}")

        placed = 0
        failed = 0

        for wl_size in pending:
            wl_id = self._next_workload_id
            self._next_workload_id += 1

            if primary_strategy == "compact":
                # First compact (defragment), then best-fit
                self._compact_nodes()
                success = self._place_best_fit(wl_size, wl_id)
            elif primary_strategy == "best_fit":
                success = self._place_best_fit(wl_size, wl_id)
            elif primary_strategy == "first_fit":
                success = self._place_first_fit(wl_size, wl_id)
            elif primary_strategy == "split_workload":
                success = self._place_split(wl_size, wl_id)
            else:
                success = self._place_best_fit(wl_size, wl_id)

            if success:
                placed += 1
            else:
                failed += 1

        feedback.append(f"Placed {placed}/{placed + failed} workloads (sizes: {pending})")
        if failed > 0:
            feedback.append(f"⚠️ {failed} workloads could not be placed!")

        # Compute reward
        # Placement success
        placement_ratio = placed / max(1, placed + failed)

        # Fragmentation reduction
        avg_frag = sum(self._fragmentation_score(n["node_id"]) for n in self._nodes) / len(self._nodes)

        # Utilisation balance
        target_gpu = self._task_cfg["target_gpu_util_pct"]
        util_reward = 0.0
        for node in self._nodes:
            nid = node["node_id"]
            slots = self._slot_maps[nid]
            used = sum(1 for s in slots if s != 0)
            util = used / max(1, len(slots))
            util_reward += max(0.0, 1.0 - 2.0 * abs(util - target_gpu))
        util_reward /= len(self._nodes)

        # Weighted reward
        reward = placement_ratio * 0.4 + (1.0 - avg_frag) * 0.3 + util_reward * 0.3

        # Budget penalty
        budget = self._task_cfg.get("budget_per_step")
        if budget:
            total_cost = sum(n["cost_per_step"] for n in self._nodes)
            if total_cost > budget:
                reward *= 0.5
                feedback.append(f"⚠️ Over budget! Cost {total_cost:.0f} > Budget {budget:.0f}")

        # Compaction overhead penalty
        if primary_strategy == "compact":
            reward *= 0.9  # 10% penalty for migration overhead
            feedback.append("ℹ️ Compact: 10% overhead for defragmentation")

        reward = max(0.0, min(1.0, reward))
        return {"feedback_lines": feedback, "step_reward": reward}

    # --- Fragmentation placement helpers ---
    def _find_contiguous_free(self, nid: str, size: int) -> int:
        """Find start index of contiguous free block of given size. Returns -1 if none."""
        slots = self._slot_maps[nid]
        for i in range(len(slots) - size + 1):
            if all(s == 0 for s in slots[i:i + size]):
                return i
        return -1

    def _place_best_fit(self, size: int, wl_id: int) -> bool:
        """Best-fit: place in node with smallest sufficient contiguous block."""
        best_node = None
        best_start = -1
        best_free = float("inf")

        for node in self._nodes:
            nid = node["node_id"]
            start = self._find_contiguous_free(nid, size)
            if start >= 0:
                free = self._slot_maps[nid].count(0)
                if free < best_free:
                    best_free = free
                    best_node = nid
                    best_start = start

        if best_node is not None and best_start >= 0:
            for i in range(size):
                self._slot_maps[best_node][best_start + i] = wl_id
            return True
        return False

    def _place_first_fit(self, size: int, wl_id: int) -> bool:
        """First-fit: place in first node with sufficient contiguous block."""
        for node in self._nodes:
            nid = node["node_id"]
            start = self._find_contiguous_free(nid, size)
            if start >= 0:
                for i in range(size):
                    self._slot_maps[nid][start + i] = wl_id
                return True
        return False

    def _place_split(self, size: int, wl_id: int) -> bool:
        """Split workload across multiple nodes if no single node has enough."""
        # First try contiguous placement
        if self._place_first_fit(size, wl_id):
            return True

        # Split across nodes
        remaining = size
        for node in self._nodes:
            nid = node["node_id"]
            slots = self._slot_maps[nid]
            for i in range(len(slots)):
                if slots[i] == 0 and remaining > 0:
                    slots[i] = wl_id
                    remaining -= 1
            if remaining == 0:
                return True
        return remaining == 0

    def _compact_nodes(self) -> None:
        """Defragment all nodes by moving allocated slots to front."""
        for node in self._nodes:
            nid = node["node_id"]
            slots = self._slot_maps[nid]
            # Gather non-zero (allocated) entries, push to front
            allocated = [s for s in slots if s != 0]
            free = [0] * (len(slots) - len(allocated))
            self._slot_maps[nid] = allocated + free

    def _compute_score(self) -> float:
        if not self._step_rewards:
            return 0.0
        return max(0.0, min(1.0, sum(self._step_rewards) / len(self._step_rewards)))
