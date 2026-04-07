"""Cloud resource reinforcement-learning environment (Gymnasium wrapper).

Wraps the GPU+CPU cloud management environment for local RL training with
stable-baselines3 and similar libraries.

Observation space (12 features per node, flattened):
    [gpu_util, cpu_util, mem_util, gpu_vram_used, gpu_vram_cap,
     cpu_usage, cpu_cap, gpu_temp, ambient_temp, cooling_level,
     fragmentation_score, cost_per_step]

Action space: Discrete(4) — task-specific mapping
    Task 1 (gpu_cpu_allocation):   0=maintain, 1=allocate_high, 2=allocate_low, 3=migrate
    Task 2 (thermal_management):   0=maintain, 1=increase_cooling, 2=decrease_cooling, 3=migrate_load
    Task 3 (heuristic_fragmentation): 0=best_fit, 1=first_fit, 2=compact, 3=split_workload
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency for local inspection
    gym = object
    spaces = None


# Number of features per node in observation
_FEATURES_PER_NODE = 12
_MAX_NODES = 5  # maximum nodes across all tasks


class CloudResourceEnv(gym.Env if spaces is not None else object):
    """Gymnasium wrapper for the Cloud GPU+CPU management environment."""

    metadata = {"render_modes": ["human"]}

    # Action mappings per task
    ACTION_MAP = {
        "gpu_cpu_allocation": {0: "maintain", 1: "allocate_high", 2: "allocate_low", 3: "migrate"},
        "thermal_management": {0: "maintain", 1: "increase_cooling", 2: "decrease_cooling", 3: "migrate_load"},
        "heuristic_fragmentation": {0: "best_fit", 1: "first_fit", 2: "compact", 3: "split_workload"},
    }

    TASK_CONFIGS = {
        "gpu_cpu_allocation": {"num_nodes": 3, "max_steps": 8},
        "thermal_management": {"num_nodes": 4, "max_steps": 10},
        "heuristic_fragmentation": {"num_nodes": 5, "max_steps": 12},
    }

    def __init__(self, task: str = "gpu_cpu_allocation", seed: int = 42):
        super().__init__()

        if task not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task}. Valid: {list(self.TASK_CONFIGS.keys())}")

        self.task = task
        self._seed = seed

        # We import and use the server environment directly for local training
        from server.cloud_environment import CloudResourceEnvironment
        self._env = CloudResourceEnvironment()

        cfg = self.TASK_CONFIGS[task]
        self.num_nodes = cfg["num_nodes"]
        self.max_steps = cfg["max_steps"]

        if spaces is not None:
            self.action_space = spaces.Discrete(4)
            # Observation: flattened node metrics (12 features × max_nodes, padded)
            obs_size = _FEATURES_PER_NODE * _MAX_NODES
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(obs_size,),
                dtype=np.float32,
            )

        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed

        obs_result = self._env.reset(seed=self._seed, task=self.task)
        self.current_step = 0
        return self._obs_from_env(), {}

    def step(self, action):
        if action not in self.ACTION_MAP[self.task]:
            raise ValueError(f"Invalid action {action}. Allowed: 0-3")

        action_name = self.ACTION_MAP[self.task][action]

        # Apply same action to all nodes
        decisions = {f"node_{i}": action_name for i in range(self.num_nodes)}
        decisions_str = json.dumps(decisions)

        # Use the internal process
        result = self._env._process_action(decisions_str)

        # Advance internal timestep
        self._env._timestep += 1
        self.current_step += 1

        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("done", False))
        truncated = False

        obs = self._obs_from_env()

        info = {
            "task": self.task,
            "timestep": self.current_step,
            "feedback": result.get("feedback", ""),
            "score": result.get("score", 0.0),
        }

        return obs, reward, terminated, truncated, info

    def _obs_from_env(self) -> np.ndarray:
        """Extract flattened observation from environment state."""
        state = self._env._build_cluster_state()
        obs = np.zeros(_FEATURES_PER_NODE * _MAX_NODES, dtype=np.float32)

        for i, node in enumerate(state.get("nodes", [])):
            if i >= _MAX_NODES:
                break
            offset = i * _FEATURES_PER_NODE
            obs[offset + 0] = node.get("gpu_utilization_pct", 0.0) / 100.0
            obs[offset + 1] = node.get("cpu_utilization_pct", 0.0) / 100.0
            obs[offset + 2] = node.get("memory_utilization_pct", 0.0) / 100.0
            obs[offset + 3] = node.get("gpu_vram_used_gb", 0.0)
            obs[offset + 4] = node.get("gpu_vram_capacity_gb", 0.0)
            obs[offset + 5] = node.get("cpu_usage", 0.0)
            obs[offset + 6] = node.get("cpu_capacity", 0.0)
            obs[offset + 7] = node.get("gpu_temp_celsius", 0.0) / 100.0  # normalise
            obs[offset + 8] = node.get("ambient_temp_celsius", 25.0) / 50.0  # normalise
            obs[offset + 9] = float(node.get("cooling_level", 0)) / 3.0
            obs[offset + 10] = node.get("fragmentation_score", 0.0)
            obs[offset + 11] = node.get("cost_per_step", 0.0) / 100.0  # normalise

        return obs

    def render(self):
        state = self._env._build_cluster_state()
        print(f"=== Step {self.current_step} | Task: {self.task} ===")
        for node in state.get("nodes", []):
            print(
                f"  {node['node_id']} ({node['gpu_type']}) | "
                f"GPU={node['gpu_utilization_pct']:.1f}% "
                f"CPU={node['cpu_utilization_pct']:.1f}% "
                f"Temp={node['gpu_temp_celsius']:.1f}°C "
                f"Frag={node['fragmentation_score']:.2f} "
                f"Cost=${node['cost_per_step']:.1f}"
            )
        if state.get("budget_per_step"):
            print(f"  Budget: ${state['budget_remaining']:.1f} remaining")