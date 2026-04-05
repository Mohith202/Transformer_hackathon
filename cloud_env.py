"""Cloud resource reinforcement-learning environment."""

# pyright: reportMissingImports=false

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency for local inspection
    gym = object
    spaces = None


class CloudResourceEnv(gym.Env if spaces is not None else object):
    metadata = {"render_modes": ["human"]}
    _NA_VALUES = {"", "na", "n/a", "nan", "null", "none", "-"}

    def __init__(self, data_path):
        super().__init__()

        self.data_path = Path(data_path)
        self.rows = self._load_rows()
        if not self.rows:
            raise ValueError(
                f"No usable rows found in dataset at {self.data_path}. "
                "Expected servers_usage.csv, servers_specs.csv, and flavors.csv with valid records."
            )

        if spaces is not None:
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=0.0,
                high=np.finfo(np.float32).max,
                shape=(4,),
                dtype=np.float32,
            )

        self.current_step = 0
        self.max_steps = len(self.rows) - 1

    def _read_csv(self, filename):
        with (self.data_path / filename).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @classmethod
    def _to_float(cls, value, default=0.0):
        if value is None:
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = str(value).strip().lower()
        if cleaned in cls._NA_VALUES:
            return float(default)
        try:
            return float(cleaned)
        except ValueError:
            return float(default)

    @classmethod
    def _to_int(cls, value, default=0):
        return int(cls._to_float(value, default=default))

    def _load_rows(self):
        usage_rows = self._read_csv("servers_usage.csv")
        spec_rows = self._read_csv("servers_specs.csv")
        flavor_rows = self._read_csv("flavors.csv")

        flavor_caps = {}
        for row in flavor_rows:
            flavor_id = row.get("flavor_id")
            if not flavor_id:
                continue
            flavor_caps[flavor_id] = {
                "cpu_capacity": self._to_float(row.get("vcpu"), default=0.0),
                "memory_capacity": self._to_float(row.get("ram"), default=0.0),
            }

        latest_flavor_by_server = {}
        for row in spec_rows:
            server_id = row.get("server_id")
            if not server_id:
                continue
            timestamp = self._to_int(row.get("timestamp"), default=0)
            previous = latest_flavor_by_server.get(server_id)
            if previous is None or timestamp >= previous[0]:
                latest_flavor_by_server[server_id] = (timestamp, row.get("flavor_id"))

        rows = []
        for row in usage_rows:
            server_id = row.get("server_id")
            if not server_id:
                continue

            flavor_id = latest_flavor_by_server.get(server_id, (0, None))[1]
            caps = flavor_caps.get(flavor_id or "", {"cpu_capacity": 0.0, "memory_capacity": 0.0})

            rows.append(
                {
                    "timestamp": self._to_int(row.get("timestamp"), default=0),
                    "server_id": server_id,
                    "cpu_usage": self._to_float(row.get("vcpu_usage"), default=0.0),
                    "memory_usage": self._to_float(row.get("ram_usage"), default=0.0),
                    "cpu_capacity": float(caps["cpu_capacity"]),
                    "memory_capacity": float(caps["memory_capacity"]),
                }
            )

        rows.sort(key=lambda item: item["timestamp"])
        return rows

    # 🔁 Reset environment
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        return self._get_obs(), {}

    # 🎮 Step function
    def step(self, action):
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}. Allowed actions are 0, 1, 2.")

        row = self.rows[self.current_step]

        cpu_usage = float(row["cpu_usage"])
        mem_usage = float(row["memory_usage"])

        cpu_cap = float(row["cpu_capacity"])
        mem_cap = float(row["memory_capacity"])

        # --- Apply action ---
        # 0 = do nothing
        # 1 = scale up (reduce usage artificially)
        # 2 = scale down (increase usage artificially)

        if action == 1:
            cpu_usage *= 0.9
            mem_usage *= 0.9
        elif action == 2:
            cpu_usage *= 1.1
            mem_usage *= 1.1

        # Reward is computed before clipping so overloads are penalized.
        overload_penalty = 0.0
        if cpu_usage > cpu_cap or mem_usage > mem_cap:
            overload_penalty = -10.0

        efficiency = -abs(cpu_usage - 0.7 * cpu_cap)
        reward = float(efficiency + overload_penalty)

        # Clip values for the next observation.
        cpu_usage = np.clip(cpu_usage, 0, cpu_cap if cpu_cap > 0 else cpu_usage)
        mem_usage = np.clip(mem_usage, 0, mem_cap if mem_cap > 0 else mem_usage)

        # --- Next step ---
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        next_obs = self._get_obs() if not terminated else np.array([
            cpu_usage,
            mem_usage,
            cpu_cap,
            mem_cap,
        ], dtype=np.float32)

        info = {
            "server_id": row["server_id"],
            "timestamp": row["timestamp"],
            "adjusted_cpu_usage": float(cpu_usage),
            "adjusted_memory_usage": float(mem_usage),
            "overloaded": bool(overload_penalty < 0),
        }

        return next_obs, reward, terminated, False, info

    # 👁️ Observation
    def _get_obs(self):
        row = self.rows[self.current_step]

        return np.array([
            row["cpu_usage"],
            row["memory_usage"],
            row["cpu_capacity"],
            row["memory_capacity"]
        ], dtype=np.float32)

    def render(self):
        if not self.rows:
            print("Step: 0 (no data)")
            return
        row = self.rows[self.current_step]
        print(
            f"Step: {self.current_step} | server={row['server_id']} | "
            f"cpu={row['cpu_usage']:.2f}/{row['cpu_capacity']:.2f} | "
            f"mem={row['memory_usage']:.2f}/{row['memory_capacity']:.2f}"
        )