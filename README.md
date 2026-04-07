---
title: Cloud GPU+CPU Resource Management Environment
emoji: ☁️🔥
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv-0.2.3
  - openenv
---

## Hugging Face Space Deployment

This Space runs the **Cloud GPU+CPU Resource Management** OpenEnv environment.

- OpenEnv pinned ref: `0.2.3`
- Hub tag: `openenv`
- **Runs on HF Spaces free tier** (2 vCPUs, 16GB RAM — no physical GPU needed)

### Connecting from Code

```python
from cloud_resource_env import CloudResourceClient

env = CloudResourceClient(base_url="https://huggingface.co/spaces/<your-username>/cloud_resource_env")
```

# Cloud GPU+CPU Resource Management Environment

A real-world OpenEnv environment that simulates **cloud GPU and CPU resource
management** with three progressively harder tasks covering allocation,
thermal management, and heuristic fragmentation.

## Environment Overview

| Component | Description |
|---|---|
| **Domain** | Cloud GPU+CPU infrastructure management |
| **State** | GPU/CPU utilisation, VRAM, temperature, fragmentation, cost |
| **Actions** | Task-specific (see below) — 4 actions per task |
| **Reward** | Multi-objective: utilisation efficiency + thermal safety + fragmentation + cost |
| **Score** | Normalised cumulative reward ∈ [0.0, 1.0] |

## Tasks (3 difficulty levels)

| Task | Difficulty | Nodes | Steps | Focus |
|---|---|---|---|---|
| `gpu_cpu_allocation` | Easy | 3 | 8 | GPU+CPU allocation with cost optimisation |
| `thermal_management` | Medium | 4 | 10 | Temperature monitoring, cooling, load migration |
| `heuristic_fragmentation` | Hard | 5 | 12 | Fragmented GPU placement + defragmentation |

### Task 1: GPU+CPU Allocation (`gpu_cpu_allocation`)

Manage a cluster of mixed GPU nodes (T4, A100, H100, V100, L4) with both GPU
and CPU resources. Optimise throughput while staying within budget.

| Action | Effect |
|---|---|
| `allocate_high` | Capacity × 1.5, Cost × 1.5 |
| `allocate_low` | Capacity ÷ 1.5, Cost ÷ 1.5 |
| `maintain` | No change |
| `migrate` | Move 30% load to other nodes |

### Task 2: Thermal Management (`thermal_management`)

Monitor GPU temperatures and ambient temperature. Prevent thermal throttling
by redistributing load or adjusting cooling levels.

| Action | Effect |
|---|---|
| `increase_cooling` | Cooling level +1 (max 3), reduces GPU temp ~5°C |
| `decrease_cooling` | Cooling level -1 (min 0), saves energy |
| `migrate_load` | Move 40% load to coolest node |
| `maintain` | No change |

**Temperature zones:**
- 🟢 Safe: 55°C – 75°C
- 🟡 Warning: 75°C – max threshold
- 🔴 Critical: Above max threshold → thermal throttle!

### Task 3: Heuristic Fragmentation (`heuristic_fragmentation`)

Place workloads in a fragmented GPU cluster. Each node has 8 GPU slots;
workloads need contiguous blocks (1, 2, 4, or 8 slots).

| Action | Effect |
|---|---|
| `best_fit` | Place in node with smallest sufficient free block |
| `first_fit` | Place in first node with enough space |
| `compact` | Defragment first, then best-fit (10% overhead) |
| `split_workload` | Split across nodes if needed |

## MCP Tools

| Tool | Description |
|---|---|
| `get_cluster_state()` | Returns metrics for all GPU+CPU nodes |
| `get_task_info()` | Returns task description, objectives, valid actions |
| `take_action(decisions)` | Applies decisions, advances timestep, returns reward |

## Observation Space (per node)

| Field | Description |
|---|---|
| `gpu_utilization_pct` | GPU compute utilisation (%) |
| `cpu_utilization_pct` | CPU utilisation (%) |
| `gpu_vram_used_gb` / `gpu_vram_capacity_gb` | VRAM usage |
| `cpu_usage` / `cpu_capacity` | CPU cores usage |
| `memory_usage_gb` / `memory_capacity_gb` | RAM usage |
| `gpu_temp_celsius` | Current GPU temperature |
| `ambient_temp_celsius` | Outside/data center temperature |
| `cooling_level` | Cooling intensity (0-3) |
| `thermal_throttle` | Whether GPU is throttling |
| `fragmentation_score` | How fragmented free GPU slots are (0-1) |
| `cost_per_step` | Running cost |
| `power_draw_watts` | Power consumption |

## Quick Start (Async)

```python
import asyncio
from cloud_resource_env import CloudResourceClient

async def main():
    client = await CloudResourceClient.from_docker_image("cloud-resource-env:latest")
    async with client:
        # Task 1: GPU+CPU allocation
        await client.reset(task="gpu_cpu_allocation")
        state = await client.call_tool("get_cluster_state")
        result = await client.call_tool(
            "take_action",
            decisions='{"node_0": "allocate_high", "node_1": "maintain", "node_2": "migrate"}'
        )

        # Task 2: Thermal management
        await client.reset(task="thermal_management")
        state = await client.call_tool("get_cluster_state")
        result = await client.call_tool(
            "take_action",
            decisions='{"node_0": "increase_cooling", "node_1": "migrate_load", "node_2": "maintain", "node_3": "maintain"}'
        )

asyncio.run(main())
```

## Quick Start (Sync)

```python
from cloud_resource_env import CloudResourceClient

with CloudResourceClient(base_url="http://localhost:8000").sync() as env:
    env.reset(task="heuristic_fragmentation")
    state = env.call_tool("get_cluster_state")
    result = env.call_tool(
        "take_action",
        decisions='{"node_0": "best_fit", "node_1": "compact", "node_2": "first_fit", "node_3": "best_fit", "node_4": "split_workload"}'
    )
```

## GPU Node Types

| Node | GPU | VRAM | CPU | RAM | Cost/step | TDP |
|---|---|---|---|---|---|---|
| T4-node | T4 | 16 GB | 4 cores | 16 GB | $8 | 70W |
| A100-node | A100 | 40 GB | 8 cores | 64 GB | $30 | 250W |
| H100-node | H100 | 80 GB | 16 cores | 128 GB | $55 | 350W |
| V100-node | V100 | 32 GB | 8 cores | 32 GB | $18 | 300W |
| L4-node | L4 | 24 GB | 4 cores | 32 GB | $12 | 72W |

## Setup & Installation

```bash
# Install the environment
pip install -e .

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or build and run Docker
docker build -t cloud-resource-env:latest .
docker run -p 8000:8000 cloud-resource-env:latest

# Train with PPO
python train.py --task gpu_cpu_allocation --timesteps 5000
python train.py --task all  # train on all tasks
```

## Project Structure

```
cloud_resource_env/
├── __init__.py            # Package exports
├── models.py              # CloudAction, CloudObservation
├── client.py              # CloudResourceClient (MCPToolClient)
├── cloud_env.py           # Gymnasium wrapper for RL training
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── Dockerfile             # Container image (HF Spaces compatible)
├── inference.py           # LLM inference with task-specific prompts
├── train.py               # PPO training script
├── README.md              # This file
└── server/
    ├── __init__.py        # Server exports
    ├── app.py             # FastAPI application
    └── cloud_environment.py  # Core environment logic (3 tasks)
```
