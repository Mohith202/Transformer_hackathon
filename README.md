---
title: Cloud Resource Management Environment
emoji: ☁️
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

This Space runs the **Cloud Resource Management** OpenEnv environment.

- OpenEnv pinned ref: `0.2.3`
- Hub tag: `openenv`

### Connecting from Code

```python
from cloud_resource_env import CloudResourceClient

env = CloudResourceClient(base_url="https://huggingface.co/spaces/<your-username>/cloud_resource_env")
```

# Cloud Resource Management Environment

A real-world OpenEnv environment that simulates cloud GPU/CPU resource management.
An AI agent observes server metrics (CPU usage, memory usage, capacity) and makes
scaling decisions to optimize resource utilization, avoid overloads, and manage costs.

Built using data patterns from the [Google Open Cloud Dataset](https://github.com/google/cluster-data).

## Environment Overview

| Component | Description |
|---|---|
| **Domain** | Cloud infrastructure autoscaling |
| **State** | Server CPU/memory usage, capacity, cost |
| **Actions** | `scale_up`, `scale_down`, `maintain` per server |
| **Reward** | Efficiency score based on proximity to 70% utilization target |
| **Score** | Normalized cumulative reward ∈ [0.0, 1.0] |

## Tasks (3 difficulty levels)

| Task | Difficulty | Servers | Steps | Special |
|---|---|---|---|---|
| `single_server_scaling` | Easy | 1 | 5 | Basic optimization |
| `multi_server_balancing` | Medium | 3 | 8 | Load balancing |
| `cost_optimized_planning` | Hard | 5 | 12 | Budget constraint |

## MCP Tools

| Tool | Description |
|---|---|
| `get_cluster_state()` | Returns current metrics for all servers |
| `get_task_info()` | Returns task description and objectives |
| `take_action(decisions)` | Applies scaling decisions, advances timestep, returns reward |

## Quick Start (Async)

```python
import asyncio
from cloud_resource_env import CloudResourceClient

async def main():
    client = await CloudResourceClient.from_docker_image("cloud-resource-env:latest")
    async with client:
        await client.reset(task="single_server_scaling")
        state = await client.call_tool("get_cluster_state")
        print(state)

        result = await client.call_tool(
            "take_action",
            decisions='{"server_0": "scale_up"}'
        )
        print(f"Reward: {result['reward']}, Done: {result['done']}")

asyncio.run(main())
```

## Quick Start (Sync)

```python
from cloud_resource_env import CloudResourceClient

with CloudResourceClient(base_url="http://localhost:8000").sync() as env:
    env.reset(task="multi_server_balancing")
    state = env.call_tool("get_cluster_state")
    result = env.call_tool("take_action", decisions='{"server_0": "maintain", "server_1": "scale_up", "server_2": "scale_down"}')
```

## Reward Function

Each step's reward is calculated per server:

1. **Efficiency**: `max(0, 1 - 2 × |cpu_utilization - 0.70|)` → 1.0 when at 70%, 0.0 when ≤20% or ≥100%
2. **Overload penalty**: -0.5 if CPU or memory exceeds capacity
3. **Budget penalty** (hard task): reward × 0.5 if total cost exceeds budget

Final score = mean of all step rewards, clamped to [0.0, 1.0].

## Action Space

For each server, choose one of:

| Action | Effect |
|---|---|
| `scale_up` | Capacity × 1.5, Cost × 1.5 |
| `scale_down` | Capacity ÷ 1.5, Cost ÷ 1.5 |
| `maintain` | No change |

## Observation Space

Each server exposes:
- `cpu_capacity` / `memory_capacity` – current max resources
- `cpu_usage` / `memory_usage` – current consumption
- `cpu_utilization_pct` / `memory_utilization_pct` – percentage
- `cost_per_step` – running cost

## Setup & Installation

```bash
# Install the environment
pip install -e cloud_resource_env/

# Run the server locally
cd cloud_resource_env && uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or build and run Docker
docker build -t cloud-resource-env:latest -f cloud_resource_env/Dockerfile cloud_resource_env/
docker run -p 8000:8000 cloud-resource-env:latest
```

## Project Structure

```
cloud_resource_env/
├── __init__.py            # Package exports
├── models.py              # CloudAction, CloudObservation
├── client.py              # CloudResourceClient (MCPToolClient)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── Dockerfile             # Container image
├── README.md              # This file
└── server/
    ├── __init__.py        # Server exports
    ├── app.py             # FastAPI application
    └── cloud_environment.py  # Core environment logic
```
