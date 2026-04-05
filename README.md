# Cloud Resource Management – OpenEnv RL Environment

> **Meta OpenEnv Hackathon** submission – A real-world reinforcement learning
> environment for cloud GPU/CPU resource autoscaling built on the
> [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

## Overview

This environment simulates a cloud infrastructure where an AI agent must make
optimal **scaling decisions** (scale up / scale down / maintain) for virtual
servers to keep CPU utilization near a 70 % target, avoid overloads, and manage
costs – all driven by workload patterns derived from the
[Google Cloud Dataset](https://github.com/google/cluster-data).

### Key Features

| Feature | Detail |
|---|---|
| **Framework** | OpenEnv v0.2.3 (MCP tools + standard `step()`/`reset()`/`state()`) |
| **3 graded tasks** | Easy → Medium → Hard with automatic graders |
| **Real-world data** | Workload patterns from Google Cloud dataset |
| **LLM Agent** | Inference via OpenAI Client (`inference.py`) |
| **Deployment** | Docker + Hugging Face Spaces |

## Tasks

| # | Task Name | Servers | Steps | Constraint | Difficulty |
|---|---|---|---|---|---|
| 1 | `single_server_scaling` | 1 | 5 | – | Easy |
| 2 | `multi_server_balancing` | 3 | 8 | Load balance | Medium |
| 3 | `cost_optimized_planning` | 5 | 12 | Budget ≤ 150/step | Hard |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
cd cloud_resource_env && uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# 3. Run inference (set env vars first)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your-token>"
python inference.py
```

## Docker

```bash
docker build -t cloud-resource-env:latest -f cloud_resource_env/Dockerfile cloud_resource_env/
docker run -p 8000:8000 cloud-resource-env:latest
```

Then run inference:
```bash
export IMAGE_NAME="cloud-resource-env:latest"
python inference.py
```

## Project Structure

```
├── inference.py                  # Root inference script (REQUIRED)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── cloud_resource_env/           # OpenEnv environment package
│   ├── __init__.py
│   ├── models.py                 # CloudAction, CloudObservation
│   ├── client.py                 # CloudResourceClient
│   ├── openenv.yaml              # OpenEnv manifest
│   ├── pyproject.toml            # Package config
│   ├── Dockerfile                # Container image
│   ├── README.md                 # HF Space metadata
│   └── server/
│       ├── __init__.py
│       ├── app.py                # FastAPI application
│       └── cloud_environment.py  # Core environment logic
├── cloud_env.py                  # Gymnasium RL environment (training)
├── train.py                      # PPO training script
├── cloud_rl_model.zip            # Pre-trained RL model
└── Dataset on .../               # Google Cloud dataset (source)
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace / API key |
| `IMAGE_NAME` | No | Docker image name (if using Docker) |
| `ENV_URL` | No | Environment URL (default: `http://localhost:8000`) |

## Reward & Scoring

- **Per-step reward**: efficiency score based on proximity to 70% CPU target
- **Overload penalty**: −0.5 if any server exceeds 100% utilization
- **Budget penalty** (hard task): reward × 0.5 if cost exceeds budget
- **Final score**: mean of all step rewards ∈ [0.0, 1.0]

## License

BSD-3-Clause (same as OpenEnv)
