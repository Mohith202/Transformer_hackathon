"""
Inference Script – Cloud GPU+CPU Resource Management OpenEnv Environment
========================================================================
Mandatory variables (set in environment configuration):
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from cloud_resource_env import CloudResourceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "cloud_resource_env"
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

TASKS = [
    {"name": "gpu_cpu_allocation", "max_steps": 2 },
    {"name": "thermal_management", "max_steps": 2 },
    {"name": "heuristic_fragmentation", "max_steps": 2},
]

TEMPERATURE = 0.4
MAX_TOKENS = 500
SUCCESS_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Task-specific system prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "gpu_cpu_allocation": textwrap.dedent("""\
        You are an expert cloud GPU+CPU infrastructure manager. You observe
        cluster node metrics (GPU utilisation, CPU utilisation, VRAM, memory,
        cost) and decide allocation actions to optimise throughput and cost.

        Rules:
        - For each node, choose one action:
            "allocate_high" — increase GPU+CPU capacity by 50% (costs more)
            "allocate_low"  — decrease capacity by 33% (saves money)
            "maintain"      — no change
            "migrate"       — move 30% of this node's load to other nodes
        - Target GPU utilisation: ~70%.
        - Target CPU utilisation: ~70%.
        - Avoid overloads (utilisation > 100%).
        - Stay within budget if one is specified.

        Respond with ONLY a valid JSON object mapping node_id to action. Example:
        {"node_0": "maintain", "node_1": "allocate_high", "node_2": "migrate"}

        No explanation, no markdown, just pure JSON.
    """),

    "thermal_management": textwrap.dedent("""\
        You are an expert cloud thermal management engineer. You monitor GPU
        temperatures and ambient temperature, and decide cooling and load
        redistribution actions to prevent thermal throttling.

        Rules:
        - For each node, choose one action:
            "increase_cooling" — increase cooling level (costs energy)
            "decrease_cooling" — decrease cooling level (saves energy)
            "migrate_load"     — move 40% of load to the coolest node
            "maintain"         — no change
        - Safe temperature zone: 55°C – 75°C.
        - CRITICAL: If GPU temperature exceeds max threshold → thermal throttle!
        - Balance: keep temps safe WITHOUT excessive cooling cost.
        - When ambient temperature is high, proactively increase cooling.

        Respond with ONLY a valid JSON object mapping node_id to action. Example:
        {"node_0": "increase_cooling", "node_1": "maintain", "node_2": "migrate_load", "node_3": "maintain"}

        No explanation, no markdown, just pure JSON.
    """),

    "heuristic_fragmentation": textwrap.dedent("""\
        You are an expert GPU cluster scheduler. You manage a fragmented GPU
        cluster where nodes have 8 GPU slots each and workloads need contiguous
        blocks. Choose allocation and defragmentation strategies.

        Rules:
        - For each node, choose one strategy:
            "best_fit"       — place workload in node with smallest sufficient free block
            "first_fit"      — place workload in first node with free space
            "compact"        — defragment first (move allocated to front), then best-fit
            "split_workload" — if no contiguous block, split across nodes
        - All nodes use the majority-vote strategy for this step.
        - The pending workloads have varying GPU requirements (1, 2, 4, or 8 slots).
        - Goal: place all pending workloads, minimise fragmentation.
        - Compaction has a 10% overhead penalty.

        Respond with ONLY a valid JSON object mapping node_id to strategy. Example:
        {"node_0": "best_fit", "node_1": "best_fit", "node_2": "compact", "node_3": "first_fit", "node_4": "best_fit"}

        No explanation, no markdown, just pure JSON.
    """),
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def build_user_prompt(
    cluster_state: Dict[str, Any],
    task_info: Dict[str, Any],
    step_num: int,
    history: List[str],
) -> str:
    state_json = json.dumps(cluster_state, indent=2)
    hist_block = "\n".join(history[-4:]) if history else "None"

    # Build task-specific context
    extra_context = ""
    task_name = task_info.get("task_name", "")

    if task_name == "thermal_management":
        nodes = cluster_state.get("nodes", [])
        hot_nodes = [n for n in nodes if n.get("gpu_temp_celsius", 0) > 75]
        if hot_nodes:
            hot_list = ", ".join(
                f"{n['node_id']}={n['gpu_temp_celsius']:.1f}°C" for n in hot_nodes
            )
            extra_context += f"\n⚠️ HOT NODES: {hot_list}\n"
        ambient = cluster_state.get("ambient_temp_celsius", 25)
        extra_context += f"Ambient temperature: {ambient:.1f}°C\n"

    elif task_name == "heuristic_fragmentation":
        pending = cluster_state.get("pending_workloads", [])
        frag = cluster_state.get("cluster_fragmentation", 0)
        extra_context += f"\nPending workloads (GPU slots needed): {pending}\n"
        extra_context += f"Cluster fragmentation: {frag:.3f}\n"

    return textwrap.dedent(f"""\
Task: {task_info.get('task_name', 'unknown')} ({task_info.get('difficulty', '?')})
Objective: {task_info.get('description', '')}
Valid actions: {task_info.get('valid_actions', [])}
Target GPU utilisation: {task_info.get('target_gpu_utilization_pct', 70)}%
Target CPU utilisation: {task_info.get('target_cpu_utilization_pct', 70)}%
Budget per step: {task_info.get('budget_per_step', 'N/A')}

Step {step_num} of {task_info.get('max_steps', '?')}
{extra_context}
Current cluster state:
{state_json}

Recent history:
{hist_block}

Decide actions for each node. Respond with JSON only.
""")


def get_llm_decision(
    client: OpenAI,
    cluster_state: Dict[str, Any],
    task_info: Dict[str, Any],
    step_num: int,
    history: List[str],
    node_ids: List[str],
) -> str:
    task_name = task_info.get("task_name", "gpu_cpu_allocation")
    system_prompt = SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["gpu_cpu_allocation"])
    user_prompt = build_user_prompt(cluster_state, task_info, step_num, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Validate JSON
        parsed = json.loads(text)
        # Ensure all node_ids are present
        valid_actions = task_info.get("valid_actions", ["maintain"])
        default_action = valid_actions[0] if valid_actions else "maintain"
        for nid in node_ids:
            if nid not in parsed:
                parsed[nid] = default_action
        return json.dumps(parsed)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        valid_actions = task_info.get("valid_actions", ["maintain"])
        default_action = valid_actions[0] if valid_actions else "maintain"
        fallback = {nid: default_action for nid in node_ids}
        return json.dumps(fallback)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def run_task(
    llm_client: OpenAI,
    env: Any,
    task_name: str,
    max_steps: int,
) -> None:
    """Run a single task episode and emit structured logs."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment with task selection
        await env.reset(task=task_name)

        # Get initial state and task info
        cluster_state = await env.call_tool("get_cluster_state")
        task_info = await env.call_tool("get_task_info")

        # Parse cluster_state if it's a string
        if isinstance(cluster_state, str):
            cluster_state = json.loads(cluster_state)
        if isinstance(task_info, str):
            task_info = json.loads(task_info)

        node_ids = [n["node_id"] for n in cluster_state.get("nodes", [])]
        history: List[str] = []

        for step in range(1, max_steps + 1):
            # Get LLM decision
            decisions = get_llm_decision(
                llm_client, cluster_state, task_info, step, history, node_ids
            )

            # Take action
            result = await env.call_tool("take_action", decisions=decisions)
            if isinstance(result, str):
                result = json.loads(result)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            error_msg = result.get("error")

            rewards.append(reward)
            steps_taken = step
            score = float(result.get("score", 0.0))

            log_step(step=step, action=decisions, reward=reward, done=done, error=error_msg)

            history.append(
                f"Step {step}: {decisions} -> reward {reward:+.2f}"
            )

            if done:
                break

            # Update cluster state for next iteration
            cluster_state = result.get("cluster_state", cluster_state)
            if isinstance(cluster_state, str):
                cluster_state = json.loads(cluster_state)

        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Connect to environment (Docker or URL)
    if LOCAL_IMAGE_NAME:
        env = await CloudResourceClient.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = CloudResourceClient(base_url=ENV_URL)

    try:
        async with env:
            for task_cfg in TASKS:
                await run_task(
                    llm_client=llm_client,
                    env=env,
                    task_name=task_cfg["name"],
                    max_steps=task_cfg["max_steps"],
                )
    except Exception as e:
        print(f"[DEBUG] env error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
