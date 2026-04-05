"""
Inference Script – Cloud Resource Management OpenEnv Environment
================================================================
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
    {"name": "single_server_scaling", "max_steps": 1},
    {"name": "multi_server_balancing", "max_steps": 2},
    {"name": "cost_optimized_planning", "max_steps": 3},
]

TEMPERATURE = 0.4
MAX_TOKENS = 300
SUCCESS_THRESHOLD = 0.3

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert cloud infrastructure manager. You observe server metrics and
decide scaling actions to keep CPU utilization near 70% of capacity.

Rules:
- For each server, choose one action: "scale_up", "scale_down", or "maintain".
- scale_up:   increases capacity by 50% (costs more)
- scale_down: decreases capacity by 33% (saves money)
- maintain:   no change
- Avoid overloads: utilization must stay below 100%.
- Target utilization: ~70% CPU.

Respond with ONLY a valid JSON object mapping server_id to action. Example:
{"server_0": "maintain", "server_1": "scale_up"}

No explanation, no markdown, just pure JSON.
""")


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
    return textwrap.dedent(f"""\
Task: {task_info.get('task_name', 'unknown')} ({task_info.get('difficulty', '?')})
Objective: {task_info.get('description', '')}
Target CPU utilization: {task_info.get('target_cpu_utilization_pct', 70)}%
Budget per step: {task_info.get('budget_per_step', 'N/A')}

Step {step_num} of {task_info.get('max_steps', '?')}

Current cluster state:
{state_json}

Recent history:
{hist_block}

Decide scaling actions for each server. Respond with JSON only.
""")


def get_llm_decision(
    client: OpenAI,
    cluster_state: Dict[str, Any],
    task_info: Dict[str, Any],
    step_num: int,
    history: List[str],
    server_ids: List[str],
) -> str:
    user_prompt = build_user_prompt(cluster_state, task_info, step_num, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
        # Ensure all server_ids are present
        for sid in server_ids:
            if sid not in parsed:
                parsed[sid] = "maintain"
        return json.dumps(parsed)
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Fallback: maintain all servers
        fallback = {sid: "maintain" for sid in server_ids}
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

        server_ids = [s["server_id"] for s in cluster_state.get("servers", [])]
        history: List[str] = []

        for step in range(1, max_steps + 1):
            # Get LLM decision
            decisions = get_llm_decision(
                llm_client, cluster_state, task_info, step, history, server_ids
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
