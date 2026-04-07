# pyright: reportMissingImports=false

"""Train a PPO agent on the Cloud GPU+CPU Resource Management Environment.

Supports all 3 tasks:
  - gpu_cpu_allocation
  - thermal_management
  - heuristic_fragmentation

Usage:
    python train.py                              # default task
    python train.py --task thermal_management    # specific task
    python train.py --task all                   # train on all tasks
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from cloud_env import CloudResourceEnv


ALL_TASKS = ["gpu_cpu_allocation", "thermal_management", "heuristic_fragmentation"]


def train_task(task: str, timesteps: int = 2000, project_root: Path = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parent

    print(f"\n{'='*60}")
    print(f"Training on task: {task}")
    print(f"{'='*60}")

    env = CloudResourceEnv(task=task)
    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=timesteps)
    model_path = project_root / f"cloud_rl_{task}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Quick evaluation
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        env.render()
        if terminated or truncated:
            break

    print(f"\nEvaluation — Task: {task} | Steps: {steps} | Total reward: {total_reward:.2f}")
    print(f"  Score: {info.get('score', 0.0):.4f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Cloud GPU+CPU env")
    parser.add_argument(
        "--task",
        default="gpu_cpu_allocation",
        choices=ALL_TASKS + ["all"],
        help="Task to train on (default: gpu_cpu_allocation)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
        help="Total timesteps for training (default: 2000)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.task == "all":
        results = {}
        for task in ALL_TASKS:
            reward = train_task(task, args.timesteps, project_root)
            results[task] = reward
        print(f"\n{'='*60}")
        print("All tasks trained!")
        for task, reward in results.items():
            print(f"  {task}: {reward:.2f}")
    else:
        train_task(args.task, args.timesteps, project_root)


if __name__ == "__main__":
    main()