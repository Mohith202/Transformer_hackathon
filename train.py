# pyright: reportMissingImports=false

from pathlib import Path

from stable_baselines3 import PPO

from cloud_env import CloudResourceEnv


def main():
	project_root = Path(__file__).resolve().parent
	data_path = project_root / "Dataset on Resource Allocation and Usage for a Private Cloud"

	env = CloudResourceEnv(data_path)
	model = PPO("MlpPolicy", env, verbose=1)

	model.learn(total_timesteps=1000)
	model.save(project_root / "cloud_rl_model")

	obs, _ = env.reset()
	total_reward = 0.0
	for _ in range(100):
		action, _ = model.predict(obs, deterministic=True)
		obs, reward, terminated, truncated, _ = env.step(int(action))
		total_reward += reward
		if terminated or truncated:
			break

	print(f"Quick evaluation reward: {total_reward:.2f}")


if __name__ == "__main__":
	main()