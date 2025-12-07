import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, TD3, SAC

def evaluate_model(model, env, episodes=20):
    """Run deterministic evaluation and return list of episodic returns."""
    returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        returns.append(total_reward)

    return returns


def main():
    env = gym.make("Reacher-v5")  # no render for batch evaluation

    models = {
        "PPO": PPO.load("./models/ppo_final"),
        "TD3": TD3.load("./models/td3_final"),
        "SAC": SAC.load("./models/sac_final"),
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        returns = evaluate_model(model, env, episodes=20)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        print(f"{name} -> Mean Return: {mean_ret:.2f}, Std: {std_ret:.2f}")

        for ep, ret in enumerate(returns):
            results.append({
                "algorithm": name,
                "episode": ep + 1,
                "return": ret
            })

    df = pd.DataFrame(results)
    df.to_csv("./plots/eval_results.csv", index=False)

    print("\nSaved evaluation results to plots/eval_results.csv")

    summary = df.groupby("algorithm")["return"].agg(["mean", "std", "min", "max"])
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()