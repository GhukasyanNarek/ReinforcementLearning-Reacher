import gymnasium as gym
from stable_baselines3 import SAC

def evaluate(model_path="../models/sac_final.zip", episodes=20):
    env = gym.make("Reacher-v5", render_mode="human")
    model = SAC.load(model_path)

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep+1}, Return = {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate()