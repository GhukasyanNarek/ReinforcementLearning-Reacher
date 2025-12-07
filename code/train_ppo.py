import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

def main():
    env = Monitor(gym.make("Reacher-v5"))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="../logs/ppo/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="../models/checkpoints/ppo/",
        name_prefix="ppo_checkpoint"
    )

    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    model.save("../models/ppo_final.zip")

if __name__ == "__main__":
    main()