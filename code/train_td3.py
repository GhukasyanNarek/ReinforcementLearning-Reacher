import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

def main():
    env = Monitor(gym.make("Reacher-v5"))

    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="../logs/td3/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="../models/checkpoints/td3/",
        name_prefix="td3_checkpoint"
    )

    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    model.save("../models/td3_final.zip")

if __name__ == "__main__":
    main()