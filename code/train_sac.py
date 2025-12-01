import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

def main():
    env = Monitor(gym.make("Reacher-v5"))

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/sac/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./models/sac/",
        name_prefix="sac_checkpoint"
    )

    model.learn(total_timesteps=300_000, callback=checkpoint_callback)
    model.save("./models/sac_final")

if __name__ == "__main__":
    main()