import os
import pickle

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.not_good = 0
        self.mean_rewards = []

    def _plot_rewards(self):
        plt.plot(np.arange(len(self.mean_rewards)), self.mean_rewards)
        plt.savefig("ppo.png")
        plt.close()

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                self.mean_rewards.append(mean_reward)
                self._plot_rewards()
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    self.not_good = 0
                else:
                    self.not_good += 1
        return True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)


def make_configured_env(config, env_name, seed):
    def _init():
        env = gym.make(env_name)
        env.unwrapped.configure(config)
        env.reset(seed=seed)
        return env

    return _init


env_name = "roundabout-v0"

n_envs = 8
envs = [make_configured_env(config, env_name, seed) for seed in range(n_envs)]

vec_env = SubprocVecEnv(envs, start_method="fork")
vec_env = VecMonitor(vec_env, "ppo_roundabout")
# env = Monitor(env,"ppo_roundabout")
callback = SaveOnBestTrainingRewardCallback(
    check_freq=max(256 // n_envs, 1), log_dir="ppo_roundabout"
)
model = PPO(
    "CnnPolicy",  # "MlpPolicy",
    vec_env,
    verbose=1,
    batch_size=128,
    ent_coef=0.001,
    n_steps=2048,
    clip_range=0.2,
    n_epochs=6,
    gae_lambda=0.95,
    gamma=0.99,
)
# model = DQN("MlpPolicy", vec_env, verbose=1,batch_size=128, tau=1.0)
model.learn(total_timesteps=5e5, log_interval=20, callback=callback)
model.save("ppo_roundabout_save")
