import os
import pickle

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _plot_rewards(self, rewards):
        plt.plot(np.arange(len(rewards)), rewards)
        plt.savefig("ppo.png")
        plt.close()

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            self._plot_rewards(y)
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 69

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)


def make_configured_env(config):
    def _init():
        env = gym.make("roundabout-v0")
        env.unwrapped.configure(config)
        return env

    return _init


# env_name = "roundabout-v0"
# env = gym.make(env_name)
# env.unwrapped.configure(config)

n_envs = 32
vec_env = make_vec_env(make_configured_env(config), n_envs=n_envs, seed=SEED)
vec_env = VecMonitor(vec_env, "ppo_roundabout")
# env = Monitor(env,"ppo_roundabout/ppo")

callback = SaveOnBestTrainingRewardCallback(
    check_freq=max(100 // n_envs, 1), log_dir="ppo_roundabout"
)
model = PPO("MlpPolicy", vec_env, verbose=1, batch_size=128)
# model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e5, log_interval=20, callback=callback)
model.save("ppo_roundabout_save")
