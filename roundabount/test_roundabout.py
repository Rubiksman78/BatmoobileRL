import pickle

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)

env_name = "roundabout-v0"
env = gym.make(env_name, render_mode="human")
env.unwrapped.configure(config)

model = PPO.load("ppo_roundabout/best_model.zip", env)
model = DQN.load("dqn_roundabout/best_model.zip", env)

all_rewards = []
for i in range(10):
    total_reward = 0
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += rewards
        done = terminated or truncated
        if done:
            break
    print(total_reward)
    all_rewards.append(total_reward)
sns.histplot(all_rewards, bins=200, kde=True, stat='count', element='bars', linewidth=0)
plt.savefig("test_dqn.png")
plt.close()