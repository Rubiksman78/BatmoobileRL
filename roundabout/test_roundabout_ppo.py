# Samuel Sithakoul script

import pickle
import time

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
from stable_baselines3 import DQN, PPO

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)

env_name = "roundabout-v0"
env = gym.make(env_name, render_mode="human")
env.unwrapped.configure(config)
env.reset()

# model = PPO.load("ppo_roundabout/best_model_occupancygrid.zip", env)
model = PPO.load("ppo_roundabout/best_model.zip", env)
all_rewards = []
n_experiments = 100
for i in tqdm.tqdm(range(n_experiments)):
    total_reward = 0
    obs, _ = env.reset()
    next_state = obs
    while True:
        action, _ = model.predict(next_state)
        obs, rewards, terminated, truncated, _ = env.step(action)
        env.render()
        time.sleep(0.5)
        total_reward += rewards
        done = terminated or truncated
        next_state = obs
        if done:
            break
    time.sleep(1)
    all_rewards.append(total_reward)

# Plot distribution
sns.displot(all_rewards, kde=True, bins=100)
plt.savefig("ppo_test.png")
