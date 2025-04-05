import pickle

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)

env_name = "roundabout-v0"
env = gym.make(env_name, render_mode="human")
env.unwrapped.configure(config)

model = PPO.load("ppo_roundabout/best_model.zip", env)

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
