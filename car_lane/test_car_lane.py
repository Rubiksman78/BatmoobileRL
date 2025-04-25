# Samuel Sithakoul script

import pickle
import time

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tqdm
from network import DQN

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)

env = gym.make("highway-fast-v0",render_mode="human")
env.unwrapped.configure(config)
# env.unwrapped.config["duration"] = 60
env.reset()

n_actions = env.action_space.n
state, info = env.reset()
n_observations = state.shape[0] * state.shape[1] * state.shape[2]
dqn = DQN(n_observations, n_actions)
dqn.load_state_dict(torch.load("dqn.pth"))


def select_action(state):
    with torch.no_grad():
        return dqn(state).max(1).indices.view(1, 1)


all_rewards = []
for i in tqdm.tqdm(range(10)):
    state, info = env.reset()
    total_reward = 0
    while True:
        state = torch.from_numpy(state).float()
        state = state.view(-1).unsqueeze(0)
        action = select_action(state)
        env.render()
        state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
        if done:
            break
    all_rewards.append(total_reward)

sns.displot(all_rewards, kde=True, bins=100)
plt.savefig("dqn_test.png")
