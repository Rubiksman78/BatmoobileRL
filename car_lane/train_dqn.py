import math
import pickle
import random
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from network import DQN

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)
env = gym.make("highway-fast-v0")
env.unwrapped.configure(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_sate", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


total_steps = 2e5
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 0.1 * total_steps
TAU = 0.005
LR = 2e-4
START_LEARNING = 200
TARGET_UPDATE_INTERVAL = 50

n_actions = env.action_space.n
state, info = env.reset()
n_observations = state.shape[0] * state.shape[1] * state.shape[2]

print("Number of observations", n_observations)
print("Number of actions", n_actions)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(15000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(rewards)
    plt.savefig("dqn.png")


def optimize_model():
    if len(memory) < START_LEARNING:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_sate)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_sate if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


rewards = []
n_steps = 0
while n_steps < total_steps:
    state, info = env.reset()
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # state has shape (7,8,8), flatten the last three dimensions and unsqueeze first one
    state = (
        torch.tensor(state, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
    )
    avg_reward = 0
    count_steps = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        observation = torch.tensor(
            observation, dtype=torch.float32, device=device
        ).view(-1)
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = observation.unsqueeze(0)
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()
        avg_reward += reward.item()
        count_steps += 1
        if total_steps % TARGET_UPDATE_INTERVAL == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
        if done:
            break
    n_steps += count_steps
    rewards.append(avg_reward / count_steps)
    plot_rewards(rewards)
    torch.save(policy_net.state_dict(), "dqn2.pth")
