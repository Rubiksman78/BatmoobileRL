import os
import random
import pickle
import copy
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buffer():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.pos = 0
        self.max_size = 25000
        self.buffer = deque(maxlen=self.max_size)

    def push(self, state_a, action, reward, next_state_a, done):
        self.buffer.append([state_a, action, reward, next_state_a, done])

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state_a, action, reward, next_state_a, done = map(np.stack, zip(*batch))
        return state_a, action, reward, next_state_a, done
    
    def __len__(self):
        return len(self.buffer)


class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        obs_dim = np.array(env.observation_space.shape).prod()
        action_dim = np.prod(env.action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(action_dim, 256)
        self.fc3 = nn.Linear(256 + 256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, state, action):
        x1 = F.relu(self.fc1(state))
        x2 = F.relu(self.fc2(action))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Actor(nn.Module):
    def __init__(self, env, state_dim):
        super().__init__()
        obs_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.state_dim = state_dim
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
    
    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        mean = self(state)
        log_std = torch.full_like(mean, 0.0)
        dist = Normal(mean, log_std.exp())
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum()
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma
        self.action_dim = action_dim

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)

class DDPG_agent():
    def __init__(self, env, continueTraining = False):
        self.env = env
        self.total_rewards = []
        self.avg_reward = []
        self.a_loss = []
        self.c_loss = []
        self.gamma = 0.9
        self.lr_c = 2e-5
        self.lr_a = 2e-5
        self.tau = 0.2
        self.device = device
        self.env = env
        self.max_episodes = 1000
        self.max_steps_per_episode = 1500
        self.state_dim = np.prod(env.observation_space.shape)
        self.noise = GaussianNoise(action_dim=np.prod(env.action_space.shape))
        
        self.actor = Actor(self.env, self.state_dim).to(device)
        self.critic = Critic(self.env).to(device)
        
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=0.95)
        
        self.critic_criterion = nn.MSELoss()

        self.save_interval = 20
        self.file_name = "ddpg_checkpoint.pt"
        
        self.avg_lengths = []
        self.all_rewards = []
        
        self.batch_size = 128
        self.replay_buffer = Buffer(self.batch_size)
        self.begin_step = 0
        
        if continueTraining:
            checkpoint = torch.load(self.file_name,weights_only=False)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.avg_lengths = checkpoint['avg_lengths']
            self.all_rewards = checkpoint['all_rewards']
            self.begin_step = checkpoint['step']

    def soft_update(self, net, target_net, tau=0.005):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def update(self, global_step):
        obs, action, reward, next_obs, done = self.replay_buffer.sample()
        obs = torch.FloatTensor(obs).reshape(self.batch_size, -1).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).reshape(self.batch_size, -1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            target_action = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, target_action)
            target_value = reward + self.gamma * (1 - done) * target_q
        current_q = self.critic(obs, action)
        critic_loss = self.critic_criterion(current_q, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action = self.actor(obs)
        actor_loss = -self.critic(obs, actor_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        if self.actor_scheduler:
            self.actor_scheduler.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()
    
    def rollout(self, global_step):
        obs = self.env.reset()
        ep_reward = 0
        ep_step = 0
        done = False
        obs = obs[0]

        for time_episode in range(self.max_steps_per_episode):
            obs_array = obs
            obs_array = obs_array.flatten().astype(np.float32)
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actor(obs_tensor).cpu().numpy()[0]
            action += self.noise.sample()
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, done, info, _ = self.env.step(action)
            self.replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            ep_step += 1
            global_step += 1
            if len(self.replay_buffer) > self.batch_size:
                self.update(global_step)
            if done:
                break

        return ep_reward, ep_step
    
    def learn(self):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(16, 6)
        global_step = 0
        for step in range(self.begin_step, self.max_episodes):
            ep_reward, ep_step = self.rollout(global_step)
            global_step += ep_step
            self.avg_lengths.append(ep_step)
            self.all_rewards.append(ep_reward)
            ax[0].clear()
            ax[1].clear()
            ax[0].plot(np.arange(len(self.avg_lengths)), self.avg_lengths)
            ax[0].set_title("Avg length")
            ax[1].set_title("Avg reward")
            ax[1].plot(np.arange(len(self.all_rewards)), self.all_rewards)
            plt.savefig("ddpg.png")
            if step % self.save_interval == 0:
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_actor': self.target_actor.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'avg_lengths': self.avg_lengths,
                    'all_rewards': self.all_rewards,
                    'step': step,
                }, self.file_name)
                print("saved")

    def test(self):
        checkpoint = torch.load(self.file_name,weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        state, _ = self.env.reset()
        total_reward = 0
        while True:
            state = torch.Tensor(state).to(device).reshape(1, -1)
            action, _, _ = self.actor.select_action(state)
            action = action.detach().cpu().numpy()
            state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            self.env.render()
            if terminated or truncated:
                break
        print(f"Total reward: {total_reward}")


# seed
SEED = 69

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)

trainModel = False
if trainModel:
    env = gym.make("racetrack-v0")
    env.unwrapped.configure(config)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    ddpg = DDPG_agent(env, continueTraining=True)
    ddpg.learn()
else:
    env = gym.make("racetrack-v0", render_mode="human")
    env.unwrapped.configure(config)
    print(env.observation_space.shape)
    print(env.action_space.shape)
    ddpg = DDPG_agent(env, continueTraining=True)
    print("testing")
    ddpg.test()