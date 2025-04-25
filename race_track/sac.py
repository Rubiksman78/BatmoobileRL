# Samuel Sithakoul script

import pickle
import random
import time
from collections import deque, namedtuple

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_nn = nn.Sequential(nn.Linear(self.state_dim, 256), nn.ReLU())
        self.actor_mean = nn.Linear(256, self.action_dim)
        self.actor_log_std = nn.Linear(256, self.action_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - min_action) / 2.0, dtype=torch.float32, device=device
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + min_action) / 2.0, dtype=torch.float32, device=device
            ),
        )
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, state):
        x = self.actor_nn(state)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )
        return mean, log_std

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(device)
        mean, log_std = self(state)
        dist = Normal(mean, log_std.exp())
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum()
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):  # Q Value
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.critic_nn = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.critic_nn(x)
        return x


class SAC:
    def __init__(self, env: gym.Env):
        self.env = env
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.min_action = float(self.env.action_space.low[0])
        self.policy_lr = 3e-4
        self.critic_lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.buffer_size = int(1e6)
        self.learning_starts = 16
        self.memory = ReplayMemory(self.buffer_size)
        self.batch_size = 16
        self.max_steps_per_episode = 5000
        self.max_episodes = 5000
        self.policy_frequency = 2
        self.actor = Actor(
            self.state_dim, self.action_dim, self.max_action, self.min_action
        ).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.critic_lr,
        )
        self.avg_lengths = []
        self.all_rewards = []
        self.auto_alpha = True
        self._init_alpha()

    def _init_alpha(self):
        if not self.auto_alpha:
            self.alpha = 0.2
        else:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.action_dim).to(device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.critic_lr)

    def update(self, global_step):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state).to(device)
        action = torch.cat(batch.action).to(device)
        next_state = torch.cat(batch.next_state).to(device)
        reward = torch.cat(batch.reward).to(device)
        done = torch.cat(batch.done).to(device)

        with torch.no_grad():
            next_state_action, next_state_logp, _ = self.actor.select_action(next_state)
            q1_next_target = self.critic_target1(next_state, next_state_action)
            q2_next_target = self.critic_target2(next_state, next_state_action)
            min_q_next = (
                torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_logp
            )
            min_q_next = min_q_next.squeeze()
            next_q_value = reward + self.gamma * (1 - done) * min_q_next
        q1_values = self.critic1(state, action).view(-1)
        q2_values = self.critic2(state, action).view(-1)
        q1_loss = F.mse_loss(q1_values, next_q_value)
        q2_loss = F.mse_loss(q2_values, next_q_value)
        q_loss = q1_loss + q2_loss
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        if global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.select_action(state)
                q1_pi = self.critic1(state, pi)
                q2_pi = self.critic2(state, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (self.alpha * log_pi - min_q_pi).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                if self.auto_alpha:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.select_action(state)
                    alpha_loss = (
                        -self.log_alpha.exp() * (log_pi + self.target_entropy)
                    ).mean()
                    self.alpha_opt.zero_grad()
                    alpha_loss.backward()
                    self.alpha_opt.step()
                    self.alpha = self.log_alpha.exp().item()
        with torch.no_grad():
            for param, target_param in zip(
                self.critic1.parameters(), self.critic_target1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.critic2.parameters(), self.critic_target2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def rollout(self, global_step):
        ep_reward = 0
        ep_step = 0
        state, _ = self.env.reset()
        for t in range(self.max_steps_per_episode):
            state = state.reshape(1, -1)
            if len(self.memory) < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.actor.select_action(state)
                action = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            ep_reward += reward
            done = terminated | truncated
            next_state = next_state.reshape(1, -1)
            real_next_state = next_state.copy()
            state_tensor = torch.Tensor(state)
            action_tensor = torch.Tensor(action).reshape(1, -1)
            next_state_tensor = torch.Tensor(real_next_state)
            reward_tensor = torch.Tensor([reward])
            done_tensor = torch.Tensor([done])
            self.memory.push(
                state_tensor,
                action_tensor,
                next_state_tensor,
                reward_tensor,
                done_tensor,
            )
            state = next_state
            if len(self.memory) > self.learning_starts:
                self.update(global_step)
            if done:
                break
            ep_step += 1
        return ep_reward, ep_step

    def learn(self):
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(16, 6)
        global_step = 0
        for step in range(self.max_episodes):
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
            plt.savefig("sac.png")
            if step % 20 == 0:
                torch.save(self.actor.state_dict(), "sac_actor.pt")
                torch.save(self.critic1.state_dict(), "sac_critic1.pt")
                torch.save(self.critic2.state_dict(), "sac_critic2.pt")

    def test(self, plot=False, simulations=100):
        self.actor.load_state_dict(torch.load("sac_actor.pt"))
        all_rewards = []
        for _ in tqdm.tqdm(range(simulations)):
            state, _ = self.env.reset()
            total_reward = 0
            while True:
                state = torch.Tensor(state).to(device).reshape(1, -1)
                action, _, _ = self.actor.select_action(state)
                action = action.detach().cpu().numpy()
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                if not plot:
                    self.env.render()
                    time.sleep(0.1)
                if terminated or truncated:
                    break
            if plot:
                all_rewards.append(total_reward)
            else:
                print(f"Total reward: {total_reward}")
        if plot:
            sns.displot(all_rewards, kde=True, bins=100)
            plt.savefig("sac_test.png")


# seed
SEED = 69
LEARN = False

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open("config.pkl", "rb") as f:
    config = pickle.load(f)
print(config)
env = gym.make("racetrack-v0", render_mode="human")
env.unwrapped.configure(config)
print(env.observation_space.shape)
print(env.action_space.shape)
sac = SAC(env)
if LEARN:
    sac.learn()
else:
    sac.test(plot=False, simulations=10)
