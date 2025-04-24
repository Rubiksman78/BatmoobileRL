import gymnasium as gym
import pickle
import torch
import matplotlib.pyplot as plt
from IPython import display
from network import DQN
import highway_env
import time

with open("config.pkl", "rb") as f:
  config = pickle.load(f)
print(config)

env = gym.make("roundabout-v0", render_mode='rgb_array')
env.unwrapped.configure(config)
env.unwrapped.config["duration"] = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render())
    plt.title("%s | Step: %s" % (step, info))
    plt.axis('off')
    plt.pause(0.001)
    display.display(plt.gcf())
    display.clear_output(wait=True)

n_actions = env.action_space.n
state, info = env.reset()
n_observations = state.shape[0]*state.shape[1]
dqn = DQN(n_observations, n_actions).to(device)
dqn.load_state_dict(torch.load("dqn2.pth"))

def select_action(state):
    with torch.no_grad():
        return dqn(state).max(1).indices.view(1,1)
  
    
state,info = env.reset()
for t in range(1000):
    state = torch.from_numpy(state).float().to(device)
    state = state.view(-1).unsqueeze(0)
    action = select_action(state)
    env.render()
    state, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
    # show_state(env,step=t,info="")   
    if done:
        time.sleep(10)
        break

env.close()
