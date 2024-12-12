import gymnasium as gym
import pygame
import time
import ale_py
import shimmy
print(ale_py.__version__, shimmy.__version__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
# Initialize the Ms. Pac-Man environment

from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np

from gymnasium import logger as gymlogger
from gymnasium.wrappers import RecordVideo
gymlogger.min_level = 40 # Error only

import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
# env = gym.make("ALE/MsPacman-v5", render_mode="human")

# Reset the environment to start a new game
# obs, info = env.reset()
random.seed(time.time())


from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400,900))
display.start()


def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else:
    print("Could not find video")

def wrap_env(env):
    env = RecordVideo(env, './video')
    return env

def get_action_dqn(network, state, epsilon, epsilon_decay):
    if random.random() < epsilon:
        action = torch.randint(0,9, (1,)).item()
    else:
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = network(state)

        action = torch.argmax(q_values, dim=1).item()
    epsilon *= epsilon_decay
    epsilon = max(epsilon, 0.01)
    return action, epsilon

def prepare_batch(memory, batch_size):
    batches = random.choices(memory, k=batch_size)
    state, action, next_state, reward, done = zip(*batches)
    state = [s for s in state]
    state = torch.tensor(state)
    action = [a for a in action]
    action = torch.tensor(action)
    next_state = [s for s in next_state]
    next_state = torch.tensor(next_state)
    reward = [r for r in reward]
    reward = torch.tensor(reward)
    done = [d for d in done]
    done = torch.tensor(done, dtype=torch.float32)
    return state, action, next_state, reward, done

def learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update):
    state, action, next_state, reward, done = batch
    part1 = reward + gamma * torch.max(target_network(next_state), dim=1)[0] * (1 - done)
    part2 = q_network(state)[torch.arange(32), action]
    loss = F.mse_loss(part2, part1)

    optim.zero_grad()
    loss.backward()
    optim.step()
    if global_step % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())

class QNetwork(nn.module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden_size = 8
        self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, action_size))
    def forward(self,x):
        return self.net(x)

def main():
    lr = 1e-3
    epochs = 500
    start_training = 1000
    gamma = 0.99
    batch_size = 32
    epsilon = 1
    epsilon_decay = .9999
    target_update = 1000
    learn_frequency = 2

    ## state_size = 4
    action_size = 9
    env = gym.make("ALE/MsPacman-v5", render_mode="human")
    state, _ = env.reset()
    state_size = state.size()
    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())

    optim = torch.optim.Adam(q_network.parameters(), lr=lr)

    memory = []
    results_dqn = []
    global_step = 0
    loop = tqdm(total=epochs, position=0, leave=False)
    for epoch in range(epochs):
        last_epoch = (epoch+1 == epochs)

        if last_epoch:
            env = wrap_env(env)

        state, _ = env.reset()
        done = False
        cum_reward = 0

        while not done and cum_reward < 200:
            action, epsilon = get_action_dqn(q_network, state, epsilon, epsilon_decay)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((state, action, next_state, reward, done))

            cum_reward += reward
            global_step += 1
            state = next_state

            if global_step > start_training and global_step % learn_frequency == 0:
                batch = prepare_batch(memory, batch_size)

                learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update)
        env.close()
        results_dqn.append(cum_reward)
        loop.update(1)
        loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))

    return results_dqn




# Run the game for a set number of steps
# for _ in range(1000):
#     # Take a random action
#     print(env.action_space)
#     action = env.action_space.sample()
    
#     # Step through the environment with the selected action
#     obs, reward, terminated, truncated, info = env.step(action)
    
#     # End the game if terminated or truncated
#     if terminated or truncated:
#         obs, info = env.reset()

#     # Add a slight delay to control the speed of rendering
#     time.sleep(0.02)

# # Close the environment when done
# env.close()
results_dqn = main()
show_video()