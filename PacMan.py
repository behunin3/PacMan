import gymnasium as gym
import pygame
import time
import ale_py
import shimmy
from enum import Enum
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import ResizeObservation
# print(ale_py.__version__, shimmy.__version__)

# Initialize the Ms. Pac-Man environment
env = gym.make("ALE/MsPacman-v5", render_mode="human")
env = GrayscaleObservation(env, keep_dim=True) # convert to grayscale to reduce information (210, 160, 3) ==> (210,160,1)
env = ResizeObservation(env, shape=(84,84)) # reduce obs size to reduce information

obs, info = env.reset()
epochs = 1000

for i in range(epochs):
    action = env.action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print('obs shape: ', obs.shape)
        print('reward: ', reward)
        print('terminated: ', terminated)
        print('truncated: ', truncated)
        print('info: ', info)
    
    # End the game if terminated or truncated
    if terminated or truncated:
        obs, info = env.reset()

    # Add a slight delay to control the speed of rendering
    time.sleep(0.02)

env.close()