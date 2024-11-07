import gymnasium as gym
# import gym_pacman

class PacMan():
    def __init__(self):
        self.env = gym.make("ALE/Pacman-v5")

print('hello world')