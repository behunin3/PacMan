import gymnasium as gym
import pygame
import time
import ale_py
import shimmy
print(ale_py.__version__, shimmy.__version__)
# Initialize the Ms. Pac-Man environment
env = gym.make("ALE/MsPacman-v5", render_mode="human")

# Reset the environment to start a new game
obs, info = env.reset()

# Run the game for a set number of steps
for _ in range(1000):
    # Take a random action
    action = env.action_space.sample()
    
    # Step through the environment with the selected action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # End the game if terminated or truncated
    if terminated or truncated:
        obs, info = env.reset()

    # Add a slight delay to control the speed of rendering
    time.sleep(0.02)

# Close the environment when done
env.close()