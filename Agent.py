import time
import random
import torch
import torch.nn.functional as F
from QNetwork import QNetwork
from Cell import Cell
from tqdm import tqdm
from enum import Enum
import matplotlib.pyplot as plt
import pygame
import os
import numpy as np

GRAY = (224,224,224)
DARK_GRAY = (100,100,100)
YELLOW = (255,255,0)
GREEN = (0,255,0)
DARK_GREEN = (0,153,0)
PURPLE = (153,51,255)
DARK_PINK = (102,0,51)
WHITE = (255,255,255)

class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

class Agent():
    def __init__(self):
        random.seed(time.time())
        self.row = 18
        self.col = 14
        self.lives = 3
        action_space = 3 # left, straight, right
        state_space = 27 # my row, my col, cell_left, cell_straight, cell_right, direction
        self.network = QNetwork(state_space, action_space)
        self.target_network = QNetwork(state_space, action_space)
        self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon = 1
        self.epsilon_decay = .9999
        self.target_update = 500
        self.gamma = 0.99
        self.lr = 1e-3
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.direction = Direction.LEFT if random.random() <= 0.5 else Direction.RIGHT
        # print(os.listdir(os.getcwd()))
        self.maze = self.read_maze('map.txt')
        self.maze[self.row][self.col] = Cell.PACMAN
        
        # self.display.fill(GRAY)
        # pygame.display.flip()
        # count = 0
        # for i in range(len(self.maze)):
        #     for j in range(len(self.maze[0])):
        #         if self.maze[i][j] == Cell.BALL or self.maze[i][j] == Cell.POWERBALL:
        #             count += 1
        # print('count: ', count)

    def get_channels(self):
        rows, cols = len(self.maze), len(self.maze[0])

        wall_channel = np.zeros((rows, cols), dtype=int)
        ball_channel = np.zeros((rows, cols), dtype=int)
        powerball_channel = np.zeros((rows, cols), dtype=int)
        ghost_channel = np.zeros((rows, cols), dtype=int)
        empty_channel = np.zeros((rows, cols), dtype=int)
        pacman_channel = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
                tile = self.maze[r][c]
                if tile == Cell.WALL or tile == Cell.GATE:
                    wall_channel[r, c] = 1
                elif tile == Cell.BALL:
                    ball_channel[r, c] = 1
                elif tile == Cell.POWERBALL:
                    powerball_channel[r, c] = 1
                elif tile == Cell.GHOST:
                    ghost_channel[r, c] = 1
                elif tile == Cell.PACMAN:
                    pacman_channel[r, c] = 1
                else:
                    empty_channel[r, c] = 1
        
        channels = np.stack([wall_channel, ball_channel, powerball_channel, ghost_channel, empty_channel, pacman_channel])
        channels = channels.flatten()
        return channels

    def get_surroundings(self):
        neighbors = [
            (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2),
            (-1,-2), (-1,-1), (-1,0), (-1,1), (-1,2),
            (0,-2), (0,-1),           (0,1), (0,2),
            (1,-2), (1,-1), (1,0),  (1,1), (1,2),
            (2,-2), (2,-1), (2,0), (2,1), (2,2)
        ]

        surroundings = []
        for dr, dc in neighbors:
            new_row, new_col = self.row + dr, self.col + dc
            if 0 <= new_row < len(self.maze) and 0 <= new_col < len(self.maze[0]):
                surroundings.append(self.maze[new_row][new_col].value)
            else:
                surroundings.append(-1)
        return surroundings

        

    def read_maze(self, file_path):
        maze = []
        with open(file_path, 'r') as file:
            for line in file:
                row = [Cell(int(char)) for char in line.strip()]
                maze.append(row)
        return maze

    def get_action(self, network, state, epsilon, epsilon_decay):
        neighbors = self.cells_around_me(self.direction)
        if random.random() < epsilon:
            action = torch.randint(0,3, (1,)).item()
        else:
            tstate = []
            for item in state:
                if isinstance(item, Cell) or isinstance(item, Direction):
                    tstate.append(float(item.value))
                else:
                    tstate.append(float(item))
            state = torch.tensor(tstate)
            # state = state.unsqueeze(0)
            with torch.no_grad():
                q_values = network(state)
            action = torch.argmax(q_values, dim=0).item()
        # if neighbors[action] == Cell.GATE or neighbors[action] == Cell.WALL:
        #     action = (action + 1) % 3
        # if neighbors[action] == Cell.GATE or neighbors[action] == Cell.WALL:
        #     action = (action + 1) % 3
        epsilon *= epsilon_decay
        epsilon = max(epsilon, 0.01)
        return action, epsilon

    def draw_maze(self):
        block_size = 20

        while True:
            self.display.fill(GRAY)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            for row in range(len(self.maze)):
                for col in range(len(self.maze[0])):
                    cell = self.maze[row][col]    
                    rect = pygame.Rect(col * block_size, row*block_size, block_size, block_size)
                    if cell == Cell.WALL:
                        pygame.draw.rect(self.display, DARK_GRAY, rect)
                    elif cell == Cell.GATE:
                        pygame.draw.rect(self.display, PURPLE, rect)
                    elif cell == Cell.BALL:
                        pygame.draw.rect(self.display, GREEN, rect)
                    elif cell == Cell.POWERBALL:
                        pygame.draw.rect(self.display, DARK_GREEN, rect)
                    elif cell == Cell.PACMAN:
                        pygame.draw.rect(self.display, YELLOW, rect)
                    elif cell == Cell.GHOST:
                        pygame.draw.rect(self.display, DARK_PINK, rect)
                    else:
                        pygame.draw.rect(self.display, WHITE, rect) 
            pygame.display.flip()

    def get_state(self):
        surroundings = self.get_surroundings()
        scalars = np.array([self.row, self.col, self.direction.value])
        state = np.concatenate((scalars, surroundings))
        return state
    
    def reset(self):
        self.maze = self.read_maze('map.txt')
        self.row = 18
        self.col = 14
        self.lives = 3
        self.direction = Direction.LEFT if random.random() <= 0.5 else Direction.RIGHT
        # left, straight, right = self.cells_around_me(self.direction)
        # channels = self.get_channels()
        return self.get_state()
        

    def cells_around_me(self, direction):
        if direction == Direction.LEFT:
            return [self.maze[self.row+1][self.col], self.maze[self.row][self.col-1], self.maze[self.row-1][self.col]] # down, left, up
        elif direction == Direction.UP:
            return [self.maze[self.row][self.col-1], self.maze[self.row-1][self.col], self.maze[self.row][self.col+1]] # left, up, right
        elif direction == Direction.RIGHT:
            return [self.maze[self.row-1][self.col], self.maze[self.row][self.col+1], self.maze[self.row+1][self.col]] # up, right, down
        else:
            return [self.maze[self.row][self.col+1], self.maze[self.row+1][self.col], self.maze[self.row][self.col-1]] # right, down, left


    def step(self, action, direction):
        # actual = Direction.LEFT
        reward = 0
        if direction == Direction.LEFT:
            if action == 0:
                actual = Direction.DOWN
            elif action == 1:
                actual = Direction.LEFT
            else:
                actual = Direction.UP
        elif direction == Direction.UP:
            if action == 0:
                actual = Direction.LEFT
            elif action == 1:
                actual = Direction.UP
            else:
                actual = Direction.RIGHT
        elif direction == Direction.RIGHT:
            if action == 0:
                actual = Direction.UP
            elif action == 1:
                actual = Direction.RIGHT
            else:
                actual = Direction.DOWN
        else:
            if action == 0:
                actual = Direction.RIGHT
            elif action == 1:
                actual = Direction.DOWN
            else:
                actual = Direction.LEFT
        
        if actual == Direction.LEFT:
            r = self.row
            c = self.col - 1
        elif actual == Direction.UP:
            r = self.row - 1
            c = self.col
        elif actual == Direction.RIGHT:
            r = self.row
            c = self.col + 1
        else:
            r = self.row + 1
            c = self.col

        terminated = False
        if self.maze[r][c] != Cell.GATE and self.maze[r][c] != Cell.WALL:
            if self.maze[r][c] == Cell.GHOST:
                terminated = True
            # elif self.maze[r][c] == Cell.BLANK:
            #     reward -= 1
            elif self.maze[r][c] == Cell.BALL or self.maze[r][c] == Cell.POWERBALL:
                reward += 1
            self.maze[r][c] = Cell.PACMAN
            self.maze[self.row][self.col] = Cell.BLANK
            self.row = r
            self.col = c
            self.direction = actual

        # left, straight, right = self.cells_around_me(self.direction)
        # channels = self.get_channels()
        next_state = self.get_state()
        # return next_state, reward, terminated, truncated
        return next_state, reward, terminated, False

    def prepare_batch(self, memory, batch_size):
        batches = random.choices(memory, k=batch_size)
        state, action, next_state, reward, done = zip(*batches)
        state = [[float(s.value) if isinstance(s, Enum) else float(s) for s in row] for row in state]
        state = torch.tensor(state)
        action = [a for a in action]
        action = torch.tensor(action)
        next_state = [[float(s.value) if isinstance(s, Enum) else float(s) for s in row] for row in next_state]
        next_state = torch.tensor(next_state)
        reward = [r for r in reward]
        reward = torch.tensor(reward)
        done = [d for d in done]
        done = torch.tensor(done, dtype=torch.float32)
        return state, action, next_state, reward, done
    
    def learn_dqn(self, batch, optim, q_network, target_network, gamma, global_step, target_update):
        clip_value=1.0
        state, action, next_state, reward, done = batch
        part1 = reward + gamma * torch.max(target_network(next_state), dim=1)[0] * (1 - done)
        part2 = q_network(state)[torch.arange(32), action]
        loss = F.mse_loss(part2, part1)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), clip_value)
        optim.step()
        if global_step % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
        return loss
    
    def train(self, visualize=True):
        if visualize:
            self.display = pygame.display.set_mode((580,400))
            pygame.display.set_caption("Pacman AI")
        block_size = 20
        
        epochs = 100
        start_training = 200
        batch_size = 32
        learn_frequency = 2

        memory = []
        results = []
        num_iters = []
        losses = []
        global_step = 0
        loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):
            # self.draw_maze()
            state = self.reset()
            done = False
            cum_reward = 0
            start_time = time.time()
            if visualize:
                self.display.fill(DARK_GRAY)
            epoch_iters = 0
            while not done and cum_reward < 286 and time.time() - start_time < 1: # there are 278 balls and 8 powerballs
                epoch_iters += 1
                if visualize:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                    for row in range(len(self.maze)):
                        for col in range(len(self.maze[0])):
                            cell = self.maze[row][col]    
                            rect = pygame.Rect(col * block_size, row*block_size, block_size, block_size)
                            if cell == Cell.WALL:
                                pygame.draw.rect(self.display, DARK_GRAY, rect)
                            elif cell == Cell.GATE:
                                pygame.draw.rect(self.display, PURPLE, rect)
                            elif cell == Cell.BALL:
                                pygame.draw.rect(self.display, GREEN, rect)
                            elif cell == Cell.POWERBALL:
                                pygame.draw.rect(self.display, DARK_GREEN, rect)
                            elif cell == Cell.PACMAN:
                                pygame.draw.rect(self.display, YELLOW, rect)
                            elif cell == Cell.GHOST:
                                pygame.draw.rect(self.display, DARK_PINK, rect)
                            else:
                                pygame.draw.rect(self.display, WHITE, rect) 
                    pygame.display.flip()
                action, self.epsilon = self.get_action(self.network, state, self.epsilon, self.epsilon_decay)
                next_state, reward, terminated, truncated = self.step(action, self.direction)
                done = terminated or truncated
                memory.append((state, action, next_state, reward, done))
                # print(reward, cum_reward)
                cum_reward += reward
                global_step += 1
                state = next_state
                if global_step > start_training and global_step % learn_frequency == 0:
                    batch = self.prepare_batch(memory, batch_size)
                    loss = self.learn_dqn(batch, self.optim, self.network, self.target_network, self.gamma, global_step, self.target_update)
                    losses.append(loss)

                results.append(cum_reward)
                loop.update(1)
                # print('epoch: ', epoch, 'cum_reward:', cum_reward)
                loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))
                # self.display.fill(GRAY)
                if visualize:
                    pygame.display.flip()
            num_iters.append(epoch_iters)
        return results, num_iters, losses
        
a = Agent()
# a.draw_maze()
results, iters, losses = a.train(False)
# print(type(results))
plt.plot(results)
plt.title("Cumulative Reward vs Time gen 4")
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.show()
plt.plot(iters)
plt.title("Num Iters per Epoch")
plt.xlabel('epoch')
plt.ylabel('Iters')
plt.show()

python_list = [t.detach().tolist() for t in losses]  # Use detach() if needed
# flat_list = [item for sublist in python_list for item in sublist]
plt.plot(python_list)
plt.title("Loss per Epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()