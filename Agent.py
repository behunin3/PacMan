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
        state_space = 6 # my row, my col, cell_left, cell_straight, cell_right, direction
        self.network = QNetwork(state_space, action_space)
        self.target_network = QNetwork(state_space, action_space)
        self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon = 1
        self.epsilon_decay = .9999
        self.target_update = 1000
        self.gamma = 0.99
        self.lr = 1e-3
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.direction = Direction.LEFT if random.random() <= 0.5 else Direction.RIGHT
        # print(os.listdir(os.getcwd()))
        self.maze = self.read_maze('map.txt')
        self.maze[self.row][self.col] = Cell.PACMAN
        self.display = pygame.display.set_mode((580,400))
        pygame.display.set_caption("Pacman AI")
        self.display.fill(GRAY)
        pygame.display.flip()
        # count = 0
        # for i in range(len(self.maze)):
        #     for j in range(len(self.maze[0])):
        #         if self.maze[i][j] == Cell.BALL or self.maze[i][j] == Cell.POWERBALL:
        #             count += 1
        # print('count: ', count)

    def read_maze(self, file_path):
        maze = []
        with open(file_path, 'r') as file:
            for line in file:
                row = [Cell(int(char)) for char in line.strip()]
                maze.append(row)
        return maze

    def get_action(self, network, state, epsilon, epsilon_decay):
        if random.random() < epsilon:
            action = torch.randint(0,3, (1,)).item()
            neighbors = self.cells_around_me(self.direction)
            if neighbors[action] == Cell.GATE or neighbors[action] == Cell.WALL:
                action = (action + 1) % 3
            if neighbors[action] == Cell.GATE or neighbors[action] == Cell.WALL:
                action = (action + 1) % 3
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
    
    def reset(self):
        self.maze = self.read_maze('map.txt')
        self.row = 18
        self.col = 14
        self.lives = 3
        self.direction = Direction.LEFT if random.random() <= 0.5 else Direction.RIGHT
        left, straight, right = self.cells_around_me(self.direction)
        state = [self.row, self.col, left, straight, right, self.direction]
        return state

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
            elif self.maze[r][c] == Cell.BALL or self.maze[r][c] == Cell.POWERBALL:
                reward += 1
            self.maze[r][c] = Cell.PACMAN
            self.maze[self.row][self.col] = Cell.BLANK
            self.row = r
            self.col = c
            self.direction = actual

        left, straight, right = self.cells_around_me(self.direction)
        next_state = [self.row, self.col, left, straight, right, self.direction] 
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
        state, action, next_state, reward, done = batch
        part1 = reward + gamma * torch.max(target_network(next_state), dim=1)[0] * (1 - done)
        part2 = q_network(state)[torch.arange(32), action]
        loss = F.mse_loss(part2, part1)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if global_step % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
    
    def train(self):
        block_size = 20
        
        epochs = 500
        start_training = 100
        batch_size = 32
        learn_frequency = 2

        memory = []
        results = []
        global_step = 0
        loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):
            # self.draw_maze()
            state = self.reset()
            done = False
            cum_reward = 0
            start_time = time.time()
            self.display.fill(DARK_GRAY)

            while not done and cum_reward < 286 and time.time() - start_time < 1: # there are 278 balls and 8 powerballs
                
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
                    self.learn_dqn(batch, self.optim, self.network, self.target_network, self.gamma, global_step, self.target_update)

                results.append(cum_reward)
                loop.update(1)
                # print('epoch: ', epoch, 'cum_reward:', cum_reward)
                loop.set_description('Episodes: {} Reward: {}'.format(epoch, cum_reward))
                # self.display.fill(GRAY)
                pygame.display.flip()

        return results
        
a = Agent()
# a.draw_maze()
results = a.train()
# print(type(results))
plt.plot(results)
plt.title("Cumulative Reward vs Time gen 0")
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.show()