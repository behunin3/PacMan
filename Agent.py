import time
import random
import torch
import torch.nn.functional as F
from QNetwork import QNetwork
from Cell import Cell
from tqdm import tqdm
from enum import Enum

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
        state_space = 11 # my position, ghost positions, what cells are around me
        self.network = QNetwork(state_space, action_space)
        self.target_network = QNetwork(state_space, action_space)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epsilon = 1
        self.epsilon_decay = .9999
        self.target_update = 1000
        self.gamma = 0.99
        self.lr = 1e-3
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.direction = Direction.LEFT
        self.maze = self.read_maze('map.txt')

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
        else:
            state = torch.tensor(state)
            state = state.unsqueeze(0)
            with torch.no_grad():
                q_values = network(state)
            
            action = torch.argmax(q_values, dim=1).item()
        epsilon *= epsilon_decay
        epsilon = max(epsilon, 0.01)
        return action, epsilon
    
    def reset(self):
        pass

    def step(self, action, direction):
        actual = Direction.LEFT
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
        
        if self.maze[r][c] != Cell.GATE and self.maze[r][c] != Cell.WALL:
            if self.maze[r][c] == Cell.BALL or self.maze[r][c] == Cell.POWERBALL:
                reward += 1
            self.maze[r][c] = Cell.PACMAN
            self.maze[self.row][self.col] = Cell.BLANK
            self.row = r
            self.col = c

        next_state = [] # TODO I HAVE NO IDEA WHAT THIS SHOULD LOOK LIKE
        # return next_state, reward, terminated, truncated

    def prepare_batch(self, memory, batch_size):
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
        epochs = 500
        start_training = 1000
        batch_size = 32
        learn_frequency = 2

        memory = []
        results = []
        global_step = 0
        loop = tqdm(total=epochs, position=0, leave=False)
        for epoch in range(epochs):
            state = self.reset()
            done = False
            cum_reward = 0

            while not done and cum_reward < 200:
                action, epsilon = self.get_action(self.network, state, self.epsilon, self.epsilon_decay)
                next_state, reward, terminated, truncated = self.step(action, self.direction)
                done = terminated, truncated
                memory.append((state, action, next_state, reward, done))

                cum_reward += reward
                global_step += 1
                state = next_state
                if global_step > start_training and global_step % learn_frequency == 0:
                    batch = self.prepare_batch(memory, batch_size)
                    self.learn_dqn(batch, self.optim, self.network, self.target_network, self.gamma, global_step, self.target_update)

                results.append(cum_reward)
                loop.update(1)
                loop.set_description('Episodes: {} Reward: {}').format(epoch, cum_reward)

            return results