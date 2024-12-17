import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque

# Define the Deep Q-Network (DQN) Model
class DQNModel(tf.keras.Model):
    def __init__(self, action_size):
        super(DQNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_size)

    def call(self, x):
        x = tf.cast(x, tf.float32) / 255.0  # Normalize pixel values
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.00025
        self.memory = deque(maxlen=20000)
        self.batch_size = 32

        # Initialize main and target networks
        self.model = DQNModel(action_size)
        self.target_model = DQNModel(action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Copy weights to target model
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model(np.expand_dims(state, axis=0))
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model(np.expand_dims(next_state, axis=0))[0])
            with tf.GradientTape() as tape:
                q_values = self.model(np.expand_dims(state, axis=0))
                loss = tf.keras.losses.mean_squared_error([target], [q_values[0][action]])
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train the Agent
def train_dqn(episodes=1000):
    env = gym.make('MsPacman-v0')
    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = DQNAgent(state_shape, action_size)

    for e in range(episodes):
        state = env.reset()
        state = tf.image.rgb_to_grayscale(state)
        state = tf.image.resize(state, (84, 84)).numpy()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = tf.image.rgb_to_grayscale(next_state)
            next_state = tf.image.resize(next_state, (84, 84)).numpy()

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.replay()

        agent.update_target_model()
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}")

    env.close()

# Run training
if __name__ == "__main__":
    train_dqn(episodes=500)
