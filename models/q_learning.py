#import sys
#sys.path.append('../')

from datetime import datetime
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from Model import Model
from environment import Environment


class Q_learning(Model):
    def __init__(self, state_space, action_space, num_episodes, learning_rate, discount_factor, exploration_rate, min_exploration_rate):
        super().__init__(state_space, action_space, discount_factor)
        self.state_space = state_space
        self.action_space = action_space
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.reward_trace = []

    def initialize_policy(self):
        return np.zeros((len(state_space), len(action_space)))

    def generate_episode(self, env, min_length=4):
        # Randomly generate the initial state (for each episode) beginning with the first period to period k−4.
        start = random.randint(0, len(env.data) - min_length)
        end = random.randint(start + min_length, len(env.data))
        return [i for i in range(start, end)]

    def get_state_index(self, state_space):
        return self.state_space.index(state_space)

    def get_action_index(self, action_space):
        return self.action_space.index(action_space)

    def choose_action(self, state_index):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(len(self.action_space))
        else:
            return np.argmax(self.q_table[state_index])  # choose action with max Q

    def update_q_value(self, state_index, action_index, reward, next_state_index):

        max_future_q = np.max(self.q_table[next_state_index])
        current_q = self.q_table[state_index][action_index]
        # refresh formula
        self.q_table[state_index][action_index] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        # self.q_table[state_index] /= np.sum(self.q_table[state_index])

    def train(self, episode, env):
        total_reward = 0
        for i in range(len(episode) - 1):
            state_index = episode[i]

            if state_index >= len(self.state_space):
                print(f"Error: state_index {state_index} is out of bounds.")
                continue

            state = self.state_space[state_index]
            next_state_index = episode[i + 1]

            if next_state_index >= len(self.state_space):
                print(f"Error: next_state_index {next_state_index} is out of bounds.")
                continue

            next_state = self.state_space[next_state_index]

            action_index = self.choose_action(state_index)  # Choose an action based on the current state index
            action = self.action_space[action_index]  # Get the corresponding action
            reward = env.get_reward(action, next_state)  # Get the reward
            total_reward += reward  # Accumulate the reward

            # Update Q-values
            self.update_q_value(state_index, action_index, reward, next_state_index)

        self.reward_trace.append(total_reward / len(episode))

    def learn(self, env, n_episodes=100, verbose_freq=10):

        for i in range(n_episodes):
            episode = self.generate_episode(env)  # generate an episode
            self.train(episode, env)  # train in that episode

    def test(self, env):
        action_index = self.choose_action(0)
        action = self.action_space[action_index]
        result = env.data[["Date"]]

        result["Return"] = (
            action[0] * env.data["NASDAQ_Returns"]
            + action[1] * env.data["MSCI_Returns"]
        )
        return result

all_actions = []
env = Environment()

if __name__ == "__main__":
    data = env.preprocess_data()
    print(data.columns)

    data[["NASDAQ_Returns", "MSCI_Returns"]] = (
            data[["NASDAQ_Returns", "MSCI_Returns"]] / 100)  # rescale for pytoch linear

    train_env = Environment()
    test_env = Environment()

    state_space = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
    action_space = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]

    #parameters
    num_episodes = 500
    learning_rate = 0.01
    discount_factor = 0.99
    exploration_rate = 0.5
    min_exploration_rate = 0.01

    q_learning_model = Q_learning(
        state_space=state_space,
        action_space=action_space,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        min_exploration_rate=min_exploration_rate
    )


    # q_learning_model.update_q_value(state_index, action_index, reward, next_state_index)
    q_learning_model.learn(train_env, n_episodes = num_episodes, verbose_freq = 10)
    print("Q-table after training:")
    print(q_learning_model.q_table)


