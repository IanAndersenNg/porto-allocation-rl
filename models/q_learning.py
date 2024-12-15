from datetime import datetime
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from Model import Model
from environment import Environment, preprocess_data


class Q_learning(Model):
    def __init__(
        self,
        state_space,
        action_space,
        num_episodes,
        learning_rate,
        discount_factor,
        exploration_rate,
        min_exploration_rate,
    ):
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

        print(f"State space length: {len(self.state_space)}")

    def initialize_policy(self):
        return np.zeros((len(self.state_space), len(self.action_space)))

    def generate_episode(self, env, min_length=4):
        start = random.randint(0, len(env.data) - min_length)
        end = random.randint(start + min_length, len(env.data))
        return [i for i in range(start, end)]

    def choose_action(self, state_index):
        state_index = state_index % len(self.state_space)
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(len(self.action_space))
        else:
            # choose action with max Q
            return np.argmax(self.q_table[state_index])

    def update_q_value(self, state_index, action_index, reward, next_state_index):
        max_future_q = np.max(self.q_table[next_state_index])
        current_q = self.q_table[state_index][action_index]
        self.q_table[state_index][action_index] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

    def train(self, episode, env):
        total_reward = 0
        for i in range(len(episode) - 1):
            # episode裡面是一堆data index, 給定一個index, 使用env.get_discrete_state 得到 00, 01, 10, 11
            state_index = episode[i] % len(self.state_space)

            if state_index >= len(self.state_space):
                print(f"Error: state_index {state_index} is out of bounds.")
                continue

            next_state_index = episode[i + 1] % len(self.state_space)


            if next_state_index >= len(self.state_space):
                print(f"Error: next_state_index {next_state_index} is out of bounds.")
                continue

            state = env.get_continuous_state(index=episode[i])
            next_state = env.get_continuous_state(index=episode[i + 1])

            # Choose an action based on the current state index
            action_index = self.choose_action(state_index)
            # Get the corresponding action
            action = self.action_space[action_index]
            # env.get_reward(action, i + 1)
            reward = env.get_reward(action, index = episode[i+1])  # Get the reward
            total_reward += reward  # Accumulate the reward

            # Update Q-values
            self.update_q_value(state_index, action_index, reward, next_state_index)

        self.reward_trace.append(total_reward / len(episode))

    def learn(self, env, n_episodes=100):
        for episode_idx in range(n_episodes):
            episode = self.generate_episode(env)

            # print start of episode
            print(f"Starting episode {episode_idx+1}/{n_episodes}")

            # Train on this episode
            self.train(episode, env)

            # Optional: Print progress every 10 episodes
            if (episode_idx + 1) % 10 == 0:
                avg_reward = np.mean(self.reward_trace[-10:])
                print(
                    f"Episode {episode_idx + 1}/{n_episodes} completed. Average reward (last 10): {avg_reward:.2f}"
                )

        print("Learning complete.")

    def test(self, env):
        action_index = self.choose_action(0)
        action = self.action_space[action_index]
        result = env.data[["Date"]]

        result["Return"] = (
            action[0] * env.data["AGG_Returns"] + action[1] * env.data["MSCI_Returns"]
        )
        return result


if __name__ == "__main__":

    data = preprocess_data()
    env = Environment(data=data)
    print("Columns:", data.columns)

    data[["AGG_Returns", "MSCI_Returns"]] = data[["AGG_Returns", "MSCI_Returns"]]

    # train and test environment
    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")]
    )
    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2020-01-01", "%Y-%m-%d")]
    )

    # define space
    state_space = ["11", "10", "01", "00"]
    action_space = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]

    # parameters
    num_episodes = 500
    learning_rate = 0.01
    discount_factor = 0.99
    exploration_rate = 0.5
    min_exploration_rate = 0.01

    # create model
    q_learning_model = Q_learning(
        state_space=state_space,
        action_space=action_space,
        num_episodes=num_episodes,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        min_exploration_rate=min_exploration_rate,
    )

    # train model
    q_learning_model.learn(train_env, n_episodes=num_episodes)

    # output q-table
    print("Q-table after training:")
    print(q_learning_model.q_table)
