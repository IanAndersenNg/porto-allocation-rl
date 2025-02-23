from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.base_model import Model
from environment import Environment, preprocess_data
import argparse


class Q_learning(Model):
    name = "q_learning"

    def __init__(
        self,
        state_space,
        action_space,
        learning_rate,
        discount_factor,
        exploration_rate,
    ):
        super().__init__(state_space, action_space, discount_factor)
        self.state_indices = {state: i for i, state in enumerate(state_space)}
        self.action_indices = {action: i for i, action in enumerate(action_space)}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((len(state_space), len(action_space)))
        self.reward_trace = []
        self.episode_reward = []

        # print(f"State space length: {len(self.state_space)}")

    def initialize_policy(self):
        return np.zeros((len(self.state_space), len(self.action_space)))

    def choose_action(self, state):
        state_index = self.state_indices[state]
        return self.action_space[np.argmax(self.q_table[state_index])]

    def update_q_value(self, state, action, reward, next_state):
        # Follow the definition of Model
        state_index = self.state_indices[state]
        next_state_index = self.state_indices[next_state]
        action_index = self.action_indices[action]

        max_future_q = np.max(self.q_table[next_state_index])
        current_q = self.q_table[state_index][action_index]
        self.q_table[state_index][action_index] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

    def learn(self, episode, env):
        total_reward = 0
        for i in range(len(episode) - 1):
            # get discrete state and convert it to the index in the q table
            state = env.get_discrete_state(index=episode[i])
            next_state = env.get_discrete_state(index=episode[i + 1])
            # epsilon greedy
            if np.random.rand() < self.exploration_rate:
                action = self.action_space[np.random.choice(len(self.action_space))]
            else:
                # Choose an action based on the current state index
                action = self.choose_action(state)
            # env.get_reward(action, i + 1)
            reward = env.get_reward(action, index=episode[i + 1])  # Get the reward
            total_reward += reward  # Accumulate the reward

            # Update Q-values
            self.update_q_value(state, action, reward, next_state)

        self.reward_trace.append(total_reward / len(episode))
        self.episode_reward.append(total_reward)

    def train(self, env, n_episodes=100, verbose_freq=None):
        for episode_idx in range(n_episodes):
            episode = self.generate_episode(env)

            # print start of episode
            if verbose_freq and not (episode_idx + 1) % verbose_freq:
                print(f"Starting episode {episode_idx+1}/{n_episodes}")

            # Train on this episode
            self.learn(episode, env)

            # Optional: Print progress every 10 episodes
            if verbose_freq and (episode_idx + 1) % verbose_freq == 0:
                avg_reward = np.mean(self.reward_trace[-10:])
                print(
                    f"Episode {episode_idx + 1}/{n_episodes} completed. Average reward (last 10): {avg_reward:.2f}"
                )
                print(self.q_table)

        print("Learning complete.")
        self.df_q_table = pd.DataFrame(
            self.q_table, index=self.state_space, columns=self.action_space
        )
        self.plot_rewards()

    def test(self, env):
        actions = []
        # Choose action every day in testing dataset
        for i in range(len(env.data)):
            state = env.get_discrete_state(index=i)
            action = self.choose_action(state)
            actions.append(action)

        result = env.data[["Date"]].reset_index(drop=True)
        result = pd.concat(
            [result, pd.DataFrame(actions, columns=env.asset_names)], axis=1
        )
        result[env.asset_names] = result[env.asset_names].shift()

        return result.dropna().reset_index(drop=True)

    def plot_rewards(self):
        # Plotting the total reward per episode
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(self.episode_reward)),
            self.episode_reward,
            label="Episode Rewards",
            color="b",
        )
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward Curve Over Episodes")
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsr",
        action="store_true",
        help="Sets the reward function to be differential sharpe ratio",
    )
    dsr_reward = parser.parse_args().dsr
    data = preprocess_data()
    print("Columns:", data.columns)

    data[["AGG_Returns", "MSCI_Returns"]] = data[["AGG_Returns", "MSCI_Returns"]]

    # train and test environment
    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")],
        use_sharpe_ratio_reward=dsr_reward,
    )
    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2019-12-31", "%Y-%m-%d")],
        use_sharpe_ratio_reward=dsr_reward,
    )
    # define space
    state_space = ["11", "10", "01", "00"]
    # action_space = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
    action_space = [(0, 1), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (1, 0)]

    # parameters
    num_episodes = 1000
    learning_rate = 0.01
    discount_factor = 0.99
    exploration_rate = 0.1

    # create model
    q_learning_model = Q_learning(
        state_space=state_space,
        action_space=action_space,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
    )

    # train model
    q_learning_model.train(train_env, n_episodes=num_episodes, verbose_freq=100)

    # output q-table
    print("Q-table after training:")
    print(q_learning_model.df_q_table)
    result = q_learning_model.test(test_env)
    print(result)
