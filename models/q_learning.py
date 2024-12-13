import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Model import Model
from environment import Environment
# Environment class from your provided code
'''
class Environment:
    def __init__(self):
        self.data = self.preprocess_data()

    def preprocess_data(self):
        nasdaq_etf = pd.read_excel('data.xlsx', skiprows=4, usecols=[0, 1], names=["Date", "NASDAQ_Returns"])
        nasdaq_etf["NASDAQ_Returns"] = pd.to_numeric(nasdaq_etf["NASDAQ_Returns"], errors="coerce")
        nasdaq_etf["Date"] = pd.to_datetime(nasdaq_etf["Date"], errors="coerce")
        nasdaq_etf = nasdaq_etf.dropna().reset_index(drop=True)

        em_etf = pd.read_excel('data.xlsx', skiprows=4, usecols=[2, 3], names=["Date", "MSCI_Returns"])
        em_etf["MSCI_Returns"] = pd.to_numeric(em_etf["MSCI_Returns"], errors="coerce")
        em_etf["Date"] = pd.to_datetime(em_etf["Date"], errors="coerce")
        em_etf = em_etf.dropna().reset_index(drop=True)

        return pd.merge(nasdaq_etf, em_etf, on="Date", how="inner")

    def _get_row(self, date=None, index=None):
        if index is not None and date is not None:
            raise ValueError("Provide either 'index' or 'date', not both.")
        if index is not None:
            return self.data.iloc[index]
        if date is not None:
            return self.data.loc[self.data["Date"] == date]
        raise ValueError("Either 'index' or 'date' must be provided.")

    def get_state(self, date=None, index=None):
        row = self._get_row(date=date, index=index)
        returns = row.iloc[1:3]
        state = "".join(["1" if r > 0 else "0" for r in returns])
        return state

    def get_reward(self, action, date=None, index=None):
        row = self._get_row(date=date, index=index)
        returns = row.iloc[1:]
        portfolio_return = np.dot(action, returns)
        portfolio_return = max(portfolio_return, -0.999)  # 限制下界，避免log(0)
        log_return = np.log(1 + portfolio_return)
        return log_return
'''

class Q_learning(Model):
    def __init__(self, state_space, action_space, num_episodes, learning_rate, discount_factor, exploration_rate, exploration_decay, min_exploration_rate):
        super().__init__(state_space, action_space, discount_factor)
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

# Q-learning parameters
num_episodes = 500
learning_rate = 0.01
discount_factor = 0.99
exploration_rate = 0.5
exploration_decay = 0.99
min_exploration_rate = 0.01

env = Environment()

# Initialize Q-table
q_table = {}
for row in env.data.itertuples(index=False):
    state = "".join(["1" if r > 0 else "0" for r in [row.NASDAQ_Returns, row.MSCI_Returns]])
    q_table[state] = [0, 0]  # Two actions: invest in NASDAQ or MSCI

# Record metrics
all_actions = []
all_rewards = []
episode_rewards = []

# Training loop
for episode in range(num_episodes):
    state = env.get_state(index=0)
    total_reward = 0

    for t in range(len(env.data)):
        # Choose action using epsilon-greedy
        if np.random.rand() < exploration_rate:
            action_index = np.random.choice(len(q_table[state]))
        else:
            action_index = np.argmax(q_table[state])

        action = [1 if i == action_index else 0 for i in range(2)]

        # Record action
        all_actions.append(action_index)

        # Get reward and next state
        reward = env.get_reward(action=action, index=t)
        next_state = env.get_state(index=t + 1) if t + 1 < len(env.data) else None

        # Update Q-value
        if next_state is not None:
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][action_index] += learning_rate * (
                reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action_index]
            )
        else:
            q_table[state][action_index] += learning_rate * (reward - q_table[state][action_index])

        # Update state and total reward
        state = next_state
        total_reward += reward

    # Decay exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)
    episode_rewards.append(total_reward)

# Visualization functions
def plot_rewards(rewards):
    plt.plot(range(len(rewards)), rewards, label='Episode Rewards', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.legend()
    plt.show()

def plot_action_distribution(actions):
    unique, counts = np.unique(actions, return_counts=True)
    plt.bar(unique, counts, color='skyblue', tick_label=['NASDAQ', 'MSCI'])
    plt.xlabel('Asset')
    plt.ylabel('Action Count')
    plt.title('Action Distribution Across Episodes')
    plt.show()

# Plot results
plot_rewards(episode_rewards)
plot_action_distribution(all_actions)
