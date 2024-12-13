from models.model import Model
from datetime import datetime
from environment import Environment, preprocess_data
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

'''
A monte carlo model with epsilon greedy policy.

- Episode: A complete run of the model from the start to the end of the training data.
- Epsilon: The probability of choosing a random action instead of the best action for exploring instead of exploiting.
- State-Action Dictionary: A dictionary that stores the average reward and count of each state-action pair.

'''

class MonteCarlo(Model):
    def __init__(self, cleaned_data, actions, epsilon=0.1):
        super().__init__(state_space=None, action_space=actions)
        self.cleaned_data = cleaned_data
        self.actions = actions
        self.epsilon = epsilon
        self.state_action_dict = self.initialize_policy()

    def initialize_policy(self):
        return {(state, action): (0, 0) 
                for state in ['11', '10', '01', '00'] 
                for action in range(len(self.actions))}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.actions) - 1)  # Explore
        return self.calculate_average_reward(state) # Exploit

    def update_policy(self, state, action, reward):
        reward_sum, count = self.state_action_dict[(state, action)]
        self.state_action_dict[(state, action)] = (reward_sum + reward, count + 1)

    def train(self, episodes, train_env):
        n_steps = len(train_env.data)

        for episode in range(episodes):
            for t in range(n_steps):
                state = train_env.get_state(index=t)

                action_index = self.choose_action(state)
                action = self.actions[action_index]

                reward = train_env.get_reward(action, index=t)
                self.update_policy(state, action_index, reward)

        optimum_action_dict = {
            state: np.argmax(self.calculate_average_reward(state))
            for state in ['11', '10', '01', '00']}

        return optimum_action_dict

    
    def test(self, optimum_action_dict, test_env):
        rewards = []

        for t in range(len(test_env.data)):
            state = test_env.get_state(index=t)

            action_index = optimum_action_dict[state]
            action = self.actions[action_index]

            reward = test_env.get_reward(action, index=t)
            rewards.append(reward)

        return rewards
    

    def calculate_average_reward(self, state):
        return [
            self.state_action_dict[(state, action)][0] / max(1, self.state_action_dict[(state, action)][1])
            for action in range(len(self.actions))
        ]


if __name__ == "__main__":
    
    data = preprocess_data()
    actions = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
    monte_carlo = MonteCarlo(data, actions, epsilon=0.1)

    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")])
    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2020-01-01", "%Y-%m-%d")])

    results_df, optimal_action = monte_carlo.train(n_episodes= 1000, env=train_env)
    test_portfolio_value, rewards = monte_carlo.test(optimal_action, test_env)