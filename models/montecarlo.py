from models.model import Model
from environment import Environment
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

'''
A monte carlo model with epsilon greedy policy.

- Episode: A complete run of the model from the start to the end of the training data.
- Epsilon: The probability of choosing a random action instead of the best action for exploring instead of exploiting.
- Portfolio Value: init to be 100 at first and then it will be updated based on the reward on each episode to track the value.

'''

class MonteCarlo(Model):
    def __init__(self, cleaned_data, actions, epsilon=0.1):
        super().__init__(state_space=None, action_space=actions)
        self.cleaned_data = cleaned_data
        self.actions = actions
        self.epsilon = epsilon
        self.nasdaq_train, self.msci_train, self.nasdaq_test, self.msci_test = self.split_data()
    
    @staticmethod
    def calculate_reward(action, nasdaq_return, msci_return):
        weight_nasdaq, weight_msci = action
        portfolio_return = (weight_nasdaq / 100) * nasdaq_return + (weight_msci / 100) * msci_return
        return portfolio_return
    

    def split_data(self):
        train_data = self.cleaned_data[self.cleaned_data['NASDAQ_Date'] < '2020-01-01']
        test_data = self.cleaned_data[self.cleaned_data['NASDAQ_Date'] >= '2020-01-01']

        nasdaq_train = train_data['NASDAQ_Returns'].values
        msci_train = train_data['MSCI_Returns'].values
        nasdaq_test = test_data['NASDAQ_Returns'].values
        msci_test = test_data['MSCI_Returns'].values

        return nasdaq_train, msci_train, nasdaq_test, msci_test


    def train(self, n_episodes):
        '''
        train the monte carlo model for n_episodes and return the results.
        '''

        results = []
        best_action_index = 0
        n_steps = len(self.nasdaq_train)

        for episode in range(n_episodes):
            portfolio_value = 100  # Initial portfolio value
            episode_rewards = []

            for t in range(n_steps):
                if random.uniform(0, 1) < self.epsilon:
                    action_index = random.randint(0, len(self.actions) - 1)  # Explore
                else:
                    action_index = best_action_index  # Exploit

                action = self.actions[action_index]
     
                nasdaq_return = self.nasdaq_train[t]
                msci_return = self.msci_train[t]

                reward = self.calculate_reward(action, nasdaq_return, msci_return)
                portfolio_value *= (1 + reward / 100)
                episode_rewards.append(reward)

            avg_reward = np.mean(episode_rewards)
            if episode == 0 or avg_reward > results[best_action_index]['average_reward']:
                best_action_index = action_index

            results.append({
                'episode': episode,
                'final_portfolio_value': portfolio_value,
                'average_reward': avg_reward,
            })

        return pd.DataFrame(results), self.actions[best_action_index]
    
    def test(self, optimal_action):
        portfolio_value_test = 100
        rewards = []

        for t in range(len(self.nasdaq_test)):
            action = optimal_action

            nasdaq_return = self.nasdaq_test[t]
            msci_return = self.msci_test[t]

            reward = self.calculate_reward(action, nasdaq_return, msci_return)

            portfolio_value_test *= (1 + reward / 100)
            rewards.append(reward)

        return portfolio_value_test, rewards

    
    @staticmethod
    def visualize_results(results_df, rewards, test_portfolio_value):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(results_df['episode'], results_df['final_portfolio_value'], label='Portfolio Value')
        plt.xlabel('Episode')
        plt.ylabel('Portfolio Value')
        plt.title('Training Portfolio Value by Episode')
        plt.legend()


        plt.subplot(1, 2, 2)
        plt.plot(range(len(rewards)), np.cumsum(rewards), label='Cumulative Reward')
        plt.axhline(y=test_portfolio_value, color='r', linestyle='--', label='Final Portfolio Value')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Reward')
        plt.title('Test Cumulative Rewards')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    environment = Environment()
    cleaned_data = environment.data

    actions = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
    monte_carlo = MonteCarlo(cleaned_data, actions, epsilon=0.1)

    # Train the model
    results_df, optimal_action = monte_carlo.train()

    # Test the model
    test_portfolio_value, rewards = monte_carlo.test(optimal_action)

    # Visualize results
    MonteCarlo.visualize_results(results_df, rewards, test_portfolio_value)

    # Print final portfolio value for test data
    print(f"Final portfolio value on test data: {test_portfolio_value}")
    print(f"Optimal action determined during training: {optimal_action}")

