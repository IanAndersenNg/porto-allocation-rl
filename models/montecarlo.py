from .base_model import Model
import numpy as np
import random
import pandas as pd
from tabulate import tabulate

'''
A monte carlo model with epsilon greedy policy.

- Episode: A complete run of the model from the start to the end of the training data.
- Epsilon: The probability of choosing a random action instead of the best action for exploring instead of exploiting.
- State-Action Dictionary: A dictionary that stores the average reward and count of each state-action pair.
- Reward: The reward is the next day reward of the action taken on the current day.
- States: ['11', '10', '01', '00']
- Actions: [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
'''

class MonteCarlo(Model):
    def __init__(self, actions, epsilon=0.0, state_space = ['11', '10', '01', '00']):
        super().__init__(state_space = state_space, action_space = actions)
        self.epsilon = epsilon
        self.state_action_dict = self.initialize_policy()
        self.optimum_action_dict = {}

    def initialize_policy(self):
        return {(state, action): (0,0) 
                for state in self.state_space
                for action in range(len(self.action_space))}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, len(self.action_space) - 1)  # Explore
        return np.argmax(self.calculate_average_reward(state)) # Exploit

    def update_policy(self, state, action, reward):
        reward_sum, count = self.state_action_dict[(state, action)]
        self.state_action_dict[(state, action)] = (round(reward_sum + reward, 2), count + 1)

    def train_with_logs(self, episodes, train_env):

        actions_taken  = []

        for i in range(episodes):
            episode_indexes  = self.generate_episode(train_env)

            print("Starting episode : ", i)
            print("____________________________________________________")

            for t in episode_indexes:
                state = train_env.get_discrete_state(index = t)
                action_index = self.choose_action(state)
                action = self.action_space[action_index]

                print("Chosen State: ", state)
                print("Chosen action: ", action)
                actions_taken.append(action)

                reward = train_env.get_reward(action, index =  t + 1)
                print("reward this iter: ", reward)
                self.update_policy(state, action_index, reward)

                print("State action dict after iter: ")
                print(self.pretty_print_state_action_dict(self.state_action_dict))
                print("____________________________________________________")

        print("State action dict after all episodes: ")
        # print("actions taken : ", actions_taken)
        print(self.pretty_print_state_action_dict(self.state_action_dict))

        self.optimum_action_dict = {
            state: np.argmax(self.calculate_average_reward(state))
            for state in self.state_space
        }

        return self.optimum_action_dict
    

    def train(self, episodes, train_env):

        for i in range(episodes):
            episode_indexes  = self.generate_episode(train_env)

            for t in episode_indexes:
                state = train_env.get_discrete_state(index = t)
                action_index = self.choose_action(state)
                action = self.action_space[action_index]

                reward = train_env.get_reward(action, index =  t + 1)
                self.update_policy(state, action_index, reward)
        
        print("State action dict after all episodes: ")
        print(self.pretty_print_state_action_dict(self.state_action_dict))

        self.optimum_action_dict = {
            state: np.argmax(self.calculate_average_reward(state))
            for state in self.state_space
        }

        return self.optimum_action_dict

    
    # returns the optimum action of each state
    def test(self, test_env):
        res_actions = []

        for t in range(len(test_env.data) - 1):            
            state = test_env.get_discrete_state(index = t)
            action_index = self.optimum_action_dict[state]
            action = self.action_space[action_index]
            res_actions.append(action)

        return res_actions
    
    # cum sum divided by count
    def calculate_average_reward(self, state):
        return [
            self.state_action_dict[(state, action)][0] / max(1, self.state_action_dict[(state, action)][1])
            for action in range(len(self.action_space))
        ]

    def pretty_print_state_action_dict(self, dict):
        states = sorted(set(state for state, action in dict.keys()))
        actions = sorted(set(action for state, action in dict.keys()))

        table = [[""] + actions]

        for state in states:
            row = [state]
            for action in actions:
                if (state, action) in dict:
                    reward_sum, count = dict[(state, action)]
                    row.append(f"({reward_sum}, {count})")
                else:
                    row.append("(0, 0)")
            table.append(row)

        print(tabulate(table, headers="firstrow", tablefmt="grid"))