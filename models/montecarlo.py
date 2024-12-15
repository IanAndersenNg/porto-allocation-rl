from .model import Model
import numpy as np
import random

'''
A monte carlo model with epsilon greedy policy.

- Episode: A complete run of the model from the start to the end of the training data.
- Epsilon: The probability of choosing a random action instead of the best action for exploring instead of exploiting.
- State-Action Dictionary: A dictionary that stores the average reward and count of each state-action pair.
- Reward: The reward is the next day reward of the action taken on the current day.

'''

class MonteCarlo(Model):
    def __init__(self, cleaned_data, actions, epsilon=0.1, state_space = ['11', '10', '01', '00']):
        super().__init__(state_space = state_space, action_space = actions)
        self.cleaned_data = cleaned_data
        self.epsilon = epsilon
        self.state_action_dict = self.initialize_policy()

    def initialize_policy(self):
        return {(state, action): (0, 0) 
                for state in self.state_space
                for action in range(len(self.action_space))}

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.action_space) - 1)  # Explore
        return np.argmax(self.calculate_average_reward(state)) # Exploit

    def update_policy(self, state, action, reward):
        reward_sum, count = self.state_action_dict[(state, action)]
        self.state_action_dict[(state, action)] = (reward_sum + reward, count + 1)

    def train(self, episodes, train_env):
        n_steps = len(train_env.data)

        for episode in range(episodes):
            for t in range(n_steps - 1):
                state = train_env.get_discrete_state(index = t)

                action_index = self.choose_action(state)
                action = self.action_space[action_index]

                reward = train_env.get_reward(action, index =  t + 1)
                self.update_policy(state, action_index, reward)

        optimum_action_dict = {
            state: np.argmax(self.calculate_average_reward(state))
            for state in self.state_space
        }

        return optimum_action_dict

    
    # returns the optimum action of each state
    def test(self, optimum_action_dict, test_env):
        res_actions = []

        for t in range(len(test_env.data) - 1):            
            state = test_env.get_discrete_state(index = t)
            action_index = optimum_action_dict[state]
            action = self.action_space[action_index]
            res_actions.append(action)

        return res_actions
    
    # cum sum divided by count
    def calculate_average_reward(self, state):
        return [
            self.state_action_dict[(state, action)][0] / max(1, self.state_action_dict[(state, action)][1])
            for action in range(len(self.action_space))
        ]