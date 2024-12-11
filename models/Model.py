import numpy as np
import random

class Model:
    def __init__(self, state_space, action_space, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.policy = self.initialize_policy()

    def initialize_policy(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def choose_action(self, state):
        raise NotImplementedError("This method should be overridden by subclasses")

    def update_policy(self, state, action, reward, next_state):
        raise NotImplementedError("This method should be overridden by subclasses")

    def train(self, episodes):
        raise NotImplementedError("This method should be overridden by subclasses")
        
        
class GradientTD(Model):
    def __init__(self, state_space, action_space, gamma=0.9, lambda_=0.9, alpha=0.1, epsilon=0.01):
        super().__init__(state_space, action_space, gamma)
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epsilon = epsilon
        self.e = np.array([0, 0])
        self.reward_trace = []

    def initialize_policy(self):
        self.theta = np.array([random.random(), random.random()])
        
    def choose_action(self):
        # Note that the policy does not depends on the state as described in the article
        return np.array([self.theta[0], 1 - self.theta[0]])

    def value_func(self, state):
        return self.theta[0] * (state[0] - state[1]) + self.theta[1]

    def update_policy(self, state, action, reward, next_state):
        delta = reward + self.gamma * self.value_func(next_state) - self.value_func(state)
        gradient = np.array([state[0] - state[1], 1])
#         gradient = np.array(
#             [
#                 (self.theta[0] ** 2 + self.theta[1]) * state[1],
#                 self.theta[1]
#             ]
#         )
        self.e = self.gamma * self.lambda_ * self.e + gradient # * self.value_func(state)
        print(self.e, gradient)
        self.theta += self.alpha * delta * self.e
        self.theta[0] = max(self.action_space[0], self.theta[0])
        self.theta[0] = min(self.action_space[1], self.theta[0])
        return
    
    def generate_episode(self, env, min_length=4):
        # Randomly generate the initial state (for each episode) beginning with the first period to period k−4.
        start = random.randint(0, len(env.data) - min_length)
        end = random.randint(start + min_length, len(env.data))
        return [i for i in range(start, end)]
        
            
    def train(self, episode, env):
        """
        given the current state (st) at the end of the current trading period and action (at), 
        the allocation for next trading period, the reward (rt) is computed at the end of the next trading period based on action (at).
        """
        total_reward = 0
        for i in range(len(episode) - 1):
            action = self.choose_action()
            # epsilon greedy: if probability < epsilon, draw a number in [0, 1]
            if random.random() < self.epsilon:
                theta = random.random()
                action = np.array([theta, 1 - theta])
            state = env.get_continuous_state(index=episode[i])
            next_state = env.get_continuous_state(index=episode[i + 1])
            reward = env.get_reward(action, index=episode[i + 1])
            total_reward += reward
            self.update_policy(state, action, reward, next_state)
        self.reward_trace.append(total_reward / len(episode))
            
        
    def learn(self, env, n_episodes=10, threshold=None, verbose=False):
        for i in range(n_episodes):
            if verbose and not i % 100:
                print(self.theta)
            episode = self.generate_episode(env)
            self.e = np.array([0, 0])
            previous_theta = np.copy(self.theta)
            self.train(episode, env)
            diff = abs(self.theta - previous_theta)
            if threshold:
                # Stop earlier if the difference is smaller than the threshold
                if diff[0] < threshold and diff[1] < threshold:
                    return
        return
