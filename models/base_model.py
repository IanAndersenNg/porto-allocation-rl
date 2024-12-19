import random


class Model:
    def __init__(self, state_space, action_space, gamma=0.99, epsilon=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = self.initialize_policy()

    def initialize_policy(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def choose_action(self, state):
        raise NotImplementedError("This method should be overridden by subclasses")

    def update_policy(self, state, action, reward, next_state):
        raise NotImplementedError("This method should be overridden by subclasses")

    def train(self, episodes, env):
        raise NotImplementedError("This method should be overridden by subclasses")

    def test(self, env):
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_episode(self, env, min_length=4):
        # Randomly generate the initial state (for each episode) beginning with the first period to period k−4.
        random.seed(9)
        start = random.randint(0, len(env.data) - min_length)
        end = random.randint(start + min_length, len(env.data))
        return [i for i in range(start, end)]
