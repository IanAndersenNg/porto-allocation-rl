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