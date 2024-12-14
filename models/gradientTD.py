from datetime import datetime
import random
import numpy as np
import torch
from environment import Environment, preprocess_data
from models.Model import Model
# import sys

# sys.path.append("../")


# import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, state):
        return self.linear(torch.Tensor([[state]]))


class GradientTD(Model):
    def __init__(
        self, state_space, action_space, gamma=0.9, lambda_=0.9, alpha=0.1, epsilon=0.01
    ):
        super().__init__(state_space, action_space, gamma)
        self.lambda_ = lambda_
        self.alpha = alpha
        self.epsilon = epsilon
        # self.e = np.array([0, 0])
        self.reward_trace = []
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)

    def initialize_policy(self):
        self.model = Net()

    def choose_action(self):
        # Note that the policy does not depends on the state as described in the paper
        theta = self.model.linear.weight.data.detach().numpy()[0][0]
        return np.array([theta, 1 - theta])

    def value_func(self, state):
        return self.model(state[0] - state[1])
        # return self.theta[0] * (state[0] - state[1]) + self.theta[1]

    def update_policy(self, state, action, reward, next_state):
        # delta = reward + self.gamma * self.value_func(next_state) - self.value_func(state)
        target = torch.Tensor([reward]) + self.gamma * self.value_func(next_state)
        # Use pytorch to calculate gradient
        loss = torch.nn.MSELoss()
        delta = loss(target, self.value_func(state))
        self.optimizer.zero_grad()
        delta.backward()
        # print(self.model.linear.weight.grad, self.model.linear.bias.grad)
        self.optimizer.step()
        self.model.linear.weight.data.clamp_(self.action_space[0], self.action_space[1])
        return

    def generate_episode(self, env, min_length=4):
        # Randomly generate the initial state (for each episode) beginning with the first period to period k−4.
        start = random.randint(0, len(env.data) - min_length)
        end = random.randint(start + min_length, len(env.data))
        return [i for i in range(start, end)]

    def train(self, episode, env):
        """
        Given the current state (st) at the end of the current trading period and action (at),
        the allocation for next trading period, the reward (rt) is computed at the end of the next trading period based on action (at).
        """
        total_reward = 0
        for i in range(len(episode) - 1):
            # print(episode[i])
            action = self.choose_action()
            # epsilon greedy: if probability < epsilon, draw a number in [0, 1]
            if random.random() < self.epsilon:
                theta = random.random()
                action = np.array([theta, 1 - theta])
            # print("a:", action)
            state = env.get_continuous_state(index=episode[i])
            next_state = env.get_continuous_state(index=episode[i + 1])
            reward = env.get_reward(action, index=episode[i + 1])
            total_reward += reward

            self.update_policy(state, action, reward, next_state)
        self.reward_trace.append(total_reward / len(episode))

    def learn(self, env, n_episodes=10, verbose_freq=None):
        for i in range(n_episodes):
            if verbose_freq and not i % verbose_freq:
                # print(self.theta)
                print(
                    "weight:",
                    self.model.linear.weight.item(),
                    "bias:",
                    self.model.linear.bias.item(),
                )
            episode = self.generate_episode(env)
            # self.e = np.array([0, 0])
            self.train(episode, env)
        return

    def test(self, env):
        action = self.choose_action()
        result = env.data[["Date"]]
        # Not sure if this is the right way to calcuate return
        result["Return"] = (
            action[0] * env.data[env.asset_names[0]]
            + action[1] * env.data[env.asset_names[1]]
        )
        return result


if __name__ == "__main__":  # python3 -m models.gradientTD
    data = preprocess_data()
    data[["AGG_Returns", "MSCI_Returns"]] = (
        data[["AGG_Returns", "MSCI_Returns"]] / 100
    )  # rescale for pytoch linear
    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")]
    )

    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2020-01-01", "%Y-%m-%d")]
    )
    GTD = GradientTD(
        state_space=np.array([-np.inf, np.inf]),
        action_space=[0, 1],
        lambda_=0.01,
        alpha=0.1,
        gamma=0.9,
    )
    GTD.learn(train_env, 100, verbose_freq=10)
    print("trained action:", GTD.choose_action())
    # plt.plot(GTD.reward_trace)
    result = GTD.test(test_env)
    print(result)