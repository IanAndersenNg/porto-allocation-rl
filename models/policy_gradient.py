import torch
import pandas as pd
import numpy as np
from datetime import datetime
from environment import Environment, preprocess_data
from models.base_model import Model
import argparse

# Using a neural network to learn our policy parameters for one continuous action


class PolicyNetwork(torch.nn.Module):
    # Takes in observations and outputs actions mu and sigma
    def __init__(self, observation_space, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.input_layer = torch.nn.Linear(observation_space, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 2)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)

        x = torch.nn.functional.relu(x)
        # actions
        action_parameters = self.output_layer(x)
        action_parameters = torch.sigmoid(action_parameters)
        return action_parameters


class PolicyGradient(Model):
    name = "policy_gradient"

    def __init__(
        self, state_space, action_space, model_hidden_size, gamma=0.9, alpha=0.1
    ):
        self.hidden_size = model_hidden_size
        super().__init__(state_space, action_space, gamma)
        self.alpha = alpha
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.reward_trace = [0]

    def initialize_policy(self):
        self.model = PolicyNetwork(len(self.state_space), self.hidden_size)

    def choose_action(self, state):
        """Selects an action given state
        Return:
        - action.item() (float): continuous action
        - log_action (float): log of probability density of action

        """

        # create state tensor
        state_tensor = torch.Tensor(state).unsqueeze(0)
        state_tensor.required_grad = True

        # forward pass through network
        action_parameters = self.model(state_tensor)

        # get mean and std, get normal distribution
        mu, sigma = action_parameters[:, :1], torch.exp(action_parameters[:, 1:])
        m = torch.distributions.Normal(mu[:, 0], sigma[:, 0])

        # sample action, get log probability
        action = m.sample()
        log_action = m.log_prob(action)
        action = torch.sigmoid(action)  # squeeze it into [0, 1]
        theta = action.item()
        return (theta, 1 - theta), log_action

    def update_policy(self, log_proba_actions, rewards):
        loss = []
        for r, log_proba in zip(rewards, log_proba_actions):
            loss.append(-r * log_proba)

        # Backpropagation
        self.optimizer.zero_grad()
        sum(loss).backward()
        self.optimizer.step()

    def process_rewards(self, rewards):
        """Converts our rewards history into cumulative discounted rewards
        Args:
        - rewards (Array): array of rewards

        Returns:
        - G (Array): array of cumulative discounted rewards
        """
        # Calculate Gt (cumulative discounted rewards)
        G = []

        # track cumulative reward
        total_r = 0

        # iterate rewards from Gt to G0
        for r in reversed(rewards):

            # Base case: G(T) = r(T)
            # Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
            total_r = r + total_r * self.gamma

            # add to front of G
            G.insert(0, total_r)

        # whitening rewards
        G = torch.tensor(G)
        G = (G - G.mean()) / G.std()

        return G

    def learn(self, episode, env):
        """
        Given the current state (st) at the end of the current trading period and action (at),
        the allocation for next trading period, the reward (rt) is computed at the end of the next trading period based on action (at).
        """
        total_reward = 0
        rewards = []
        log_proba_actions = []
        # generate trajectory
        for i in range(len(episode) - 1):
            # print(episode[i])
            # print("a:", action)
            state = env.get_continuous_state(index=episode[i]).astype(float)
            #             print(state.values)
            action, log_proba = self.choose_action(state.values)
            reward = env.get_reward(action, index=episode[i + 1])
            total_reward += reward
            rewards.append(reward)
            log_proba_actions.append(log_proba)
        rewards = self.process_rewards(rewards)

        self.update_policy(rewards, log_proba_actions)
        self.reward_trace.append(total_reward / len(episode))

    def train(self, env, n_episodes=10, verbose_freq=None):
        for i in range(n_episodes):
            if verbose_freq and not i % verbose_freq:
                print(f"Episode {i} / {n_episodes}, Reward:", self.reward_trace[-1])
            episode = self.generate_episode(env)

            # self.e = np.array([0, 0])
            self.learn(episode, env)
        return

    def test(self, env):
        actions = []
        # Choose action every day in testing dataset
        for i in range(len(env.data)):
            state = env.get_continuous_state(index=i).astype(float)
            action, _ = self.choose_action(np.array(state.values))
            actions.append(action)

        result = env.data[["Date"]].reset_index(drop=True)
        result = pd.concat(
            [result, pd.DataFrame(actions, columns=env.asset_names)], axis=1
        )
        result[env.asset_names] = result[env.asset_names].shift()
        return result.dropna().reset_index(drop=True)


if __name__ == "__main__":  # python3 -m models.gradientTD
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsr",
        action="store_true",
        help="Sets the reward function to be differential sharpe ratio",
    )
    dsr_reward = parser.parse_args().dsr
    data = preprocess_data()
    data[["AGG_Returns", "MSCI_Returns"]] = (
        data[["AGG_Returns", "MSCI_Returns"]] / 100
    )  # rescale for pytoch linear
    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")],
        use_sharpe_ratio_reward=dsr_reward,
    )

    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2019-12-31", "%Y-%m-%d")],
        use_sharpe_ratio_reward=dsr_reward,
    )
    PG = PolicyGradient(
        state_space=np.array([-np.inf, np.inf]),
        action_space=[0, 1],
        model_hidden_size=8,
        alpha=0.1,
        gamma=0.9,
    )
    PG.train(train_env, 100, verbose_freq=10)
    # plt.plot(GTD.reward_trace)
    result = PG.test(test_env)
    print(result)
