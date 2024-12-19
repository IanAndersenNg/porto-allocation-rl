import numpy as np
from datetime import datetime
from models.q_learning import Q_learning
from environment import Environment, preprocess_data


class SARSA(Q_learning):
    def update_policy(self, state, action, reward, next_state):
        state_index = self.state_indices[state]
        next_state_index = self.state_indices[next_state]
        action_index = self.action_indices[action]
        # epsilon greedy
        if np.random.rand() < self.exploration_rate:
            next_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            # Choose an action based on the current state index
            next_action = self.choose_action(state)
        next_action_index = self.action_indices[next_action]

        current_q = self.q_table[state_index][action_index]
        self.q_table[state_index][action_index] = current_q + self.learning_rate * (
            reward
            + self.discount_factor * self.q_table[next_state_index][next_action_index]
            - current_q
        )


if __name__ == "__main__":

    data = preprocess_data()
    data[["AGG_Returns", "MSCI_Returns"]] = data[["AGG_Returns", "MSCI_Returns"]]

    # train and test environment
    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")]
    )
    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2020-01-01", "%Y-%m-%d")]
    )

    # define space
    state_space = ["11", "10", "01", "00"]
    action_space = [(0, 1), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (1, 0)]

    # parameters
    num_episodes = 1000
    learning_rate = 0.01
    discount_factor = 0.99
    exploration_rate = 0.1
    min_exploration_rate = 0.01

    # create model
    sarsa_model = SARSA(
        state_space=state_space,
        action_space=action_space,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
    )

    # train model
    sarsa_model.learn(train_env, n_episodes=num_episodes, verbose_freq=100)

    # output q-table
    print("Q-table after training:")
    print(sarsa_model.df_q_table)
    result = sarsa_model.test(test_env)
    print(result)
