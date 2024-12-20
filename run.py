import argparse
import numpy as np
from models import Q_learning, SARSA, GradientTD, MonteCarlo
from datetime import datetime
from environment import Environment, preprocess_data

result_folder = "results"


def main(algo, n_episodes, dsr):
    print("Run", algo.name)
    result_file_name = f"{algo.name}_dsr.csv" if dsr else f"{algo.name}.csv"
    data = preprocess_data()
    data[["AGG_Returns", "MSCI_Returns"]] = data[["AGG_Returns", "MSCI_Returns"]] / 100

    # train and test environment
    train_env = Environment(
        data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")],
        use_sharpe_ratio_reward=dsr_reward,
    )
    test_env = Environment(
        data[data["Date"] >= datetime.strptime("2019-12-31", "%Y-%m-%d")],
        use_sharpe_ratio_reward=dsr_reward,
    )
    algo.train(train_env, n_episodes=n_episodes, verbose_freq=n_episodes // 10)
    algo.test(train_env).to_csv(f"{result_folder}/train_{result_file_name}", index=False)
    result = algo.test(test_env)
    result.to_csv(f"{result_folder}/{result_file_name}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dsr",
        action="store_true",
        help="Sets the reward function to be differential sharpe ratio",
    )
    dsr_reward = parser.parse_args().dsr
    n_episodes = 1000
    discrete_state_space = ["11", "10", "01", "00"]
    discrete_action_space = [(0, 1), (0.25, 0.75), (0.50, 0.50), (0.75, 0.25), (1, 0)]
    continuous_state_space = np.array([-np.inf, np.inf])
    continuous_action_space = [0, 1]
    q_learning_model = Q_learning(
        state_space=discrete_state_space,
        action_space=discrete_action_space,
        learning_rate=0.01,
        discount_factor=0.99,
        exploration_rate=0.1,
    )

    sarsa_model = SARSA(
        state_space=discrete_state_space,
        action_space=discrete_action_space,
        learning_rate=0.01,
        discount_factor=0.99,
        exploration_rate=0.1,
    )
    gtd_model = GradientTD(
        state_space=continuous_state_space,
        action_space=continuous_action_space,
        lambda_=0.01,
        alpha=0.1,
        gamma=0.9,
    )

    monte_carlo_model = MonteCarlo(
        discrete_action_space,
        epsilon=0.3
    )

    main(q_learning_model, n_episodes, dsr_reward)
    main(sarsa_model, n_episodes, dsr_reward)
    main(gtd_model, n_episodes, dsr_reward)
    main(monte_carlo_model, n_episodes, dsr_reward)
