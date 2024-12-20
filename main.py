from environment import Environment, preprocess_data
from datetime import datetime
from models.base_model import Model
from models.montecarlo import MonteCarlo

data = preprocess_data()
actions = [(0, 1), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1, 0)]
monte_carlo = MonteCarlo(actions, epsilon=0.3)

train_env = Environment(
    data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")])

test_env = Environment(
    data[data["Date"] >= datetime.strptime("2020-01-01", "%Y-%m-%d")])

optimum_action_dict = monte_carlo.train(1000, train_env)
monte_carlo_actions = monte_carlo.test(test_env)

print("Optimum state action dict : ", optimum_action_dict)
print("Monte Carlo actions: ", monte_carlo_actions[0:10])
print("Monte Carlo actions length: ", len(monte_carlo_actions))