from environment import Environment, preprocess_data
from datetime import datetime
from models.base_model import Model
from models.montecarlo import MonteCarlo

data = preprocess_data()
actions = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
monte_carlo = MonteCarlo(data, actions, epsilon=0.4)

train_env = Environment(
    data[data["Date"] < datetime.strptime("2020-01-01", "%Y-%m-%d")])

test_env = Environment(
    data[data["Date"] >= datetime.strptime("2020-01-01", "%Y-%m-%d")])

optimum_action_dict = monte_carlo.train(10, train_env)
monte_carlo_actions = monte_carlo.test(optimum_action_dict, test_env)

print("Optimum state action dict : ", optimum_action_dict)
print("Monte Carlo actions: ", monte_carlo_actions[0:10])
print("Monte Carlo actions length: ", len(monte_carlo_actions))