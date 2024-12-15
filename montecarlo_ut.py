import unittest
import numpy as np
from models.montecarlo import MonteCarlo
from environment import Environment
import pandas as pd
from datetime import datetime

class TestMonteCarlo(unittest.TestCase):

    def setUp(self):
        mock_data = pd.DataFrame({
            "Date": pd.date_range(start="2020-01-01", periods=10, freq="D"),
            "MSCI_Returns": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0],
            "AGG_Returns": [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]
        })
        
        self.actions = [(0, 100), (25, 75), (50, 50), (75, 25), (100, 0)]
        self.monte_carlo = MonteCarlo(mock_data, self.actions, epsilon=0.1)
        self.train_env = Environment(mock_data)

    def test_train(self):
        optimum_action_dict = self.monte_carlo.train(10, self.train_env)
        expected_optimum_action_dict = {
            '11': 0, '10': 1, '01': 2, '00': 3
        }

        self.assertEqual(
            optimum_action_dict, expected_optimum_action_dict,
            "The optimum action dictionary does not match the expected result."
        )

if __name__ == "__main__":
    unittest.main()
