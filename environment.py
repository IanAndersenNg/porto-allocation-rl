import pandas as pd
import numpy as np

class Environment:
    def __init__(self):
        self.data = self.preprocess_data()

    def preprocess_data(self):
        nasdaq_etf = pd.read_excel("./data/data.xlsx", skiprows=4, usecols=[0, 1], names=["Date", "NASDAQ_Returns"])
        nasdaq_etf["NASDAQ_Returns"] = pd.to_numeric(nasdaq_etf["NASDAQ_Returns"], errors="coerce")
        nasdaq_etf["Date"] = pd.to_datetime(nasdaq_etf["Date"], errors="coerce")
        nasdaq_etf = nasdaq_etf.dropna().reset_index(drop=True)

        em_etf = pd.read_excel("./data/data.xlsx", skiprows=4, usecols=[2, 3], names=["Date", "MSCI_Returns"])
        em_etf["MSCI_Returns"] = pd.to_numeric(em_etf["MSCI_Returns"], errors="coerce")
        em_etf["Date"] = pd.to_datetime(em_etf["Date"], errors="coerce")
        em_etf = em_etf.dropna().reset_index(drop=True)

        return pd.merge(nasdaq_etf, em_etf, on="Date", how="inner")

    def _get_row(self, date = None, index = None):
        if index is not None and date is not None:
            raise ValueError("Provide either 'index' or 'date', not both.")
        if index is not None:
            return self.data.iloc[index]
        if date is not None:
            return self.data.loc[data["Date"] == date]
        raise ValueError("Either 'index' or 'date' must be provided.")

    def get_state(self, date = None, index = None):
        """
        Returns state representation based on asset returns in string form

        Inputs:
            index (int): Row index
            date (string / timestamp): Row date
        """
        row = self._get_row(date = date, index = index)
        returns = row.iloc[1:3]
        state = "".join(["1" if r > 0 else "0" for r in returns])
        return state

    def get_reward(self, action, date = None, index = None):
        """
        Returns log return + Sharpe ratio as the reward
        
        Inputs:
            action (list of numbers): Portfolio weights for each asset.
            index (int): Row index
            date (string / timestamp): Row date
        """
        row = self._get_row(date = date, index = index)
        returns = row.iloc[1:]
        portfolio_return = np.dot(action, returns)
        log_return = np.log(1 + portfolio_return)

        # assume risk-free rate of 0.02 for Sharpe ratio calculation
        sharpe_ratio = (log_return - 0.02) / (np.std(returns) if np.std(returns) > 0 else 1)
        reward = log_return + sharpe_ratio  # TODO - POC only, calculations to be finalized later
        return reward