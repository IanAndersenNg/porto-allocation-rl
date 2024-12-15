import pandas as pd
import numpy as np


def preprocess_data():
    datafile = "./data/data1.xlsx"
    agg_etf = pd.read_excel(
        datafile, skiprows=4, usecols=[8, 9], names=["Date", "AGG_Returns"]
    )
    agg_etf["AGG_Returns"] = pd.to_numeric(agg_etf["AGG_Returns"], errors="coerce")
    agg_etf["Date"] = pd.to_datetime(agg_etf["Date"], errors="coerce")
    agg_etf = agg_etf.dropna().reset_index(drop=True)

    em_etf = pd.read_excel(
        datafile, skiprows=4, usecols=[0, 1], names=["Date", "MSCI_Returns"]
    )
    em_etf["MSCI_Returns"] = pd.to_numeric(em_etf["MSCI_Returns"], errors="coerce")
    em_etf["Date"] = pd.to_datetime(em_etf["Date"], errors="coerce")
    em_etf = em_etf.dropna().reset_index(drop=True)

    return pd.merge(agg_etf, em_etf, on="Date", how="inner")


class Environment:
    def __init__(self, data):
        self.data = data
        self.asset_names = [col for col in data.columns if col != "Date"]

    def _get_row(self, date=None, index=None):
        if index is not None and date is not None:
            raise ValueError("Provide either 'index' or 'date', not both.")
        if index is not None:
            return self.data.iloc[index]
        if date is not None:
            return self.data.loc[self.data["Date"] == date]
        raise ValueError("Either 'index' or 'date' must be provided.")

    def get_discrete_state(self, date=None, index=None):
        """
        Returns state representation based on asset returns in string form

        Inputs:
            index (int): Row index
            date (string / timestamp): Row date
        """
        #         row = self._get_row(date = date, index = index)
        #         returns = row.iloc[1:3]
        returns = self.get_continuous_state(date=date, index=index)
        state = "".join(["1" if r > 0 else "0" for r in returns])
        return state

    def get_continuous_state(self, date=None, index=None):
        row = self._get_row(date=date, index=index)
        returns = row.iloc[1:3]
        return returns

    def get_reward(self, action, date=None, index=None):
        """
        Returns log return + Sharpe ratio as the reward

        Inputs:
            action (list of numbers): Portfolio weights for each asset.
            index (int): Row index
            date (string / timestamp): Row date
        """
        #         row = self._get_row(date = date, index = index)
        #         returns = row.iloc[1:]
        returns = self.get_continuous_state(date=date, index=index)
        portfolio_return = np.dot(action, returns)
        # I just found out for the continuous agent, the reward is simply the weighted sum
        return portfolio_return
        log_return = np.log(1 + portfolio_return)

        # assume risk-free rate of 0.02 for Sharpe ratio calculation
        # sharpe_ratio = (log_return - 0.02) / (np.std(returns) if np.std(returns) > 0 else 1)
        # reward = log_return + sharpe_ratio  # TODO - POC only, calculations to be finalized later
        reward = log_return
        return reward
