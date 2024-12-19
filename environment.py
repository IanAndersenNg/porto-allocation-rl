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
    def __init__(self, data, use_sharpe_ratio_reward = False):
        self.data = data
        self.asset_names = [col for col in data.columns if col != "Date"]
        self.use_sharpe_ratio_reward = use_sharpe_ratio_reward
        self.prev_A = 0
        self.prev_B = 0

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
        Returns portfolio return or Sharpe ratio as the reward

        Inputs:
            action (list of numbers): Portfolio weights for each asset.
            index (int): Row index
            date (string / timestamp): Row date
        """
        #         row = self._get_row(date = date, index = index)
        #         returns = row.iloc[1:]
        returns = self.get_continuous_state(date=date, index=index)
        portfolio_return = np.dot(action, returns)
        if not self.use_sharpe_ratio_reward:
            return portfolio_return

        if not self.use_sharpe_ratio_reward:
            return portfolio_return

        # time scale of around 1 decade, following eta value of paper
        eta = 0.1
        A_t = eta * portfolio_return + (1 - eta) * self.prev_A
        B_t = eta * portfolio_return ** 2 + (1 - eta) * self.prev_B

        # deltas A and B are referenced from the paper (equation 3.3)
        delta_A = portfolio_return - self.prev_A
        delta_B = portfolio_return ** 2 - self.prev_B

        dsr_denominator = (self.prev_B - self.prev_A ** 2) ** (3 / 2)
        diff_sharpe_ratio = (self.prev_B * delta_A - 0.5 * self.prev_A * delta_B) / dsr_denominator if dsr_denominator != 0 else 0
        self.prev_A = A_t
        self.prev_B = B_t
        # I just found out for the continuous agent, the reward is simply the weighted sum
        return diff_sharpe_ratio
