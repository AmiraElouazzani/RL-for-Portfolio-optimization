"""
This module handles all data loading and basic preprocessing:
- Downloads stock price data from Yahoo Finance using yfinance
- Computes log returns for use in correlation and analysis
"""

import yfinance as yf
import numpy as np
import pandas as pd

def download_price_data(tickers, start="2015-01-01", end="2020-01-01"):
    """
    Downloads adjusted close price data from Yahoo Finance.
    
    Args:
        tickers (list of str): Stock tickers.
        start (str): Start date in 'YYYY-MM-DD'.
        end (str): End date in 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Adjusted close price data.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    return data['Adj Close']

def compute_log_returns(price_data):
    """
    Computes log returns from price data.

    Args:
        price_data (pd.DataFrame): Adjusted close price data.

    Returns:
        pd.DataFrame: Log return time series.
    """
    return np.log(price_data / price_data.shift(1)).dropna()


# tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
# prices = download_price_data(tickers)
# log_returns = compute_log_returns(prices)