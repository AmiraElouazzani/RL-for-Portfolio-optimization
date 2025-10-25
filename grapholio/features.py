"""
This module handles financial feature extraction for visualization:
- Computes node positions using average return and volatility (std dev)
"""


import pandas as pd

def compute_financial_positions(log_returns, date, window=30):
    """
    Computes 2D positions for each asset using financial features:
    - X = average return over window
    - Y = volatility (standard deviation) over window

    Args:
        log_returns (pd.DataFrame): Log return time series.
        date (str): Date (YYYY-MM-DD) to end the window on.
        window (int): Number of days in the rolling window.

    Returns:
        dict: Dictionary mapping tickers to (x, y) positions.
    """
    # Get trailing window of returns up to 'date'
    window_data = log_returns.loc[:date].tail(window)

    # Compute mean and std dev for each asset
    avg_returns = window_data.mean()
    volatility = window_data.std()

    # Create position dictionary
    pos = {}
    for ticker in avg_returns.index:
        x = avg_returns[ticker]
        y = volatility[ticker]
        pos[ticker] = (x, y)
    
    return pos

# tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
# prices = download_price_data(tickers)
# log_returns = compute_log_returns(prices)
# pos = compute_financial_positions(log_returns, date="2019-01-02", window=30)