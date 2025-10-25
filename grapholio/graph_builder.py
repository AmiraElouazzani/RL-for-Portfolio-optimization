"""
This module builds graph structures from financial data:
- Computes rolling correlation matrices
- Builds undirected weighted graphs from correlations
"""

import pandas as pd
import networkx as nx

def compute_rolling_correlation(log_returns, window=30):
    """
    Computes rolling correlation matrices over a time window.

    Args:
        log_returns (pd.DataFrame): Log return time series.
        window (int): Rolling window size (days).

    Returns:
        pd.core.groupby.GroupBy: MultiIndex correlation matrices (date, ticker1, ticker2).
    """
    return log_returns.rolling(window=window).corr()

def build_graph(corr_matrix, threshold=0.6):
    """
    Builds an undirected correlation graph from a correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix for one date.
        threshold (float): Correlation threshold for adding edges.

    Returns:
        networkx.Graph: Graph with tickers as nodes and edges weighted by correlation.
    """
    G = nx.Graph()

    # Add all nodes (to keep unconnected nodes visible)
    for ticker in corr_matrix.columns:
        G.add_node(ticker)

    # Add edges based on correlation threshold
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] > threshold:
                G.add_edge(i, j, weight=corr_matrix.loc[i, j])

    return G

# tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
# prices = download_price_data(tickers)
# log_returns = compute_log_returns(prices)
# rolling_corrs = compute_rolling_correlation(log_returns, window=30)
# corr_matrix = rolling_corrs.loc["2019-01-02"]
# G = build_graph(corr_matrix, threshold=0.6)