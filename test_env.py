"""
test_env.py

Tests the FinancialGraphEnv by stepping through it using equal weights or random allocations.
Also plots weights and rewards over time for visualization.
"""

import numpy as np
import pandas as pd
from rl.env import FinancialGraphEnv
from grapholio.data_loader import download_price_data, compute_log_returns
from grapholio.graph_builder import compute_rolling_correlation, build_graph
from grapholio.features import compute_financial_positions

import matplotlib.pyplot as plt

# 1. Setup data
tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
prices = download_price_data(tickers, start="2018-01-01", end="2019-01-01")
log_returns = compute_log_returns(prices)
rolling_corrs = compute_rolling_correlation(log_returns, window=30)

# 2. Build graph snapshots with node features
dates = pd.date_range("2018-03-01", "2018-04-01").strftime('%Y-%m-%d')
graphs_by_date = {}

for date in dates:
    try:
        G = build_graph(rolling_corrs.loc[date], threshold=0.5)
        pos = compute_financial_positions(log_returns, date)
        node_features = np.array([pos[t] for t in G.nodes()])
        graphs_by_date[date] = (G, node_features)
    except:
        continue

# 3. Initialize environment
env = FinancialGraphEnv(graphs_by_date, log_returns)

# 4. Run test loop
obs = env.reset()
rewards = []
weight_history = []

for _ in range(len(graphs_by_date) - 1):
    # Use equal weights or random weights
    
    # action = np.ones(env.n_assets) / env.n_assets
    action = np.random.dirichlet(np.ones(env.n_assets))  # test random

    obs, reward, done, info = env.step(action)

    rewards.append(reward)
    weight_history.append(env.portfolio_weights.copy())

    print(f"{info['date']} | Reward: {reward:.5f} | Weights: {env.portfolio_weights}")

    if done:
        break

# 5. Plot weights over time
weight_history = np.array(weight_history)
plt.figure(figsize=(10, 5))
for i, ticker in enumerate(tickers):
    plt.plot(weight_history[:, i], label=ticker)
plt.title("Portfolio Weights Over Time")
plt.xlabel("Time Step")
plt.ylabel("Weight")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Plot cumulative return
cumulative_return = np.cumsum(rewards)
plt.figure(figsize=(10, 4))
plt.plot(cumulative_return, color='green')
plt.title("Cumulative Portfolio Return")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()
