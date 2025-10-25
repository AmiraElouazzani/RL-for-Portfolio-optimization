"""
test_agent.py

Tests the GNNPolicy on real graph data.
- Loads some graph snapshots
- Feeds them into the GNN
- Prints the resulting portfolio weights
"""

import torch
import numpy as np
import pandas as pd

from grapholio.data_loader import download_price_data, compute_log_returns
from grapholio.graph_builder import compute_rolling_correlation, build_graph
from grapholio.features import compute_financial_positions
from rl.agent import GNNPolicy

from torch_geometric.data import Data

# 1. Load and preprocess data
tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ']
prices = download_price_data(tickers, start="2018-01-01", end="2019-01-01")
log_returns = compute_log_returns(prices)
rolling_corrs = compute_rolling_correlation(log_returns, window=30)

# 2. Pick some test dates
dates = pd.date_range("2018-03-01", "2018-03-10").strftime('%Y-%m-%d')
graphs_by_date = {}

for date in dates:
    try:
        G = build_graph(rolling_corrs.loc[date], threshold=0.5)
        pos = compute_financial_positions(log_returns, date)
        node_features = np.array([pos[t] for t in G.nodes()])
        graphs_by_date[date] = (G, node_features)
    except:
        continue

# 3. Create and load GNNPolicy
input_dim = 2            # [return, volatility]
hidden_dim = 64
num_assets = len(tickers)

policy = GNNPolicy(input_dim, hidden_dim, num_assets)

# 4. Feed a sample graph to the policy
for date, (G, node_features) in graphs_by_date.items():
    # Map nodes to indices
    node_list = list(G.nodes)
    node_to_idx = {name: i for i, name in enumerate(node_list)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Sort features to match index
    x = torch.tensor(np.array([node_features[node_to_idx[name]] for name in node_list]), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    # Forward pass through GNN
    with torch.no_grad():
        weights = policy(data)

    print(f"\n📅 {date}")
    for t, w in zip(tickers, weights[0]):
        print(f"{t}: {w.item():.4f}")
