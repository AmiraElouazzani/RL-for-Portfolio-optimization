"""
env.py

This module defines a Gym-compatible custom environment for reinforcement learning.
It uses financial graph data as observations (via GNNs) and allows an agent to allocate
a portfolio across multiple assets.

State = Graph (features from return/volatility, adjacency from correlation)
Action = Portfolio weights for each asset (continuous)
Reward = Portfolio return or Sharpe ratio over the time step

Requires:
- precomputed log returns
- dynamic graphs per date
- PyTorch Geometric or DGL for GNNs
"""


import gym
import numpy as np
from gym import spaces

import torch
from torch_geometric.data import Data
import networkx as nx

class FinancialGraphEnv(gym.Env):
    """
    GNN-based financial portfolio environment.
    Each state is a graph observation (PyTorch Geometric Data object).
    """

    def __init__(self, graphs_by_date, log_returns, initial_cash=1.0):
        super(FinancialGraphEnv, self).__init__()

        self.graphs_by_date = graphs_by_date  # dict: date -> (nx_graph, node_features)
        self.dates = sorted(list(graphs_by_date.keys()))
        self.log_returns = log_returns
        self.initial_cash = initial_cash

        self.current_step = 0
        self.n_assets = len(self.log_returns.columns)
        self.cash = initial_cash
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets  # equal weight

        # Action: weight allocation for each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation: PyG graph (not part of gym space system)
        self.observation_space = None  # unused, handled manually

    def _get_observation(self):
        """
        Convert NetworkX graph + features into PyTorch Geometric Data object.
        """
        date = self.dates[self.current_step]
        G, node_features = self.graphs_by_date[date]

        # Map node names to integer indices
        node_list = list(G.nodes)
        node_to_idx = {name: i for i, name in enumerate(node_list)}

        # Convert edges to index pairs
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Ensure node features are in the correct order
        x = torch.tensor(np.array([node_features[node_to_idx[name]] for name in node_list]), dtype=torch.float)

        return Data(x=x, edge_index=edge_index)


    def _get_reward(self):
        """
        Compute portfolio return from previous step to current step.
        """
        current_date = self.dates[self.current_step]
        prev_date = self.dates[self.current_step - 1]

        # Daily returns for each asset between prev and current
        asset_returns = self.log_returns.loc[current_date]

        portfolio_return = np.dot(self.portfolio_weights, asset_returns)
        return portfolio_return

    def reset(self):
        self.current_step = 1  # skip step 0 (no previous return)
        self.cash = self.initial_cash
        self.portfolio_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_observation()

    def step(self, action):
        """
        Step through environment with portfolio allocation (action).
        """
        # Normalize weights
        action = np.clip(action, 0, 1)
        action /= action.sum() + 1e-8
        self.portfolio_weights = action

        reward = self._get_reward()

        self.current_step += 1
        done = self.current_step >= len(self.dates)

        obs = self._get_observation() if not done else None
        info = {"date": self.dates[self.current_step - 1]}

        return obs, reward, done, info



# obs = env.reset()

# for _ in range(10):
#     action = np.ones(env.n_assets) / env.n_assets  # equal allocation
#     obs, reward, done, info = env.step(action)
#     print(f"{info['date']} | reward: {reward:.4f}")
#     if done:
#         break
