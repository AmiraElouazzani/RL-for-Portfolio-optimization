"""
Defines a PyTorch GNN-based policy network that maps a graph observation
to a portfolio allocation (i.e., action vector).

Includes:
- GNN encoder (GraphConv)
- Global pooling to combine node features
- MLP to output action logits (weights)

Classes:
- GNNPolicy: maps graph → portfolio weights (softmax)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool

class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_assets):
        """
        Args:
            input_dim (int): Dimension of node features (e.g., 2 for [return, vol])
            hidden_dim (int): Hidden layer size in GNN
            num_assets (int): Output size = number of assets (portfolio weights)
        """
        super(GNNPolicy, self).__init__()

        # GNN encoder
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # Policy head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_assets)

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): Graph observation with .x and .edge_index

        Returns:
            torch.Tensor: Softmax-normalized portfolio weights (size: [num_assets])
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pooling across nodes to get graph-level embedding
        pooled = global_mean_pool(x, batch)

        # Policy head
        out = F.relu(self.fc1(pooled))
        logits = self.fc2(out)

        # Portfolio weights = softmax over assets
        weights = F.softmax(logits, dim=-1)

        return weights
