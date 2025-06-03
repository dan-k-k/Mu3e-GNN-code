# defineGNNmodel.py

import copy
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, GCNConv, GINEConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, classification_report, accuracy_score, confusion_matrix
import numpy as np

# Define a GCN model that also uses extra graph-level features.
class GCNMultiClass(nn.Module):
    def __init__(self, in_channels, hidden_channels, extra_features):
        super().__init__()
        # Edge network for the first NNConv layer.
        self.edge_nn = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels * hidden_channels)
        )
        self.conv1 = NNConv(in_channels, hidden_channels, self.edge_nn, aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # Edge network for the second NNConv layer.
        self.edge_nn2 = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_channels * hidden_channels)
        )
        self.conv2 = NNConv(hidden_channels, hidden_channels, self.edge_nn2, aggr='mean')
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.dropout = nn.Dropout(0.2)
        # The final layer now outputs 3 logits (one for each class).
        self.fc = nn.Linear(hidden_channels + extra_features, 3)
    
    def forward(self, data):
        # Extract features from nodes, edges, and the batch assignment.
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # If edge_attr is missing, create a dummy tensor.
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            edge_attr = torch.ones((edge_index.size(1), 1), device=x.device, dtype=torch.float)
            
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling over node features.
        pooled = global_mean_pool(x, batch)
        # Get extra graph-level features (e.g. normalized helix parameters).
        extra = data.graph_attr.to(x.device)
        combined = torch.cat([pooled, extra], dim=1)
        # The output is now of size [batch_size, 3]
        return self.fc(combined)


# Define a GINE model that also uses extra graph-level features.
class GINEMultiClass(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_channels, extra_features):
        """
        in_channels    : # dims of raw node features (here 11)
        hidden_channels: # hidden dims for GINE (here 64)
        edge_channels  : # dims of raw edge features (here 5)
        extra_features : # dims of your graph_attr (here 10)
        """
        super().__init__()

        # 1a) A small linear to lift raw node‐features → hidden_channels
        self.lin_node = nn.Linear(in_channels, hidden_channels)

        # 1b) A small linear to lift raw edge‐features → hidden_channels
        self.lin_edge = nn.Linear(edge_channels, hidden_channels)

        # 2) First GINEConv: message‐network (MLP) that maps "[hidden + hidden] → hidden"
        self.gine_nn1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINEConv(self.gine_nn1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        # 3) Second GINEConv: same structure
        self.gine_nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv2 = GINEConv(self.gine_nn2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # 4) Dropout between layers
        self.dropout = nn.Dropout(0.2)

        # 5) Final fully‐connected head: (hidden + extra_features) → 3 logits
        self.fc = nn.Linear(hidden_channels + extra_features, 3)
    
    def forward(self, data):
        """
        data.x         : [total_num_nodes, in_channels=11]
        data.edge_index: [2, total_num_edges]
        data.edge_attr : [total_num_edges, edge_channels=5]
        data.batch     : [total_num_nodes] indicating graph‐membership
        data.graph_attr: [batch_size, extra_features=10]
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        batch     = data.batch

        x = self.lin_node(x)            # → [N, hidden_channels]
        x = F.relu(x)

        if edge_attr is None:
            # In case no edge_attr was stored, we create a dummy zero‐tensor
            device = x.device
            edge_attr = torch.zeros((edge_index.size(1), self.lin_edge.in_features),
                                    device=device, dtype=torch.float)
        edge_emb = self.lin_edge(edge_attr)  # → [E, hidden_channels]
        edge_emb = F.relu(edge_emb)

        x = self.conv1(x, edge_index, edge_emb)  # → [N, hidden_channels]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # We must re‐use the same edge_emb each time, because GINEConv
        # expects node‐ and edge‐features to share the same dimensionality.
        x = self.conv2(x, edge_index, edge_emb)  # → [N, hidden_channels]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        pooled = global_mean_pool(x, batch)  # → [batch_size, hidden_channels]

        extra = data.graph_attr.to(pooled.device)  # → [batch_size, extra_features]
        combined = torch.cat([pooled, extra], dim=1)  # → [batch_size, hidden + extra]

        return self.fc(combined)