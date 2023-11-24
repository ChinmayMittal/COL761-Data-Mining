import torch
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn.pool import *
import torch.nn.init as init
from encoder import NodeEncoder, EdgeEncoder
from enum import Enum, auto

class GNN_TYPE(Enum):
    GCN = auto()
    GGNN = auto()
    GIN = auto()
    GAT = auto()
    SAGE = auto()

    @staticmethod
    def from_string(s):
        try:
            return GNN_TYPE[s]
        except KeyError:
            raise ValueError()

    def get_layer(self, in_dim, out_dim):
        if self is GNN_TYPE.GCN:
            return GCNConv(
                in_channels=in_dim,
                out_channels=out_dim, add_self_loops=False)
        elif self is GNN_TYPE.GIN:
            return GINConv(torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.BatchNorm1d(out_dim), torch.nn.ReLU(),
                                         torch.nn.Linear(out_dim, out_dim), torch.nn.BatchNorm1d(out_dim), torch.nn.ReLU()))
        elif self is GNN_TYPE.GAT:
            # 4-heads, although the paper by Velickovic et al. had used 6-8 heads.
            # The output will be the concatenation of the heads, yielding a vector of size out_dim
            num_heads = 4
            return GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        elif self is GNN_TYPE.SAGE:
            return SAGEConv(in_dim, out_dim, aggr='mean')

class GNNRegressor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, in_channels, gnn_type, num_layers, use_norm_layers):
        super().__init__()
        self.gnn_type = gnn_type
        self.use_norm_layers = use_norm_layers
        self.layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        self.edge_linear = torch.nn.Linear(in_channels, 1)
        self.dropout = torch.nn.Dropout(p = 0.5)

        self.linear = torch.nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            self.layers.append(gnn_type.get_layer(
                    in_dim=hidden_channels,
                    out_dim=hidden_channels))
            if use_norm_layers:
                self.norm_layers.append(torch.nn.LayerNorm(hidden_channels))
        
        self.linear2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.num_layers = num_layers
        self.node_encoder = NodeEncoder(in_channels)
        self.edge_encoder = EdgeEncoder(in_channels)

        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.fc2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.fc3 = torch.nn.Linear(32, out_channels)

        self.edge_embeddings = torch.nn.Embedding(60, in_channels)

        init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.edge_linear.weight)

    def forward(self, x, edge_index, edge_attr, batch, size):
        x = self.node_encoder(x.to(torch.long))
        enc_attr = self.edge_encoder(edge_attr.to(torch.long))
        temp = self.edge_linear(enc_attr)
        temp = torch.sigmoid(temp)
        x = self.linear(x)
        x = torch.relu(x)
        for i in range(self.num_layers):
            if self.gnn_type == GNN_TYPE.GAT or self.gnn_type == GNN_TYPE.GCN:
                new_x = self.layers[i](x, edge_index, temp).relu()
            else:
                new_x = self.layers[i](x, edge_index).relu()
            x = new_x+x
            if self.use_norm_layers:
                x = self.norm_layers[i](x)
        # x = self.linear2(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        x = global_add_pool(x, batch)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        


        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.fc3(x)

        # x = F.sigmoid(x)
        # print(x)
        # x = x[root_mask, :]
        return x