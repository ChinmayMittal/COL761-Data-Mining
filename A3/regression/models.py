import torch
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn.pool import *
import torch.nn.init as init
from encoder import NodeEncoder, EdgeEncoder

class GNNRegressor(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = GATConv(emb_dim, hidden_channels//2, heads = 2)
        self.conv2 = GATConv(hidden_channels, hidden_channels//2, heads = 2)
        self.edge_linear = torch.nn.Linear(emb_dim, 1)
        self.dropout = torch.nn.Dropout(p = 0.5)

        self.fc1 = torch.nn.Linear(hidden_channels, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.fc2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)

        self.fc3 = torch.nn.Linear(32, 16)
        self.bn3 = torch.nn.BatchNorm1d(16)

        self.fc4 = torch.nn.Linear(16, out_channels)
        self.node_encoder = NodeEncoder(emb_dim)
        self.edge_encoder = EdgeEncoder(emb_dim)

        #Initializations

        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.edge_linear.weight)

    def forward(self, x, edge_index, edge_attr, batch, size):
        x = self.node_encoder(x.to(torch.long))
        edge_attr = self.edge_encoder(edge_attr.to(torch.long))
        temp = self.edge_linear(edge_attr)
        temp = torch.relu(temp)
        
        x = self.conv1(x, edge_index, temp).relu()
        x = self.conv2(x, edge_index, temp).relu()
        x = global_mean_pool(x, batch)
        # print(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)


        return x