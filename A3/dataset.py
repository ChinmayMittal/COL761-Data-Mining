import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from evaluate import Evaluator


class GraphDataset(Dataset):
    
    def __init__(self, dataset_path):
        super().__init__()        
        self.dataset_path = dataset_path
        
        dataset_edge_features = np.loadtxt(os.path.join(dataset_path, "edge_features.csv.gz"), dtype=float, delimiter=',')
        dataset_edges = np.loadtxt(os.path.join(dataset_path, "edges.csv.gz"), dtype=int, delimiter=',')
        dataset_graph_labels = np.loadtxt(os.path.join(dataset_path, "graph_labels.csv.gz"), dtype=float, delimiter=',')
        dataset_node_features = np.loadtxt(os.path.join(dataset_path, "node_features.csv.gz"), dtype=float, delimiter=',')
        dataset_num_nodes = np.loadtxt(os.path.join(dataset_path, "num_nodes.csv.gz"), dtype=int, delimiter=',')
        dataset_num_edges = np.loadtxt(os.path.join(dataset_path, "num_edges.csv.gz"), dtype=int, delimiter=',')

        num_graphs = dataset_num_nodes.size
        self.graphs = []
        self.data = []

        self.nan_cnt = 0
        self.one_cnt = 0
        self.zero_cnt = 0
        for i in range(0, num_graphs):
            num_node = dataset_num_nodes[i]
            num_edge = dataset_num_edges[i]
            edges = dataset_edges[np.sum(dataset_num_edges[:i]) : np.sum(dataset_num_edges[:i+1]), :]
            edge_features = torch.FloatTensor(dataset_edge_features[np.sum(dataset_num_edges[:i]) : np.sum(dataset_num_edges[:i+1]), :])
            node_features = torch.FloatTensor(dataset_node_features[np.sum(dataset_num_nodes[:i]) : np.sum(dataset_num_nodes[:i+1]), :])
            node_data = dict((j, node_features[j, :]) for j in range(num_node))
            
            if(np.isnan(dataset_graph_labels[i])):
                self.nan_cnt += 1
                continue
            
            self.zero_cnt += (dataset_graph_labels[i] == 0)
            self.one_cnt += (dataset_graph_labels[i] == 1)
            label = float(dataset_graph_labels[i])
            
            G = nx.Graph(y=label)
            label = torch.tensor(label)
            G.add_nodes_from([i for i in range(num_node)])
            for j, e in enumerate(edges):
                G.add_edge(e[0], e[1], edge_attr=edge_features[j])
            nx.set_node_attributes(G, node_data, name="X")
            edges = torch.tensor(list(edges))
            d = Data(edge_index=edges.T, x=node_features, edge_attr=edge_features, y = label)
            self.data.append(d)
            self.graphs.append(G)
        
        print(f"Dataset Path: {self.dataset_path}")
        print("Number of nan: ", self.nan_cnt)
        print("Number of 1: ", self.one_cnt)
        print("Number of 0: ", self.zero_cnt)
        
    def len(self):
        return len(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def get(self, idx):
        return self.data[idx]
    
    def get_graph(self, idx):
        return self.graphs
    
X_train = GraphDataset("./dataset/dataset_2/train")
X_val =  GraphDataset("./dataset/dataset_2/valid")
BATCH_SIZE = 128

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(X_val,batch_size=BATCH_SIZE, shuffle=True)

# Basic GCN implementation

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

device = torch.device('cpu')

from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import lr_scheduler
from encoder import NodeEncoder, EdgeEncoder




class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = GATConv(emb_dim, hidden_channels, heads = 2)
        self.conv2 = GATConv(2*hidden_channels, hidden_channels, heads = 1)

        #NOTE: We cannot pass edge attr to GraphSage and GIN conv layers
        # self.conv1 = SAGEConv(emb_dim, hidden_channels, aggr='mean')  # 'mean' aggregation
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')

        # self.conv1 = GINConv(torch.nn.Sequential(
        #     torch.nn.Linear(emb_dim, hidden_channels),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_channels, hidden_channels)
        # ))

        # self.conv2 = GINConv(torch.nn.Sequential(
        #     torch.nn.Linear(hidden_channels, hidden_channels),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_channels, hidden_channels)
        # ))

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
        # print(x.shape)
        x = self.node_encoder(x.to(torch.long))
        # print(x.shape)
        edge_attr = self.edge_encoder(edge_attr.to(torch.long))
        temp = self.edge_linear(edge_attr)
        temp = torch.relu(temp)
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        x = self.conv1(x, edge_index, temp).relu()
        x = self.conv2(x, edge_index, temp).relu()
        x = global_mean_pool(x, batch)
        # print(x.shape)
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
        
        # x = torch.relu(x)
        
        # print(x.shape)
        x = F.sigmoid(x)
        # print(x)
        # x = x[root_mask, :]
        return x
    
def calculate_accuracy(outputs, targets):
    predictions = (outputs > 0.5).float()  # Convert probabilities to binary predictions (0 or 1)
    correct_predictions = (predictions == targets).float()
    accuracy = correct_predictions.sum().item() / targets.shape[1]
    return accuracy


model = GCN(64, 1, 128)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=1e-3)

    
optimizer.zero_grad()
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    model.to(device)
    total_loss = 0.0
    total_correct_train = 0
    total_samples_train = 0
    for i, batch in enumerate(train_loader):
        # Move batch to device
        batch.to(device)
        weights = torch.ones_like(batch.y.reshape(1,-1))
        weights[batch.y.reshape(1,-1) == 1] = 2.0
        loss_fun = torch.nn.BCELoss(weight= weights)

        out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])

        optimizer.zero_grad()
        loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
        # print(float(loss))

        # Backward pass and optimization
        loss.mean().backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss+=loss.item()
        total_correct_train += calculate_accuracy(out.reshape(1, -1), batch.y.reshape(1, -1)) * batch.y.shape[0]
        total_samples_train += batch.y.shape[0]
    # scheduler.step()
    average_loss = total_loss / len(train_loader)
    accuracy_train = total_correct_train / total_samples_train

     # Validation phase
    model.eval()
    with torch.no_grad():
        total_bce = 0.0
        total_correct_val = 0
        total_samples_val = 0

        for i, batch in enumerate(val_loader):
             # Move batch to device
            batch.to(device)
            # Forward pass
            loss_fun = torch.nn.BCELoss()
            if(batch.edge_index.shape[0] == 0):
                continue
            out = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
            # print(out.shape)
            loss = loss_fun(out.reshape(1, -1), batch.y.reshape(1, -1))
            loss = loss.mean()

            total_bce += loss.item()
            total_correct_val += calculate_accuracy(out.reshape(1, -1), batch.y.reshape(1, -1)) * batch.y.shape[0]
            total_samples_val += batch.y.shape[0]

        average_bce_val = total_bce / len(val_loader)
        accuracy_val = total_correct_val / total_samples_val
    # Print logs
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy_train:.4f}, Val BCE: {average_bce_val:.4f}, Val Accuracy: {accuracy_val:.4f}")


# Make sure to clear the computation graph after the loop
torch.cuda.empty_cache()


evaluator = Evaluator('dataset-2')
# benchmark = DataLoader(X_train_dat, batch_size= len(X_train_dat), shuffle = True)
val_loader = DataLoader(X_val, batch_size= len(X_val), shuffle = True)

for i, batch in enumerate(val_loader):
    model.eval()
    # Move batch to device
    batch.to(device)
    # Forward pass
    y_pred = model(batch.x, batch.edge_index.to(torch.long), batch.edge_attr, batch.batch, batch.y.shape[0])
    y_true = batch.y
    y_true = y_true.unsqueeze(1)
    # print(y_pred.shape)
    # print(y_true.shape)
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)