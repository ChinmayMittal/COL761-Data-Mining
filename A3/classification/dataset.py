import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Dataset, Data


class GraphDataset(Dataset):
    
    def __init__(self, dataset_path, labels=True):
        super().__init__()        
        self.dataset_path = dataset_path
        
        dataset_edge_features = np.loadtxt(os.path.join(dataset_path, "edge_features.csv.gz"), dtype=float, delimiter=',')
        dataset_edges = np.loadtxt(os.path.join(dataset_path, "edges.csv.gz"), dtype=int, delimiter=',')
        
        if labels:
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
            
            if labels:
                if(np.isnan(dataset_graph_labels[i])):
                    self.nan_cnt += 1
                    continue
            
            
                self.zero_cnt += (dataset_graph_labels[i] == 0)
                self.one_cnt += (dataset_graph_labels[i] == 1)
                label = float(dataset_graph_labels[i])
            
            else:
                label = -1

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
        
        if labels:
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
        return self.graphs[idx]
