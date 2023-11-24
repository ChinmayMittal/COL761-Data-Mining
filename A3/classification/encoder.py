import torch


def get_node_feature_dims():
    return [119, 5, 12, 12, 10, 6, 6, 2, 2]


def get_edge_feature_dims():
    return [5, 6, 2]


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_embedding_list = torch.nn.ModuleList()
        full_node_feature_dims = get_node_feature_dims()
        for i, dim in enumerate(full_node_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.node_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.node_embedding_list[i](x[:,i])
        return x_embedding


class EdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(EdgeEncoder, self).__init__()
        full_edge_feature_dims = get_edge_feature_dims()
        self.edge_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_edge_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.edge_embedding_list.append(emb)

    def forward(self, edge_attr):
        edge_embedding = 0
        for i in range(edge_attr.shape[1]):
            edge_embedding += self.edge_embedding_list[i](edge_attr[:,i])
        return edge_embedding
