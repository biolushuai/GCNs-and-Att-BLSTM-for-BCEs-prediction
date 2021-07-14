from layers import *
from config import DefaultConfig
configs = DefaultConfig()


class ResCNNModel(nn.Module):
    def __init__(self):
        super(ResCNNModel, self).__init__()
        window = configs.window_size
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        dropout_rate = configs.dropout_rate
        self.rescnn = ResCNN(1, 1, window)
        self.linear1 = nn.Sequential(
            nn.Linear(feature_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, indices):
        out = torch.unsqueeze(torch.unsqueeze(vertex, 0), 0)
        out = self.rescnn(out)
        out = torch.squeeze(out)
        out = out[indices]
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BiLSTMResCNN(nn.Module):
    def __init__(self):
        super(BiLSTMResCNN, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        window = configs.window_size

        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.rescnn = ResCNN(1, 1, window)

        self.linear1 = nn.Sequential(
            nn.Linear(feature_dim + 2 * num_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)

        local_out = torch.unsqueeze(torch.unsqueeze(vertex, 0), 0)
        local_out = self.rescnn(local_out)
        local_out = torch.squeeze(local_out)
        local_out = local_out[indices]

        out = torch.cat((local_out, global_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class NodeAverageModel(nn.Module):
    def __init__(self):
        super(NodeAverageModel, self).__init__()
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        self.hidden_dims = configs.hidden_dim
        self.hidden_dims.insert(0, feature_dim)
        self.node_avers = nn.ModuleList([NodeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                         for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, nh_indices, indices):
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_avers[i](vertex, nh_indices)
        out = vertex[indices]
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BiLSTMNodeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMNodeAverageModel, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        self.hidden_dims = configs.hidden_dim
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_avers = nn.ModuleList([NodeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                         for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, nh_indices, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_avers[i](vertex, nh_indices)
        local_out = vertex[indices]
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class NodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(NodeEdgeAverageModel, self).__init__()
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        mlp_dim = configs.mlp_dim
        self.hidden_dims = configs.hidden_dim
        self.hidden_dims.insert(0, feature_dim)
        self.nodeedge_avers = nn.ModuleList([NodeEdgeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                         for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, edge, nh_indices, indices):
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.nodeedge_avers[i](vertex, edge, nh_indices)
        out = vertex[indices]
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BiLSTMNodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMNodeEdgeAverageModel, self).__init__()
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim
        self.hidden_dims = configs.hidden_dim
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_edge_avers = nn.ModuleList([NodeEdgeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                             for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, edge, nh_indices, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_edge_avers[i](vertex, edge, nh_indices)
        local_out = vertex[indices]
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class BiLSTMResCNNNodeEdgeAverageModel(nn.Module):
    def __init__(self):
        super(BiLSTMResCNNNodeEdgeAverageModel, self).__init__()
        window = configs.window
        num_hidden = configs.num_hidden
        num_layer = configs.num_layer
        dropout_rate = configs.dropout_rate
        feature_dim = configs.feature_dim

        self.cnnres = ResCNN(1, 1, window)
        self.hidden_dims = configs.hidden_dim
        self.hidden_dims.insert(0, feature_dim)
        self.bilstm = BiLSTMAttentionLayer(num_hidden, num_layer)
        self.node_edge_avers = nn.ModuleList([NodeEdgeAverageLayer(self.hidden_dims[i], self.hidden_dims[i + 1], dropout_rate)
                                             for i in range(len(self.hidden_dims[:-1]))])
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_dims[-1] + 2 * num_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, vertex, edge, nh_indices, indices):
        batch_size = len(indices)
        global_vertex = vertex.repeat(batch_size, 1, 1)
        global_out = self.bilstm(global_vertex)

        vertex = torch.unsqueeze(torch.unsqueeze(vertex, 0), 0)
        vertex = self.cnnres(vertex)
        vertex = torch.squeeze(vertex)
        for i in range(len(self.hidden_dims[:-1])):
            vertex = self.node_edge_avers[i](vertex, edge, nh_indices)
        local_out = vertex[indices]
        out = torch.cat((global_out, local_out), 1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out