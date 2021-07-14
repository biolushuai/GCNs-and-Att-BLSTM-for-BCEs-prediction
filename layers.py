import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

from config import DefaultConfig

configs = DefaultConfig()


class BiLSTMAttentionLayer(nn.Module):
    def __init__(self, num_hidden, num_layers):
        super(BiLSTMAttentionLayer, self).__init__()
        self.feature_dim = configs.feature_dim

        self.encoder = nn.LSTM(input_size=self.feature_dim,
                               hidden_size=num_hidden,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        self.w_omega = nn.Parameter(torch.Tensor(num_hidden * 2, num_hidden * 2))
        self.u_omega = nn.Parameter(torch.Tensor(num_hidden * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs)
        print(outputs.shape)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = outputs * att_score
        outs = torch.sum(scored_x, dim=1)
        print(outs.shape)
        return outs


class ResCNN(nn.Module):
    def __init__(self, in_planes, planes, window, stride=1):
        super(ResCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=2 * window + 1, stride=stride, padding=window, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=2 * window + 1, stride=stride, padding=window, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=2 * window + 1, stride=stride, padding=window, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv1(out))
        out += x
        out = F.relu(out)
        return out


class NodeAverageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(NodeAverageLayer, self).__init__()
        self.center_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.nh_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        center_std = 1. / np.prod(self.center_weight.shape[0:-1])
        nh_std = 1. / np.prod(self.nh_weight.shape[0:-1])
        self.center_weight.data.uniform_(-center_std, center_std)
        self.nh_weight.data.uniform_(-nh_std, nh_std)
        self.bias.data.fill_(0)

    def forward(self, vertex, nh_indices):
        nh_size = nh_indices.shape[1]
        zc = torch.mm(vertex, self.center_weight)
        zn = torch.mm(vertex, self.nh_weight)
        zn = zn[torch.squeeze(nh_indices)]
        zn = torch.sum(zn, axis=1)
        zn = torch.div(zn, nh_size)
        z = zc + zn + self.bias
        z = self.activation(z)
        z = self.dropout(z)
        return z


class NodeEdgeAverageLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate):
        super(NodeEdgeAverageLayer, self).__init__()
        self.center_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.nh_weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.edge_weight = Parameter(torch.FloatTensor(2, out_dim))
        self.bias = Parameter(torch.FloatTensor(out_dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()

    def reset_parameters(self):
        center_std = 1. / np.prod(self.center_weight.shape[0:-1])
        nh_std = 1. / np.prod(self.nh_weight.shape[0:-1])
        edge_std = 1. / np.prod(self.edge_weight.shape[0:-1])
        self.center_weight.data.uniform_(-center_std, center_std)
        self.nh_weight.data.uniform_(-nh_std, nh_std)
        self.edge_weight.data.uniform_(-edge_std, edge_std)
        self.bias.data.fill_(0)

    def forward(self, vertex, edge, nh_indices):
        nh_size = nh_indices.shape[1]
        zc = torch.mm(vertex, self.center_weight)
        zn = torch.mm(vertex, self.nh_weight)
        ze = torch.tensordot(edge, self.edge_weight, ([-1], [0]))

        zn = zn[torch.squeeze(nh_indices)]
        zn = torch.div(torch.sum(zn, 1), nh_size)
        ze = torch.div(torch.sum(ze, 1), nh_size)

        z = zc + zn + ze + self.bias
        z = self.activation(z)
        z = self.dropout(z)
        return z