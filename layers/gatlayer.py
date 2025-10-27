import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATLayer(MessagePassing):
    """
    Homogeneous single-head GAT layer. Use with
    'out = layer(x, edge_index)', where x: [N, Fin], edge_index: [2, E]
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, x, edge_index):
        """
        x: [N, Fin], edge_index: [2, E]
        """
        z = self.fc(x)
        out = self.propagate(edge_index, x=z)
        return F.relu(out)

    def message(self, x_j, x_i, index):
        """
        x_j: source node features (after fc) [E, Fout]
        x_i: target node features (after fc) [E, Fout]
        index: target indices per edge, used for attention softmax
        """
        e = self.attn_fc(torch.cat([x_j, x_i], dim=-1))
        a = F.leaky_relu(e)
        alpha = softmax(a, index)
        return alpha * x_j

    def update(self, aggr_out):
        return aggr_out


class GATLayer_(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer_, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, h):
        return F.relu(self.linear(h))
