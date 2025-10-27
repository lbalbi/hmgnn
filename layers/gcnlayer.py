import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GCNLayer(MessagePassing):
    """
    Bipartite-capable attention layer (PyG).
    If x is a Tensor -> homogeneous case.
    If x is a tuple (x_src, x_dst) -> bipartite case.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')  # sum aggregation
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)   # for src
        self.fc2 = nn.Linear(in_dim, out_dim, bias=False)   # for dst (used in bipartite)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, x, edge_index, size=None):
        """
        x: Tensor [N,F] or (x_src [Ns,F], x_dst [Nd,F])
        edge_index: LongTensor [2,E]
        size: optional (Ns, Nd) for bipartite
        """
        if isinstance(x, tuple):
            x_src, x_dst = x
            z_src = self.fc1(x_src)
            z_dst = self.fc2(x_dst)
            out = self.propagate(edge_index, x=(z_src, z_dst), size=size)  # -> [Nd, out_dim]
        else:
            z = self.fc1(x)  # [N, out_dim]
            out = self.propagate(edge_index, x=z)  # homogeneous -> [N, out_dim]
        return F.relu(out)

    def message(self, x_j, x_i, index):
        """
        x_j: source node transformed features [E, out_dim]
        x_i: target node transformed features [E, out_dim]
        index: target indices per edge (for softmax)
        """
        e = self.attn_fc(torch.cat([x_j, x_i], dim=-1))  # [E,1]
        a = F.leaky_relu(e)
        alpha = softmax(a, index)                        # normalize over incoming edges of each dst
        return alpha * x_j                               # weighted messages

    def update(self, aggr_out):
        return aggr_out


class GCNLayer_homogeneous(MessagePassing):
    """
    Homogeneous attention layer (PyG).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def forward(self, x, edge_index):
        z = self.fc(x)  # [N, out_dim]
        out = self.propagate(edge_index, x=z)  # [N, out_dim]
        return F.relu(out)

    def message(self, x_j, x_i, index):
        e = self.attn_fc(torch.cat([x_j, x_i], dim=-1))  # [E,1]
        a = F.leaky_relu(e)
        alpha = softmax(a, index)
        return alpha * x_j

    def update(self, aggr_out):
        return aggr_out


class GCNLayer_(nn.Module):
    """
    Pure MLP over node features; already PyG-compatible.
    """
    def __init__(self, in_dim, out_dim):
        super(GCNLayer_, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, h):
        return F.relu(self.linear(h))
