import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.fc2 = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        e  = self.attn_fc(z2)
        return {"e": F.leaky_relu(e)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h     = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):

        if isinstance(h, tuple):
            h_src, h_dst = h
            z_src = self.fc1(h_src)
            z_dst = self.fc2(h_dst)
            g.srcdata["z"] = z_src
            g.dstdata["z"] = z_dst
        else:
            z = self.fc1(h)
            g.ndata["z"] = z

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        if isinstance(h, tuple):
            return g.dstdata.pop("h")
        else:
            return g.ndata.pop("h")



class GCNLayer_homogeneous(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer_homogeneous, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):

        z = self.fc(h)
        g.ndata["z"] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")



class GCNLayer_(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer_, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.linear.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, h):
        h = self.linear(h)
        return F.relu(h)