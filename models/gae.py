import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv
from typing import Tuple, List
class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=5, res=True, norm=False, scale=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.scale = scale
        if scale: self.scale_norm = nn.LayerNorm(h_feats)
        self.norm = norm
        if norm:
            self.norms = torch.nn.ModuleList()
            for _ in range(layer - 1):
                self.norms.append(nn.LayerNorm(h_feats))
        self.act = F.relu

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.scale:
            x = self.scale_norm(x)
        ori = x
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.res:
                x += ori
            if self.norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()
class DotPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        x = (x_i * x_j).sum(dim=-1)
        return x.squeeze()
class LorentzPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        n = x_i.size(1)
        x = torch.sum(x_i[:, 0:n//2] * x_j[:, 0:n//2], dim=-1) - torch.sum(x_i[:, n//2:] * x_j[:, n//2:], dim=-1)
        return x.squeeze()

def drop_edge(g, dpe = 0.2):
    g = g.clone()
    eids = torch.randperm(g.number_of_edges())[:int(g.number_of_edges() * dpe)].to(g.device)
    g.remove_edges(eids)
    g = dgl.add_self_loop(g)
    return g
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,
                 norm=False, tailact=False, norm_affine=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if norm:
            self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
        self.lins.append(nn.ReLU())
        if dropout > 0:
            self.lins.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if norm:
                self.lins.append(nn.LayerNorm(hidden_channels), elementwise_affine=norm_affine)
            self.lins.append(nn.ReLU())
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        if tailact:
            self.lins.append(nn.LayerNorm(out_channels), elementwise_affine=norm_affine)
            self.lins.append(nn.ReLU())
            self.lins.append(nn.Dropout(dropout))

    def forward(self, x):
        x = self.lins(x)
        return x.squeeze()
    


class GCN_GAE(nn.Module):
    """
    Refined-GAE model for link classification.
    """
    def __init__(self, in_feats, hidden_dim, out_dim: int, n_layers: int = 2,
                relu=False, linear=False, prop_step=2, dropout=0.2, residual=0.1, 
                e_etypes: List[Tuple[str, str, str]] = None,
                ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"), n_type: str = "node"):
        
        super(GCN_GAE, self).__init__()
        input_dim = list(in_feats.values())[0]
        self.n_type = n_type
        self.ppi_etype = ppi_etype
        self.conv1 = GraphConv(input_dim, input_dim)
        self.conv2 = GraphConv(input_dim, input_dim)
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        self.linear = linear
        if linear: self.mlps = nn.ModuleList([MLP(hidden_dim, hidden_dim, 2, dropout) for _ in range(prop_step)])
        self.classify = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, g, edge_index):

        f = g.ndata['feat']
        h = self.conv1(g, f).flatten(1) + self.residual * f
        for i in range(1, self.prop_step):
            h = F.relu(h)
            if self.linear: h = self.mlps[i](h)
            h = self.conv2(g, h).flatten(1) + self.residual * f

        src_ids, dst_ids = edge_index
        hs = h[src_ids]
        hd = h[dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)
        z = h
        return z, torch.sigmoid(logits)



class GCN_multilayers(nn.Module):
        
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, drop_edge=False, relu=False, linear=False, prop_step=2, dropout=0.2, residual=0, conv='GCN'):
        super(GCN_multilayers, self).__init__()
        if conv == 'GCN':
            self.convs = nn.ModuleList([GraphConv(in_feats, h_feats)])
            for _ in range(prop_step - 1):
                self.convs.append(GraphConv(h_feats, h_feats))
        elif conv == 'SAGE':
            self.convs = nn.ModuleList([SAGEConv(in_feats, h_feats, 'mean')])
            for _ in range(prop_step - 1):
                self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
        elif conv == 'GAT':
            self.convs = nn.ModuleList([GATConv(in_feats, h_feats // 4, 4)])
            for _ in range(prop_step - 1):
                self.convs.append(GATConv(h_feats, h_feats // 4, 4))
        elif conv == 'GIN':
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, 0.2)])
            self.convs = nn.ModuleList([GINConv(self.mlps[0], 'mean')])
            for _ in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, 0.2))
                self.convs.append(GINConv(self.mlps[-1], 'mean'))
        self.norm = norm
        self.drop_edge = drop_edge
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        self.linear = linear
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if linear:
            self.mlps = nn.ModuleList([MLP(h_feats, h_feats, 2, dropout) for _ in range(prop_step)])

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        x = self.dp(x)
        return x
    
    def forward(self, g, in_feat):
        ori = in_feat
        if self.drop_edge:
            g = drop_edge(g)
        h = self.conv1(g, in_feat).flatten(1) + self.residual * ori
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            if self.linear:
                h = self.mlps[i](h)
            h = self.conv2(g, h).flatten(1) + self.residual * ori
        return h

    
class PureGCN(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden=256, dp=0, relu=False, norm=False, res=False):
        super().__init__()
        self.lin = nn.Linear(input_dim, hidden)
        self.conv = GraphConv(hidden, hidden, weight=False, bias=False)
        self.num_layers = num_layers
        self.dp = dp
        self.norm = norm
        self.res = res
        self.relu = relu
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

    def forward(self, adj_t, x, e_feat=None):
        x = self.lin(x)
        ori = x
        for i in range(self.num_layers):
            if i != 0 and self.res:
                x = x + ori
            if self.norm:
                x = self.norms[i](x)
            if self.relu:
                x = F.relu(x)
            if self.dp > 0:
                x = F.dropout(x, p=self.dp, training=self.training)
            x = self.conv(adj_t, x, edge_weight=e_feat)
        return x

class PureGCN_no_para(nn.Module):
    def __init__(self, input_dim, num_layers=2, relu=False, norm=False, res=False):
        super().__init__()
        self.conv = GraphConv(input_dim, input_dim, weight=False, bias=False)
        self.num_layers = num_layers
        self.norm = norm
        self.res = res
        self.relu = relu

    def forward(self, adj_t, x, e_feat=None):
        ori = x
        for i in range(self.num_layers):
            if i != 0 and self.res:
                x = x + ori
            if self.norm:
                x = F.layer_norm(x, x.shape[1:])
            if self.relu:
                x = F.relu(x)
            x = self.conv(adj_t, x, edge_weight=e_feat)
        return x       