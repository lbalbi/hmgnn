import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class SignedGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear_pos = nn.Linear(in_feats, out_feats)
        self.linear_neg = nn.Linear(in_feats, out_feats)
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Reinitialize the parameters of the layer.
        """
        self.linear_pos.reset_parameters()
        self.linear_neg.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linear_pos.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_neg.weight, gain=gain)

    def forward(self, g_pos, g_neg, h_pos, h_neg):
        """
        g_pos: DGL graph with positive edges
        g_neg: DGL (sub-)graph with negative edges
        h_pos: positive representation
        h_neg: negative representation
        """
        with g_pos.local_scope():
            g_pos.ndata['h'] = h_pos
            g_pos.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h_neigh'))
            h_pos_new = self.linear_pos(g_pos.ndata['h_neigh'])

        with g_neg.local_scope():
            g_neg.ndata['h'] = h_neg
            g_neg.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h_neigh'))
            h_neg_new = self.linear_neg(g_neg.ndata['h_neigh'])

        return F.relu(h_pos_new), F.relu(h_neg_new)
