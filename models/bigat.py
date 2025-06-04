import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import GATLayer
class BiGAT(torch.nn.module):
    def __init__(self, relations,
                 name = "BiGAT",
                 in_channels = 128,
                 hidden_channels = 256,
                 out_channels = 1,
                 dropout = 0.01,
                 lr = 0.001,
                 weight_decay = 0.0001):
        super(BiGAT, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.name = name
        self.relations = relations
    
        
    def layer_makeup(self):
        self.relations = [rel for rel in self.relations if not None]
        self.hg_pos = dglnn.HeteroGraphConv({rel: dglnn.GATConv(self.in_channels, self.hidden_channels, 1, 'mean')
                                                   for rel in self.relations}, aggregate='mean')
        self.hg_neg = dglnn.HeteroGraphConv({rel: dglnn.GATConv(self.hidden_channels, self.hidden_channels//2, 1, 'mean') 
                                             for rel in self.relations}, aggregate='mean')
        self.linear = torch.nn.Linear(self.hidden_channels, self.out_channels)

    def reset_parameters(self):
        self.hg.reset_parameters()
        self.hg2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, pos_graph, neg_graph):
        pos_out = self.hg_pos(pos_graph, pos_graph.ndata['feat'])
        neg_out = self.hg_neg(neg_graph, neg_graph.ndata['feat'])

        out = torch.cat([pos_out, neg_out], dim=0)
        out = self.linear(out)
        return out



class BiGAT_origin(torch.nn.module):
    def __init__(self, g_pos, g_neg, relations, in_channels=128, hidden_channels=256, out_channels=1, num_heads=1):
        super(BiGAT_origin, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.relations = relations
        self.num_heads = num_heads

        self.hg = dglnn.HeteroGraphConv({rel: GATLayer(g_pos, self.in_channels, self.hidden_channels, self.num_heads, 'mean')
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: GATLayer(g_neg, self.hidden_channels, self.hidden_channels//2, self.num_heads, 'mean')
                                                   for rel in self.relations}, aggregate='mean')
        self.linear = torch.nn.Linear(self.hidden_channels, self.out_channels)


    def forward(self):
        out = torch.cat([self.hg["positive"], self.hg["negative"]], dim=0)
        out = self.linear(out)
        return torch.sigmoid(out)