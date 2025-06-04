import dgl.nn as dglnn
import torch

class BiGAT:
    def __init__(self, relations,
                 name = "BiGAT",
                 in_channels = 128,
                 hidden_channels = 256,
                 out_channels = 1,
                 dropout = 0.01,
                 lr = 0.001,
                 weight_decay = 0.0001):
        
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
        self.hg_pos = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(self.in_channels, self.hidden_channels,'mean') 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg_neg = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(self.hidden_channels, self.in_channels,'mean') for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(self.hidden_channels, self.out_channels)

    def reset_parameters(self):
        self.hg.reset_parameters()
        self.hg2.reset_parameters()
        self.linear.reset_parameters()
        pass

    def forward(self, pos_graph, neg_graph):
        pos_out = self.hg_pos(pos_graph, pos_graph.ndata['feat'])
        neg_out = self.hg_neg(neg_graph, neg_graph.ndata['feat'])

        out = torch.cat([pos_out, neg_out], dim=0)
        out = self.linear(out)
        return out