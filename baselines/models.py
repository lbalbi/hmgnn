import torch, torch_geometric as tg
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F


## relational GNN implementation

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, rel_names, subtract_rel):
        super(RGCNLayer, self).__init__()
        self.rel_names = rel_names
        self.subtract_rel = subtract_rel
        self.weights = nn.ModuleDict({ rel: nn.Linear(in_feat, out_feat, bias=False) for rel in rel_names })
        self.self_loop = nn.Linear(in_feat, out_feat, bias=False)

    def forward(self, g, features):
        with g.local_scope():
            g.ndata['h'] = features
            messages = {}

            for rel in self.rel_names:
                g.update_all(message_func=lambda edges: {'msg': self.weights[rel](edges.src['h'])},
                    reduce_func=dgl.function.sum(msg='msg', out='h_{}'.format(rel)), etype=rel)
                messages[rel] = g.ndata.pop('h_{}'.format(rel))

            aggregated_msg = sum(msg for rel, msg in messages.items() if rel != self.subtract_rel)
            final_msg = aggregated_msg - messages[self.subtract_rel]
            h_self = self.self_loop(features)
            h = final_msg + h_self
            return F.relu(h)


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, rel_names, num_layers=2):
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.node_embeddings = nn.Embedding(num_nodes, h_dim)
        for _ in range(num_layers - 1):
            self.layers.append(RGCNLayer(h_dim, h_dim/2, rel_names))
        self.final_layer = RGCNLayer(h_dim, out_dim, rel_names)

    def forward(self, g, e_types, n_pairs, predictee_n, mask="all", val=False):
        h = self.node_embeddings.weight
        for layer in self.layers:
            h = layer(g, h)
        g.ndata['h'] = h
        h_ = []

        for e_type in e_types:
            h = torch.cat([g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]],g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]]], 1)
            if mask != "all" and val==False: h = h[g.edges[e_type].data[mask] == True]
            if mask != "all" and val==True: h = h[g.edges[e_type].data[mask] == False]
            h_.append(h)
        h = torch.cat([g.nodes[predictee_n].data["h"][n_pairs[:,0]], g.nodes[predictee_n].data["h"][n_pairs[:,1]]], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        h = self.final_layer(g, h)
        return torch.sigmoid(h)


## heterophilic semantics-aware GNN
class HPGCN_sem(nn.Module):
    """
    Heterophilic Graph Convolutional Network (HPGCN) with modeling of edges of different types/ different 
    types of opposing valued edges
    """
    pass


##  heterophilic binary dgl GNN
class HPGCN(nn.Module):
    """Heterophilic Graph Convolutional Network (HPGCN)
    For modeling the relations between nodes connected through a negative edge"""
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
            dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HPGCN, self).__init__()
        


## heterogeneous dgl GNNs
class HeteroGCN_dgl(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
    relations=[], dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HeteroGCN_dgl, self).__init__()
        self.relations = [ rel for rel in relations if not None]
        self.name = "HeteroGCN"
        self.hg = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(n_features, n_hidden) 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(n_hidden, n_features) for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(n_hidden, n_classes)

    def reset_parameters(self):

        for layer in self.hg.mods: self.hg.mods[layer].reset_parameters()
        for layer in self.hg2.mods: self.hg2.mods[layer].reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, n_pairs, e_types, predictee_n,  mask = "all", val=False):

        h = self.hg(g, g.ndata["x"])
        h = {key: nn.functional.relu(x) for key, x in h.items()}
        h = self.hg2(g, h)
        g.ndata["h"] = h
        h_ = []
        
        for e_type in e_types:
            h = torch.cat([g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]],g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]]], 1)
            if mask != "all" and val==False: h = h[g.edges[e_type].data[mask] == True]
            if mask != "all" and val==True: h = h[g.edges[e_type].data[mask] == False]
            h_.append(h)

        h = torch.cat([g.nodes[predictee_n].data["h"][n_pairs[:,0]], g.nodes[predictee_n].data["h"][n_pairs[:,1]]], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        return torch.sigmoid(self.linear(nn.functional.relu(h_)))


class HeteroSAGE_dgl(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
    relations=[], dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HeteroSAGE_dgl, self).__init__()
        self.relations = [ rel for rel in relations if not None]
        self.name = "HeteroSAGE"
        self.hg = dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(n_features, n_hidden,'mean') 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(n_hidden, n_features,'mean') for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(n_hidden, n_classes)


    def reset_parameters(self):

        for layer in self.hg.mods: self.hg.mods[layer].reset_parameters()
        for layer in self.hg2.mods: self.hg2.mods[layer].reset_parameters()
        self.linear.reset_parameters()


    def forward(self, g, n_pairs, e_types, predictee_n, mask = "all", val=False):

        h = self.hg(g, g.ndata["x"])
        h = {key: nn.functional.relu(x) for key, x in h.items()}
        h = self.hg2(g, h)
        g.ndata["h"] = h
        h_ = []
        
        for e_type in e_types:
            h = torch.cat([g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]],g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]]], 1)
            if mask != "all" and val==False: h = h[g.edges[e_type].data[mask] == True]
            if mask != "all" and val==True: h = h[g.edges[e_type].data[mask] == False]
            h_.append(h)

        h = torch.cat([g.nodes[predictee_n].data["h"][n_pairs[:,0]], g.nodes[predictee_n].data["h"][n_pairs[:,1]]], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        return torch.sigmoid(self.linear(nn.functional.relu(h_)))


class HeteroGAT_dgl(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
    relations=[], dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HeteroGAT_dgl, self).__init__()
        self.relations = [ rel for rel in relations if not None]
        self.name = "HeteroGAT"
        self.hg = dglnn.HeteroGraphConv({rel: dglnn.GATConv(n_features, n_hidden,1) 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: dglnn.GATConv(n_hidden, n_features,1) for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(n_features * 2, n_classes)

    def reset_parameters(self):

        for layer in self.hg.mods: self.hg.mods[layer].reset_parameters()
        for layer in self.hg2.mods: self.hg2.mods[layer].reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, n_pairs, e_types, predictee_n, mask = "all", val=False):

        h = self.hg(g, g.ndata["x"])
        h = {key: nn.functional.relu(x) for key, x in h.items()}
        h = self.hg2(g, h)
        g.ndata["h"] = h
        h_ = []

        for e_type in e_types:

            h = torch.cat([torch.squeeze(g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]]),torch.squeeze(g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]])], 1)
            if mask != "all" and val==False: h = h[g.edges[e_type].data[mask] == True]
            if mask != "all" and val==True: h = h[g.edges[e_type].data[mask] == False]
            h_.append(h)
        
        h = torch.cat([torch.squeeze(g.nodes[predictee_n].data["h"][n_pairs[:,0]]), 
                       torch.squeeze(g.nodes[predictee_n].data["h"][n_pairs[:,1]])], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        return torch.sigmoid(self.linear(nn.functional.relu(h_)))


## two-type dgl GNNs
class BiGCN_dgl(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
                 relations=[], dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(BiGCN_dgl, self).__init__()

        self.relations = [rel for rel in relations if not None]
        self.name = "BiGCN"
        self.hg = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(n_features, n_hidden) 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(n_hidden, n_features) for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(n_hidden, n_classes)


    def reset_parameters(self):

        for layer in self.hg.mods: self.hg.mods[layer].reset_parameters()
        for layer in self.hg2.mods: self.hg2.mods[layer].reset_parameters()
        self.linear.reset_parameters()


    def forward(self, g, n_pairs, e_types, predictee_n = "node", mask = "all", val=False):

        nfeats = g.nodes[predictee_n].data["x"].float()
        feats = {predictee_n: nfeats}
        h = self.hg(g, feats)
        h = {key: nn.functional.relu(x) for key, x in h.items()}
        h = self.hg2(g, h)
        g.ndata["h"] = h[predictee_n]
        h_ = []
        
        for e_type in e_types:
            h = torch.cat([g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]],g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]]], 1)
            h = h[g.edges[e_type].data[mask] == True]
            h_.append(h)

        h = torch.cat([g.nodes[predictee_n].data["h"][n_pairs[:,0]], g.nodes[predictee_n].data["h"][n_pairs[:,1]]], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        return torch.sigmoid(self.linear(nn.functional.relu(h_)))
    

class BiSAGE_dgl(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
    relations=[], dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(BiSAGE_dgl, self).__init__()
        
        self.relations = [rel for rel in relations if not None]
        self.name = "BiSAGE"
        self.hg = dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(n_features, n_hidden,'mean') 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(n_hidden, n_features,'mean') for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(n_hidden, n_classes)


    def reset_parameters(self):

        for layer in self.hg.mods: self.hg.mods[layer].reset_parameters()
        for layer in self.hg2.mods: self.hg2.mods[layer].reset_parameters()
        self.linear.reset_parameters()


    def forward(self, g, n_pairs, e_types, predictee_n = "node", mask = "all", val=False):

        nfeats = g.nodes[predictee_n].data["x"].float()
        feats = {predictee_n: nfeats}
        h = self.hg(g, feats)
        h = {key: nn.functional.relu(x) for key, x in h.items()}
        h = self.hg2(g, h)
        g.ndata["h"] = h[predictee_n]
        h_ = []
        
        for e_type in e_types:
            h = torch.cat([g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]],g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]]], 1)
            h = h[g.edges[e_type].data[mask] == True]
            h_.append(h)

        h = torch.cat([g.nodes[predictee_n].data["h"][n_pairs[:,0]], g.nodes[predictee_n].data["h"][n_pairs[:,1]]], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        return torch.sigmoid(self.linear(nn.functional.relu(h_)))


class BiGAT_dgl(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, 
    relations=[], dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(BiGAT_dgl, self).__init__()
        self.relations = [rel for rel in relations if not None]
        self.name = "BiGAT"
        self.hg = dglnn.HeteroGraphConv({rel: dglnn.GATConv(n_features, n_hidden,1) 
                                                   for rel in self.relations}, aggregate='mean')
        self.hg2 = dglnn.HeteroGraphConv({rel: dglnn.GATConv(n_hidden, n_features,1) for rel in self.relations},
                                          aggregate='mean')
        self.linear = torch.nn.Linear(n_features * 2, n_classes)


    def reset_parameters(self):

        for layer in self.hg.mods: self.hg.mods[layer].reset_parameters()
        for layer in self.hg2.mods: self.hg2.mods[layer].reset_parameters()
        self.linear.reset_parameters()


    def forward(self, g, n_pairs, e_types, predictee_n = "node", mask = "all", val=False):

        nfeats = g.nodes[predictee_n].data["x"].float()
        feats = {predictee_n: nfeats}
        h = self.hg(g, feats)
        h = {key: nn.functional.relu(x) for key, x in h.items()}
        h = self.hg2(g, h)
        g.ndata["h"] = h[predictee_n]
        h_ = []
        
        for e_type in e_types:
            h = torch.cat([torch.squeeze(g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[0]]), 
                           torch.squeeze(g.nodes[predictee_n].data["h"][g.edges(etype=e_type)[1]])], 1)
            h = h[g.edges[e_type].data[mask] == True]
            h_.append(h)
        h = torch.cat([torch.squeeze(g.nodes[predictee_n].data["h"][n_pairs[:,0]]), 
                       torch.squeeze(g.nodes[predictee_n].data["h"][n_pairs[:,1]])], 1)
        h_.append(h)
        h_ = torch.cat(h_, dim=0)
        return torch.sigmoid(self.linear(nn.functional.relu(h_)))






## homogeneous GNNs from pytorch
class SAGE_pyg(torch.nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(SAGE_pyg, self).__init__()
        self.name = "GraphSAGE"
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.sage_1 = tg.nn.SAGEConv(self.n_features, self.n_hidden, 'mean')
        self.drop = torch.nn.Dropout(self.dropout)
        self.relu = torch.nn.ReLU()
        self.sage_2 = tg.nn.SAGEConv(self.n_hidden, self.n_features, 'mean')
        self.linear = torch.nn.Linear(self.n_hidden, self.n_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def reset_parameters(self):

        self.sage_1.reset_parameters()
        self.sage_2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, features, mask):
        
        h = self.sage_1(features, g)
        h = self.drop(self.relu(h))
        h = self.sage_2(h, g)

        mask = mask.T
        p1 = h[mask[0]].detach()
        p2 = h[mask[1]].detach()
        h = torch.cat((p1, p2), 1)
        return torch.sigmoid(self.linear(h))


class GAT_pyg(torch.nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(GAT_pyg, self).__init__()
        self.name = "GAT"
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.gat_1 = tg.nn.GATConv(self.n_features, self.n_hidden, heads=1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.gat_2 = tg.nn.GATConv(self.n_hidden, self.n_features, heads=1)
        self.linear = torch.nn.Linear(self.n_hidden, self.n_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def reset_parameters(self):

        self.gat_1.reset_parameters()
        self.gat_2.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, features, mask):
        
        h = self.gat_1(features, g)
        h = self.drop(self.relu(h))
        h = self.gat_2(h, g)

        mask = mask.T
        p1 = h[mask[0]].detach()
        p2 = h[mask[1]].detach()
        h = torch.cat((p1, p2), 1)
        return torch.sigmoid(self.linear(h))


class GCN_pyg(torch.nn.Module):

    def __init__(self, n_features = 128, n_classes = 1,  n_hidden = 256, dropout = 0.0, lr= 0.01,
                      weight_decay = 0.01):
        super(GCN_pyg, self).__init__()
        self.name = "GCN"
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.gconv_a = tg.nn.GCNConv(self.n_features, self.n_hidden)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.gconv_b = tg.nn.GCNConv(self.n_hidden, self.n_features)
        self.linear = torch.nn.Linear(self.n_hidden, self.n_classes)

    def reset_parameters(self):

        self.gconv_a.reset_parameters()
        self.gconv_b.reset_parameters()
        self.linear.reset_parameters()


    def forward(self, g, features, mask):

        h = self.gconv_a(features, g)
        h = self.drop(self.relu(h))
        h = self.gconv_b(h, g)
        
        mask = mask.T
        p1 = h[mask[0]].detach()
        p2 = h[mask[1]].detach()
        h = torch.cat((p1, p2), 1)
        return torch.sigmoid(self.linear(h))


## heterogeneous pytorch GNNs
class HeteroGAT_pyg(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HeteroGAT_pyg, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.gconv_a = tg.nn.HeteroConv({('protein', "has_+annotation", "class"): tg.nn.GATConv(self.n_features, self.n_hidden),
                    ('protein', "has_-annotation", "class"): tg.nn.GATConv(self.n_features, self.n_hidden),
                    ('class', "has_link", "class"): tg.nn.GraphConv(self.n_features, self.n_hidden),
                    ('protein', "has_ppi", "protein"): tg.nn.GATConv(self.n_features, self.n_hidden)})

        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.gconv_b = tg.nn.HeteroConv({('protein', "has_+annotation", "class"): tg.nn.GATConv(self.n_hidden, self.n_features),
                    ('protein', "has_-annotation", "class"): tg.nn.GATConv(self.n_hidden, self.n_features),
                    ('class', "has_link", "class"): tg.nn.GraphConv(self.n_features, self.n_hidden),
                    ('protein', "has_ppi", "protein"): tg.nn.GATConv(self.n_hidden, self.n_features)})

        self.linear = torch.nn.Linear(self.n_features*2, self.n_classes)

    def reset_parameters(self):
        
        self.gconv_a.reset_parameters()
        self.gconv_b.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, features, mask):
        
        h = self.gconv_a(g, features)
        h = self.drop(self.relu(h))
        h = self.gconv_b(g, h)
        mask = mask.T
        p1 = h['protein'][mask[0]].detach()
        p2 = h['protein'][mask[1]].detach()
        h = torch.cat((p1, p2), 1)
        return torch.sigmoid(self.linear(h))


class HeteroGCN_pyg(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HeteroGCN_pyg, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.gconv_a = tg.nn.HeteroConv({('protein', "has_+annotation", "class"): tg.nn.GraphConv(self.n_features, self.n_hidden),
                    ('protein', "has_-annotation", "class"): tg.nn.GraphConv(self.n_features, self.n_hidden),
                    ('class', "has_link", "class"): tg.nn.GraphConv(self.n_features, self.n_hidden),
                    ('protein', "has_ppi", "protein"): tg.nn.GraphConv(self.n_features, self.n_hidden)})
        self.gconv_b = tg.nn.HeteroConv({('protein', "has_+annotation", "class"): tg.nn.GraphConv(self.n_hidden, self.n_features),
                    ('protein', "has_-annotation", "class"): tg.nn.GraphConv(self.n_hidden, self.n_features),
                    ('class', "has_link", "class"): tg.nn.GraphConv(self.n_hidden, self.n_features),
                    ('protein', "has_ppi", "protein"): tg.nn.GraphConv(self.n_hidden, self.n_features)})

        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.linear = torch.nn.Linear(self.n_features*2, self.n_classes)

    def reset_parameters(self):
        for layer in self.gconv_a.mods:
            layer.reset_parameters()
        for layer in self.gconv_b.mods:    
            self.gconv_b.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, features, mask):

        h = self.gconv_a(x_dict = features, edge_index_dict = g)
        h = self.drop(self.relu(h))
        h = {key: x.relu() for key, x in h.items()}
        h = self.gconv_b(h, g)

        mask = mask.T
        p1 = h['protein'][mask[0]].detach()
        p2 = h['protein'][mask[1]].detach()
        h = torch.cat((p1, p2), 1)
        return torch.sigmoid(self.linear(h))


class HeteroSAGE_pyg(nn.Module):
    def __init__(self, n_features = 128, n_classes = 1, n_hidden = 256, dropout = 0.0, lr= 0.01, weight_decay = 0.01):
        super(HeteroSAGE_pyg, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        self.gconv_a = tg.nn.HeteroConv({('protein', "has_+annotation", "class"): tg.nn.SAGEConv(self.n_features, self.n_hidden, 'mean'),
                    ('protein', "has_-annotation", "class"): tg.nn.SAGEConv(self.n_features, self.n_hidden, 'mean'),
                    ('class', "has_link", "class"): tg.nn.SAGEConv(self.n_features, self.n_hidden, 'mean'),
                    ('protein', "has_ppi", "protein"): tg.nn.SAGEConv(self.n_features, self.n_hidden, 'mean')})
        self.gconv_b = tg.nn.HeteroConv({('protein', "has_+annotation", "class"): tg.nn.SAGEConv(self.n_hidden, self.n_features, 'mean'),
                    ('protein', "has_-annotation", "class"): tg.nn.SAGEConv(self.n_hidden, self.n_features, 'mean'),
                    ('class', "has_link", "class"): tg.nn.SAGEConv(self.n_hidden, self.n_features, 'mean'),
                    ('protein', "has_ppi", "protein"): tg.nn.SAGEConv(self.n_hidden, self.n_features, 'mean')})

        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.dropout)
        self.linear = torch.nn.Linear(self.n_features *2, self.n_classes)

    def reset_parameters(self):
        for layer in self.gconv_a.mods:
            layer.reset_parameters()
        for layer in self.gconv_b.mods:    
            self.gconv_b.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, g, features, mask):

        h = self.gconv_a(x_dict = features, edge_index_dict = g)
        h = self.drop(self.relu(h))
        h = {key: x.relu() for key, x in h.items()}
        h = self.gconv_b(h, g)

        mask = mask.T
        p1 = h['protein'][mask[0]].detach()
        p2 = h['protein'][mask[1]].detach()
        h = torch.cat((p1, p2), 1)
        return torch.sigmoid(self.linear(h))