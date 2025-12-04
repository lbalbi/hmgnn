import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from typing import List, Tuple, Optional


class SHGCN(nn.Module):
    """ Heterogeneous GCN that computes two node embeddings:
    - z_pos: from all edge types except 'neg_statement'
    - z_neg: from only the 'neg_statement' edge type
    It returns (z_pos, z_neg, out), where z_pos, z_neg: [num_nodes(n_type), hidden_dim]
    and out is [num_candidate_edges, out_dim] (sigmoid-probabilities for classfication pairs)
    """

    def __init__(self, in_dim, hidden_dim: int, out_dim: int = 1, num_layers: int = 2,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"), n_type: str = "node",
        e_etypes: Optional[List[Tuple[str, str, str]]] = None, neg_rel_name: str = "neg_statement"):
        super().__init__()
        assert e_etypes is not None and len(e_etypes) > 0, "e_etypes must be provided"
        self.n_type = n_type
        self.ppi_etype = ppi_etype
        self.e_types = e_etypes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.neg_rel_name = neg_rel_name

        if isinstance(in_dim, dict): base_in_ch = in_dim[self.n_type]
        else: base_in_ch = int(in_dim)

        neg_candidates = [et for et in e_etypes if et[1] == neg_rel_name]
        self.neg_etype = None if len(neg_candidates) == 0 else neg_candidates[0]
        self.pos_etypes = [et for et in e_etypes if et != self.neg_etype]
        if self.neg_etype is None: print("[HGCN] WARNING: No edge type with rel =", neg_rel_name,
                  "found in e_etypes. z_neg will be zeros.")

        self.convs_pos = nn.ModuleList()
        self.convs_neg = nn.ModuleList()

        for layer in range(num_layers):
            in_ch = base_in_ch if layer == 0 else hidden_dim
            conv_dict_pos = {et: GCNConv(in_ch, hidden_dim, add_self_loops=True, normalize=True)
                for et in self.pos_etypes}
            conv_dict_neg = {}
            if self.neg_etype is not None:
                conv_dict_neg[self.neg_etype] = GCNConv(in_ch, hidden_dim, add_self_loops=True, normalize=True)
                
            self.convs_pos.append(HeteroConv(conv_dict_pos, aggr="mean"))
            self.convs_neg.append(HeteroConv(conv_dict_neg, aggr="mean"))
        self.classify = nn.Linear(4 * hidden_dim, out_dim)


    def forward(self, data, edge_index_pairs: torch.Tensor):

        x_dict = data.x_dict
        full_edge_index_dict = data.edge_index_dict
        pos_edge_index_dict = {}
        neg_edge_index_dict = {}

        for et, eidx in full_edge_index_dict.items():
            if et == self.neg_etype: neg_edge_index_dict[et] = eidx
            else: pos_edge_index_dict[et] = eidx

        h_pos_dict = x_dict
        for hetero_conv in self.convs_pos:
            h_pos_dict = hetero_conv(h_pos_dict, pos_edge_index_dict)
            h_pos_dict = {nt: F.relu(h) for nt, h in h_pos_dict.items()}

        if self.neg_etype is None:
            h_neg_dict = {nt: torch.zeros_like(h_pos_dict[nt]) for nt in h_pos_dict.keys()}
        else:
            h_neg_dict = x_dict
            for hetero_conv in self.convs_neg:
                h_neg_dict = hetero_conv(h_neg_dict, neg_edge_index_dict)
                h_neg_dict = {nt: F.relu(h) for nt, h in h_neg_dict.items()}

        z_pos = h_pos_dict[self.n_type]
        z_neg = h_neg_dict[self.n_type]
        src_ids, dst_ids = edge_index_pairs
        z_cat = torch.cat([z_pos, z_neg], dim=-1)
        hs = z_cat[src_ids]
        hd = z_cat[dst_ids]
        h_pair = torch.cat([hs, hd], dim=-1)
        logits = self.classify(h_pair)
        out = torch.sigmoid(logits)

        return z_pos, z_neg, out
