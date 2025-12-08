import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from typing import List, Tuple, Optional, Dict

from torch_geometric.data import HeteroData


class HGCN(nn.Module):
    """  Heterogeneous GCN for link classification.
    Encoder: HeteroConv + GCNConv over a HeteroData graph.
    Decoder: relation-agnostic MLP on [h_u || h_v].
    Made compatible with RA_HGCN:
      - encode(data) -> h_dict
      - score_triples(z, edge_index, rel_ids=None) -> (logits, probs)
      - forward(data, edge_index, rel_ids=None) -> (z, logits, probs)
    """

    def __init__(self, in_dim, hidden_dim: int, out_dim: int,
        n_layers: int = 2, ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node", e_etypes: Optional[List[Tuple[str, str, str]]] = None):
        super().__init__()

        if e_etypes is None:
            raise ValueError("HGCN requires e_etypes (list of edge types).")

        self.ppi_etype = ppi_etype
        self.e_types = e_etypes
        self.n_type = n_type
        self.hidden_dim = hidden_dim
        if isinstance(in_dim, dict): base_in = in_dim.get(n_type, next(iter(in_dim.values())))
        else: base_in = int(in_dim)
        self.in_dim_nt = base_in

        convs = []
        for layer in range(n_layers):
            in_ch = base_in if layer == 0 else hidden_dim
            conv_dict = {}
            for (src, rel, dst) in e_etypes:
                conv_dict[(src, rel, dst)] = GCNConv(in_ch, hidden_dim,
                    add_self_loops=True, normalize=True)
            convs.append(HeteroConv(conv_dict, aggr="mean"))
        self.convs = nn.ModuleList(convs)
        self.classify = nn.Linear(2 * hidden_dim, 1)

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        h_dict = x_dict
        for hetero_conv in self.convs:
            h_dict = hetero_conv(h_dict, edge_index_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}
        return h_dict

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor,
        rel_ids: Optional[torch.Tensor] = None):
        src_ids, dst_ids = edge_index
        h_u = z[src_ids]
        h_v = z[dst_ids]
        h_pair = torch.cat([h_u, h_v], dim=-1)
        logits = self.classify(h_pair).view(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: torch.Tensor,
        rel_ids: Optional[torch.Tensor] = None):
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs


# class HGCN(nn.Module):
#     """ Heterogeneous GCN for link classification, expects a torch_geometric.data.HeteroData instance.
#     data.x_dict: {ntype: Tensor[num_nodes_ntype, in_dim_ntype]}; data.edge_index_dict: { (src, rel, dst): LongTensor[2, E_rel]}
#     The `edge_index` passed to forward() are the candidate node pairs (on `n_type`) to classify.
#     """

#     def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"), 
#                  n_type: str = "node", e_etypes: Optional[List[Tuple[str, str, str]]] = None):
                 
#         super().__init__()
#         self.ppi_etype = ppi_etype
#         self.e_types = e_etypes
#         self.n_type = n_type

#         self.convs = nn.ModuleList()
#         for _ in range(n_layers):
#             in_ch = in_dim[n_type] if _ == 0 else hidden_dim
#             conv_dict = {(src, rel, dst): GCNConv(in_ch, hidden_dim, add_self_loops=True, normalize=True)
#                 for (src, rel, dst) in e_etypes}
#             self.convs.append(HeteroConv(conv_dict, aggr='mean'))
#         self.classify = nn.Linear(2 * hidden_dim, out_dim)


#     def forward(self, data, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x_dict = data.x_dict
#         edge_index_dict = data.edge_index_dict
#         h_dict = x_dict
#         for hetero_conv in self.convs:
#             h_dict = hetero_conv(h_dict, edge_index_dict)
#             h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}
#         src_ids, dst_ids = edge_index
#         hs = h_dict[self.n_type][src_ids]
#         hd = h_dict[self.n_type][dst_ids]
#         h_pair = torch.cat([hs, hd], dim=1)
#         logits = self.classify(h_pair)
        
#         z = h_dict[self.n_type]
#         return z, torch.sigmoid(logits)
