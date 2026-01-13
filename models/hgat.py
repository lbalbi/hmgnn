import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from typing import List, Tuple, Dict, Optional


class HGAT(nn.Module):
    """
    Heterogeneous GAT for link classification.
    Uses classic multi-head concatenation (concat=True) while keeping the total
    embedding size fixed by setting head_dim = hidden_dim // heads.

    Expects a torch_geometric.data.HeteroData instance:
      data.x_dict: {ntype: Tensor[num_nodes_ntype, in_dim_ntype]}
      data.edge_index_dict: {(src, rel, dst): LongTensor[2, E_rel]}
    The `edge_index` passed to forward() are the candidate node pairs (on `n_type`) to classify.
    """

    def __init__(self, in_dim: Dict[str, int], hidden_dim: int,
        out_dim: int, n_layers: int = 2, heads: int = 4, dropout: float = 0.0,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node", e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        aggr: str = "mean"):
        super().__init__()
        if e_etypes is None:
            raise ValueError("HGAT requires e_etypes (list of (src, rel, dst) edge types).")
        self.ppi_etype = ppi_etype
        self.e_types = e_etypes
        self.n_type = n_type
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}) for concat=True.")
        head_dim = hidden_dim // heads

        self.convs = nn.ModuleList()
        for layer in range(n_layers):
            conv_dict = {}
            for (src, rel, dst) in e_etypes:
                # For hetero GAT, GATConv supports bipartite input via (in_src, in_dst).
                if layer == 0:in_ch = (in_dim[src], in_dim[dst]) if src != dst else in_dim[src]
                else: in_ch = (hidden_dim, hidden_dim) if src != dst else hidden_dim
                conv_dict[(src, rel, dst)] = GATConv(in_ch, head_dim, heads=heads,
                    concat=True, dropout=dropout, add_self_loops=(src == dst))
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))
        self.classify = nn.Linear(2 * hidden_dim, out_dim)


    def forward(self, data, edge_index: torch.Tensor):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        h_dict = x_dict
        for hetero_conv in self.convs:
            out_dict = hetero_conv(h_dict, edge_index_dict)
            h_dict = {nt: F.relu(out_dict.get(nt, h_dict[nt])) for nt in h_dict.keys()}

        src_ids, dst_ids = edge_index
        hs = h_dict[self.n_type][src_ids]
        hd = h_dict[self.n_type][dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)
        z = h_dict[self.n_type]
        return z, torch.sigmoid(logits)
