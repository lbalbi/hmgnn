import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from typing import List, Tuple, Optional


class HGCN(nn.Module):
    """ Heterogeneous GCN for link classification, expects a torch_geometric.data.HeteroData instance.
    data.x_dict: {ntype: Tensor[num_nodes_ntype, in_dim_ntype]}; data.edge_index_dict: { (src, rel, dst): LongTensor[2, E_rel]}
    The `edge_index` passed to forward() are the candidate node pairs (on `n_type`) to classify.
    """

    def __init__(self, hidden_dim: int, out_dim: int, n_layers: int = 2, ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"), 
                 n_type: str = "node", e_etypes: Optional[List[Tuple[str, str, str]]] = None):
        super().__init__()
        self.ppi_etype = ppi_etype
        self.e_types = e_etypes
        self.n_type = n_type

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {(src, rel, dst): GCNConv(-1, hidden_dim, add_self_loops=True, normalize=True)
                for (src, rel, dst) in e_etypes}
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        self.classify = nn.Linear(2 * hidden_dim, out_dim)


    def forward(self, data, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        h_dict = x_dict
        for hetero_conv in self.convs:
            h_dict = hetero_conv(h_dict, edge_index_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}
        src_ids, dst_ids = edge_index
        hs = h_dict[self.n_type][src_ids]
        hd = h_dict[self.n_type][dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)

        z = h_dict[self.n_type]
        return z, torch.sigmoid(logits)
