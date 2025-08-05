import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLHeteroGraph
from dgl.nn.pytorch import HeteroGraphConv, GraphConv
from typing import List, Tuple, Dict


class HGCN(nn.Module):
    """
    Heterogeneous GCN model for link classification on a specific edge relation (e.g. PPI).

    Args:
        in_feats (Dict[str, int]): Input feature sizes for each node type.
        hidden_dim (int): Hidden embedding dimension.
        out_dim (int): Output dimension for classification (e.g., number of classes).
        n_layers (int): Number of GCN layers.
        ppi_etype (Tuple[str, str, str]): Canonical edge type to classify, e.g. ("protein", "PPI", "protein").
    """
    def __init__(
        self,
        in_feats: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node",
        e_etypes: List[Tuple[str, str, str]] = None,
    ):
        super(HGCN, self).__init__()
        self.ppi_etype = ppi_etype
        self.e_types = e_etypes
        self.n_type = n_type
        self.layers = nn.ModuleList()
        in_dim_maps = [in_feats] + [{etype: hidden_dim for etype in in_feats} for _ in range(n_layers - 1)]
        self.layers = nn.ModuleList([HeteroGraphConv({rel: GraphConv(in_dim_maps[layer_idx][rel], hidden_dim)
                    for src, rel, dst in e_etypes}, aggregate="mean") for layer_idx in range(n_layers)])

        self.classify = nn.Linear(2 * hidden_dim, out_dim)



    def forward(self, graph: DGLHeteroGraph, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h_dict = {ntype: graph.nodes[ntype].data["feat"] for ntype in graph.ntypes}

        for layer in self.layers:
            h_dict = layer(graph, h_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}

        src_ids, dst_ids = edge_index
        hs = h_dict[self.n_type][src_ids]
        hd = h_dict[self.n_type][dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)
        z = h_dict[self.n_type]
        
        return z, torch.sigmoid(logits)