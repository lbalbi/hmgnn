import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLHeteroGraph
from dgl.nn.pytorch import HeteroGraphConv
from layers import GATLayer
from typing import List, Tuple, Dict


class HeteroGCN(nn.Module):
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
        ppi_etype: Tuple[str, str, str] = None
    ):
        super().__init__()
        self.ppi_etype = ppi_etype

        self.layers = nn.ModuleList()
        self.layers.append(
            HeteroGraphConv({ntype: GATLayer(in_feats[ntype], hidden_dim) for ntype in in_feats
            }, 1, aggregate='mean'))

        for _ in range(n_layers - 1):
            self.layers.append(
                HeteroGraphConv({ ntype: GATLayer(hidden_dim, hidden_dim) for ntype in in_feats
                }, 1, aggregate='mean'))
        self.classify = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, graph: DGLHeteroGraph, pairs: List[Tuple[int, int]]) -> torch.Tensor:

        h_dict = {ntype: graph.nodes[ntype].data['feat']
            for ntype in graph.ntypes}
        
        for layer in self.layers:
            h_dict = layer(graph, h_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        if len(pairs) == 0:
            return torch.empty(0, self.classify.out_features, device=next(self.parameters()).device)
        src_ids = torch.tensor([u for u, _ in pairs], dtype=torch.long,
                               device=next(self.parameters()).device)
        dst_ids = torch.tensor([v for _, v in pairs], dtype=torch.long,
                               device=next(self.parameters()).device)

        mask = graph.has_edges_between(src_ids, dst_ids, etype=self.ppi_etype)
        valid_src = src_ids[mask]
        valid_dst = dst_ids[mask]
        if valid_src.numel() == 0:
            return torch.empty(0, self.classify.out_features, device=valid_src.device)

        src_ntype, _, dst_ntype = self.ppi_etype
        hs = h_dict[src_ntype][valid_src]
        hd = h_dict[dst_ntype][valid_dst]
        h_pair = torch.cat([hs, hd], dim=1)
        out = self.classify(h_pair)
        return torch.sigmoid(out)