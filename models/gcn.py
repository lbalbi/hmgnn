
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from typing import List, Tuple, Dict


class GCN(nn.Module):
    """
    Homogeneous GCN model for link classification.
    Args:
        in_feats (Dict[str, int]): Input feature sizes for node type. Only the first value is used.
        hidden_dim (int): Hidden embedding dimension.
        out_dim (int): Output dimension for classification (e.g., number of classes).
        n_layers (int): Number of GCN layers.
        ppi_etype (Tuple[str, str, str]): Canonical edge type to classify, e.g. ("node", "PPI", "node").
        n_type (str): Node type (unused in homogeneous graphs, but kept for compatibility).
        e_etypes (List[Tuple[str, str, str]]): Edge types (unused here, for compatibility).
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
        super(GCN, self).__init__()
        input_dim = list(in_feats.values())[0]
        self.n_type = n_type
        self.ppi_etype = ppi_etype

        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
        self.classify = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, graph, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = graph.ndata['feat']

        for layer in self.layers:
            h = layer(graph, h)
            h = F.relu(h)

        src_ids, dst_ids = edge_index
        hs = h[src_ids]
        hd = h[dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)
        z = h

        return z, torch.sigmoid(logits)
