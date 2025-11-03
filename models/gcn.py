import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Tuple, Dict

class GCN(nn.Module):
    """
    Homogeneous GCN model for link classification (PyTorch Geometric).

    Args:
        in_feats (Dict[str, int]): Input feature sizes for node type. Only the first value is used.
        hidden_dim (int): Hidden embedding dimension.
        out_dim (int): Output dimension for classification (e.g., number of classes).
        n_layers (int): Number of GCN layers.
        ppi_etype (Tuple[str, str, str]): Unused; kept for API compatibility.
        n_type (str): Unused in homogeneous graphs; kept for compatibility.
        e_etypes (List[Tuple[str, str, str]]): Unused; kept for compatibility.
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
        super().__init__()
        input_dim = list(in_feats.values())[0]
        self.n_type = n_type
        self.ppi_etype = ppi_etype

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True))
        self.classify = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, features, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        full_edge_index = edge_index
        h = features
        for conv in self.convs:
            h = conv(h, full_edge_index)
            h = F.relu(h)
        src_ids, dst_ids = edge_index
        hs = h[src_ids]
        hd = h[dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)
        z = h
        return z, torch.sigmoid(logits)
