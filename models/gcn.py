import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Tuple, Dict, Optional, Union
from torch_geometric.data import HeteroData, Data


class GCN(nn.Module):
    """
    Homogeneous GCN encoder that can consume a HeteroData batch by
    collapsing all edges of the main node type into a single edge_index.
    Decoder is relation-aware in the same style as RA_HGCN:
      - embeds relation IDs
      - MLP over [h_u, e_r, h_v] -> logit
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        n_type: str = "node",
        num_layers: int = 2,
        rel2id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        if isinstance(in_dim, dict):
            if n_type in in_dim: in_dim = int(in_dim[n_type])
            else: in_dim = int(next(iter(in_dim.values())))
        else: in_dim = int(in_dim)

        self.n_type = n_type
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # ----- GCN encoder over a single homogeneous graph -----
        convs: List[GCNConv] = []
        for layer in range(num_layers):
            in_ch = in_dim if layer == 0 else hidden_dim
            convs.append(GCNConv(in_ch, hidden_dim))
        self.convs = nn.ModuleList(convs)

        ## ----- Relation embeddings for the decoder -----
        if rel2id is None:
        #     # fall back to 1 relation (won't crash if forgot to pass)
            self.num_rel = 1
        else:
            self.num_rel = len(rel2id)

        self.rel_emb = nn.Embedding(self.num_rel, hidden_dim)

        ## ----- Triple classifier: [h_u, e_r, h_v] -> logit -----
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_homogeneous(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collapse all edges of node type `self.n_type` into a single homogeneous
        edge_index.  Edge types are ignored.
        """
        x = data[self.n_type].x
        device = x.device

        edges = []
        for (s, r, d), eidx in data.edge_index_dict.items():
            if s == self.n_type and d == self.n_type:
                edges.append(eidx)

        if len(edges) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(edges, dim=1).to(device)

        return x, edge_index

    # ------------------------------------------------------------------
    # Public API: encode / score_triples / forward
    # ------------------------------------------------------------------
    def encode(self, data: Union[HeteroData, Data]) -> Dict[str, torch.Tensor]:
        """
        If data is HeteroData, collapse to homogeneous graph and run GCNConv.
        If data is already homogeneous Data, just use its x and edge_index.
        Returns: { self.n_type: node_embeddings }
        """
        if isinstance(data, HeteroData):
            x, edge_index = self._build_homogeneous(data)
        else:
            x, edge_index = data.x, data.edge_index

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        return {self.n_type: h}

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor, rel_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Same signature as RA_HGCN.score_triples.
        z: [num_nodes, hidden_dim]
        edge_index: [2, B]
        rel_ids: [B]
        """
        src, dst = edge_index
        h_u = z[src]
        h_v = z[dst]

        if rel_ids.dim() == 0:
            rel_ids = rel_ids.view(-1)
        # rel_ids = rel_ids.clamp(min=0, max=self.num_rel - 1)
        if (rel_ids < 0).any() or (rel_ids >= self.num_rel).any():
            bad = rel_ids[(rel_ids < 0) | (rel_ids >= self.num_rel)][:10].tolist()
            raise ValueError(f"rel_ids out of range (num_rel={self.num_rel}). Examples: {bad}")

        e_r = self.rel_emb(rel_ids)
        h_pair = torch.cat([h_u, e_r, h_v], dim=-1)
        #h_pair = torch.cat([h_u, h_v], dim=-1)
        logits = self.classify(h_pair).view(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(
        self,
        data: Union[HeteroData, Data],
        edge_index: torch.Tensor,
        rel_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience forward if you ever call the model directly.
        """
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs


# class GCN(nn.Module):
#     """
#     Homogeneous GCN model for link classification (PyTorch Geometric).

#     Args:
#         in_feats (Dict[str, int]): Input feature sizes for node type. Only the first value is used.
#         hidden_dim (int): Hidden embedding dimension.
#         out_dim (int): Output dimension for classification (e.g., number of classes).
#         n_layers (int): Number of GCN layers.
#         ppi_etype (Tuple[str, str, str]): Unused; kept for API compatibility.
#         n_type (str): Unused in homogeneous graphs; kept for compatibility.
#         e_etypes (List[Tuple[str, str, str]]): Unused; kept for compatibility.
#     """

#     def __init__(
#         self,
#         in_feats: Dict[str, int],
#         hidden_dim: int,
#         out_dim: int,
#         n_layers: int = 2,
#         ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
#         n_type: str = "node",
#         e_etypes: List[Tuple[str, str, str]] = None,
#     ):
#         super().__init__()
#         input_dim = list(in_feats.values())[0]
#         self.n_type = n_type
#         self.ppi_etype = ppi_etype

#         self.convs = nn.ModuleList()
#         self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=False, normalize=True))
#         for _ in range(n_layers - 1):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=True))
#         self.classify = nn.Linear(2 * hidden_dim, out_dim)

#     def forward(self, features, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

#         full_edge_index = edge_index
#         h = features
#         for conv in self.convs:
#             h = conv(h, full_edge_index)
#             h = F.relu(h)
#         src_ids, dst_ids = edge_index
#         hs = h[src_ids]
#         hd = h[dst_ids]
#         h_pair = torch.cat([hs, hd], dim=1)
#         logits = self.classify(h_pair)
#         z = h
#         return z, torch.sigmoid(logits)
