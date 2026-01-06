import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional


class SRA_HGCN(nn.Module):
    """  Relation-aware HGCN with SHGCN-style *dual-view* node embeddings.
    Encoder has two node embedding streams:
          * positive-view: mean over all *positive* relations (those NOT starting with `neg_prefix`)
          * negative-view: mean over all *negative* relations (those starting with `neg_prefix`)
    For each layer and each view, applies the same GCNConv weights to each relation
     in that view and average the resulting messages across relations.
    Decoder (relation-aware triple classifier):
      - Concatenate the two node views: h = [h_pos || h_neg]  (dim = 2*hidden_dim)
      - For triple (u, r, v), build [h_u || e_r || h_v] and score with an MLP -> logit
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, e_etypes: List[Tuple[str, str, str]],
        n_type: str = "node", num_layers: int = 2, rel2id: Optional[Dict[str, int]] = None,
        neg_prefix: str = "NOT_"):
        super().__init__()
        self.n_type = n_type
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.neg_prefix = str(neg_prefix)
        self.dual_view = True

        if rel2id is None:
            rel_names = sorted({rel for (_, rel, _) in e_etypes if not str(rel).startswith(self.neg_prefix)})
            rel2id = {r: i for i, r in enumerate(rel_names)}
        self.rel2id: Dict[str, int] = rel2id
        self.num_rel = len(self.rel2id) if len(self.rel2id) > 0 else 1

        pos_etypes: List[Tuple[str, str, str]] = []
        neg_etypes: List[Tuple[str, str, str]] = []
        for (src_nt, rel, dst_nt) in e_etypes:
            if src_nt != self.n_type or dst_nt != self.n_type: continue
            rel_s = str(rel)
            if rel_s.startswith(self.neg_prefix): neg_etypes.append((src_nt, rel, dst_nt))
            else: pos_etypes.append((src_nt, rel, dst_nt))

        self.pos_etypes = pos_etypes
        self.neg_etypes = neg_etypes
        self.pos_convs = nn.ModuleList()
        self.neg_convs = nn.ModuleList()

        for layer in range(self.num_layers):
            in_ch = self.in_dim if layer == 0 else self.hidden_dim
            self.pos_convs.append(GCNConv(in_ch, self.hidden_dim, add_self_loops=True, normalize=True, cached=False))
            self.neg_convs.append(GCNConv(in_ch, self.hidden_dim, add_self_loops=True, normalize=True, cached=False))
        self.rel_emb = nn.Embedding(self.num_rel, self.hidden_dim)
        triple_in_dim = (2 * self.hidden_dim) + self.hidden_dim + (2 * self.hidden_dim)
        self.classify = nn.Sequential(nn.Linear(triple_in_dim, self.hidden_dim),
            nn.ReLU(), nn.Linear(self.hidden_dim, 1))

    @staticmethod
    def _empty_edge_index(device: torch.device) -> torch.Tensor:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    def _mean_over_relations(self, conv: GCNConv, x: torch.Tensor,
        edge_index_list: List[torch.Tensor]) -> torch.Tensor:
        """Apply `conv` to `x` for each edge_index in `edge_index_list` and average outputs."""
        device = x.device
        if len(edge_index_list) == 0: return conv(x, self._empty_edge_index(device))
        out = None
        for ei in edge_index_list:
            if ei.device != device: ei = ei.to(device)
            y = conv(x, ei)
            out = y if out is None else (out + y)
        return out / float(len(edge_index_list))

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x = data[self.n_type].x
        edge_index_dict = data.edge_index_dict
        h_pos = x
        h_neg = x

        for layer in range(self.num_layers):
            pos_edges = [edge_index_dict[et] for et in self.pos_etypes if et in edge_index_dict]
            neg_edges = [edge_index_dict[et] for et in self.neg_etypes if et in edge_index_dict]
            h_pos = self._mean_over_relations(self.pos_convs[layer], h_pos, pos_edges)
            h_neg = self._mean_over_relations(self.neg_convs[layer], h_neg, neg_edges)
            h_pos = F.relu(h_pos)
            h_neg = F.relu(h_neg)

        h_cat = torch.cat([h_pos, h_neg], dim=-1)
        return {self.n_type: h_cat, f"{self.n_type}_pos": h_pos,
            f"{self.n_type}_neg": h_neg}

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor,
        rel_ids: torch.Tensor):
        """Score triples (u,r,v) given node embeddings z and relation IDs."""
        src, dst = edge_index
        if src.device != z.device:
            src = src.to(z.device)
            dst = dst.to(z.device)
        if rel_ids.device != z.device: rel_ids = rel_ids.to(z.device)
        if rel_ids.dim() == 0: rel_ids = rel_ids.view(-1)
        rel_ids = rel_ids.clamp(min=0, max=self.num_rel - 1)

        h_u = z[src]
        h_v = z[dst]
        e_r = self.rel_emb(rel_ids)
        h_triple = torch.cat([h_u, e_r, h_v], dim=-1)
        logits = self.classify(h_triple).view(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: torch.Tensor,
        rel_ids: torch.Tensor):
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import HeteroData
# from typing import Dict, List, Tuple, Optional

# class SRA_HGCN(nn.Module):
#     """
#     Relation-aware HGCN with SHGCN-style *dual-view* node embeddings.
#     Encoder has two node embedding streams:
#           * positive-view: mean over all *positive* relations (those NOT starting with `neg_prefix`)
#           * negative-view: mean over all *negative* relations (those starting with `neg_prefix`)
#     For each layer and each view, applies the same GCNConv weights to each relation
#      in that view and average the resulting messages across relations.
#     Decoder (relation-aware triple classifier):
#       - Concatenate the two node views: h = [h_pos || h_neg]  (dim = 2*hidden_dim)
#       - For triple (u, r, v), build [h_u || e_r || h_v] and score with an MLP -> logit
#     """

#     def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, e_etypes: List[Tuple[str, str, str]],
#         n_type: str = "node", num_layers: int = 2, rel2id: Optional[Dict[str, int]] = None,
#         neg_prefix: str = "NOT_"):
#         super().__init__()
#         self.n_type = n_type
#         self.in_dim = int(in_dim)
#         self.hidden_dim = int(hidden_dim)
#         self.out_dim = int(out_dim)
#         self.num_layers = int(num_layers)
#         self.neg_prefix = str(neg_prefix)

#         if rel2id is None:
#             rel_names = sorted({rel for (_, rel, _) in e_etypes if not str(rel).startswith(self.neg_prefix)})
#             rel2id = {r: i for i, r in enumerate(rel_names)}
#         self.rel2id: Dict[str, int] = rel2id
#         self.num_rel = len(self.rel2id) if len(self.rel2id) > 0 else 1
#         pos_etypes: List[Tuple[str, str, str]] = []
#         neg_etypes: List[Tuple[str, str, str]] = []
#         for (src_nt, rel, dst_nt) in e_etypes:
#             if src_nt != self.n_type or dst_nt != self.n_type: continue
#             rel_s = str(rel)
#             if rel_s.startswith(self.neg_prefix): neg_etypes.append((src_nt, rel, dst_nt))
#             else: pos_etypes.append((src_nt, rel, dst_nt))
#         self.pos_etypes = pos_etypes
#         self.neg_etypes = neg_etypes
#         self.pos_convs = nn.ModuleList()
#         self.neg_convs = nn.ModuleList()

#         for layer in range(self.num_layers):
#             in_ch = self.in_dim if layer == 0 else self.hidden_dim
#             self.pos_convs.append(GCNConv(in_ch, self.hidden_dim, add_self_loops=True, normalize=True, cached=False))
#             self.neg_convs.append(GCNConv(in_ch, self.hidden_dim, add_self_loops=True, normalize=True, cached=False))
#         self.rel_emb = nn.Embedding(self.num_rel, self.hidden_dim)
#         triple_in_dim = (2 * self.hidden_dim) + self.hidden_dim + (2 * self.hidden_dim)
#         self.classify = nn.Sequential(nn.Linear(triple_in_dim, self.hidden_dim),
#             nn.ReLU(), nn.Linear(self.hidden_dim, 1))

#     @staticmethod
#     def _empty_edge_index(device: torch.device) -> torch.Tensor:
#         return torch.empty((2, 0), dtype=torch.long, device=device)

#     def _mean_over_relations(self, conv: GCNConv, x: torch.Tensor,
#         edge_index_list: List[torch.Tensor]) -> torch.Tensor:
#         """Apply `conv` to `x` for each edge_index in `edge_index_list` and average outputs."""
#         device = x.device
#         if len(edge_index_list) == 0: return conv(x, self._empty_edge_index(device))
#         out = None
#         for ei in edge_index_list:
#             if ei.device != device: ei = ei.to(device)
#             y = conv(x, ei)
#             out = y if out is None else (out + y)
#         return out / float(len(edge_index_list))

#     def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
#         x = data[self.n_type].x
#         edge_index_dict = data.edge_index_dict
#         h_pos = x
#         h_neg = x
#         for layer in range(self.num_layers):
#             pos_edges = [edge_index_dict[et] for et in self.pos_etypes if et in edge_index_dict]
#             neg_edges = [edge_index_dict[et] for et in self.neg_etypes if et in edge_index_dict]
#             h_pos = self._mean_over_relations(self.pos_convs[layer], h_pos, pos_edges)
#             h_neg = self._mean_over_relations(self.neg_convs[layer], h_neg, neg_edges)
#             h_pos = F.relu(h_pos)
#             h_neg = F.relu(h_neg)

#         h_cat = torch.cat([h_pos, h_neg], dim=-1)
#         return {self.n_type: h_cat, f"{self.n_type}_pos": h_pos,
#             f"{self.n_type}_neg": h_neg}

#     def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor,
#         rel_ids: torch.Tensor):
#         """Score triples (u,r,v) given node embeddings z and relation IDs."""
#         src, dst = edge_index
#         if src.device != z.device:
#             src = src.to(z.device)
#             dst = dst.to(z.device)
#         if rel_ids.device != z.device: rel_ids = rel_ids.to(z.device)
#         if rel_ids.dim() == 0: rel_ids = rel_ids.view(-1)
#         rel_ids = rel_ids.clamp(min=0, max=self.num_rel - 1)

#         h_u = z[src]
#         h_v = z[dst]
#         e_r = self.rel_emb(rel_ids)
#         h_triple = torch.cat([h_u, e_r, h_v], dim=-1)
#         logits = self.classify(h_triple).view(-1)
#         probs = torch.sigmoid(logits)
#         return logits, probs

#     def forward(self, data: HeteroData, edge_index: torch.Tensor,
#         rel_ids: torch.Tensor):
#         h_dict = self.encode(data)
#         z = h_dict[self.n_type]
#         logits, probs = self.score_triples(z, edge_index, rel_ids)
#         return z, logits, probs
