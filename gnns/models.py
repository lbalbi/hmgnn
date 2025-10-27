import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import HeteroData
from typing import Optional, Iterator, Tuple, List, Dict


def _edge_pairs_emb(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Concatenate embeddings of (src, dst) along last dim: [E, 2*F]."""
    return torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)

def _maybe_mask(edge_store: Dict, edge_index: torch.Tensor, mask_name: str, val: bool) -> torch.Tensor:
    """Return indices of edges selected by mask if present; else all edges."""
    if mask_name == "all": return torch.arange(edge_index.size(1), device=edge_index.device)
    m = edge_store.get(mask_name, None)
    if m is None: return torch.arange(edge_index.size(1), device=edge_index.device)
    return torch.where(~m if val else m)[0]


class RGCNLayer(nn.Module):
    """
    Simple relation-wise layer:
    For each relation r, apply W_r to src features and aggregate into dst by sum.
    Then sum over relations, optionally subtract the contribution of one 'subtract_rel'.
    Includes self-loop transform.
    """
    def __init__(self, in_feat: int, out_feat: int, rel_names: List[str], subtract_rel: Optional[str] = None):
        super().__init__()
        self.rel_names = list(rel_names)
        self.subtract_rel = subtract_rel
        self.weights = nn.ModuleDict({rel: nn.Linear(in_feat, out_feat, bias=False) for rel in self.rel_names})
        self.self_loop = nn.Linear(in_feat, out_feat, bias=False)

    def forward(self, g: HeteroData, ntype: str, h: torch.Tensor) -> torch.Tensor:
        num_nodes = h.size(0)
        device = h.device
        out_per_rel = {}

        for rel in self.rel_names:
            et = (ntype, rel, ntype)
            if et not in g.edge_types or 'edge_index' not in g[et]:
                out_per_rel[rel] = torch.zeros(num_nodes, self.self_loop.out_features, device=device)
                continue
            edge_index = g[et].edge_index
            src = edge_index[0]
            dst = edge_index[1]
            msg = self.weights[rel](h)[src]
            out = torch.zeros(num_nodes, msg.size(1), device=device)
            out.index_add_(0, dst, msg)
            out_per_rel[rel] = out
        if len(out_per_rel) == 0: agg = torch.zeros(num_nodes, self.self_loop.out_features, device=device)
        else:
            agg = sum(v for k, v in out_per_rel.items() if k != self.subtract_rel)
            if self.subtract_rel in out_per_rel: agg = agg - out_per_rel[self.subtract_rel]
        h_self = self.self_loop(h)
        h_out = agg + h_self
        return F.relu(h_out)


class RGCN(nn.Module):
    """
    Retains the original idea: build node embeddings via stacked RGCNLayer.
    For pair scoring, we concatenate (u,v) embeddings and pass through a final MLP.
    """
    def __init__(self, num_nodes: int, h_dim: int, out_dim: int, rel_names: List[str],
                 subtract_rel: Optional[str] = None, num_layers: int = 2, predictee_n: str = "node"):
        super().__init__()
        assert num_layers >= 1
        self.predictee_n = predictee_n
        self.emb = nn.Embedding(num_nodes, h_dim)
        layers: List[nn.Module] = []
        in_c = h_dim
        for _ in range(num_layers - 1):
            layers.append(RGCNLayer(in_c, in_c, rel_names, subtract_rel))
        self.layers = nn.ModuleList(layers)
        self.final_node = RGCNLayer(in_c, out_dim, rel_names, subtract_rel)
        self.scorer = nn.Sequential(nn.ReLU(), nn.Linear(2 * out_dim, 1))

    def forward(self, g: HeteroData, e_types: List[str], n_pairs: torch.Tensor,
                predictee_n: str = "node", mask: str = "all", val: bool = False) -> torch.Tensor:
        h = self.emb.weight
        for layer in self.layers:
            h = layer(g, predictee_n, h)
        h = self.final_node(g, predictee_n, h)

        pair_feats = []
        for rel in e_types:
            et = (predictee_n, rel, predictee_n)
            if et not in g.edge_types or 'edge_index' not in g[et]: continue
            ei = g[et].edge_index
            keep = _maybe_mask(g[et], ei, mask, val)
            if keep.numel() == 0: continue
            pair_feats.append(_edge_pairs_emb(h, ei[:, keep]))

        if n_pairs.numel(): pair_feats.append(torch.cat([h[n_pairs[:, 0]], h[n_pairs[:, 1]]], dim=1))
        if len(pair_feats) == 0: return torch.empty(0, 1, device=h.device)
        z = torch.cat(pair_feats, dim=0)
        return torch.sigmoid(self.scorer(z))



class HeteroGCN_dgl(nn.Module):
    """PyG replacement (keeps class name & forward signature)."""
    def __init__(self, n_features=128, n_classes=1, n_hidden=256, relations=None,
                 dropout=0.0, lr=0.01, weight_decay=0.01):
        super().__init__()
        relations = relations or []
        self.relations = [rel for rel in relations if rel is not None]
        self.name = "HeteroGCN"
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self._convs1: Dict[Tuple[str, str,]()]()_
