import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData


class _NBFMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lin_self = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_msg = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        h: Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        etype2id: Dict[Tuple[str, str, str], int],
        rel_emb: nn.Embedding,
    ) -> Tensor:
        num_nodes = h.size(0)
        agg = torch.zeros_like(h)
        deg = torch.zeros((num_nodes, 1), device=h.device, dtype=h.dtype)

        for et, eidx in edge_index_dict.items():
            rid = etype2id.get(et, None)
            if rid is None:
                continue
            if eidx is None or eidx.numel() == 0:
                continue

            src, dst = eidx
            rvec = rel_emb.weight[rid].unsqueeze(0)
            msg = h[src] + rvec
            gate = torch.sigmoid((msg * rvec).sum(dim=-1, keepdim=True) / math.sqrt(float(self.hidden_dim)))
            msg = msg * gate

            agg.index_add_(0, dst, msg)
            deg.index_add_(0, dst, torch.ones((dst.size(0), 1), device=h.device, dtype=h.dtype))

        agg = agg / deg.clamp(min=1.0)
        out = self.lin_self(h) + self.lin_msg(agg)
        out = F.relu(self.norm(out))
        return self.drop(out)


class NBFNET(nn.Module):
    """
    NBFNet-inspired relation-aware baseline for this branch's pair-classification API.
    It uses relation-conditioned message passing over HeteroData edge types and predicts
    candidate pairs from the resulting node embeddings.
    """

    def __init__(
        self,
        in_dim: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node",
        e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_type = n_type
        self.ppi_etype = ppi_etype
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)

        input_dim = int(in_dim[n_type]) if isinstance(in_dim, dict) else int(in_dim)
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_drop = nn.Dropout(self.dropout)

        etypes = [tuple(et) for et in (e_etypes or [])]
        self.mp_etypes = [et for et in etypes if et[0] == self.n_type and et[2] == self.n_type]
        if not self.mp_etypes:
            self.mp_etypes = [self.ppi_etype]

        self.etype2id: Dict[Tuple[str, str, str], int] = {et: i for i, et in enumerate(self.mp_etypes)}
        self.rel_emb = nn.Embedding(len(self.etype2id), self.hidden_dim)

        self.layers = nn.ModuleList(
            [_NBFMessageLayer(self.hidden_dim, dropout=self.dropout) for _ in range(int(n_layers))]
        )

        self.classify = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1 if self.out_dim == 1 else self.out_dim),
        )

    def _collect_edge_index_dict(self, data: HeteroData) -> Dict[Tuple[str, str, str], Tensor]:
        out: Dict[Tuple[str, str, str], Tensor] = {}
        for et in self.mp_etypes:
            if et not in data.edge_types:
                continue
            eidx = data[et].edge_index
            if eidx is None:
                continue
            out[et] = eidx
        return out

    def forward(self, data: HeteroData, edge_index: Tensor):
        x = data[self.n_type].x
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = self.input_drop(h)

        edge_index_dict = self._collect_edge_index_dict(data)
        for layer in self.layers:
            h = layer(h, edge_index_dict, self.etype2id, self.rel_emb)

        src_ids, dst_ids = edge_index
        hs = h[src_ids]
        hd = h[dst_ids]
        pair = torch.cat([hs, hd, torch.abs(hs - hd), hs * hd], dim=-1)
        logits = self.classify(pair)
        probs = torch.sigmoid(logits)
        return h, probs
