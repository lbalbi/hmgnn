import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, HeteroData

CLS_REL = "cls_link"
CLS_EDGE_SUFFIX = "__cls"


class _NBFMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lin_self = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_agg = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_rel_emb: Tensor,
        query: Tensor,
    ) -> Tensor:
        if edge_index.numel() == 0:
            out = self.lin_self(h) + self.lin_query(query).unsqueeze(0)
            out = F.relu(self.norm(out))
            return self.dropout(out)

        src, dst = edge_index
        msg = h[src] + edge_rel_emb
        gate = torch.sigmoid(
            (msg * query.unsqueeze(0)).sum(dim=-1, keepdim=True) / math.sqrt(float(self.hidden_dim))
        )
        msg = msg * gate

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        out = self.lin_self(h) + self.lin_agg(agg) + self.lin_query(query).unsqueeze(0)
        out = F.relu(self.norm(out))
        return self.dropout(out)


class NBFNET(nn.Module):
    """
    NBFNet-inspired baseline for link prediction under the current trainer contract:
      - encode(data) -> {n_type: node_embeddings}
      - score_triples(z, edge_index, rel_ids) -> (logits, probs)

    It caches message-passing edges in `encode`, then performs relation-conditioned
    Bellman-Ford-style propagation in `score_triples` (once per unique relation id
    in the batch) before DistMult scoring on (h, r, t).
    """

    def __init__(
        self,
        in_dim: Union[int, Dict[str, int]],
        hidden_dim: int,
        out_dim: int,
        e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        n_type: str = "node",
        num_layers: int = 3,
        rel2id: Optional[Dict[str, int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if isinstance(in_dim, dict):
            if n_type in in_dim:
                in_dim = int(in_dim[n_type])
            else:
                in_dim = int(next(iter(in_dim.values())))
        else:
            in_dim = int(in_dim)

        self.n_type = str(n_type)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        if rel2id is None:
            rel_names = sorted({str(rel) for (_, rel, _) in (e_etypes or [])})
            rel2id = {rel: i for i, rel in enumerate(rel_names)}
        self.rel2id = dict(rel2id)
        self.num_rel = max(1, len(self.rel2id))

        mp_rel_names = sorted(
            {
                str(rel)
                for (_, rel, _) in (e_etypes or [])
                if rel is not None
                and str(rel) != CLS_REL
                and not str(rel).endswith(CLS_EDGE_SUFFIX)
            }
        )
        if not mp_rel_names:
            mp_rel_names = ["__dummy_rel__"]
        self.mp_rel2id = {rel: i for i, rel in enumerate(mp_rel_names)}
        self.num_mp_rel = len(self.mp_rel2id)

        self.node_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.input_norm = nn.LayerNorm(self.hidden_dim)
        self.input_drop = nn.Dropout(self.dropout)

        self.rel_emb = nn.Embedding(self.num_rel, self.hidden_dim)
        self.mp_rel_emb = nn.Embedding(self.num_mp_rel, self.hidden_dim)

        self.layers = nn.ModuleList(
            [_NBFMessageLayer(self.hidden_dim, dropout=self.dropout) for _ in range(self.num_layers)]
        )

        self._cached_edge_index: Optional[Tensor] = None
        self._cached_edge_type: Optional[Tensor] = None

    def _build_relational(self, data: HeteroData) -> Tuple[Tensor, Tensor, Tensor]:
        x = data[self.n_type].x
        device = x.device
        edge_indices: List[Tensor] = []
        edge_types: List[Tensor] = []

        for (src_nt, rel, dst_nt), eidx in data.edge_index_dict.items():
            if src_nt != self.n_type or dst_nt != self.n_type:
                continue
            if eidx is None or eidx.numel() == 0:
                continue
            rel_name = str(rel)
            if rel_name == CLS_REL or rel_name.endswith(CLS_EDGE_SUFFIX):
                continue
            rid = self.mp_rel2id.get(rel_name, None)
            if rid is None:
                continue

            eidx = eidx.to(device)
            edge_indices.append(eidx)
            edge_types.append(torch.full((eidx.size(1),), rid, dtype=torch.long, device=device))

        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_type = torch.cat(edge_types, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)

        return x, edge_index, edge_type

    def encode(self, data: Union[HeteroData, Data]) -> Dict[str, Tensor]:
        if isinstance(data, HeteroData):
            x, edge_index, edge_type = self._build_relational(data)
        else:
            x = data.x
            edge_index = data.edge_index.to(x.device)
            if hasattr(data, "edge_type") and data.edge_type is not None:
                edge_type = data.edge_type.to(x.device).long()
            else:
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)

        z = self.node_proj(x)
        z = self.input_norm(z)
        z = F.relu(z)
        z = self.input_drop(z)

        self._cached_edge_index = edge_index
        self._cached_edge_type = edge_type
        return {self.n_type: z}

    def _propagate_query(self, z: Tensor, query: Tensor) -> Tensor:
        if self._cached_edge_index is None or self._cached_edge_type is None:
            raise RuntimeError("NBFNET.score_triples called before encode; no cached graph found.")

        edge_index = self._cached_edge_index
        edge_type = self._cached_edge_type
        edge_rel = (
            self.mp_rel_emb(edge_type)
            if edge_type.numel() > 0
            else z.new_empty((0, self.hidden_dim))
        )

        h = z + query.unsqueeze(0)
        for layer in self.layers:
            h = layer(h, edge_index, edge_rel, query)
        return h

    def score_triples(self, z: Tensor, edge_index: Tensor, rel_ids: Tensor):
        if rel_ids.dim() == 0:
            rel_ids = rel_ids.view(-1)
        rel_ids = rel_ids.long()

        if rel_ids.numel() != edge_index.size(1):
            raise ValueError(
                f"Mismatch between rel_ids ({rel_ids.numel()}) and edge_index columns ({edge_index.size(1)})."
            )
        if rel_ids.numel() and ((rel_ids < 0).any() or (rel_ids >= self.num_rel).any()):
            bad = rel_ids[(rel_ids < 0) | (rel_ids >= self.num_rel)][:10].tolist()
            raise ValueError(f"rel_ids out of range (num_rel={self.num_rel}). Examples: {bad}")

        src, dst = edge_index
        logits = z.new_empty((rel_ids.size(0),), dtype=z.dtype)
        unique_rel = torch.unique(rel_ids).detach().cpu().tolist()

        query_states: Dict[int, Tensor] = {}
        for rid in unique_rel:
            query = self.rel_emb.weight[int(rid)]
            query_states[int(rid)] = self._propagate_query(z, query)

        for rid, state in query_states.items():
            mask = rel_ids == int(rid)
            if not torch.any(mask):
                continue
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            b_src = src[idx]
            b_dst = dst[idx]

            h_src = state[b_src]
            h_dst = state[b_dst]
            q = self.rel_emb.weight[int(rid)].unsqueeze(0).expand_as(h_src)
            logits[idx] = (h_src * q * h_dst).sum(dim=-1)

        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: Union[HeteroData, Data], edge_index: Tensor, rel_ids: Tensor):
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs
