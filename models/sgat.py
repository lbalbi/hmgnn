# signed_gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, dropout_edge
from torch_geometric.nn import GATv2Conv

CLS_REL = "cls_link"


class SignedGATConv(nn.Module):
    """
    Signed-GAT layer with two channels:
      h_pos (balanced/friend) and h_neg (unbalanced/enemy)

    Update (SGCN-style routing, but attention-based aggregation):
      pos <- f([ self_pos , GAT_pos(h_pos), GAT_neg(h_neg) ])
      neg <- f([ self_neg , GAT_pos(h_neg), GAT_neg(h_pos) ])
    """
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 2,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.dropout = float(dropout)

        # We keep output dimension = hidden_dim by using concat=True then projecting back,
        # OR use concat=False directly. Here: concat=False keeps output dim hidden_dim.
        self.gat_pos = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=attn_dropout,
            add_self_loops=False,
        )
        self.gat_neg = GATv2Conv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            heads=self.heads,
            concat=False,
            dropout=attn_dropout,
            add_self_loops=False,
        )

        self.lin_pos = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.lin_neg = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

        self.norm_pos = nn.LayerNorm(self.hidden_dim) if norm else nn.Identity()
        self.norm_neg = nn.LayerNorm(self.hidden_dim) if norm else nn.Identity()

    def forward(
        self,
        h_pos: Tensor,
        h_neg: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Attention aggregates
        pos_from_pos = self.gat_pos(h_pos, pos_edge_index) if pos_edge_index.numel() else h_pos.new_zeros(h_pos.shape)
        neg_from_neg = self.gat_neg(h_neg, neg_edge_index) if neg_edge_index.numel() else h_neg.new_zeros(h_neg.shape)

        pos_from_neg = self.gat_pos(h_neg, pos_edge_index) if pos_edge_index.numel() else h_neg.new_zeros(h_neg.shape)
        neg_from_pos = self.gat_neg(h_pos, neg_edge_index) if neg_edge_index.numel() else h_pos.new_zeros(h_pos.shape)

        pos_in = torch.cat([h_pos, pos_from_pos, neg_from_neg], dim=-1)
        neg_in = torch.cat([h_neg, pos_from_neg, neg_from_pos], dim=-1)

        h_pos = self.lin_pos(pos_in)
        h_neg = self.lin_neg(neg_in)

        h_pos = self.norm_pos(h_pos)
        h_neg = self.norm_neg(h_neg)

        h_pos = F.relu(h_pos)
        h_neg = F.relu(h_neg)

        if self.dropout > 0:
            h_pos = F.dropout(h_pos, p=self.dropout, training=self.training)
            h_neg = F.dropout(h_neg, p=self.dropout, training=self.training)

        return h_pos, h_neg


class SGAT(nn.Module):
    """
    Signed GAT for your framework.

    Contract:
      - encode(data) -> dict with keys:
            'node' (2H), 'node_pos'(H), 'node_neg'(H)
      - score_triples(z, edge_index, rel_ids) -> (logits, probs)
      - forward(...) -> choose 2-return or 3-return depending on trainer
    """

    def __init__(
        self,
        in_dim,
        hidden_dim: int,
        out_dim: int = 1,
        n_layers: int = 2,
        heads: int = 2,
        attn_dropout: float = 0.0,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node",
        e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        rel2id: Optional[Dict[str, int]] = None,
        neg_prefix: str = "NOT_",
        pos_rel_name: str = "pos_statement",
        neg_rel_name: str = "neg_statement",
        dropout: float = 0.2,
        drop_edge_p: float = 0.0,
        add_self_loops_flag: bool = True,
        make_undirected: bool = True,
        norm: bool = True,
        return_probs_only: bool = True,  # <- set True if trainer uses BCELoss
    ):
        super().__init__()
        self.n_type = n_type
        self.ppi_etype = ppi_etype
        self.e_etypes = e_etypes or []
        self.neg_prefix = str(neg_prefix)
        self.pos_rel_name = str(pos_rel_name)
        self.neg_rel_name = str(neg_rel_name)

        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.drop_edge_p = float(drop_edge_p)
        self.add_self_loops_flag = bool(add_self_loops_flag)
        self.make_undirected = bool(make_undirected)

        # Trainer detection uses this; keep False and let trainer infer via keys.
        self.dual_view = False

        self.return_probs_only = bool(return_probs_only)

        if isinstance(in_dim, dict):
            in_dim_ = int(in_dim.get(n_type, next(iter(in_dim.values()))))
        else:
            in_dim_ = int(in_dim)

        self.input_proj = nn.Linear(in_dim_, self.hidden_dim)

        self.layers = nn.ModuleList([
            SignedGATConv(
                hidden_dim=self.hidden_dim,
                heads=heads,
                dropout=self.dropout,
                attn_dropout=attn_dropout,
                norm=norm
            )
            for _ in range(int(n_layers))
        ])

        # DistMult over signed embedding space z=[z_pos||z_neg] (dim=2H).
        if rel2id is None:
            rel_names = sorted({str(rel) for (_, rel, _) in (self.e_etypes or [])})
            self.num_rel = max(1, len(rel_names))
        else:
            self.num_rel = max(1, len(rel2id))
        self.rel_emb = nn.Embedding(self.num_rel, 2 * self.hidden_dim)

    def _get_node_features(self, data: HeteroData) -> Tensor:
        nt = data[self.n_type]
        if hasattr(nt, "x") and nt.x is not None:
            return nt.x
        if hasattr(nt, "feat") and getattr(nt, "feat") is not None:
            return getattr(nt, "feat")
        raise ValueError(f"{self.__class__.__name__}: node features not found at data['{self.n_type}'].x (or .feat).")

    def _split_pos_neg_edges(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        device = self._get_node_features(data).device
        pos_parts: List[Tensor] = []
        neg_parts: List[Tensor] = []

        for (src, rel, dst) in data.edge_types:
            if src != self.n_type or dst != self.n_type:
                continue
            if rel == CLS_REL:
                continue
            store = data[(src, rel, dst)]
            if "edge_index" not in store or store.edge_index is None or store.edge_index.numel() == 0:
                continue

            eidx = store.edge_index.to(device)
            is_neg = (rel == self.neg_rel_name) or (isinstance(rel, str) and rel.startswith(self.neg_prefix))
            (neg_parts if is_neg else pos_parts).append(eidx)

        def _cat(parts: List[Tensor]) -> Tensor:
            if not parts:
                return torch.empty((2, 0), dtype=torch.long, device=device)
            return torch.cat(parts, dim=1)

        pos_edge_index = _cat(pos_parts)
        neg_edge_index = _cat(neg_parts)

        if self.make_undirected:
            if pos_edge_index.numel() > 0:
                pos_edge_index = torch.cat([pos_edge_index, pos_edge_index.flip(0)], dim=1)
            if neg_edge_index.numel() > 0:
                neg_edge_index = torch.cat([neg_edge_index, neg_edge_index.flip(0)], dim=1)

        if self.training and self.drop_edge_p > 0.0:
            if pos_edge_index.numel() > 0:
                pos_edge_index, _ = dropout_edge(pos_edge_index, p=self.drop_edge_p)
            if neg_edge_index.numel() > 0:
                neg_edge_index, _ = dropout_edge(neg_edge_index, p=self.drop_edge_p)

        if self.add_self_loops_flag:
            x = self._get_node_features(data)
            if x.size(0) > 0:
                pos_edge_index, _ = add_self_loops(pos_edge_index, num_nodes=x.size(0))

        return pos_edge_index, neg_edge_index

    def encode(self, data: HeteroData) -> Dict[str, Tensor]:
        x = self._get_node_features(data)
        n = x.size(0)

        xh = self.input_proj(x)
        xh = F.relu(xh)
        if self.dropout > 0:
            xh = F.dropout(xh, p=self.dropout, training=self.training)

        # Two channels
        h_pos = xh
        h_neg = xh.new_zeros((n, self.hidden_dim))

        pos_edge_index, neg_edge_index = self._split_pos_neg_edges(data)

        for layer in self.layers:
            h_pos, h_neg = layer(h_pos, h_neg, pos_edge_index, neg_edge_index)

        z_pos = h_pos
        z_neg = h_neg
        z = torch.cat([z_pos, z_neg], dim=-1)

        return {
            self.n_type: z,
            f"{self.n_type}_pos": z_pos,
            f"{self.n_type}_neg": z_neg,
        }

    def score_triples(
        self,
        z: Tensor,
        edge_index: Tensor,
        rel_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        zu = z[src]
        zv = z[dst]
        if rel_ids is not None:
            rel_ids = rel_ids.long()
            if (rel_ids < 0).any() or (rel_ids >= self.num_rel).any():
                bad = rel_ids[(rel_ids < 0) | (rel_ids >= self.num_rel)][:10].tolist()
                raise ValueError(f"rel_ids out of range (num_rel={self.num_rel}). Examples: {bad}")
            er = self.rel_emb(rel_ids)
            logits = (zu * er * zv).sum(dim=-1)
        else:
            logits = (zu * zv).sum(dim=-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: Tensor, rel_ids: Optional[Tensor] = None):
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits
