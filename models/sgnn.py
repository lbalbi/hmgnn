import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops, dropout_edge

try:
    from torch_geometric.utils import scatter
except Exception:
    from torch_scatter import scatter


def _mean_aggregate(x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Mean aggregation of src features into dst nodes.
    edge_index: [2, E] with messages src -> dst
    """
    if edge_index is None or edge_index.numel() == 0:
        return x.new_zeros((num_nodes, x.size(-1)))
    src, dst = edge_index[0], edge_index[1]
    out = scatter(x[src], dst, dim=0, dim_size=num_nodes, reduce="mean")
    return out


class SignedConv(nn.Module):
    """  SGCN-style signed update with two channels: h_pos, h_neg
    Update intuition:
      pos uses: self pos + pos-neighbors pos + neg-neighbors neg
      neg uses: self neg + pos-neighbors neg + neg-neighbors pos
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.2, norm: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lin_pos = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.lin_neg = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.norm_pos = nn.LayerNorm(self.hidden_dim) if norm else nn.Identity()
        self.norm_neg = nn.LayerNorm(self.hidden_dim) if norm else nn.Identity()

    def forward(self, h_pos: Tensor, h_neg: Tensor, pos_edge_index: Tensor,
        neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        n = h_pos.size(0)

        pos_from_pos = _mean_aggregate(h_pos, pos_edge_index, n)
        neg_from_neg = _mean_aggregate(h_neg, neg_edge_index, n)
        pos_from_neg = _mean_aggregate(h_neg, pos_edge_index, n)
        neg_from_pos = _mean_aggregate(h_pos, neg_edge_index, n)
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


class SGNN(nn.Module):
    """   Simple Signed GNN for your framework. Framework contract:
      - encode(batch_or_graph) -> h_dict
      - score_triples(z, edge_index_local, rel_ids) -> (logits, probs)
    Your trainer will call _maybe_switch_to_dual() and switch the contrastive loss
    if h_dict contains: 'node_pos', 'node_neg', and 'node' with node.size(1) == 2 * node_pos.size(1)
    """

    def __init__(self, in_dim, hidden_dim: int, out_dim: int = 1, n_layers: int = 2,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"), n_type: str = "node",
        e_etypes: Optional[List[Tuple[str, str, str]]] = None, neg_prefix: str = "NOT_",
        pos_rel_name: str = "pos_statement", neg_rel_name: str = "neg_statement",
        dropout: float = 0.2, drop_edge_p: float = 0.0,
        add_self_loops_flag: bool = True, make_undirected: bool = True,
        norm: bool = True):
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
        self.dual_view = False

        if isinstance(in_dim, dict):
            in_dim_ = int(in_dim.get(n_type, next(iter(in_dim.values()))))
        else: in_dim_ = int(in_dim)

        self.input_proj = nn.Linear(in_dim_, self.hidden_dim)
        self.layers = nn.ModuleList(
            [SignedConv(self.hidden_dim, dropout=self.dropout, norm=norm) for _ in range(int(n_layers))])

        pair_dim = 4 * self.hidden_dim
        self.classify = nn.Sequential(nn.Linear(pair_dim, self.hidden_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1 if self.out_dim == 1 else self.out_dim))

    def _get_node_features(self, data: HeteroData) -> Tensor:
        nt = data[self.n_type]
        if hasattr(nt, "x") and nt.x is not None: return nt.x
        if hasattr(nt, "feat") and getattr(nt, "feat") is not None: return getattr(nt, "feat")
        raise ValueError(
            f"{self.__class__.__name__}: node features not found at data['{self.n_type}'].x (or .feat)."
        )

    def _split_pos_neg_edges(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        """  Builds two homogeneous edge_index tensors:
          pos_edge_index: "positive" edges; neg_edge_index: "negative" edges
          If edge type name starts with neg_prefix (e.g., 'NOT_') => negative
          Else => positive

        Works both for:
          - graphs that contain ('node','pos_statement','node') / ('node','neg_statement','node')
          - graphs that contain many relations including NOT_* negatives
        """
        device = self._get_node_features(data).device
        pos_parts: List[Tensor] = []
        neg_parts: List[Tensor] = []

        for (src, rel, dst) in data.edge_types:
            if src != self.n_type or dst != self.n_type: continue
            store = data[(src, rel, dst)]
            if "edge_index" not in store or store.edge_index is None or store.edge_index.numel() == 0:
                continue
            eidx = store.edge_index.to(device)

            is_neg = False
            if rel == self.neg_rel_name: is_neg = True
            elif isinstance(rel, str) and rel.startswith(self.neg_prefix): is_neg = True

            if is_neg: neg_parts.append(eidx)
            else: pos_parts.append(eidx)

        def _cat(parts: List[Tensor]) -> Tensor:
            if not parts: return torch.empty((2, 0), dtype=torch.long, device=device)
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

        h_pos = xh
        h_neg = xh.new_zeros((n, self.hidden_dim))
        pos_edge_index, neg_edge_index = self._split_pos_neg_edges(data)

        for layer in self.layers:
            h_pos, h_neg = layer(h_pos, h_neg, pos_edge_index, neg_edge_index)
        z_pos = h_pos
        z_neg = h_neg
        z = torch.cat([z_pos, z_neg], dim=-1)

        return {self.n_type: z, f"{self.n_type}_pos": z_pos,
            f"{self.n_type}_neg": z_neg}

    def score_triples(self, z: Tensor, edge_index: Tensor,
        rel_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """  z: (num_nodes_in_(sub)graph, 2H)
        edge_index: (2, B) local node indices
        rel_ids: ignored (relation-agnostic decoder), but kept for trainer compatibility.  """
        src, dst = edge_index[0], edge_index[1]
        zu = z[src]
        zv = z[dst]
        h_pair = torch.cat([zu, zv], dim=-1)

        logits = self.classify(h_pair)
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: Tensor, rel_ids: Optional[Tensor] = None):
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits