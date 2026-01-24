import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import RGCNConv

class RA_RGCN(nn.Module):
    """
    Encoder: RGCNConv over a *homogenized* edge_index + edge_type built from HeteroData.
    Decoder: same relation-aware MLP over [h_u, e_r, h_v] as your RA_HGCN.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        e_etypes: List[Tuple[str, str, str]],
        n_type: str = "node",
        num_layers: int = 2,
        rel2id: Optional[Dict[str, int]] = None,   # decoder relations (triple labels)
        num_bases: Optional[int] = None,           # optional parameter sharing for RGCN
        use_fast: bool = False,                    # toggle FastRGCNConv if you want
    ):
        super().__init__()

        self.n_type = n_type
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_layers = int(num_layers)
        if rel2id is None:
            dec_rel_names = sorted({rel for (_, rel, _) in e_etypes})
            rel2id = {r: i for i, r in enumerate(dec_rel_names)}
        self.dec_rel2id = dict(rel2id)
        self.num_dec_rel = len(self.dec_rel2id)

        mp_rel_names = sorted({rel for (_, rel, _) in e_etypes if rel is not None})
        self.mp_rel2id = {r: i for i, r in enumerate(mp_rel_names)}
        self.num_mp_rel = len(self.mp_rel2id)

        # ----- RGCN encoder -----
        Conv = RGCNConv
        if use_fast:
            from torch_geometric.nn import FastRGCNConv
            Conv = FastRGCNConv

        convs: List[nn.Module] = []
        for layer in range(self.num_layers):
            in_ch = self.in_dim if layer == 0 else self.hidden_dim
            convs.append(Conv(in_channels=in_ch, out_channels=self.hidden_dim,
                    num_relations=self.num_mp_rel, num_bases=num_bases,
                    aggr="mean", root_weight=True, bias=True))
        self.convs = nn.ModuleList(convs)
        # ----- decoder: relation embeddings + MLP over [h_u, e_r, h_v] -----
        self.rel_emb = nn.Embedding(self.num_dec_rel, self.hidden_dim)
        self.classify = nn.Sequential(nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(), nn.Linear(self.hidden_dim, 1))

    def _hetero_to_relational(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """  Convert HeteroData (single node type) into:
          x: [N, F]
          edge_index: [2, E]
          edge_type: [E] int64 in [0, num_mp_rel)   """
        x = data[self.n_type].x
        device = x.device
        edge_indices = []
        edge_types = []

        for (s, r, d), ei in data.edge_index_dict.items():
            if s != self.n_type or d != self.n_type: continue
            if ei is None or ei.numel() == 0: continue
            rid = self.mp_rel2id.get(r, None)
            if rid is None: continue

            edge_indices.append(ei.to(device))
            edge_types.append(torch.full((ei.size(1),), rid, dtype=torch.long, device=device))

        if not edge_indices:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            edge_type = torch.empty(0, dtype=torch.long, device=device)
        else:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_type = torch.cat(edge_types, dim=0)
        return x, edge_index, edge_type

    def encode(self, data: Union[HeteroData, Data]) -> Dict[str, torch.Tensor]:
        """   If HeteroData: build edge_index + edge_type then apply RGCNConv layers.
        If Data: expects .edge_type to exist.   """
        if isinstance(data, HeteroData):
            x, edge_index, edge_type = self._hetero_to_relational(data)
        else:
            x, edge_index = data.x, data.edge_index
            if not hasattr(data, "edge_type"):
                raise ValueError("Homogeneous Data must have `edge_type` for RGCNConv.")
            edge_type = data.edge_type.to(x.device)

        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_type)
            h = F.relu(h)
        return {self.n_type: h}

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor, rel_ids: torch.Tensor):
        src, dst = edge_index
        h_u = z[src]
        h_v = z[dst]
        # rel_ids here are for the decoder relation vocabulary
        e_r = self.rel_emb(rel_ids.clamp(0, self.num_dec_rel - 1))
        # rel_ids = rel_ids.clamp(min=0, max=self.num_rel - 1)
        if (rel_ids < 0).any() or (rel_ids >= self.num_dec_rel).any():
            bad = rel_ids[(rel_ids < 0) | (rel_ids >= self.num_dec_rel)][:10].tolist()
            raise ValueError(f"rel_ids out of range (num_rel={self.num_dec_rel}). Examples: {bad}")

        h_pair = torch.cat([h_u, e_r, h_v], dim=-1)
        logits = self.classify(h_pair).view(-1)
        probs = torch.sigmoid(logits)
        return logits, probs