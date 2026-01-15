import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GATConv, HeteroConv


class RA_HGAT(nn.Module):
    """  Relation-aware heterogeneous GAT.
    Encoder: Multi-layer message passing over a HeteroData graph using HeteroConv + GATConv.
    Decoder: relation-aware scoring function s(u, r, v) via:
        h_u, e_r, h_v -> MLP -> logit
    Uses classic GAT concatenation (concat=True) and keeps embedding size fixed:
      head_dim = hidden_dim // heads
      heads * head_dim == hidden_dim
    """

    def __init__(self, in_dim: Union[int, Dict[str, int]], hidden_dim: int, out_dim: int,
        e_etypes: List[Tuple[str, str, str]], n_type: str = "node",
        num_layers: int = 2, rel2id: Optional[Dict[str, int]] = None,
        heads: int = 4, attn_dropout: float = 0.0, aggr: str = "mean"):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}).")
        head_dim = hidden_dim // heads

        self.n_type = n_type
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.e_etypes = e_etypes
        self.heads = heads
        self.head_dim = head_dim

        # If rel2id not provided, infer from edge types like your RA_HGCN
        if rel2id is None:
            rel_names = sorted({rel for (_, rel, _) in e_etypes})
            rel2id = {r: i for i, r in enumerate(rel_names)}
        self.rel2id: Dict[str, int] = rel2id
        self.num_rel = len(self.rel2id)

        # --- normalize in_dim to a per-node-type dict ---
        if isinstance(in_dim, dict): node_in_dim = in_dim
        else:
            ntypes = {s for (s, _, _) in e_etypes} | {d for (_, _, d) in e_etypes}
            node_in_dim = {nt: in_dim for nt in ntypes}

        # ----- Heterogeneous GAT encoder -----
        convs: List[HeteroConv] = []
        for _layer in range(num_layers):
            conv_dict = {}
            for (src_nt, rel, dst_nt) in e_etypes:
                if _layer == 0:
                    src_in = node_in_dim[src_nt]
                    dst_in = node_in_dim[dst_nt]
                else: src_in = dst_in = hidden_dim
                conv_dict[(src_nt, rel, dst_nt)] = GATConv(
                    in_channels=(src_in, dst_in), out_channels=head_dim,
                    heads=heads, concat=True, dropout=attn_dropout,
                    add_self_loops=(src_nt == dst_nt),  # avoid invalid loops on bipartite edges
                    bias=True)
            convs.append(HeteroConv(conv_dict, aggr=aggr))
        self.convs = nn.ModuleList(convs)

        # ----- Relation embeddings + decoder (same as RA_HGCN) -----
        self.rel_emb = nn.Embedding(self.num_rel, hidden_dim)
        self.classify = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        h_dict = x_dict
        for hetero_conv in self.convs:
            h_dict = hetero_conv(h_dict, edge_index_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}
        return h_dict

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor,
        rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [num_nodes, hidden_dim] embeddings for self.n_type
        edge_index: [2, B]
        rel_ids: [B] in [0, num_rel)
        """
        src, dst = edge_index
        h_u, h_v = z[src], z[dst]
        if rel_ids.dim() == 0: rel_ids = rel_ids.view(-1)
        rel_ids = rel_ids.clamp(min=0, max=self.num_rel - 1)

        e_r = self.rel_emb(rel_ids)
        h_pair = torch.cat([h_u, e_r, h_v], dim=-1)
        logits = self.classify(h_pair).view(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: torch.Tensor,
        rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs
