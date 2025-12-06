import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional

class RA_HGCN(nn.Module):
    """ Heterogeneous GCN with a relation-aware edge classifier.
    Encoder: Multi-layer message passing over a HeteroData graph using HeteroConv + GCNConv.
    Decoder:
        - Relation-aware scoring function s(u, r, v).
        - For each triple (u, r, v):
              h_u, h_v : node embeddings
              e_r      : learned relation embedding
          Concatenates [h_u, e_r, h_v] and feeds to an MLP -> scalar logit.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
        e_etypes: List[Tuple[str, str, str]], n_type: str = "node",
        num_layers: int = 2, rel2id: Optional[Dict[str, int]] = None):
        super().__init__()

        self.n_type = n_type
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        if rel2id is None:
            rel_names = sorted({rel for (_, rel, _) in e_etypes})
            rel2id = {r: i for i, r in enumerate(rel_names)}
        self.rel2id: Dict[str, int] = rel2id
        self.num_rel = len(self.rel2id)

        convs: List[HeteroConv] = []
        for layer in range(num_layers):
            conv_dict = {}
            for (src_nt, rel, dst_nt) in e_etypes:
                in_ch = hidden_dim if layer > 0 else in_dim
                conv_dict[(src_nt, rel, dst_nt)] = GCNConv(in_ch, hidden_dim)
            convs.append(HeteroConv(conv_dict, aggr="mean"))
        self.convs = nn.ModuleList(convs)
        self.rel_emb = nn.Embedding(self.num_rel, hidden_dim)

        self.classify = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1))


    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """ Encode nodes of a HeteroData graph into embeddings per node type.
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        h_dict = x_dict
        for hetero_conv in self.convs:
            h_dict = hetero_conv(h_dict, edge_index_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}
        return h_dict

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor,
        rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Score triples (u, r, v) given node embeddings and relation IDs.
        Args:
            z: Tensor[num_nodes, hidden_dim] node embeddings for self.n_type.
            edge_index: LongTensor[2, B] with source and target node indices.
            rel_ids: LongTensor[B] with relation IDs in [0, num_rel).
        """
        src, dst = edge_index
        h_u = z[src]                  # (B, D)
        h_v = z[dst]                  # (B, D)
        e_r = self.rel_emb(rel_ids)   # (B, D)

        h_pair = torch.cat([h_u, e_r, h_v], dim=-1)  # (B, 3D)
        logits = self.classify(h_pair).view(-1)      # (B,)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: torch.Tensor,
        rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass. Args:
            data: HeteroData graph with all edges for message passing.
            edge_index: LongTensor[2, B] triples' (src, dst) indices.
            rel_ids: LongTensor[B] relation IDs.
        """
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs
