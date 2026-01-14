import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GATConv, HeteroConv


class GAT(nn.Module):
    """  Homogeneous GAT encoder that collapses all edges of a node type into a single edge_index.
    Decoder is relation-aware:
      - embeds relation IDs
      - MLP over [h_u, e_r, h_v] -> logit
    Uses classic GAT concat=True and keeps embedding size fixed:
      head_dim = hidden_dim // heads
      output_dim_per_layer = heads * head_dim = hidden_dim   """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
        heads: int = 4, e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        n_type: str = "node", num_layers: int = 2, rel2id: Optional[Dict[str, int]] = None,
        attn_dropout: float = 0.0):
        super().__init__()

        if isinstance(in_dim, dict):
            if n_type in in_dim: in_dim = int(in_dim[n_type])
            else: in_dim = int(next(iter(in_dim.values())))
        else: in_dim = int(in_dim)

        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads}).")
        head_dim = hidden_dim // heads

        self.n_type = n_type
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.heads = heads
        self.head_dim = head_dim

        convs: List[GATConv] = []
        for layer in range(num_layers):
            in_ch = in_dim if layer == 0 else hidden_dim
            convs.append(GATConv(in_channels=in_ch, out_channels=head_dim,
                    heads=heads, concat=True, dropout=attn_dropout,
                    add_self_loops=True, bias=True))
        self.convs = nn.ModuleList(convs)
        self.num_rel = 1 if rel2id is None else len(rel2id)
        self.rel_emb = nn.Embedding(self.num_rel, hidden_dim)
        # ----- Triple classifier: [h_u, e_r, h_v] -> logit -----
        self.classify = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _build_homogeneous(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collapse all edges of node type `self.n_type` into a single homogeneous edge_index.
        Edge types are ignored.
        """
        x = data[self.n_type].x
        device = x.device

        edges = []
        for (s, r, d), eidx in data.edge_index_dict.items():
            if s == self.n_type and d == self.n_type:
                edges.append(eidx)

        if len(edges) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else: edge_index = torch.cat(edges, dim=1).to(device)
        return x, edge_index

    def encode(self, data: Union[HeteroData, Data]) -> Dict[str, torch.Tensor]:
        """
        If data is HeteroData, collapse to homogeneous graph and run GATConv.
        If data is already homogeneous Data, just use its x and edge_index.
        Returns: { self.n_type: node_embeddings }
        """
        if isinstance(data, HeteroData):
            x, edge_index = self._build_homogeneous(data)
        else: x, edge_index = data.x, data.edge_index
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        return {self.n_type: h}

    def score_triples(self, z: torch.Tensor, edge_index: torch.Tensor,
        rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [num_nodes, hidden_dim]
        edge_index: [2, B]
        rel_ids: [B]
        """
        src, dst = edge_index
        h_u = z[src]
        h_v = z[dst]

        if rel_ids.dim() == 0: rel_ids = rel_ids.view(-1)
        rel_ids = rel_ids.clamp(min=0, max=self.num_rel - 1)
        e_r = self.rel_emb(rel_ids)
        h_pair = torch.cat([h_u, e_r, h_v], dim=-1)

        logits = self.classify(h_pair).view(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: Union[HeteroData, Data], edge_index: torch.Tensor,
        rel_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn import GATConv
# from typing import List, Tuple, Dict
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dgl.nn import GraphConv
# from typing import Dict, Tuple, List


# class GAT(nn.Module):
#     """
#     Homogeneous GAT model for link classification.
#     Args:
#         in_feats (Dict[str, int]): Input feature sizes for node type. Only the first value is used.
#         hidden_dim (int): Hidden embedding dimension.
#         out_dim (int): Output dimension for classification (e.g., number of classes).
#         n_layers (int): Number of GAT layers.
#         num_heads (int): Number of attention heads.
#         ppi_etype (Tuple[str, str, str]): Canonical edge type to classify, e.g. ("node", "PPI", "node").
#         n_type (str): Node type (unused in homogeneous graphs, but kept for compatibility).
#         e_etypes (List[Tuple[str, str, str]]): Edge types (unused here, for compatibility).
#     """

#     def __init__(
#         self,
#         in_feats: Dict[str, int],
#         hidden_dim: int,
#         out_dim: int,
#         n_layers: int = 2,
#         num_heads: int = 4,
#         ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
#         n_type: str = "node",
#         e_etypes: List[Tuple[str, str, str]] = None,
#     ):
#         super(GAT, self).__init__()
#         input_dim = list(in_feats.values())[0]
#         self.n_type = n_type
#         self.ppi_etype = ppi_etype
#         self.num_heads = num_heads

#         self.layers = nn.ModuleList()
#         self.layers.append(GATConv(input_dim, hidden_dim // num_heads, num_heads, "mean"))
#         for _ in range(n_layers - 1):
#             self.layers.append(GATConv(hidden_dim, hidden_dim // num_heads, num_heads, "mean"))
#         self.classify = nn.Linear(2 * hidden_dim, out_dim)

#     def forward(self, graph, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         h = graph.ndata['feat']

#         for layer in self.layers:
#             h = layer(graph, h).flatten(1)
#             h = F.elu(h)

#         src_ids, dst_ids = edge_index
#         hs = h[src_ids]
#         hd = h[dst_ids]
#         h_pair = torch.cat([hs, hd], dim=1)
#         logits = self.classify(h_pair)
#         z = h
#         return z, torch.sigmoid(logits)