import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge, add_self_loops
from typing import Dict, List, Optional, Tuple


class MLP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
        out_channels: int, num_layers: int = 2, dropout: float = 0.2,
        norm: bool = False, tailact: bool = False, norm_affine: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        if norm:
            layers.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(max(num_layers - 2, 0)):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if norm:
                layers.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_channels, out_channels))
        if tailact:
            layers.append(nn.LayerNorm(out_channels, elementwise_affine=norm_affine))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GCN_GAE(nn.Module):
    """
    PyG-compatible GCN-GAE-style encoder + link/triple classifier that works with your current Trainer.

    Trainer contract:
      - encode(data: HeteroData) -> Dict[str, Tensor] with key == self.n_type (default 'node')
      - score_triples(z: Tensor, edge_index: Tensor[2,B], rel_ids: Tensor[B]) -> (logits, probs)

    Encoder:
      - Runs a (homogeneous) GCN over a chosen edge set from the hetero graph.
      - By default it uses `ppi_etype` if present; otherwise it concatenates all edge_index across edge types.

    Decoder:
      - If rel2id is provided (or inferred) AND rel_ids are supplied, it can use relation embeddings.
      - If rel2id is not provided, rel_ids are ignored and scoring uses only (u,v).

    Notes:
      - This is a supervised link classifier (BCEWithLogitsLoss) baseline; it's not a full autoencoder.
      - It is safe in NeighborLoader mode (batch-local edge_index) and full-graph mode.
    """

    def __init__(
        self,
        in_dim: Dict[str, int] | int,
        hidden_dim: int,
        out_dim: int = 1,
        n_layers: int = 2,
        prop_step: int = 2,
        dropout: float = 0.2,
        residual: float = 0.1,
        linear: bool = False,
        e_etypes: Optional[List[Tuple[str, str, str]]] = None,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node",
        rel2id: Optional[Dict[str, int]] = None,
        drop_edge_p: float = 0.0,
        add_self_loops_flag: bool = True,
    ):
        super().__init__()
        self.n_type = n_type
        self.ppi_etype = ppi_etype

        if isinstance(in_dim, dict):
            in_dim_ = int(in_dim.get(n_type, list(in_dim.values())[0]))
        else:
            in_dim_ = int(in_dim)

        self.in_dim = in_dim_
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.n_layers = int(n_layers)
        self.prop_step = int(prop_step)
        self.dropout = float(dropout)
        self.residual = float(residual)
        self.linear = bool(linear)
        self.drop_edge_p = float(drop_edge_p)
        self.add_self_loops_flag = bool(add_self_loops_flag)

        # Encoder: simple GCN
        self.conv1 = GCNConv(in_dim_, self.hidden_dim, add_self_loops=False, normalize=True)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=False, normalize=True)

        # Residual projection if dims differ
        self.res_proj = nn.Identity() if in_dim_ == self.hidden_dim else nn.Linear(in_dim_, self.hidden_dim, bias=False)

        # Optional per-step MLP (kept from your original idea, but dimensionally consistent)
        if self.linear:
            self.mlps = nn.ModuleList(
                [MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout) for _ in range(max(self.prop_step, 1))]
            )
        else:
            self.mlps = None

        # Optional relation embeddings for decoding (only used if provided/inferred)
        if rel2id is None and e_etypes is not None:
            rel_names = sorted({rel for (_, rel, _) in e_etypes})
            rel2id = {r: i for i, r in enumerate(rel_names)}
        self.rel2id = rel2id
        self.num_rel = len(rel2id) if rel2id is not None else 0
        self.rel_emb = nn.Embedding(self.num_rel, self.hidden_dim) if self.num_rel > 0 else None

        # Decoder
        if self.rel_emb is None:
            self.classify = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1 if self.out_dim == 1 else self.out_dim),
            )
        else:
            self.classify = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1 if self.out_dim == 1 else self.out_dim),
            )

    def _get_node_features(self, data: HeteroData) -> Tensor:
        nt = data[self.n_type]
        if hasattr(nt, "x") and nt.x is not None:
            return nt.x
        if hasattr(nt, "feat") and getattr(nt, "feat") is not None:
            return getattr(nt, "feat")
        raise ValueError(f"{self.__class__.__name__}: node features not found at data['{self.n_type}'].x (or .feat).")

    def _get_message_edge_index(self, data: HeteroData) -> Tensor:
        # Prefer PPI edges if present
        if self.ppi_etype in data.edge_types and "edge_index" in data[self.ppi_etype]:
            eidx = data[self.ppi_etype].edge_index
            if eidx is not None and eidx.numel() > 0:
                return eidx

        # Fallback: concatenate all edge_index tensors across edge types (single node type assumption)
        parts: List[Tensor] = []
        for et, store in data.edge_items():
            # data.edge_items() yields ((src, rel, dst), edge_store)
            if isinstance(et, tuple) and len(et) == 3:
                if "edge_index" in store and store.edge_index is not None and store.edge_index.numel() > 0:
                    parts.append(store.edge_index)
        if not parts:
            # No edges -> return empty
            return torch.empty((2, 0), dtype=torch.long, device=self._get_node_features(data).device)
        return torch.cat(parts, dim=1)

    def encode(self, data: HeteroData) -> Dict[str, Tensor]:
        x = self._get_node_features(data)
        edge_index = self._get_message_edge_index(data)

        # Optional edge dropout (train-time only)
        if self.training and self.drop_edge_p > 0.0 and edge_index.numel() > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.drop_edge_p, force_undirected=False)

        if self.add_self_loops_flag and x.size(0) > 0:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x_res = self.res_proj(x)

        h = self.conv1(x, edge_index)
        h = h + self.residual * x_res

        for i in range(1, max(self.prop_step, 1)):
            h = F.relu(h)
            if self.mlps is not None:
                h = self.mlps[min(i, len(self.mlps) - 1)](h)
            h = self.conv2(h, edge_index)
            h = h + self.residual * x_res

        return {self.n_type: h}

    def score_triples(self, z: Tensor, edge_index: Tensor, rel_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        z: (num_nodes_in_batch, hidden_dim)
        edge_index: (2, B) with local node indices
        rel_ids: (B,) long
        """
        src, dst = edge_index
        hs = z[src]
        hd = z[dst]

        if self.rel_emb is not None and rel_ids is not None:
            er = self.rel_emb(rel_ids)
            h_in = torch.cat([hs, er, hd], dim=-1)
        else:
            h_in = torch.cat([hs, hd], dim=-1)

        logits = self.classify(h_in)
        # match the trainer's BCEWithLogitsLoss expectation (typically labels are (B,))
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)

        probs = torch.sigmoid(logits)
        return logits, probs

    def forward(self, data: HeteroData, edge_index: Tensor, rel_ids: Optional[Tensor] = None):
        h_dict = self.encode(data)
        z = h_dict[self.n_type]
        logits, probs = self.score_triples(z, edge_index, rel_ids)
        return z, logits, probs


# import dgl
# from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv
# from typing import Tuple, List

# class Hadamard_MLPPredictor(nn.Module):
#     def __init__(self, h_feats, dropout, layer=5, res=True, norm=False, scale=False):
#         super().__init__()
#         self.lins = torch.nn.ModuleList()
#         self.lins.append(torch.nn.Linear(h_feats, h_feats))
#         for _ in range(layer - 2):
#             self.lins.append(torch.nn.Linear(h_feats, h_feats))
#         self.lins.append(torch.nn.Linear(h_feats, 1))
#         self.dropout = dropout
#         self.res = res
#         self.scale = scale
#         if scale: self.scale_norm = nn.LayerNorm(h_feats)
#         self.norm = norm
#         if norm:
#             self.norms = torch.nn.ModuleList()
#             for _ in range(layer - 1):
#                 self.norms.append(nn.LayerNorm(h_feats))
#         self.act = F.relu

#     def forward(self, x_i, x_j):
#         x = x_i * x_j
#         if self.scale:
#             x = self.scale_norm(x)
#         ori = x
#         for i in range(len(self.lins) - 1):
#             x = self.lins[i](x)
#             if self.res:
#                 x += ori
#             if self.norm:
#                 x = self.norms[i](x)
#             x = self.act(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lins[-1](x)
#         return x.squeeze()
# class DotPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x_i, x_j):
#         x = (x_i * x_j).sum(dim=-1)
#         return x.squeeze()
# class LorentzPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x_i, x_j):
#         n = x_i.size(1)
#         x = torch.sum(x_i[:, 0:n//2] * x_j[:, 0:n//2], dim=-1) - torch.sum(x_i[:, n//2:] * x_j[:, n//2:], dim=-1)
#         return x.squeeze()

# def drop_edge(g, dpe = 0.2):
#     g = g.clone()
#     eids = torch.randperm(g.number_of_edges())[:int(g.number_of_edges() * dpe)].to(g.device)
#     g.remove_edges(eids)
#     g = dgl.add_self_loop(g)
#     return g


# class MLP(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,
#                  norm=False, tailact=False, norm_affine=True):
#         super(MLP, self).__init__()
#         self.lins = torch.nn.Sequential()
#         self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
#         if norm:
#             self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
#         self.lins.append(nn.ReLU())
#         if dropout > 0:
#             self.lins.append(nn.Dropout(dropout))
#         for _ in range(num_layers - 2):
#             self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
#             if norm:
#                 self.lins.append(nn.LayerNorm(hidden_channels), elementwise_affine=norm_affine)
#             self.lins.append(nn.ReLU())
#             if dropout > 0:
#                 self.lins.append(nn.Dropout(dropout))
#         self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
#         if tailact:
#             self.lins.append(nn.LayerNorm(out_channels), elementwise_affine=norm_affine)
#             self.lins.append(nn.ReLU())
#             self.lins.append(nn.Dropout(dropout))

#     def forward(self, x):
#         x = self.lins(x)
#         return x.squeeze()
    




# class GCN_GAE(nn.Module):
#     """
#     Refined-GAE model for link classification.
#     """
#     def __init__(self, in_feats, hidden_dim, out_dim: int, n_layers: int = 2,
#                 relu=False, linear=False, prop_step=2, dropout=0.2, residual=0.1, 
#                 e_etypes: List[Tuple[str, str, str]] = None,
#                 ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"), n_type: str = "node"):
        
#         super(GCN_GAE, self).__init__()
#         input_dim = list(in_feats.values())[0]
#         self.n_type = n_type
#         self.ppi_etype = ppi_etype
#         self.conv1 = GraphConv(input_dim, input_dim)
#         self.conv2 = GraphConv(input_dim, input_dim)
#         self.relu = relu
#         self.prop_step = prop_step
#         self.residual = residual
#         self.linear = linear
#         if linear: self.mlps = nn.ModuleList([MLP(hidden_dim, hidden_dim, 2, dropout) for _ in range(prop_step)])
#         self.classify = nn.Linear(hidden_dim, out_dim)
    
#     def forward(self, g, edge_index):

#         f = g.ndata['feat']
#         h = self.conv1(g, f).flatten(1) + self.residual * f
#         for i in range(1, self.prop_step):
#             h = F.relu(h)
#             if self.linear: h = self.mlps[i](h)
#             h = self.conv2(g, h).flatten(1) + self.residual * f

#         src_ids, dst_ids = edge_index
#         hs = h[src_ids]
#         hd = h[dst_ids]
#         h_pair = torch.cat([hs, hd], dim=1)
#         logits = self.classify(h_pair)
#         z = h
#         return z, torch.sigmoid(logits)



# class GCN_multilayers(nn.Module):
        
#     def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, drop_edge=False, relu=False, linear=False, prop_step=2, dropout=0.2, residual=0, conv='GCN'):
#         super(GCN_multilayers, self).__init__()
#         if conv == 'GCN':
#             self.convs = nn.ModuleList([GraphConv(in_feats, h_feats)])
#             for _ in range(prop_step - 1):
#                 self.convs.append(GraphConv(h_feats, h_feats))
#         elif conv == 'SAGE':
#             self.convs = nn.ModuleList([SAGEConv(in_feats, h_feats, 'mean')])
#             for _ in range(prop_step - 1):
#                 self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
#         elif conv == 'GAT':
#             self.convs = nn.ModuleList([GATConv(in_feats, h_feats // 4, 4)])
#             for _ in range(prop_step - 1):
#                 self.convs.append(GATConv(h_feats, h_feats // 4, 4))
#         elif conv == 'GIN':
#             self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, 0.2)])
#             self.convs = nn.ModuleList([GINConv(self.mlps[0], 'mean')])
#             for _ in range(prop_step - 1):
#                 self.mlps.append(MLP(h_feats, h_feats, 2, 0.2))
#                 self.convs.append(GINConv(self.mlps[-1], 'mean'))
#         self.norm = norm
#         self.drop_edge = drop_edge
#         self.relu = relu
#         self.prop_step = prop_step
#         self.residual = residual
#         self.linear = linear
#         if norm:
#             self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
#         self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         if linear:
#             self.mlps = nn.ModuleList([MLP(h_feats, h_feats, 2, dropout) for _ in range(prop_step)])

#     def _apply_norm_and_activation(self, x, i):
#         if self.norm:
#             x = self.norms[i](x)
#         if self.relu:
#             x = F.relu(x)
#         x = self.dp(x)
#         return x
    
#     def forward(self, g, in_feat):
#         ori = in_feat
#         if self.drop_edge:
#             g = drop_edge(g)
#         h = self.conv1(g, in_feat).flatten(1) + self.residual * ori
#         for i in range(1, self.prop_step):
#             h = self._apply_norm_and_activation(h, i)
#             if self.linear:
#                 h = self.mlps[i](h)
#             h = self.conv2(g, h).flatten(1) + self.residual * ori
#         return h

    
# class PureGCN(nn.Module):
#     def __init__(self, input_dim, num_layers=2, hidden=256, dp=0, relu=False, norm=False, res=False):
#         super().__init__()
#         self.lin = nn.Linear(input_dim, hidden)
#         self.conv = GraphConv(hidden, hidden, weight=False, bias=False)
#         self.num_layers = num_layers
#         self.dp = dp
#         self.norm = norm
#         self.res = res
#         self.relu = relu
#         if self.norm:
#             self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

#     def forward(self, adj_t, x, e_feat=None):
#         x = self.lin(x)
#         ori = x
#         for i in range(self.num_layers):
#             if i != 0 and self.res:
#                 x = x + ori
#             if self.norm:
#                 x = self.norms[i](x)
#             if self.relu:
#                 x = F.relu(x)
#             if self.dp > 0:
#                 x = F.dropout(x, p=self.dp, training=self.training)
#             x = self.conv(adj_t, x, edge_weight=e_feat)
#         return x

# class PureGCN_no_para(nn.Module):
#     def __init__(self, input_dim, num_layers=2, relu=False, norm=False, res=False):
#         super().__init__()
#         self.conv = GraphConv(input_dim, input_dim, weight=False, bias=False)
#         self.num_layers = num_layers
#         self.norm = norm
#         self.res = res
#         self.relu = relu

#     def forward(self, adj_t, x, e_feat=None):
#         ori = x
#         for i in range(self.num_layers):
#             if i != 0 and self.res:
#                 x = x + ori
#             if self.norm:
#                 x = F.layer_norm(x, x.shape[1:])
#             if self.relu:
#                 x = F.relu(x)
#             x = self.conv(adj_t, x, edge_weight=e_feat)
#         return x       