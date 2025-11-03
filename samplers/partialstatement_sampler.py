import random, torch
from typing import List, Tuple, Optional, Dict
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of

class PartialStatementSampler:
    """ A high-performance sampler that uses an external list of negative statements or graph-based negative edges, 
    with vectorization and no Python loops in the hot path. Assumes relations are within the same node type for statements/GO
    """
    def __init__(self, k: int = 2, go_etype: str = "link", neg_edges: Optional[List[Tuple[int, int]]] = None):
        self.k = k
        self.go_etype = go_etype
        self.external_neg = neg_edges or []
        self.pos_global: List[List[int]] = []
        self.pos_targets: set = set()
        self.direct_global: List[List[int]] = []
        self.two_hop_global: List[List[int]] = []
        self.pool_matrix_cpu: Optional[Tensor] = None
        self.pool_mask_cpu: Optional[Tensor] = None
        self.M: int = 0
        self.orig: Optional[Tensor] = None
        self.mapping: Dict[int, int] = {}
        self.batch_pool: Optional[Tensor] = None
        self.batch_mask: Optional[Tensor] = None
        self.B: int = 0
        self.device = torch.device("cpu")



    def prepare_global(self, full_g: HeteroData, pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        """ Builds pools from the full graph (PyG HeteroData) and computes, for every node u:
          - direct_global[u]: direct negative-statement neighbors
          - two_hop_global[u]: two-hop backups via GO predecessors of each direct neighbor
          - pos_global[u]: positive-statement neighbors
          - pool_matrix_cpu[u]: a padded list of candidates to sample negatives from
          - pool_mask_cpu[u]: validity mask for that row
        """
        pos_key = _find_key_by_rel(full_g, pos_etype)
        neg_key = _find_key_by_rel(full_g, neg_etype)
        src_nt, _, dst_nt = pos_key
        node_type = src_nt

        N = _num_nodes_of(full_g, node_type)
        if self.external_neg:
            src_neg, dst_neg = zip(*self.external_neg)
            src_neg = list(src_neg)
            dst_neg = list(dst_neg)
        else:
            if "edge_index" in full_g[neg_key]:
                src_neg_t, dst_neg_t = full_g[neg_key].edge_index
                src_neg = src_neg_t.tolist()
                dst_neg = dst_neg_t.tolist()
            else: src_neg, dst_neg = [], []
        if "edge_index" in full_g[pos_key]:
            src_pos_t, dst_pos_t = full_g[pos_key].edge_index
            src_pos = src_pos_t.tolist()
            dst_pos = dst_pos_t.tolist()
        else: src_pos, dst_pos = [], []

        pos_global: List[List[int]] = [[] for _ in range(N)]
        for u, v in zip(src_pos, dst_pos):
            pos_global[u].append(v)
        self.pos_global = pos_global
        self.pos_targets = set(dst_pos)
        direct_global: List[List[int]] = [[] for _ in range(N)]
        for u, v in zip(src_neg, dst_neg):
            direct_global[u].append(v)

        go_key = _find_key_by_rel(full_g, self.go_etype)
        predecessors: Dict[int, List[int]] = {}
        if "edge_index" in full_g[go_key]:
            go_src, go_dst = full_g[go_key].edge_index
            for s, d in zip(go_src.tolist(), go_dst.tolist()):
                predecessors.setdefault(d, []).append(s)

        two_hop_global: List[List[int]] = [[] for _ in range(N)]
        for u in range(N):
            if len(direct_global[u]) < self.k:
                sup = set()
                for b in direct_global[u]:
                    sup.update(predecessors.get(b, []))
                two_hop_global[u] = list(sup)
        self.direct_global = direct_global
        self.two_hop_global = two_hop_global

        pools: List[List[int]] = []
        for u in range(N):
            if len(direct_global[u]) >= self.k: pool = direct_global[u].copy()
            else: pool = list({*direct_global[u], *two_hop_global[u]})
            if len(pool) < self.k:
                excluded = set(direct_global[u]) | set(pos_global[u])
                candidates = list(self.pos_targets - excluded)
                needed = self.k - len(pool)
                if candidates:
                    if len(candidates) >= needed: pool.extend(random.sample(candidates, needed))
                    else: pool.extend(random.choices(candidates, k=needed))
            if len(pool) == 0: pool = [u]
            pools.append(pool)

        M = max(len(row) for row in pools) if pools else 0
        if M == 0:
            self.pool_matrix_cpu = torch.empty(0, 0, dtype=torch.long)
            self.pool_mask_cpu = torch.empty(0, 0, dtype=torch.bool)
            self.M = 0
            return
        pool_mat = torch.full((N, M), fill_value=0, dtype=torch.long)
        mask_mat = torch.zeros((N, M), dtype=torch.bool)
        for u, row in enumerate(pools):
            L = len(row)
            pool_mat[u, :L] = torch.tensor(row, dtype=torch.long)
            mask_mat[u, :L] = True
        self.pool_matrix_cpu = pool_mat
        self.pool_mask_cpu = mask_mat
        self.M = M



    def prepare_batch(self, batch: HeteroData):
        """ Node IDs are global and each batch subgraph retains the full node set: 
        'batch['node'].num_nodes == full.num_nodes'.
        """
        any_key = next(iter(batch.edge_types))
        self.device = batch[any_key].edge_index.device if batch[any_key].edge_index.is_cuda else torch.device("cpu")
        ntype = _find_key_by_rel(batch, self.go_etype)[0]
        B = _num_nodes_of(batch, ntype)
        self.orig = torch.arange(B, dtype=torch.long)
        self.mapping = {g.item(): i for i, g in enumerate(self.orig)}
        self.batch_pool = self.pool_matrix_cpu[self.orig].to(self.device)
        self.batch_mask = self.pool_mask_cpu[self.orig].to(self.device)
        self.B = int(B)


    def sample(self) -> Tensor:
        """ Vectorized sampling of k negatives per node, returns edge_index with (src=row_id, dst=sampled pool entry).
        """
        if self.B == 0 or self.M == 0: return torch.empty(2, 0, dtype=torch.long, device=self.device)
        probs = self.batch_mask.float()
        row_sums = probs.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0)
        if zero_rows.any():
            probs[zero_rows, :] = 1.0
            row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums
        idx = torch.multinomial(probs, self.k, replacement=True)
        dst = torch.gather(self.batch_pool, 1, idx) 
        src = torch.arange(self.B, device=self.device).unsqueeze(1).expand(-1, self.k)
        return torch.stack([src, dst], dim=0).reshape(2, -1)


    def get_contrastive_samples(self, z: Tensor, neg_ei: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Args: z: (B, D) node embeddings for the batch (local == global id order).
            neg_ei: [2, B*k] edge_index sampled by `sample()`; we use its dst as negatives.
        Returns:
            z_pos: (B, D) embeddings of each node (anchor of positive view)
            z_pos_pos: (B, D) embeddings of one positive-statement neighbor per node
            z_pos_neg: (B, k, D) embeddings of k negatives per node
        """
        B, D = z.shape
        device, z_pos = z.device, z
        pos_nb: List[int] = []
        for i, g in enumerate(self.orig.tolist()):
            opts = self.pos_global[g] if g < len(self.pos_global) else []
            if opts:
                g2 = random.choice(opts)
                pos_nb.append(self.mapping.get(g2, i))
            else: pos_nb.append(i)

        pos_nb = torch.tensor(pos_nb, device=device, dtype=torch.long)
        z_pos_pos = z[pos_nb]
        if neg_ei.numel() == 0: z_pos_neg = z_pos.unsqueeze(1).expand(-1, self.k, -1)
        else:
            neg_dst = neg_ei[1].view(B, self.k)
            z_pos_neg = z[neg_dst]
        return z_pos, z_pos_pos, z_pos_neg
