import random
import torch
from typing import List, Tuple, Optional, Dict
from torch import Tensor
from torch_geometric.data import HeteroData

from .utils import _num_nodes_of


class PartialStatementSampler:
    """
    High-performance sampler using an external list of statement edges or
    graph-based edges, in a multi-relation Wikidata-style KG.
    • Anchors:
        - Sources of relation `instance_rel` (e.g. "2" for "instance of").
        - If `instance_rel` is None or no such edges exist:
              anchors = all nodes with at least one positive or negative
                        statement (fallback: all nodes).
    • Positive statements:
        - Any edge (u, r, v) where:
              r does NOT start with `neg_prefix`
              and r != go_etype (subclass relation)
              and r != instance_rel
          These define: pos_global[u] = list of v
    • Negative statements:
        - Any edge (u, r, v) where r starts with `neg_prefix`
          These define: direct_global[u] = list of v
    • External edges:
        - If `edges_are_negative=True`:
              `external_neg` are treated as additional NEGATIVE statements (u->v).
        - If `edges_are_negative=False`:
              `external_neg` are treated as additional POSITIVE statements (u->v).
    • Ontology / GO expansion:
        - Edges (c, go_etype, b) define that c is a "subclass" of b.
        - For each anchor u, two-hop negatives are subclasses of its direct
          negative neighbors:
              two_hop_global[u] = {c | ∃b in direct_global[u] and (c, go_etype, b)}
    • Negative pool:
        - Base pool[u] = direct_global[u] ∪ two_hop_global[u].
        - If |pool[u]| < k, pad from global positive targets not already used.
        - If still empty, fallback to [u].
    """

    def __init__(self, k: int = 1, go_etype:str = "subclass_of",
        neg_edges: Optional[List[Tuple[int, int]]] = None,
        edges_are_negative: bool = True,
        instance_rel:str = "2", neg_prefix:str = "NOT_"):
        self.k = k
        self.go_etype = go_etype
        self.external_neg = neg_edges or []
        self.edges_are_negative = edges_are_negative
        self.instance_rel = instance_rel
        self.neg_prefix = neg_prefix
        self.pos_global: List[List[int]] = []
        self.pos_targets: set = set()
        self.direct_global: List[List[int]] = []
        self.two_hop_global: List[List[int]] = []
        self.pool_matrix_cpu: Optional[Tensor] = None
        self.pool_mask_cpu: Optional[Tensor] = None
        self.M: int = 0
        self.anchors: List[int] = []

        self.orig: Optional[Tensor] = None
        self.global_to_local: Dict[int, int] = {}
        self.batch_pool: Optional[Tensor] = None
        self.batch_mask: Optional[Tensor] = None
        self.batch_anchor_locals: Optional[Tensor] = None
        self.batch_anchor_globals: List[int] = []
        self.A: int = 0
        self.device = torch.device("cpu")


    def prepare_global(self, full_g: HeteroData) -> None:
        """
        Scan the full graph and build positive/negative neighbor structures,
        2-hop expansion via `go_etype`, anchor set via `instance_rel`,
        and per-node negative pools packed into matrices.
        """
        if len(full_g.node_types) != 1: raise ValueError(
                f"PartialStatementSampler assumes a single node type; got {full_g.node_types}")
        node_type = full_g.node_types[0]

        if hasattr(full_g[node_type], "num_nodes") and full_g[node_type].num_nodes is not None:
            N = int(full_g[node_type].num_nodes)
        else:
            max_id = -1
            for (_, _, _), eidx in full_g.edge_index_dict.items():
                if eidx.numel() > 0: max_id = max(max_id, int(eidx.max().item()))
            N = max_id + 1 if max_id >= 0 else 0

        self.pos_global = [[] for _ in range(N)]
        self.direct_global = [[] for _ in range(N)]
        self.two_hop_global = [[] for _ in range(N)]
        self.pos_targets = set()

        if self.external_neg:
            if self.edges_are_negative:
                for u, v in self.external_neg:
                    if 0 <= u < N and 0 <= v < N: self.direct_global[u].append(v)
            else:
                for u, v in self.external_neg:
                    if 0 <= u < N and 0 <= v < N:
                        self.pos_global[u].append(v)
                        self.pos_targets.add(v)

        predecessors: Dict[int, List[int]] = {}
        anchor_sources: set = set()

        for (src_nt, rel, dst_nt), eidx in full_g.edge_index_dict.items():
            if src_nt != node_type or dst_nt != node_type: continue
            if eidx.numel() == 0: continue

            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()
            if rel == self.go_etype:
                for s, d in zip(src_list, dst_list):
                    if 0 <= s < N and 0 <= d < N:
                        predecessors.setdefault(d, []).append(s)
                continue

            if self.instance_rel is not None and rel == self.instance_rel:
                for s in src_list:
                    if 0 <= s < N: anchor_sources.add(s)
                continue

            if rel.startswith(self.neg_prefix):
                for u, v in zip(src_list, dst_list):
                    if 0 <= u < N and 0 <= v < N: self.direct_global[u].append(v)
                continue

            for u, v in zip(src_list, dst_list):
                if 0 <= u < N and 0 <= v < N:
                    self.pos_global[u].append(v)
                    self.pos_targets.add(v)

        for u in range(N):
            if len(self.direct_global[u]) < self.k:
                sup = set()
                for b in self.direct_global[u]:
                    for c in predecessors.get(b, []):
                        sup.add(c)
                self.two_hop_global[u] = list(sup)

        pools: List[List[int]] = []
        for u in range(N):
            if len(self.direct_global[u]) >= self.k: pool = self.direct_global[u].copy()
            else: pool = list({*self.direct_global[u], *self.two_hop_global[u]})

            if len(pool) < self.k:
                excluded = set(self.direct_global[u]) | set(self.pos_global[u])
                candidates = list(self.pos_targets - excluded)
                needed = self.k - len(pool)
                if candidates:
                    if len(candidates) >= needed:
                        pool.extend(random.sample(candidates, needed))
                    else: pool.extend(random.choices(candidates, k=needed))

            if len(pool) == 0: pool = [u]
            pools.append(pool)

        M = max((len(row) for row in pools), default=0)
        if M == 0:
            self.pool_matrix_cpu = torch.empty(0, 0, dtype=torch.long)
            self.pool_mask_cpu = torch.empty(0, 0, dtype=torch.bool)
            self.M = 0
        else:
            pool_mat = torch.full((N, M), fill_value=0, dtype=torch.long)
            mask_mat = torch.zeros((N, M), dtype=torch.bool)
            for u, row in enumerate(pools):
                if not row: continue
                L = len(row)
                pool_mat[u, :L] = torch.tensor(row, dtype=torch.long)
                mask_mat[u, :L] = True
            self.pool_matrix_cpu = pool_mat
            self.pool_mask_cpu = mask_mat
            self.M = M

        # Anchors
        if self.instance_rel is not None and anchor_sources:
            anchors = sorted(anchor_sources)
        else:
            anchors = [u for u in range(N) if (self.pos_global[u] or self.direct_global[u])]
            if not anchors: anchors = list(range(N))
        self.anchors = anchors


    def prepare_batch(self, batch: HeteroData) -> None:
        """
        Assumes batch retains global node IDs (your setup: full graph).
        Restricts negative pools to anchors present in this batch.
        """
        if len(batch.edge_types) == 0:
            self.A = 0
            self.batch_pool = None
            self.batch_mask = None
            self.batch_anchor_locals = None
            self.batch_anchor_globals = []
            self.global_to_local = {}
            return

        any_key = next(iter(batch.edge_types))
        self.device = (batch[any_key].edge_index.device
            if batch[any_key].edge_index.is_cuda
            else torch.device("cpu"))

        node_type = any_key[0]
        B = _num_nodes_of(batch, node_type)

        self.orig = torch.arange(B, dtype=torch.long)
        self.global_to_local = {int(g.item()): int(i) for i, g in enumerate(self.orig)}

        anchor_globals: List[int] = []
        anchor_locals: List[int] = []
        for g in self.anchors:
            if g in self.global_to_local:
                anchor_globals.append(g)
                anchor_locals.append(self.global_to_local[g])

        if not anchor_globals:
            self.A = 0
            self.batch_pool = torch.empty(0, 0, dtype=torch.long, device=self.device)
            self.batch_mask = torch.empty(0, 0, dtype=torch.bool, device=self.device)
            self.batch_anchor_locals = torch.empty(0, dtype=torch.long, device=self.device)
            self.batch_anchor_globals = []
            return

        if self.pool_matrix_cpu is None or self.pool_mask_cpu is None:
            self.batch_pool = torch.empty(len(anchor_globals), 0, dtype=torch.long, device=self.device)
            self.batch_mask = torch.empty(len(anchor_globals), 0, dtype=torch.bool, device=self.device)
            self.M = 0
        else:
            rows = torch.tensor(anchor_globals, dtype=torch.long)
            self.batch_pool = self.pool_matrix_cpu[rows].to(self.device)
            self.batch_mask = self.pool_mask_cpu[rows].to(self.device)

        self.batch_anchor_locals = torch.tensor(anchor_locals, dtype=torch.long, device=self.device)
        self.batch_anchor_globals = anchor_globals
        self.A = len(anchor_globals)


    def sample(self) -> Tensor:
        """
        Vectorized sampling of k negatives per anchor.
        Returns edge_index [2, A*k] with local ids:
            src = anchor local id
            dst = sampled negative local id
        """
        if self.A == 0 or self.M == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)

        probs = self.batch_mask.float()
        row_sums = probs.sum(dim=1, keepdim=True)
        zero_rows = row_sums == 0
        if zero_rows.any():
            probs[zero_rows, :] = 1.0
            row_sums = probs.sum(dim=1, keepdim=True)
        probs = probs / row_sums

        idx = torch.multinomial(probs, self.k, replacement=True)
        dst = torch.gather(self.batch_pool, 1, idx)             
        src = self.batch_anchor_locals.unsqueeze(1).expand(-1, self.k)
        return torch.stack([src, dst], dim=0).reshape(2, -1)

    def get_contrastive_samples(self, z: Tensor,
        neg_ei: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        z: (B, D) node embeddings in batch-local order (here B == num_nodes).
        If `neg_ei` is None, uses self.sample().
        Returns:
            z_pos     : (A, D) anchor embeddings
            z_pos_pos : (A, D) one positive neighbor per anchor
            z_pos_neg : (A, k, D) k negatives per anchor
        """
        B, D = z.shape
        device = z.device

        if self.A == 0:
            empty = torch.empty(0, D, device=device)
            empty_neg = torch.empty(0, self.k, D, device=device)
            return empty, empty, empty_neg

        if neg_ei is None: neg_ei = self.sample()
        z_pos = z[self.batch_anchor_locals]

        pos_nb: List[int] = []
        for g, l in zip(self.batch_anchor_globals, self.batch_anchor_locals.tolist()):
            opts_global = self.pos_global[g] if g < len(self.pos_global) else []
            opts_local = [self.global_to_local[v]
                for v in opts_global
                if v in self.global_to_local]
            chosen = random.choice(opts_local) if opts_local else l
            pos_nb.append(chosen)

        pos_nb_tensor = torch.tensor(pos_nb, device=device, dtype=torch.long)
        z_pos_pos = z[pos_nb_tensor]
        if neg_ei.numel() == 0: z_pos_neg = z_pos.unsqueeze(1).expand(-1, self.k, -1)
        else:
            neg_dst = neg_ei[1].view(self.A, self.k)
            z_pos_neg = z[neg_dst]
        return z_pos, z_pos_pos, z_pos_neg
