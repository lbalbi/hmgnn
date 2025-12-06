import random
import torch
from typing import List, Tuple, Dict, Optional
from torch import Tensor
from torch_geometric.data import HeteroData

from .utils import _num_nodes_of


class RandomStatementSampler:
    """
    Random sampler over statement edges in a multi-relation KG.
    • Anchors:
        - Sources of relation `instance_rel` (e.g. "2" for "instance of").
        - If no such edges exist, fall back to all nodes with at least one
          positive statement; if still empty, all nodes.
    • Positive statements:
        - Any edge (u, r, v) where:
              r does NOT start with `neg_prefix`
              and r != subclass_rel
              and r != instance_rel
          Define: pos_global[u] and reverse pos neighbors rev_pos_global[v].
    • Candidate negatives (global):
        - If `external_negs` is given:
              global_candidates = { v | (u, v) in external_negs }.
        - Otherwise:
              global_candidates = all positive targets:
                   { v | ∃(u, r, v) positive }.
    • Per-anchor negative pool:
        - For each anchor u:
             invalid(u) = pos_global[u] ∪ rev_pos_global[u] ∪ {u}
             neg_global[u] = { v in global_candidates | v not in invalid(u) }
        - If empty, we fall back to using positive neighbors or the anchor itself
          at batch-time.
    """

    def __init__(self, k: int = 1, instance_rel: str = "2",
        subclass_rel: str = "subclass_of", neg_prefix: str = "NOT_",
        external_negs: Optional[List[Tuple[int, int]]] = None):
        self.k = k
        self.instance_rel = instance_rel
        self.subclass_rel = subclass_rel
        self.neg_prefix = neg_prefix
        self.external_negs = external_negs or []
        self.node_type: Optional[str] = None
        self.num_nodes: int = 0

        self.anchors_global: List[int] = []
        self.pos_global: List[List[int]] = []
        self.neg_global: Dict[int, List[int]] = {}
        self.batch_global: List[int] = []
        self.batch_locals: List[int] = []
        self.pos_local: List[List[int]] = []
        self.neg_cands: List[List[int]] = []
        self.N: int = 0
        self.device = torch.device("cpu")

    def prepare_global(self, full_g: HeteroData) -> None:
        """
        Build global anchors (instance sources), positive neighbors,
        and per-anchor candidate negative sets.
        """
        if len(full_g.node_types) != 1:
            raise ValueError(
                f"RandomStatementSampler assumes a single node type; got {full_g.node_types}"
            )
        self.node_type = full_g.node_types[0]

        if hasattr(full_g[self.node_type], "num_nodes") and full_g[self.node_type].num_nodes is not None:
            N = int(full_g[self.node_type].num_nodes)
        else:
            max_id = -1
            for (_, _, _), eidx in full_g.edge_index_dict.items():
                if eidx.numel() > 0: max_id = max(max_id, int(eidx.max().item()))
            N = max_id + 1 if max_id >= 0 else 0
        self.num_nodes = N
        self.pos_global = [[] for _ in range(N)]
        rev_pos_global: List[List[int]] = [[] for _ in range(N)]

        anchor_sources: set = set()
        global_pos_targets: set = set()

        for (src_nt, rel, dst_nt), eidx in full_g.edge_index_dict.items():
            if src_nt != self.node_type or dst_nt != self.node_type: continue
            if eidx.numel() == 0: continue

            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            if self.instance_rel is not None and rel == self.instance_rel:
                for s in src_list:
                    if 0 <= s < N: anchor_sources.add(s)
                continue
            if rel == self.subclass_rel: continue
            if rel.startswith(self.neg_prefix): continue

            for u, v in zip(src_list, dst_list):
                if 0 <= u < N and 0 <= v < N:
                    self.pos_global[u].append(v)
                    rev_pos_global[v].append(u)
                    global_pos_targets.add(v)

        if self.external_negs:
            global_candidates = {v for (_, v) in self.external_negs if 0 <= v < N}
        else: global_candidates = global_pos_targets

        if self.instance_rel is not None and anchor_sources:anchors = sorted(anchor_sources)
        else:
            anchors = [u for u in range(N) if self.pos_global[u]]
            if not anchors: anchors = list(range(N))
        self.anchors_global = anchors

        neg_global: Dict[int, List[int]] = {}
        for u in self.anchors_global:
            invalid = set(self.pos_global[u])
            if 0 <= u < len(rev_pos_global): invalid |= set(rev_pos_global[u])
            invalid.add(u)
            cand = [v for v in global_candidates if v not in invalid]
            neg_global[u] = cand
        self.neg_global = neg_global


    def prepare_batch(self, batch: HeteroData) -> None:
        """
        Derive batch anchors and local candidates.
        In your setup, `batch` is the full graph, so local == global IDs.
        """
        if not batch.edge_types:
            self.N = 0
            self.batch_global = []
            self.batch_locals = []
            self.pos_local = []
            self.neg_cands = []
            return

        any_key = next(iter(batch.edge_types))
        self.device = (batch[any_key].edge_index.device
            if batch[any_key].edge_index.is_cuda
            else torch.device("cpu"))

        if self.node_type is None: self.node_type = any_key[0]
        B = _num_nodes_of(batch, self.node_type)
        orig = list(range(B))
        mapping: Dict[int, int] = {g: i for i, g in enumerate(orig)}

        batch_global: List[int] = []
        batch_locals: List[int] = []
        for g in self.anchors_global:
            if g in mapping:
                batch_global.append(g)
                batch_locals.append(mapping[g])

        self.batch_global = batch_global
        self.batch_locals = batch_locals
        self.N = len(self.batch_locals)

        self.pos_local = []
        self.neg_cands = []

        for g, local in zip(self.batch_global, self.batch_locals):
            plc: List[int] = []
            for v in self.pos_global[g]:
                idx = mapping.get(int(v))
                if idx is not None: plc.append(idx)
            self.pos_local.append(plc)
            ngc: List[int] = []
            for v in self.neg_global.get(g, []):
                idx = mapping.get(int(v))
                if idx is not None: ngc.append(idx)
            if not ngc: ngc = plc if plc else [local]
            self.neg_cands.append(ngc)


    def sample(self) -> Tensor:
        """
        For each batch anchor, sample k negatives with replacement.
        Returns edge_index [2, N*k] (local coordinates).
        """
        if self.N == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)

        src_list: List[int] = []
        dst_list: List[int] = []

        for local_anchor, negs in zip(self.batch_locals, self.neg_cands):
            chosen = random.choices(negs, k=self.k)
            src_list.extend([local_anchor] * self.k)
            dst_list.extend(chosen)
        idx = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device)
        return idx

    def get_contrastive_samples(self, z: Tensor,
        neg_ei: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        z: (B, D) node embeddings (local indexing).
        If neg_ei is None, internally calls self.sample().
        Returns:
            z_anchor: (A, D) embeddings for anchors only
            z_pos_pos: (A, D) positive neighbor embeddings
            z_pos_neg: (A, k, D) negative embeddings
        """
        A = len(self.batch_locals)
        if A == 0:
            D = z.size(-1)
            empty = torch.empty(0, D, device=z.device)
            empty_neg = torch.empty(0, self.k, D, device=z.device)
            return empty, empty, empty_neg

        if neg_ei is None: neg_ei = self.sample()
        k = self.k
        z_anchor = z[self.batch_locals]
        pos_idx = [random.choice(self.pos_local[i]) if self.pos_local[i] else self.batch_locals[i]
            for i in range(A)]
        pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=z.device)
        z_pos_pos = z[pos_idx]

        neg_dst = neg_ei[1].view(A, k)
        z_pos_neg = z[neg_dst]
        return z_anchor, z_pos_pos, z_pos_neg
