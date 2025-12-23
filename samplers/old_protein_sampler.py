import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class ProteinStatementSampler:
    """  Builds protein↔protein pools using statement edges protein->class:
    share_neg[u] = proteins that share at least one DIRECT neg_statement class with u
    share_pos[u] = proteins that share at least one DIRECT pos_statement class with u
    pos_to_my_neg[u] = proteins that have POS statements to any class in u's EXPANDED neg-class set
    neg_to_my_pos[u] = proteins that have NEG statements to any class in u's DIRECT pos-class set
    Then per batch it samples:
      - 1 share_neg protein (for pulling z_neg to z_neg)
      - 1 share_pos protein (for pulling z_pos to z_pos)
      - k_neg proteins from pos_to_my_neg (for pushing z_neg(anchor) away from z_pos(other))
      - k_neg proteins from neg_to_my_pos (for pushing z_pos(anchor) away from z_neg(other))
    """

    def __init__(self, anchor_etype: str = "PPI", pos_etype: str = "pos_statement",
        neg_etype: str = "neg_statement", go_etype: str = "link", k_neg: int = 1,
        use_go_expansion_for_neg: bool = True, seed: int = 42):
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.go_etype = go_etype
        self.k_neg = int(k_neg)
        self.use_go_expansion_for_neg = bool(use_go_expansion_for_neg)
        self.seed = int(seed)
        self.device = torch.device("cpu")
        self.n_type: Optional[str] = None
        self.N: int = 0
        self.anchors_global: List[int] = []
        self.pos_classes: List[List[int]] = []
        self.neg_classes_direct: List[List[int]] = []
        self.neg_classes_expanded: List[List[int]] = []

        self.class_to_pos_proteins: Dict[int, List[int]] = {}
        self.class_to_neg_proteins: Dict[int, List[int]] = {}
        self.share_neg: Dict[int, List[int]] = {}
        self.share_pos: Dict[int, List[int]] = {}
        self.pos_to_my_neg: Dict[int, List[int]] = {}
        self.neg_to_my_pos: Dict[int, List[int]] = {}
        self.batch_anchors_local: List[int] = []
        random.seed(self.seed)


    def _build_go_predecessors(self, full_g: HeteroData) -> Dict[int, List[int]]:
        """  Builds a predecessors list for GO nodes from edge_index of go_etype.
          for edge s->d, s is a predecessor (parent/ancestor) of d.
        """
        predecessors: Dict[int, List[int]] = {}
        go_key = _find_key_by_rel(full_g, self.go_etype)
        if "edge_index" not in full_g[go_key]: return predecessors

        go_src, go_dst = full_g[go_key].edge_index
        for s, d in zip(go_src.tolist(), go_dst.tolist()):
            predecessors.setdefault(d, []).append(s)
        return predecessors

    def prepare_global(self, full_g: HeteroData):
        """ Precompute all pools for every protein.
        Call once per fold (on the full fold training graph used for contrastive sampling logic).
        """
        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_a, dst_a = full_g[anchor_key].edge_index
        src_nt, _, _ = anchor_key
        self.n_type = src_nt
        self.N = _num_nodes_of(full_g, self.n_type)

        anchors = set(src_a.tolist()) | set(dst_a.tolist())
        self.anchors_global = sorted(anchors)

        self.pos_classes = [[] for _ in range(self.N)]
        self.neg_classes_direct = [[] for _ in range(self.N)]

        pos_key = _find_key_by_rel(full_g, self.pos_etype)
        if "edge_index" in full_g[pos_key]:
            src_pos, dst_pos = full_g[pos_key].edge_index
            for u, c in zip(src_pos.tolist(), dst_pos.tolist()):
                self.pos_classes[u].append(c)
                self.class_to_pos_proteins.setdefault(c, []).append(u)

        neg_key = _find_key_by_rel(full_g, self.neg_etype)
        if "edge_index" in full_g[neg_key]:
            src_neg, dst_neg = full_g[neg_key].edge_index
            for u, c in zip(src_neg.tolist(), dst_neg.tolist()):
                self.neg_classes_direct[u].append(c)
                self.class_to_neg_proteins.setdefault(c, []).append(u)

        # GO expansion for anchor negative classes
        self.neg_classes_expanded = [[] for _ in range(self.N)]
        if self.use_go_expansion_for_neg:
            predecessors = self._build_go_predecessors(full_g)
            for u in range(self.N):
                direct = self.neg_classes_direct[u]
                if not direct:
                    self.neg_classes_expanded[u] = []
                    continue
                sup = set(direct)
                for c in direct:
                    sup.update(predecessors.get(c, []))
                self.neg_classes_expanded[u] = list(sup)
        else:
            for u in range(self.N):
                self.neg_classes_expanded[u] = list(set(self.neg_classes_direct[u]))

        self.share_neg.clear()
        self.share_pos.clear()
        self.pos_to_my_neg.clear()
        self.neg_to_my_pos.clear()

        for u in self.anchors_global:
            share_neg_set = set()
            for c in set(self.neg_classes_direct[u]):
                for p in self.class_to_neg_proteins.get(c, []):
                    if p != u: share_neg_set.add(p)
            self.share_neg[u] = list(share_neg_set)

            share_pos_set = set()
            for c in set(self.pos_classes[u]):
                for p in self.class_to_pos_proteins.get(c, []):
                    if p != u: share_pos_set.add(p)
            self.share_pos[u] = list(share_pos_set)

            pos_to_my_neg_set = set()
            for c in set(self.neg_classes_expanded[u]):
                for p in self.class_to_pos_proteins.get(c, []):
                    if p != u: pos_to_my_neg_set.add(p)
            self.pos_to_my_neg[u] = list(pos_to_my_neg_set)

            neg_to_my_pos_set = set()
            for c in set(self.pos_classes[u]):
                for p in self.class_to_neg_proteins.get(c, []):
                    if p != u: neg_to_my_pos_set.add(p)
            self.neg_to_my_pos[u] = list(neg_to_my_pos_set)


    def prepare_batch(self, batch: HeteroData):
        """
        Pick which anchors to use from this batch graph.
        We use proteins that appear in anchor_etype edges inside this batch.
        """
        some_key = next(iter(batch.edge_types))
        self.device = batch[some_key].edge_index.device
        anchor_key = _find_key_by_rel(batch, self.anchor_etype)
        src, dst = batch[anchor_key].edge_index
        anchors = torch.unique(torch.cat([src, dst], dim=0)).tolist()
        anchors = [u for u in anchors if 0 <= u < self.N]
        self.batch_anchors_local = anchors

    def _sample_one(self, pool: List[int]) -> int:
        return random.choice(pool)

    def _sample_k(self, pool: List[int], k: int) -> List[int]:
        if len(pool) >= k: return random.sample(pool, k)
        if len(pool) == 0: return []
        return [random.choice(pool) for _ in range(k)]

    def sample_batch(self) -> Dict[str, Tensor]:
        """  Returns tensors (on sampler.device):
          anchors: (B,)
          pos_same_neg: (B,) or -1 if missing
          pos_same_pos: (B,) or -1 if missing
          neg_pos_to_my_neg: (B, k_neg) with -1 for missing entries/rows
          neg_neg_to_my_pos: (B, k_neg) with -1 for missing entries/rows
        """
        device = self.device
        B = len(self.batch_anchors_local)
        if B == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {"anchors": empty, "pos_same_neg": empty, "pos_same_pos": empty,
                "neg_pos_to_my_neg": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
                "neg_neg_to_my_pos": torch.empty(0, self.k_neg, dtype=torch.long, device=device)}

        anchors_t = torch.tensor(self.batch_anchors_local, dtype=torch.long, device=device)
        pos_same_neg = torch.full((B,), -1, dtype=torch.long, device=device)
        pos_same_pos = torch.full((B,), -1, dtype=torch.long, device=device)
        neg_pos_to_my_neg = torch.full((B, self.k_neg), -1, dtype=torch.long, device=device)
        neg_neg_to_my_pos = torch.full((B, self.k_neg), -1, dtype=torch.long, device=device)

        for i, u in enumerate(self.batch_anchors_local):
            pool_share_neg = self.share_neg.get(u, [])
            pool_share_pos = self.share_pos.get(u, [])
            pool_pos_to_my_neg = self.pos_to_my_neg.get(u, [])
            pool_neg_to_my_pos = self.neg_to_my_pos.get(u, [])
            if len(pool_share_neg) > 0: pos_same_neg[i] = self._sample_one(pool_share_neg)
            if len(pool_share_pos) > 0: pos_same_pos[i] = self._sample_one(pool_share_pos)

            sampA = self._sample_k(pool_pos_to_my_neg, self.k_neg)
            if len(sampA) > 0:
                neg_pos_to_my_neg[i, :len(sampA)] = torch.tensor(sampA, dtype=torch.long, device=device)
            sampB = self._sample_k(pool_neg_to_my_pos, self.k_neg)
            if len(sampB) > 0:
                neg_neg_to_my_pos[i, :len(sampB)] = torch.tensor(sampB, dtype=torch.long, device=device)

        return {"anchors": anchors_t, "pos_same_neg": pos_same_neg, "pos_same_pos": pos_same_pos,
            "neg_pos_to_my_neg": neg_pos_to_my_neg, "neg_neg_to_my_pos": neg_neg_to_my_pos}