import random
from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .utils import _find_key_by_rel, _num_nodes_of


class ProteinStatementSampler:
    """Protein↔protein contrastive sampler.

    Changes vs previous version:
      - Never repeats proteins to fill k_neg; if fewer than k_neg exist, returns only those and pads the rest with -1.
      - Drops anchors that do not have at least 1 example for EACH required type:
          (share_neg, share_pos, pos_to_my_neg, neg_to_my_pos)
        so the contrastive loss only receives fully-formed tuples.

    Returned batch tensors:
      anchors: (B,)
      pos_same_neg: (B,)
      pos_same_pos: (B,)
      neg_pos_to_my_neg: (B, k_neg) padded with -1
      neg_neg_to_my_pos: (B, k_neg) padded with -1
    """

    def __init__(
        self,
        anchor_etype: str = "PPI",
        pos_etype: str = "pos_statement",
        neg_etype: str = "neg_statement",
        go_etype: str = "link",
        k_neg: int = 1,
        use_go_expansion_for_neg: bool = True,
        seed: int = 42,
    ):
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
        predecessors: Dict[int, List[int]] = {}
        go_key = _find_key_by_rel(full_g, self.go_etype)
        if "edge_index" not in full_g[go_key]:
            return predecessors

        go_src, go_dst = full_g[go_key].edge_index
        for s, d in zip(go_src.tolist(), go_dst.tolist()):
            predecessors.setdefault(d, []).append(s)
        return predecessors

    def prepare_global(self, full_g: HeteroData):
        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_a, dst_a = full_g[anchor_key].edge_index
        src_nt, _, _ = anchor_key
        self.n_type = src_nt
        self.N = _num_nodes_of(full_g, self.n_type)

        anchors = set(src_a.tolist()) | set(dst_a.tolist())
        self.anchors_global = sorted(anchors)

        # reset
        self.pos_classes = [[] for _ in range(self.N)]
        self.neg_classes_direct = [[] for _ in range(self.N)]
        self.class_to_pos_proteins.clear()
        self.class_to_neg_proteins.clear()

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
                    if p != u:
                        share_neg_set.add(p)
            self.share_neg[u] = list(share_neg_set)

            share_pos_set = set()
            for c in set(self.pos_classes[u]):
                for p in self.class_to_pos_proteins.get(c, []):
                    if p != u:
                        share_pos_set.add(p)
            self.share_pos[u] = list(share_pos_set)

            pos_to_my_neg_set = set()
            for c in set(self.neg_classes_expanded[u]):
                for p in self.class_to_pos_proteins.get(c, []):
                    if p != u:
                        pos_to_my_neg_set.add(p)
            self.pos_to_my_neg[u] = list(pos_to_my_neg_set)

            neg_to_my_pos_set = set()
            for c in set(self.pos_classes[u]):
                for p in self.class_to_neg_proteins.get(c, []):
                    if p != u:
                        neg_to_my_pos_set.add(p)
            self.neg_to_my_pos[u] = list(neg_to_my_pos_set)

    def prepare_batch(self, batch: HeteroData):
        some_key = next(iter(batch.edge_types))
        self.device = batch[some_key].edge_index.device

        anchor_key = _find_key_by_rel(batch, self.anchor_etype)
        src, dst = batch[anchor_key].edge_index
        anchors = torch.unique(torch.cat([src, dst], dim=0)).tolist()
        anchors = [u for u in anchors if 0 <= u < self.N]
        self.batch_anchors_local = anchors

    def _sample_one(self, pool: List[int]) -> int:
        return random.choice(pool)

    def _sample_k_no_repeat(self, pool: List[int], k: int) -> List[int]:
        """Return up to k unique samples, no repetition."""
        if not pool:
            return []
        if len(pool) <= k:
            # up to k, no repeats
            return random.sample(pool, len(pool))
        return random.sample(pool, k)

    def sample_batch(self) -> Dict[str, Tensor]:
        device = self.device

        if len(self.batch_anchors_local) == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {
                "anchors": empty,
                "pos_same_neg": empty,
                "pos_same_pos": empty,
                "neg_pos_to_my_neg": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
                "neg_neg_to_my_pos": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
            }

        kept_anchors: List[int] = []
        kept_pos_same_neg: List[int] = []
        kept_pos_same_pos: List[int] = []
        kept_negA: List[List[int]] = []
        kept_negB: List[List[int]] = []

        for u in self.batch_anchors_local:
            pool_share_neg = self.share_neg.get(u, [])
            pool_share_pos = self.share_pos.get(u, [])
            pool_pos_to_my_neg = self.pos_to_my_neg.get(u, [])
            pool_neg_to_my_pos = self.neg_to_my_pos.get(u, [])

            # requirement: must have at least 1 example for each type
            if not pool_share_neg or not pool_share_pos or not pool_pos_to_my_neg or not pool_neg_to_my_pos:
                continue

            v_same_neg = self._sample_one(pool_share_neg)
            v_same_pos = self._sample_one(pool_share_pos)
            negA = self._sample_k_no_repeat(pool_pos_to_my_neg, self.k_neg)
            negB = self._sample_k_no_repeat(pool_neg_to_my_pos, self.k_neg)

            # negA/negB must have at least 1 (already ensured by pool non-empty, but keep safe)
            if len(negA) == 0 or len(negB) == 0:
                continue

            kept_anchors.append(u)
            kept_pos_same_neg.append(v_same_neg)
            kept_pos_same_pos.append(v_same_pos)
            kept_negA.append(negA)
            kept_negB.append(negB)

        B = len(kept_anchors)
        if B == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {
                "anchors": empty,
                "pos_same_neg": empty,
                "pos_same_pos": empty,
                "neg_pos_to_my_neg": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
                "neg_neg_to_my_pos": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
            }

        anchors_t = torch.tensor(kept_anchors, dtype=torch.long, device=device)
        pos_same_neg = torch.tensor(kept_pos_same_neg, dtype=torch.long, device=device)
        pos_same_pos = torch.tensor(kept_pos_same_pos, dtype=torch.long, device=device)

        neg_pos_to_my_neg = torch.full((B, self.k_neg), -1, dtype=torch.long, device=device)
        neg_neg_to_my_pos = torch.full((B, self.k_neg), -1, dtype=torch.long, device=device)

        for i in range(B):
            negA = kept_negA[i]
            negB = kept_negB[i]
            if len(negA) > 0:
                neg_pos_to_my_neg[i, : len(negA)] = torch.tensor(negA, dtype=torch.long, device=device)
            if len(negB) > 0:
                neg_neg_to_my_pos[i, : len(negB)] = torch.tensor(negB, dtype=torch.long, device=device)

        return {
            "anchors": anchors_t,
            "pos_same_neg": pos_same_neg,
            "pos_same_pos": pos_same_pos,
            "neg_pos_to_my_neg": neg_pos_to_my_neg,
            "neg_neg_to_my_pos": neg_neg_to_my_pos,
        }
