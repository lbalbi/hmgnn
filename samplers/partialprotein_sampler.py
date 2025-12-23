# partialprotein_sampler.py
import random, torch
from typing import List, Tuple, Optional, Dict
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class PartialProteinSampler:
    """Protein-protein sampler that can use *external* statement edges when 
    passing flags like --use_nstatement_sampler / --use_pstatement_sampler to
    remove statement edges from the graph use them for contrastive learning.

    This sampler mirrors ProteinStatementSampler logic, but:
      - reads missing statement edges from externally provided lists
      - uses an *external anchor list* (proteins) instead of pulling anchors from the
        PPI edges inside the current batch
      - if fewer than k negatives exist, it returns only the available ones
        (with a boolean mask)
      - if an anchor does not have at least 1 example for EACH required example type,
        the anchor is skipped entirely

    Required pools per anchor u (must all be non-empty to keep u):
      1) share_neg[u] (one sampled protein)
      2) share_pos[u] (one sampled protein)
      3) pos_to_my_neg[u] (>=1 sampled protein; up to k_neg)
      4) neg_to_my_pos[u] (>=1 sampled protein; up to k_neg)
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
        external_pos_edges: Optional[List[Tuple[int, int]]] = None,
        external_neg_edges: Optional[List[Tuple[int, int]]] = None,
        anchors: Optional[List[int]] = None,
    ):
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.go_etype = go_etype
        self.k_neg = int(k_neg)
        self.use_go_expansion_for_neg = bool(use_go_expansion_for_neg)
        self.seed = int(seed)
        self.external_pos_edges = external_pos_edges or []
        self.external_neg_edges = external_neg_edges or []
        self.anchors_external = anchors  # optional; if None inferred from statement edges

        self.device = torch.device("cpu")
        self.n_type: Optional[str] = None
        self.N: int = 0

        # per-protein statement class lists
        self.pos_classes: List[List[int]] = []
        self.neg_classes_direct: List[List[int]] = []
        self.neg_classes_expanded: List[List[int]] = []

        # class -> proteins reverse index
        self.class_to_pos_proteins: Dict[int, List[int]] = {}
        self.class_to_neg_proteins: Dict[int, List[int]] = {}

        # protein pools
        self.share_neg: Dict[int, List[int]] = {}
        self.share_pos: Dict[int, List[int]] = {}
        self.pos_to_my_neg: Dict[int, List[int]] = {}
        self.neg_to_my_pos: Dict[int, List[int]] = {}

        # anchors used for sampling in current epoch/batch context
        self.anchors_global: List[int] = []
        self.batch_anchors_local: List[int] = []

        random.seed(self.seed)

    def _build_go_predecessors(self, full_g: HeteroData) -> Dict[int, List[int]]:
        predecessors: Dict[int, List[int]] = {}
        go_key = _find_key_by_rel(full_g, self.go_etype)
        if go_key is None or "edge_index" not in full_g[go_key]:
            return predecessors
        go_src, go_dst = full_g[go_key].edge_index
        for s, d in zip(go_src.tolist(), go_dst.tolist()):
            predecessors.setdefault(d, []).append(s)
        return predecessors

    def _iter_statement_edges(
        self, full_g: HeteroData, etype: str, external_edges: List[Tuple[int, int]]
    ) -> Tuple[List[int], List[int]]:
        """Return (src_list, dst_list) for statement edges."""
        if external_edges:
            src, dst = zip(*external_edges)
            return list(src), list(dst)

        key = _find_key_by_rel(full_g, etype)
        if key is None or "edge_index" not in full_g[key]:
            return [], []
        s, d = full_g[key].edge_index
        return s.tolist(), d.tolist()

    def prepare_global(self, full_g: HeteroData):
        """Precompute pools for every anchor protein (once per fold)."""
        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        if anchor_key is None:
            raise ValueError(f"Could not find edge type with relation='{self.anchor_etype}'")
        src_nt, _, _ = anchor_key
        self.n_type = src_nt
        self.N = _num_nodes_of(full_g, self.n_type)

        src_pos, dst_pos = self._iter_statement_edges(full_g, self.pos_etype, self.external_pos_edges)
        src_neg, dst_neg = self._iter_statement_edges(full_g, self.neg_etype, self.external_neg_edges)

        if self.anchors_external is not None:
            anchors = set(int(x) for x in self.anchors_external)
        else:
            anchors = set(src_pos) | set(src_neg)
            if not anchors:
                a_src, a_dst = full_g[anchor_key].edge_index
                anchors = set(a_src.tolist()) | set(a_dst.tolist())

        self.anchors_global = sorted([u for u in anchors if 0 <= u < self.N])

        self.pos_classes = [[] for _ in range(self.N)]
        self.neg_classes_direct = [[] for _ in range(self.N)]
        self.neg_classes_expanded = [[] for _ in range(self.N)]
        self.class_to_pos_proteins.clear()
        self.class_to_neg_proteins.clear()

        for u, c in zip(src_pos, dst_pos):
            if 0 <= u < self.N:
                self.pos_classes[u].append(c)
                self.class_to_pos_proteins.setdefault(c, []).append(u)

        for u, c in zip(src_neg, dst_neg):
            if 0 <= u < self.N:
                self.neg_classes_direct[u].append(c)
                self.class_to_neg_proteins.setdefault(c, []).append(u)

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

    def prepare_batch(self, batch: HeteroData, anchors: Optional[List[int]] = None):
        """Set device and anchor list for sample_batch()."""
        some_key = next(iter(batch.edge_types))
        self.device = batch[some_key].edge_index.device
        use = anchors if anchors is not None else self.anchors_global
        self.batch_anchors_local = [int(u) for u in use if 0 <= int(u) < self.N]

    def _sample_one(self, pool: List[int]) -> int:
        return random.choice(pool)

    def _sample_up_to_k(self, pool: List[int], k: int) -> List[int]:
        if not pool:
            return []
        if len(pool) <= k:
            return list(pool)
        return random.sample(pool, k)

    def sample_batch(self) -> Dict[str, Tensor]:
        device = self.device
        k = self.k_neg

        candidates = self.batch_anchors_local
        if not candidates:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {
                "anchors": empty,
                "pos_same_neg": empty,
                "pos_same_pos": empty,
                "neg_pos_to_my_neg": torch.empty(0, k, dtype=torch.long, device=device),
                "neg_pos_to_my_neg_mask": torch.empty(0, k, dtype=torch.bool, device=device),
                "neg_neg_to_my_pos": torch.empty(0, k, dtype=torch.long, device=device),
                "neg_neg_to_my_pos_mask": torch.empty(0, k, dtype=torch.bool, device=device),
            }

        valid_anchors: List[int] = []
        for u in candidates:
            if not self.share_neg.get(u) or not self.share_pos.get(u):
                continue
            if not self.pos_to_my_neg.get(u) or not self.neg_to_my_pos.get(u):
                continue
            valid_anchors.append(u)

        if not valid_anchors:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {
                "anchors": empty,
                "pos_same_neg": empty,
                "pos_same_pos": empty,
                "neg_pos_to_my_neg": torch.empty(0, k, dtype=torch.long, device=device),
                "neg_pos_to_my_neg_mask": torch.empty(0, k, dtype=torch.bool, device=device),
                "neg_neg_to_my_pos": torch.empty(0, k, dtype=torch.long, device=device),
                "neg_neg_to_my_pos_mask": torch.empty(0, k, dtype=torch.bool, device=device),
            }

        B = len(valid_anchors)
        anchors_t = torch.tensor(valid_anchors, dtype=torch.long, device=device)

        pos_same_neg = torch.empty((B,), dtype=torch.long, device=device)
        pos_same_pos = torch.empty((B,), dtype=torch.long, device=device)

        neg_pos_to_my_neg = torch.full((B, k), -1, dtype=torch.long, device=device)
        neg_pos_to_my_neg_mask = torch.zeros((B, k), dtype=torch.bool, device=device)
        neg_neg_to_my_pos = torch.full((B, k), -1, dtype=torch.long, device=device)
        neg_neg_to_my_pos_mask = torch.zeros((B, k), dtype=torch.bool, device=device)

        for i, u in enumerate(valid_anchors):
            pos_same_neg[i] = self._sample_one(self.share_neg[u])
            pos_same_pos[i] = self._sample_one(self.share_pos[u])

            sampA = self._sample_up_to_k(self.pos_to_my_neg[u], k)
            if sampA:
                L = len(sampA)
                neg_pos_to_my_neg[i, :L] = torch.tensor(sampA, dtype=torch.long, device=device)
                neg_pos_to_my_neg_mask[i, :L] = True

            sampB = self._sample_up_to_k(self.neg_to_my_pos[u], k)
            if sampB:
                L = len(sampB)
                neg_neg_to_my_pos[i, :L] = torch.tensor(sampB, dtype=torch.long, device=device)
                neg_neg_to_my_pos_mask[i, :L] = True

        return {
            "anchors": anchors_t,
            "pos_same_neg": pos_same_neg,
            "pos_same_pos": pos_same_pos,
            "neg_pos_to_my_neg": neg_pos_to_my_neg,
            "neg_pos_to_my_neg_mask": neg_pos_to_my_neg_mask,
            "neg_neg_to_my_pos": neg_neg_to_my_pos,
            "neg_neg_to_my_pos_mask": neg_neg_to_my_pos_mask,
        }
