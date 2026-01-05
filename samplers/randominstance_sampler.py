import random, torch
from typing import List, Optional, Tuple
from torch_geometric.data import HeteroData
from negativeinstance_sampler import NegativeInstanceSampler


class RandomInstanceSampler(NegativeInstanceSampler):
    """ For Wiki framework: random corruption-based instance sampler. Same instance-example groups 
    as NegativeStatementSampler, but:
      - Anchor negative classes are NOT taken from graph. Instead, for each anchor u, corrupts its 
      positive classes to generate synthetic negatives: neg_classes(u) = random sample of classes 
      from a global class universe excluding pos_classes(u)
    The graph's negative statement edges are still used to find other instances that have negative
    statements to those synthetic classes (so shared_neg and neg_to_u_pos can be populated).
    """

    def __init__(self, k: int = 1, instance_rel: str = "2", subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_", external_negs: Optional[List[Tuple[int, int]]] = None,
        max_corrupt_per_anchor: Optional[int] = None, seed: Optional[int] = None):
        super().__init__(k=k, subclass_rel=subclass_rel,
            neg_prefix=neg_prefix, instance_rel=instance_rel,
            neg_expansion_hops=0, seed=seed)
        self.external_negs = external_negs or []
        self.max_corrupt_per_anchor = max_corrupt_per_anchor
        self._class_universe: List[int] = []

    def _build_class_universe(self) -> None:
        seen: set[int] = set()
        for u in self.anchors:
            for c in self.pos_classes[u]:
                seen.add(int(c))
        if not seen:
            for u in self.anchors:
                for c in self.neg_classes_direct[u]:
                    seen.add(int(c))
        self._class_universe = sorted(seen)

    def _corrupt_classes(self, pos_set: set[int], n_draw: int) -> List[int]:
        if not self._class_universe: return []
        if self.max_corrupt_per_anchor is not None:
            n_draw = min(n_draw, int(self.max_corrupt_per_anchor))
        n_draw = max(1, int(n_draw))
        neg: set[int] = set()
        attempts = 0
        max_attempts = max(200, 50 + 20 * n_draw)
        while len(neg) < n_draw and attempts < max_attempts:
            c = random.choice(self._class_universe)
            attempts += 1
            if c in pos_set: continue
            neg.add(int(c))
        return list(neg)

    def prepare_global(self, full_g: HeteroData) -> None:
        super().prepare_global(full_g)
        if self.external_negs:
            N = self.num_nodes
            neg_key_list: List[int] = []
            neg_val_list: List[int] = []

            for cls, insts in self.class2neg_instances.items():
                for inst in insts.tolist():
                    neg_key_list.append(int(cls))
                    neg_val_list.append(int(inst))

            for a, b in self.external_negs:
                if not (0 <= int(a) < N and 0 <= int(b) < N): continue
                inst, cls = self._as_inst_class(int(a), int(b))
                if inst is None or cls is None: continue
                self.neg_classes_direct[inst].append(int(cls))
                neg_key_list.append(int(cls))
                neg_val_list.append(int(inst))

            self.neg_classes_direct = [list(set(xs)) for xs in self.neg_classes_direct]
            self.class2neg_instances = (
                self._group_unique_by_key(torch.tensor(neg_key_list, dtype=torch.long),
                                          torch.tensor(neg_val_list, dtype=torch.long))
                if neg_key_list else {})

        self._build_class_universe()
        N = self.num_nodes
        self.neg_classes_expanded = [[] for _ in range(N)]
        for u in self.anchors:
            pos_set = set(int(c) for c in self.pos_classes[u])
            n_draw = len(pos_set) if len(pos_set) > 0 else 1
            self.neg_classes_expanded[u] = self._corrupt_classes(pos_set, n_draw)

        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_pos = [empty for _ in range(N)]
        self.pool_shared_neg = [empty for _ in range(N)]
        self.pool_pos_to_u_neg = [empty for _ in range(N)]
        self.pool_neg_to_u_pos = [empty for _ in range(N)]

        for u in self.anchors:
            pos_cls = self.pos_classes[u]
            neg_cls = self.neg_classes_expanded[u]
            self.pool_shared_pos[u] = self._collect_instances(pos_cls, self.class2pos_instances, exclude=u)
            self.pool_shared_neg[u] = self._collect_instances(neg_cls, self.class2neg_instances, exclude=u)
            self.pool_pos_to_u_neg[u] = self._collect_instances(neg_cls, self.class2pos_instances, exclude=u)
            self.pool_neg_to_u_pos[u] = self._collect_instances(pos_cls, self.class2neg_instances, exclude=u)

        self._global2local_cpu = None
        self._in_batch_mask_cpu = None
