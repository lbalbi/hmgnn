import torch
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.data import HeteroData

from .randominstance_sampler import RandomInstanceSampler

ExternalStmt = Union[Tuple[int, int], Tuple[int, str, int], Tuple[int, int, int]]


class TypedInstanceSampler(RandomInstanceSampler):
    """
    Type-constrained variant of RandomInstanceSampler.

    For each positive (anchor -> tail) in a batch, corrupted tails are sampled only
    from entities that share at least one `instance_rel` class with the original tail.

    Concretely:
      1) Build global maps from `instance_rel` edges:
         - inst2classes[entity] = set(classes)
         - class2instances[class] = set(entities)
      2) For each batch anchor, build candidate corrupted tails as:
         union_{pos_tail t of anchor} union_{c in inst2classes[t]} class2instances[c]
         then remove the anchor and all its positive tails.
    """

    def __init__(
        self,
        k: int = 1,
        instance_rel: str = "2",
        subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_",
        external_negs: Optional[List[ExternalStmt]] = None,
        max_corrupt_per_anchor: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            k=k,
            instance_rel=instance_rel,
            subclass_rel=subclass_rel,
            neg_prefix=neg_prefix,
            external_negs=external_negs,
            max_corrupt_per_anchor=max_corrupt_per_anchor,
            seed=seed,
        )
        self._inst2classes: Dict[int, Tuple[int, ...]] = {}
        self._class2instances: Dict[int, Tuple[int, ...]] = {}
        self._batch_typed_neg_tails_by_inst: Optional[Dict[int, List[int]]] = None
        print("Using TypedInstanceSampler")

    def prepare_global(self, full_g: HeteroData) -> None:
        super().prepare_global(full_g)

        inst2classes_tmp: Dict[int, set] = {}
        class2instances_tmp: Dict[int, set] = {}

        inst_key = self._find_edge_key(full_g, self.instance_rel)
        if inst_key is None or "edge_index" not in full_g[inst_key]:
            self._inst2classes = {}
            self._class2instances = {}
            return

        eidx = full_g[inst_key].edge_index
        if eidx is None or eidx.numel() == 0:
            self._inst2classes = {}
            self._class2instances = {}
            return

        src = eidx[0].detach().cpu().tolist()
        dst = eidx[1].detach().cpu().tolist()
        N = int(self.num_nodes)

        for a, b in zip(src, dst):
            if not (0 <= int(a) < N and 0 <= int(b) < N):
                continue
            inst, cls = self._as_inst_class(int(a), int(b))
            if inst is None or cls is None:
                continue
            inst2classes_tmp.setdefault(int(inst), set()).add(int(cls))
            class2instances_tmp.setdefault(int(cls), set()).add(int(inst))

        self._inst2classes = {k: tuple(sorted(v)) for k, v in inst2classes_tmp.items()}
        self._class2instances = {k: tuple(sorted(v)) for k, v in class2instances_tmp.items()}

    def prepare_batch(self, batch: HeteroData) -> None:
        super().prepare_batch(batch)
        self._batch_typed_neg_tails_by_inst = {}

        if self._batch_global_for_row is None or self._batch_pos_tails_by_inst is None:
            return

        global_for_row = self._batch_global_for_row
        if global_for_row.numel() == 0:
            return

        g2l = {int(g): int(i) for i, g in enumerate(global_for_row.tolist())}

        for u_local, pos_tails_local in self._batch_pos_tails_by_inst.items():
            if not pos_tails_local:
                continue

            pos_local_set = set(int(x) for x in pos_tails_local)
            cand_locals: set = set()

            for t_local in pos_local_set:
                if t_local < 0 or t_local >= int(global_for_row.numel()):
                    continue
                t_global = int(global_for_row[t_local].item())
                tail_classes = self._inst2classes.get(t_global, ())
                if not tail_classes:
                    continue

                for cls in tail_classes:
                    same_type_entities = self._class2instances.get(int(cls), ())
                    for ent_global in same_type_entities:
                        ent_local = g2l.get(int(ent_global))
                        if ent_local is None:
                            continue
                        if ent_local == int(u_local):
                            continue
                        if ent_local in pos_local_set:
                            continue
                        cand_locals.add(int(ent_local))

            if cand_locals:
                self._batch_typed_neg_tails_by_inst[int(u_local)] = sorted(cand_locals)

    def get_contrastive_samples(
        self,
        z: torch.Tensor,
        anchor_nodes: Optional[torch.Tensor] = None,
        n_id: Optional[torch.Tensor] = None,
    ):
        device = z.device
        B_rows, D = z.shape
        k = self.k

        if self._batch_global_for_row is None or self._batch_pos_tails_by_inst is None:
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty_k, empty_k, empty_k, empty_k

        if anchor_nodes is None or anchor_nodes.numel() == 0:
            anchor_locals = torch.arange(B_rows, dtype=torch.long)
        else:
            anchor_locals = anchor_nodes.detach().long().cpu().unique()

        if self.max_contrastive_anchors and anchor_locals.numel() > self.max_contrastive_anchors:
            perm = torch.randperm(anchor_locals.numel())[: self.max_contrastive_anchors]
            anchor_locals = anchor_locals[perm]

        anchors_local_list: List[int] = []
        shneg_idx: List[torch.Tensor] = []
        pos2neg_idx: List[torch.Tensor] = []
        neg2pos_idx: List[torch.Tensor] = []
        shpos_idx: List[torch.Tensor] = []

        typed_neg_map = self._batch_typed_neg_tails_by_inst or {}

        for u_local in anchor_locals.tolist():
            if u_local < 0 or u_local >= B_rows:
                continue

            pos_tails = self._batch_pos_tails_by_inst.get(int(u_local), [])
            neg_tails = typed_neg_map.get(int(u_local), [])
            if not pos_tails or not neg_tails:
                continue

            cand_pos = torch.tensor(pos_tails, dtype=torch.long)
            cand_neg = torch.tensor(neg_tails, dtype=torch.long)
            cand_pos = cand_pos[cand_pos != u_local]
            cand_neg = cand_neg[cand_neg != u_local]
            if cand_pos.numel() == 0 or cand_neg.numel() == 0:
                continue

            anchors_local_list.append(int(u_local))
            shpos_idx.append(self._sample_k(cand_pos, u_local, k, device).unsqueeze(0))
            shneg_idx.append(self._sample_k(cand_pos, u_local, k, device).unsqueeze(0))
            pos2neg_idx.append(self._sample_k(cand_neg, u_local, k, device).unsqueeze(0))
            neg2pos_idx.append(self._sample_k(cand_neg, u_local, k, device).unsqueeze(0))

        if not anchors_local_list:
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty_k, empty_k, empty_k, empty_k

        anchors_local_t = torch.tensor(anchors_local_list, dtype=torch.long, device=device)
        shneg_t = torch.cat(shneg_idx, dim=0)
        pos2neg_t = torch.cat(pos2neg_idx, dim=0)
        neg2pos_t = torch.cat(neg2pos_idx, dim=0)
        shpos_t = torch.cat(shpos_idx, dim=0)

        return (z[anchors_local_t], z[shneg_t], z[pos2neg_t], z[neg2pos_t], z[shpos_t])
