from collections import defaultdict
import time
from typing import Dict, List, Optional
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .negativeinstance_sampler import NegativeInstanceSampler


class NegativeEntitySampler2(NegativeInstanceSampler):
    """Negative entity sampler (all-relations fallback only).

    Pool construction uses ontological expansion via subclass_of but does NOT
    enforce per-relation pools; it builds only the global '__any__' pools.
    """

    def __init__(self, k: int = 2, subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_", instance_rel: str = "2", neg_expansion_hops: int = 1,
        max_pool_size: Optional[int] = None):
        super().__init__(
            k=k,
            subclass_rel=subclass_rel,
            neg_prefix=neg_prefix,
            instance_rel=instance_rel,
            neg_expansion_hops=neg_expansion_hops,
        )
        self.max_pool_size = max_pool_size if max_pool_size is None else int(max_pool_size)

    def _cap_pool(self, t: Tensor) -> Tensor:
        if self.max_pool_size is None or t.numel() <= self.max_pool_size:
            return t
        idx = torch.randperm(int(t.numel()))[:int(self.max_pool_size)]
        return t[idx]

    def _expand_up(self, direct: List[int], subclass_successors: Dict[int, Tensor]) -> List[int]:
        if not direct:
            return []
        expanded = set(int(x) for x in direct)
        frontier = set(expanded)
        for _ in range(max(self.neg_expansion_hops, 0)):
            nxt: set[int] = set()
            for c in list(frontier):
                parents = subclass_successors.get(c)
                if parents is None or parents.numel() == 0:
                    continue
                for p in parents.tolist():
                    if p not in expanded:
                        expanded.add(p)
                        nxt.add(p)
            frontier = nxt
            if not frontier:
                break
        return list(expanded)

    # Signal to trainer that this sampler returns (anchor, pos_pool, neg_pool).
    contrastive_entity_pools = True

    def get_contrastive_samples(
        self,
        z: Tensor,
        anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None,
        anchor_nodes_are_unique: bool = False,
    ):
        if getattr(self, "_lazy_pool_building", False):
            if anchor_nodes is not None and anchor_nodes.numel() > 0:
                anchor_globals = anchor_nodes.detach().long().cpu()
                if not anchor_nodes_are_unique:
                    anchor_globals = torch.unique(anchor_globals)
                anchor_globals = anchor_globals[(anchor_globals >= 0) & (anchor_globals < self.num_nodes)]
                self._ensure_lazy_pools(anchor_globals.tolist())
            else:
                # Fallback: build pools for all anchors when no anchor list is provided.
                self._ensure_lazy_pools(self.anchors)
        # Use base sampler to get 4 pools, then merge into 2.
        z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos = super().get_contrastive_samples(
            z, anchor_nodes=anchor_nodes, n_id=n_id, anchor_nodes_are_unique=anchor_nodes_are_unique
        )
        if z_anchor.numel() == 0:
            empty = z_anchor.new_empty((0,) + z_shared_pos.shape[1:])
            return z_anchor, empty, empty
        z_pos = torch.cat([z_shared_neg, z_shared_pos], dim=1)
        z_neg = torch.cat([z_pos_to_u_neg, z_neg_to_u_pos], dim=1)
        return z_anchor, z_pos, z_neg

    def _ensure_lazy_pools(self, anchors: List[int]) -> None:
        for u in anchors:
            if u in self._pools_built:
                continue
            self._build_pools_for_anchor(u)
            self._ensure_anchor_caches([u])
            self._pools_built.add(u)

    def _build_pools_for_anchor(self, u: int) -> None:
        if not (0 <= u < self.num_nodes):
            return
        empty = self._empty_pool
        preds_u = self._preds_by_anchor[u]

        def _ensure_pool(pool_list: List[Optional[Dict[str, Tensor]]], idx: int) -> Dict[str, Tensor]:
            pools = pool_list[idx]
            if pools is None:
                pools = defaultdict(lambda: empty)
                pool_list[idx] = pools
            return pools

        # Instance-of co-membership positives (auto-fallback for empty preds only).
        if not preds_u and self._build_co_membership:
            if not self._inst2classes:
                # Build class2instances and inst2classes lazily.
                self._class2instances = [torch.empty(0, dtype=torch.long) for _ in range(self.num_nodes)]
                self._inst2classes = [torch.empty(0, dtype=torch.long) for _ in range(self.num_nodes)]
                if self._inst_dst.numel() > 0:
                    order = torch.argsort(self._inst_dst)
                    sorted_cls = self._inst_dst[order]
                    sorted_inst = self._inst_src[order]
                    cls_unique, counts = torch.unique_consecutive(sorted_cls, return_counts=True)
                    start = 0
                    for cls_id, cnt in zip(cls_unique.tolist(), counts.tolist()):
                        cls_instances = sorted_inst[start:start + cnt]
                        self._class2instances[int(cls_id)] = cls_instances
                        start += cnt
                if self._inst_src.numel() > 0:
                    order = torch.argsort(self._inst_src)
                    sorted_inst = self._inst_src[order]
                    sorted_cls = self._inst_dst[order]
                    inst_unique, counts = torch.unique_consecutive(sorted_inst, return_counts=True)
                    start = 0
                    for inst_id, cnt in zip(inst_unique.tolist(), counts.tolist()):
                        inst_classes = sorted_cls[start:start + cnt]
                        self._inst2classes[int(inst_id)] = inst_classes
                        start += cnt
            if self._inst2classes and self._inst2classes[u].numel() > 0:
                co_members = []
                for cls in self._inst2classes[u].tolist():
                    if 0 <= cls < self.num_nodes and self._class2instances and self._class2instances[cls].numel() > 0:
                        co_members.extend(self._class2instances[cls].tolist())
                if co_members:
                    co_members = self._unique_cpu(co_members)
                    if co_members.numel():
                        pools = _ensure_pool(self.pool_shared_pos_by_pred, u)
                        existing = pools.get(str(self.primary_pred), empty)
                        if existing.numel() > 0:
                            merged = torch.unique(torch.cat([existing, co_members]))
                            pools[str(self.primary_pred)] = merged
                        else:
                            pools[str(self.primary_pred)] = co_members

        # All-relations pooling only (always use '__any__' pools).
        pos_cls_all = []
        neg_cls_all = []
        for cls_list in self._pos_by_pred[u].values():
            pos_cls_all.extend(cls_list)
        for cls_list in self._neg_by_pred_direct[u].values():
            neg_cls_all.extend(cls_list)

        if pos_cls_all:
            pos_sup = self._expand_up(pos_cls_all, self._subclass_successors)
            pos_cls_all = list(set(pos_cls_all) | set(pos_sup))
        if neg_cls_all:
            neg_sub = self._expand_negs(neg_cls_all, self._subclass_predecessors)
            neg_cls_all = list(set(neg_cls_all) | set(neg_sub))

        if pos_cls_all:
            pools = _ensure_pool(self.pool_shared_pos_by_pred, u)
            pools["__any__"] = self._cap_pool(self._collect_instances(
                pos_cls_all, self._class2pos_all, exclude=u
            ))
            pools = _ensure_pool(self.pool_shared_neg_by_pred, u)
            pools["__any__"] = self._cap_pool(self._collect_instances(
                neg_cls_all, self._class2neg_all, exclude=u
            )) if neg_cls_all else pools.get("__any__", empty)
        if pos_cls_all:
            pools = _ensure_pool(self.pool_neg_to_u_pos_by_pred, u)
            pools["__any__"] = self._cap_pool(self._collect_instances(
                pos_cls_all, self._class2neg_all, exclude=u
            ))
        if neg_cls_all:
            pools = _ensure_pool(self.pool_pos_to_u_neg_by_pred, u)
            pools["__any__"] = self._cap_pool(self._collect_instances(
                neg_cls_all, self._class2pos_all, exclude=u
            ))
        self.preds_seen[u].add("__any__")

    def prepare_global(self, full_g: HeteroData) -> None:
        t0 = time.perf_counter()
        if len(full_g.node_types) != 1:
            raise ValueError("NegativeEntitySampler2 assumes a single node type.")
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N

        instance_nodes: set[int] = set()
        class_nodes: set[int] = set()
        inst2classes: List[Tensor] = []
        class2instances: List[Tensor] = []
        build_co_membership = False
        inst_src = torch.empty(0, dtype=torch.long)
        inst_dst = torch.empty(0, dtype=torch.long)
        if self.instance_rel is not None:
            inst_key = self._find_edge_key(full_g, self.instance_rel)
            if inst_key is not None and "edge_index" in full_g[inst_key]:
                inst_src = full_g[inst_key].edge_index[0].long().cpu()
                inst_dst = full_g[inst_key].edge_index[1].long().cpu()
                valid_mask = (inst_src >= 0) & (inst_src < N) & (inst_dst >= 0) & (inst_dst < N)
                inst_src = inst_src[valid_mask]
                inst_dst = inst_dst[valid_mask]
                instance_nodes.update(inst_src.tolist())
                class_nodes.update(inst_dst.tolist())

                # Defer co-membership index building until we know it's needed.
                build_co_membership = True
        t_inst = time.perf_counter()

        sub_key = self._find_edge_key(full_g, self.subclass_rel)
        if sub_key is not None and "edge_index" in full_g[sub_key]:
            sub_src, sub_dst = full_g[sub_key].edge_index
            class_nodes.update(int(x) for x in sub_src.tolist() if 0 <= int(x) < N)
            class_nodes.update(int(x) for x in sub_dst.tolist() if 0 <= int(x) < N)

        # Mark instance nodes; everything else is treated as class-capable.
        if instance_nodes:
            is_instance = torch.zeros(N, dtype=torch.bool)
            is_instance[list(instance_nodes)] = True
            self._is_instance = is_instance
        else:
            self._is_instance = None

        # Build subclass predecessor/successor maps for ontology expansion.
        subclass_predecessors: Dict[int, Tensor] = {}
        subclass_successors: Dict[int, Tensor] = {}
        sub_key = self._find_edge_key(full_g, self.subclass_rel)
        if sub_key is not None and "edge_index" in full_g[sub_key]:
            sub_src, sub_dst = full_g[sub_key].edge_index  # child -> parent
            tmp_child: Dict[int, List[int]] = {}
            tmp_parent: Dict[int, List[int]] = {}
            for child, parent in zip(sub_src.tolist(), sub_dst.tolist()):
                if 0 <= child < N and 0 <= parent < N:
                    tmp_child.setdefault(int(parent), []).append(int(child))
                    tmp_parent.setdefault(int(child), []).append(int(parent))
            subclass_predecessors = {p: self._unique_cpu(ch) for p, ch in tmp_child.items()}
            subclass_successors = {c: self._unique_cpu(ps) for c, ps in tmp_parent.items()}
        t_sub = time.perf_counter()

        pos_by_pred = [defaultdict(list) for _ in range(N)]
        neg_by_pred_direct = [defaultdict(list) for _ in range(N)]
        class2pos_by_pred_tmp = defaultdict(lambda: defaultdict(list))
        class2neg_by_pred_tmp = defaultdict(lambda: defaultdict(list))
        anchor_sources: set[int] = set(instance_nodes)
        inst_set = set(instance_nodes)
        anchors_with_any_src: set[int] = set()
        anchors_with_any_dst: set[int] = set()
        anchors_with_any_incident: set[int] = set()

        primary = str(self.primary_pred)
        class_level_pos_cnt = 0
        class_level_neg_cnt = 0

        skip_rev = {str(self.subclass_rel), str(self.primary_pred), "subclass_of", "instance_of"}
        for (_, rel, _), eidx in full_g.edge_index_dict.items():
            if eidx.numel() == 0 or rel == self.subclass_rel:
                continue
            rel = str(rel)
            base = self._base_pred(rel)
            is_neg_rel = rel.startswith(self.neg_prefix)
            src = eidx[0].long().cpu()
            dst = eidx[1].long().cpu()
            valid_mask = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
            src = src[valid_mask]
            dst = dst[valid_mask]
            do_reverse = base not in skip_rev
            if src.numel() == 0:
                continue

            if inst_set:
                anchors_with_any_src.update(int(x) for x in src.tolist() if int(x) in inst_set)
                anchors_with_any_dst.update(int(x) for x in dst.tolist() if int(x) in inst_set)
                anchors_with_any_incident.update(int(x) for x in src.tolist() if int(x) in inst_set)
                anchors_with_any_incident.update(int(x) for x in dst.tolist() if int(x) in inst_set)

            if self._is_instance is not None:
                src_is_inst = self._is_instance[src]
                dst_is_inst = self._is_instance[dst]
            else:
                src_is_inst = torch.zeros_like(src, dtype=torch.bool)
                dst_is_inst = torch.zeros_like(dst, dtype=torch.bool)

            # Forward: src instance -> dst class
            f_mask = src_is_inst & (~dst_is_inst)
            if f_mask.any():
                inst_ids = src[f_mask].tolist()
                cls_ids = dst[f_mask].tolist()
                for inst, cls in zip(inst_ids, cls_ids):
                    anchor_sources.add(inst)
                    if is_neg_rel:
                        neg_by_pred_direct[inst][base].append(cls)
                        class2neg_by_pred_tmp[base][cls].append(inst)
                    else:
                        pos_by_pred[inst][base].append(cls)
                        class2pos_by_pred_tmp[base][cls].append(inst)

            # Reverse: dst instance -> src class (non-structural only)
            if do_reverse:
                r_mask = dst_is_inst & (~src_is_inst)
                if r_mask.any():
                    inst_ids = dst[r_mask].tolist()
                    cls_ids = src[r_mask].tolist()
                    for inst, cls in zip(inst_ids, cls_ids):
                        anchor_sources.add(inst)
                        if is_neg_rel:
                            neg_by_pred_direct[inst][base].append(cls)
                            class2neg_by_pred_tmp[base][cls].append(inst)
                        else:
                            pos_by_pred[inst][base].append(cls)
                            class2pos_by_pred_tmp[base][cls].append(inst)

            # Count class-class assertions (diagnostic only).
            if (self._is_instance is not None) and (~src_is_inst & ~dst_is_inst).any():
                if is_neg_rel:
                    class_level_neg_cnt += int((~src_is_inst & ~dst_is_inst).sum().item())
                else:
                    class_level_pos_cnt += int((~src_is_inst & ~dst_is_inst).sum().item())
        t_scan = time.perf_counter()

        self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))
        # Pre-unique predicate class lists and precompute predicates per anchor.
        preds_by_anchor: List[List[str]] = [[] for _ in range(N)]
        for u in self.anchors:
            if pos_by_pred[u]:
                for pred, cls_list in list(pos_by_pred[u].items()):
                    if cls_list:
                        pos_by_pred[u][pred] = list(set(cls_list))
            if neg_by_pred_direct[u]:
                for pred, cls_list in list(neg_by_pred_direct[u].items()):
                    if cls_list:
                        neg_by_pred_direct[u][pred] = list(set(cls_list))
            if pos_by_pred[u] or neg_by_pred_direct[u]:
                preds_by_anchor[u] = list(set(pos_by_pred[u].keys()) | set(neg_by_pred_direct[u].keys()))
        # (debug prints removed)

        class2pos_by_pred = {pred: {c: self._unique_cpu(vs) for c, vs in cls_map.items()}
            for pred, cls_map in class2pos_by_pred_tmp.items()}
        class2neg_by_pred = {pred: {c: self._unique_cpu(vs) for c, vs in cls_map.items()}
            for pred, cls_map in class2neg_by_pred_tmp.items()}
        # Global (non-relation) pools for fallback.
        class2pos_all_tmp = defaultdict(list)
        class2neg_all_tmp = defaultdict(list)
        for pred, cls_map in class2pos_by_pred_tmp.items():
            for c, vs in cls_map.items():
                class2pos_all_tmp[int(c)].extend(vs)
        for pred, cls_map in class2neg_by_pred_tmp.items():
            for c, vs in cls_map.items():
                class2neg_all_tmp[int(c)].extend(vs)
        class2pos_all = {c: self._unique_cpu(vs) for c, vs in class2pos_all_tmp.items()}
        class2neg_all = {c: self._unique_cpu(vs) for c, vs in class2neg_all_tmp.items()}
        t_maps = time.perf_counter()

        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_pos_by_pred = [None for _ in range(N)]
        self.pool_shared_neg_by_pred = [None for _ in range(N)]
        self.pool_pos_to_u_neg_by_pred = [None for _ in range(N)]
        self.pool_neg_to_u_pos_by_pred = [None for _ in range(N)]
        self.preds_seen = [set() for _ in range(N)]
        self._lazy_pool_building = True
        self._pools_built: set[int] = set()
        self._pos_by_pred = pos_by_pred
        self._neg_by_pred_direct = neg_by_pred_direct
        self._class2pos_by_pred = class2pos_by_pred
        self._class2neg_by_pred = class2neg_by_pred
        self._class2pos_all = class2pos_all
        self._class2neg_all = class2neg_all
        self._preds_by_anchor = preds_by_anchor
        self._subclass_predecessors = subclass_predecessors
        self._subclass_successors = subclass_successors
        self._instance_nodes = instance_nodes
        self._inst_src = inst_src
        self._inst_dst = inst_dst
        self._build_co_membership = build_co_membership
        self._inst2classes = inst2classes
        self._class2instances = class2instances
        self._empty_pool = empty

        # Lazy pool construction: initialize anchor caches without materializing pools.
        self._init_anchor_caches()
        t_pools = time.perf_counter()
        # Build caches for sampling.
        self._g2l_cpu = torch.empty(self.num_nodes, dtype=torch.long)
        self._stamp_cpu = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._cur_stamp = 0
        # Anchor caches are built lazily per anchor.
        t_cache = time.perf_counter()
        # Diagnostics are skipped in lazy mode to avoid full pool materialization.
        print(
            "[NegativeEntitySampler2][timing] "
            f"instance_maps={t_inst - t0:.2f}s subclass_maps={t_sub - t_inst:.2f}s "
            f"scan_edges={t_scan - t_sub:.2f}s pred_maps={t_maps - t_scan:.2f}s "
            f"build_pools={t_pools - t_maps:.2f}s cache={t_cache - t_pools:.2f}s "
            f"total={t_cache - t0:.2f}s", flush=True)
