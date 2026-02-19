import random
import time
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

class NegativeInstanceSampler:
    """  Wiki-framework sampler that returns instance-example groups per anchor.
    Positive statements: relation does NOT start with 'NOT_' and != subclass_of
    Negative statements: relation DOES start with 'NOT_'
    Ontological expansion: subclass_of children expansion for negative classes  """
    def __init__(self, k: int = 2, subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_", instance_rel: str = "2", neg_expansion_hops: int = 3,
        max_pool_size: int = 100):
        self.k = int(k)
        self.subclass_rel = subclass_rel
        self.neg_prefix = neg_prefix
        self.instance_rel = instance_rel
        self.neg_expansion_hops = int(neg_expansion_hops)
        self.max_pool_size = int(max_pool_size)
        self.max_class_pool_size = int(max_pool_size)
        self.node_type: str = "node"
        self.num_nodes: int = 0
        self._is_instance: Optional[Tensor] = None

        self.anchors: List[int] = []
        self.pos_classes: List[List[int]] = []
        self.neg_classes_direct: List[List[int]] = []
        self.neg_classes_expanded: List[List[int]] = []

        self.pool_shared_pos: List[Tensor] = []
        self.pool_shared_neg: List[Tensor] = []
        self.pool_pos_to_u_neg: List[Tensor] = []
        self.pool_neg_to_u_pos: List[Tensor] = []

        self._g2l_cpu: Optional[Tensor] = None
        self._stamp_cpu: Optional[Tensor] = None        
        self._cur_stamp: int = 0
        self._dbg_ctr = 0
        self._dbg_every = 20

    def print_pool_stats(self, prefix: str = "[NegativeInstanceSampler]") -> None:
        if not self.anchors:
            print(f"{prefix} No anchors; skipping pool stats.")
            return

        def _lens_list(lst: List[List[int]]) -> Optional[List[int]]:
            if not lst:
                return None
            max_u = max(self.anchors) if self.anchors else -1
            if max_u >= len(lst):
                return None
            return [len(lst[u]) for u in self.anchors]

        def _lens_tensor(lst: List[Tensor]) -> Optional[List[int]]:
            if not lst:
                return None
            max_u = max(self.anchors) if self.anchors else -1
            if max_u >= len(lst):
                return None
            return [int(lst[u].numel()) for u in self.anchors]

        def _summ(name: str, vals: Optional[List[int]]) -> None:
            if not vals:
                print(f"{prefix} {name}: empty")
                return
            v = sorted(vals)
            n = len(v)
            mean = float(sum(v)) / float(n)
            median = float(v[n // 2])
            p90 = float(v[int(0.9 * (n - 1))])
            vmax = float(v[-1])
            print(f"{prefix} {name}: mean={mean:.2f} median={median:.2f} p90={p90:.2f} max={vmax:.0f}")

        _summ("pos_classes_per_anchor", _lens_list(self.pos_classes))
        _summ("neg_classes_direct_per_anchor", _lens_list(self.neg_classes_direct))
        _summ("neg_classes_expanded_per_anchor", _lens_list(self.neg_classes_expanded))
        _summ("pool_shared_pos_size", _lens_tensor(self.pool_shared_pos))
        _summ("pool_shared_neg_size", _lens_tensor(self.pool_shared_neg))
        _summ("pool_pos_to_u_neg_size", _lens_tensor(self.pool_pos_to_u_neg))
        _summ("pool_neg_to_u_pos_size", _lens_tensor(self.pool_neg_to_u_pos))

    @staticmethod
    def _find_edge_key(g: HeteroData, rel: str) -> Optional[Tuple[str, str, str]]:
        for et in g.edge_types:
            if et[1] == rel: return et
        return None

    @staticmethod
    def _infer_num_nodes(g: HeteroData, node_type: str) -> int:
        nt = g[node_type]
        if hasattr(nt, "num_nodes") and nt.num_nodes is not None:
            return int(nt.num_nodes)
        max_id = -1
        for eidx in g.edge_index_dict.values():
            if eidx.numel() > 0: max_id = max(max_id, int(eidx.max().item()))
        return max_id + 1 if max_id >= 0 else 0

    def _as_inst_class(self, a: int, b: int) -> Tuple[Optional[int], Optional[int]]:
        """  Determine (inst, cls) direction based on self._is_instance mask.
        If both are instances or both are classes, returns (None, None).  """
        if self._is_instance is None: return a, b
        ia = bool(self._is_instance[a])
        ib = bool(self._is_instance[b])
        if ia and not ib: return a, b
        if ib and not ia: return b, a
        if ia and ib: return a, b
        return a, b

    @staticmethod
    def _unique_cpu(x: List[int]) -> Tensor:
        if not x: return torch.empty(0, dtype=torch.long)
        return torch.unique(torch.tensor(x, dtype=torch.long))

    def _cap_tensor(self, t: Tensor, cap: int) -> Tensor:
        if t.numel() <= cap:
            return t
        idx = torch.randperm(int(t.numel()))[:cap]
        return t[idx]

    def _collect_instances_fast(self, classes: List[int], class2inst: Dict[int, Tensor],
        exclude: Optional[int] = None) -> Tensor:
        if not classes:
            return torch.empty(0, dtype=torch.long)
        parts: List[Tensor] = []
        for c in classes:
            t = class2inst.get(c)
            if t is None or t.numel() == 0:
                continue
            parts.append(t)
        if not parts:
            return torch.empty(0, dtype=torch.long)
        if len(parts) == 1:
            out = parts[0]
        else:
            out = torch.unique(torch.cat(parts, dim=0))
        if exclude is not None and out.numel() > 0:
            out = out[out != int(exclude)]
        return out

    def _cap_pool(self, pool: Tensor) -> Tensor:
        if pool.numel() <= self.max_pool_size:
            return pool
        idx = torch.randperm(int(pool.numel()))[: self.max_pool_size]
        return pool[idx]

    def _expand_negs(self, direct: List[int], subclass_predecessors: Dict[int, Tensor]) -> List[int]:
        if not direct: return []
        expanded = set(int(x) for x in direct)
        frontier = set(expanded)
        for _ in range(max(self.neg_expansion_hops, 0)):
            nxt: set[int] = set()
            for c in list(frontier):
                children = subclass_predecessors.get(c)
                if children is None or children.numel() == 0:
                    continue
                for ch in children.tolist():
                    if ch not in expanded:
                        expanded.add(ch)
                        nxt.add(ch)
            frontier = nxt
            if not frontier: break
        return list(expanded)

    def _bump_stamp(self) -> int:
        """  Increment stamp used for membership checks.
        If it ever gets too large, reset safely.  """
        self._cur_stamp += 1
        if self._cur_stamp >= 2_000_000_000:
            assert self._stamp_cpu is not None
            self._stamp_cpu.zero_()
            self._cur_stamp = 1
        return self._cur_stamp

    @staticmethod
    def _choose_indices(n: int, k: int) -> Tensor:
        """  CPU indices into a tensor of length n.
        Without replacement if possible, else with replacement.  """
        if n <= 0: return torch.empty(0, dtype=torch.long)
        if n >= k: return torch.randperm(n)[:k]
        return torch.randint(0, n, (k,), dtype=torch.long)

    def _sample_k_locals_cpu(self, pool_globals: Tensor, *, stamp: int,
        fallback_local: int, k: int, full_graph_mode: bool) -> Tensor:
        """   Returns CPU LongTensor [k] of LOCAL indices.
        - full_graph_mode: local == global, and all nodes are "in batch"
        - neighbor mode: only sample nodes in current batch via stamp membership   """
        if pool_globals.numel() == 0:
            return torch.full((k,), fallback_local, dtype=torch.long)

        if full_graph_mode:
            n = int(pool_globals.numel())
            idx = self._choose_indices(n, k)
            chosen_g = pool_globals[idx]
            return chosen_g.long()
        # neighbor mode: filter pool to in-batch via stamp
        assert self._stamp_cpu is not None and self._g2l_cpu is not None
        s = self._stamp_cpu
        g2l = self._g2l_cpu

        mask = (s[pool_globals] == stamp)
        pool_in = pool_globals[mask]
        if pool_in.numel() == 0:
            return torch.full((k,), fallback_local, dtype=torch.long)
        n = int(pool_in.numel())
        idx = self._choose_indices(n, k)
        chosen_g = pool_in[idx]
        chosen_l = g2l[chosen_g]
        # safety fallback (should rarely trigger)
        bad = (s[chosen_g] != stamp) | (chosen_l < 0)
        if bad.any():
            chosen_l = chosen_l.clone()
            chosen_l[bad] = fallback_local
        return chosen_l.long()

    def prepare_global(self, full_g: HeteroData) -> None:
        t0 = time.time()
        if len(full_g.node_types) != 1:
            raise ValueError("NegativeInstanceSampler assumes a single node type.")
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N
        print(f"[NegativeInstanceSampler] prepare_global: start (num_nodes={N})")
        # Detect instance nodes
        instance_nodes: set[int] = set()
        if self.instance_rel is not None:
            inst_key = self._find_edge_key(full_g, self.instance_rel)
            if inst_key is not None and "edge_index" in full_g[inst_key]:
                inst_src = full_g[inst_key].edge_index[0].tolist()
                instance_nodes.update(int(x) for x in inst_src if 0 <= int(x) < N)

        if instance_nodes:
            is_instance = torch.zeros(N, dtype=torch.bool)
            is_instance[list(instance_nodes)] = True
            self._is_instance = is_instance
        else: self._is_instance = None
        print(f"[NegativeInstanceSampler] prepare_global: instance_nodes={len(instance_nodes)}")

        self.pos_classes = [[] for _ in range(N)]
        self.neg_classes_direct = [[] for _ in range(N)]
        # Build subclass predecessors (parent -> children)
        subclass_predecessors: Dict[int, Tensor] = {}
        sub_key = self._find_edge_key(full_g, self.subclass_rel)
        if sub_key is not None and "edge_index" in full_g[sub_key]:
            sub_src, sub_dst = full_g[sub_key].edge_index
            tmp: Dict[int, List[int]] = {}
            for child, parent in zip(sub_src.tolist(), sub_dst.tolist()):
                if 0 <= child < N and 0 <= parent < N:
                    tmp.setdefault(int(parent), []).append(int(child))
            subclass_predecessors = {p: self._unique_cpu(ch) for p, ch in tmp.items()}
        print(f"[NegativeInstanceSampler] prepare_global: subclass_predecessors={len(subclass_predecessors)}")

        class2pos_tmp: Dict[int, List[int]] = {}
        class2neg_tmp: Dict[int, List[int]] = {}
        anchor_sources: set[int] = set(instance_nodes)
        # Scan all edges, collect (inst -> class) and inverse maps (class -> inst)
        t_scan = time.time()
        for (_, rel, _), eidx in full_g.edge_index_dict.items():
            if eidx.numel() == 0 or rel == self.subclass_rel:
                continue
            is_neg_rel = rel.startswith(self.neg_prefix)
            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            for a, b in zip(src_list, dst_list):
                if not (0 <= a < N and 0 <= b < N): continue
                inst, cls = self._as_inst_class(int(a), int(b))
                if inst is None or cls is None: continue
                anchor_sources.add(inst)
                if is_neg_rel:
                    self.neg_classes_direct[inst].append(cls)
                    class2neg_tmp.setdefault(cls, []).append(inst)
                else:
                    self.pos_classes[inst].append(cls)
                    class2pos_tmp.setdefault(cls, []).append(inst)
        print(f"[NegativeInstanceSampler] prepare_global: edge scan took {time.time() - t_scan:.2f}s")
        self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))

        # Expand negative classes via subclass graph
        self.neg_classes_expanded = [[] for _ in range(N)]
        t_expand = time.time()
        for u in self.anchors:
            self.neg_classes_expanded[u] = self._expand_negs(self.neg_classes_direct[u], subclass_predecessors)
        print(f"[NegativeInstanceSampler] prepare_global: neg expansion took {time.time() - t_expand:.2f}s")

        class2pos = {c: self._cap_tensor(self._unique_cpu(vs), self.max_class_pool_size)
                     for c, vs in class2pos_tmp.items()}
        class2neg = {c: self._cap_tensor(self._unique_cpu(vs), self.max_class_pool_size)
                     for c, vs in class2neg_tmp.items()}

        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_pos = [empty for _ in range(N)]
        self.pool_shared_neg = [empty for _ in range(N)]
        self.pool_pos_to_u_neg = [empty for _ in range(N)]
        self.pool_neg_to_u_pos = [empty for _ in range(N)]

        # Build pools per anchor u (global ids)
        t_pools = time.time()
        for u in self.anchors:
            pos_cls = self.pos_classes[u]
            neg_cls = self.neg_classes_expanded[u]
            if pos_cls:
                self.pool_shared_pos[u] = self._cap_pool(self._collect_instances_fast(pos_cls, class2pos, exclude=u))
            if neg_cls:
                self.pool_shared_neg[u] = self._cap_pool(self._collect_instances_fast(neg_cls, class2neg, exclude=u))
                self.pool_pos_to_u_neg[u] = self._cap_pool(self._collect_instances_fast(neg_cls, class2pos, exclude=u))
            if pos_cls:
                self.pool_neg_to_u_pos[u] = self._cap_pool(self._collect_instances_fast(pos_cls, class2neg, exclude=u))
        print(f"[NegativeInstanceSampler] prepare_global: pool build took {time.time() - t_pools:.2f}s")

        if self.anchors:
            empty_shpos = sum(1 for u in self.anchors if self.pool_shared_pos[u].numel() == 0)
            empty_shneg = sum(1 for u in self.anchors if self.pool_shared_neg[u].numel() == 0)
            empty_pos2neg = sum(1 for u in self.anchors if self.pool_pos_to_u_neg[u].numel() == 0)
            empty_neg2pos = sum(1 for u in self.anchors if self.pool_neg_to_u_pos[u].numel() == 0)
            print(
                "[NegativeInstanceSampler] anchors without pool neighbors (global pools): "
                f"shared_pos={empty_shpos}/{len(self.anchors)}, "
                f"shared_neg={empty_shneg}/{len(self.anchors)}, "
                f"pos->u_neg={empty_pos2neg}/{len(self.anchors)}, "
                f"neg->u_pos={empty_neg2pos}/{len(self.anchors)}"
            )

        # Initialize stamp buffers (CPU)
        self._g2l_cpu = torch.empty(self.num_nodes, dtype=torch.long)
        self._stamp_cpu = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._cur_stamp = 0
        print(f"[NegativeInstanceSampler] prepare_global: done in {time.time() - t0:.2f}s")

    def prepare_batch(self, batch: HeteroData, pos_index: Optional[Tensor] = None) -> None:
        # Keep as no-op to preserve your current external behavior.
        return

    def get_contrastive_samples(
        self,
        z: Tensor,
        anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None,
    ):
        """
        Returns:
            (z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos)
        with shapes:
            z_anchor:     (B,D)
            each other:   (B,k,D)
        where B = number of anchors that had non-empty pools and were valid.
        """
        device = z.device
        B_rows, D = z.shape
        k = self.k
        t_start = time.time()

        # --------------------------
        # FULL GRAPH MODE (n_id is None)
        # local == global, no in-batch filtering needed
        # --------------------------
        if n_id is None:
            t_select = time.time()
            if anchor_nodes is None or anchor_nodes.numel() == 0:
                anchor_globals_cpu = torch.tensor(self.anchors, dtype=torch.long)
            else:
                anchor_globals_cpu = torch.unique(anchor_nodes.detach().long().cpu())

            # keep valid range
            anchor_globals_cpu = anchor_globals_cpu[(anchor_globals_cpu >= 0) & (anchor_globals_cpu < self.num_nodes)]
            t_select = time.time() - t_select

            anchors_cpu: List[int] = []
            shneg_cpu, pos2neg_cpu, neg2pos_cpu, shpos_cpu = [], [], [], []

            t_sample = time.time()
            for u in anchor_globals_cpu.tolist():
                # skip anchors with no pools at all
                if (
                    self.pool_shared_pos[u].numel() == 0
                    and self.pool_shared_neg[u].numel() == 0
                    and self.pool_pos_to_u_neg[u].numel() == 0
                    and self.pool_neg_to_u_pos[u].numel() == 0
                ):
                    continue
                # Skip anchors with missing negative pools to avoid self-padding
                if (self.pool_pos_to_u_neg[u].numel() == 0) or (self.pool_neg_to_u_pos[u].numel() == 0):
                    continue

                anchors_cpu.append(u)
                shneg_cpu.append(self._sample_k_locals_cpu(self.pool_shared_neg[u], stamp=0,
                                                          fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                pos2neg_cpu.append(self._sample_k_locals_cpu(self.pool_pos_to_u_neg[u], stamp=0,
                                                             fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                neg2pos_cpu.append(self._sample_k_locals_cpu(self.pool_neg_to_u_pos[u], stamp=0,
                                                             fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                shpos_cpu.append(self._sample_k_locals_cpu(self.pool_shared_pos[u], stamp=0,
                                                          fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
            t_sample = time.time() - t_sample

            if not anchors_cpu:
                empty = torch.empty(0, D, device=device)
                empty_k = torch.empty(0, k, D, device=device)
                return empty, empty_k, empty_k, empty_k, empty_k

            t_concat = time.time()
            anchors = torch.tensor(anchors_cpu, dtype=torch.long, device=device)
            shneg = torch.cat(shneg_cpu, dim=0).to(device)
            pos2neg = torch.cat(pos2neg_cpu, dim=0).to(device)
            neg2pos = torch.cat(neg2pos_cpu, dim=0).to(device)
            shpos = torch.cat(shpos_cpu, dim=0).to(device)
            t_concat = time.time() - t_concat

            self._dbg_ctr += 1
            if self._dbg_ctr % self._dbg_every == 0:
                total = time.time() - t_start
                print(
                    "[NegativeInstanceSampler] get_contrastive_samples (full): "
                    f"select={t_select:.3f}s sample={t_sample:.3f}s concat={t_concat:.3f}s total={total:.3f}s "
                    f"anchors_in={len(anchor_globals_cpu)} anchors_out={len(anchors_cpu)}"
                )

            return z[anchors], z[shneg], z[pos2neg], z[neg2pos], z[shpos]

        # --------------------------
        # NEIGHBOR MODE (n_id provided)
        # Use stamp trick to avoid O(num_nodes) clears
        # --------------------------
        assert self._stamp_cpu is not None and self._g2l_cpu is not None

        t_select = time.time()
        n_id_cpu = n_id.detach().long().cpu()
        stamp = self._bump_stamp()

        # mark membership + map global -> local
        self._stamp_cpu[n_id_cpu] = stamp
        self._g2l_cpu[n_id_cpu] = torch.arange(n_id_cpu.numel(), dtype=torch.long)

        # Determine anchor locals and corresponding anchor globals
        if anchor_nodes is None or anchor_nodes.numel() == 0:
            # fallback: try all known anchors, but only those in batch
            anchor_globals_cpu = torch.tensor(self.anchors, dtype=torch.long)
            in_batch = (self._stamp_cpu[anchor_globals_cpu] == stamp)
            anchor_globals_cpu = anchor_globals_cpu[in_batch]
            anchor_locals_cpu = self._g2l_cpu[anchor_globals_cpu]
        else:
            # In your code, anchor_nodes are LOCAL indices when n_id is not None.
            anchor_locals_cpu = torch.unique(anchor_nodes.detach().long().cpu())
            # safety filter
            anchor_locals_cpu = anchor_locals_cpu[(anchor_locals_cpu >= 0) & (anchor_locals_cpu < n_id_cpu.numel())]
            anchor_globals_cpu = n_id_cpu[anchor_locals_cpu]
        t_select = time.time() - t_select

        anchors_local_cpu: List[int] = []
        shneg_cpu, pos2neg_cpu, neg2pos_cpu, shpos_cpu = [], [], [], []

        t_sample = time.time()
        for u_g, u_l in zip(anchor_globals_cpu.tolist(), anchor_locals_cpu.tolist()):
            if u_l < 0:
                continue
            if (
                self.pool_shared_pos[u_g].numel() == 0
                and self.pool_shared_neg[u_g].numel() == 0
                and self.pool_pos_to_u_neg[u_g].numel() == 0
                and self.pool_neg_to_u_pos[u_g].numel() == 0
            ):
                continue
            if (self.pool_pos_to_u_neg[u_g].numel() == 0) or (self.pool_neg_to_u_pos[u_g].numel() == 0):
                continue

            anchors_local_cpu.append(u_l)
            shneg_cpu.append(self._sample_k_locals_cpu(self.pool_shared_neg[u_g], stamp=stamp,
                                                      fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
            pos2neg_cpu.append(self._sample_k_locals_cpu(self.pool_pos_to_u_neg[u_g], stamp=stamp,
                                                         fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
            neg2pos_cpu.append(self._sample_k_locals_cpu(self.pool_neg_to_u_pos[u_g], stamp=stamp,
                                                         fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
            shpos_cpu.append(self._sample_k_locals_cpu(self.pool_shared_pos[u_g], stamp=stamp,
                                                      fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
        t_sample = time.time() - t_sample

        if not anchors_local_cpu:
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty_k, empty_k, empty_k, empty_k

        t_concat = time.time()
        anchors_local = torch.tensor(anchors_local_cpu, dtype=torch.long, device=device)
        shneg = torch.cat(shneg_cpu, dim=0).to(device)
        pos2neg = torch.cat(pos2neg_cpu, dim=0).to(device)
        neg2pos = torch.cat(neg2pos_cpu, dim=0).to(device)
        shpos = torch.cat(shpos_cpu, dim=0).to(device)
        t_concat = time.time() - t_concat

        self._dbg_ctr += 1
        if self._dbg_ctr % self._dbg_every == 0:
            total = time.time() - t_start
            print(
                "[NegativeInstanceSampler] get_contrastive_samples (neighbor): "
                f"select={t_select:.3f}s sample={t_sample:.3f}s concat={t_concat:.3f}s total={total:.3f}s "
                f"anchors_in={len(anchor_globals_cpu)} anchors_out={len(anchors_local_cpu)} n_id={len(n_id_cpu)}"
            )
        return z[anchors_local], z[shneg], z[pos2neg], z[neg2pos], z[shpos]


# import random
# from typing import Dict, List, Optional, Tuple
# import torch
# from torch import Tensor
# from torch_geometric.data import HeteroData


# class NegativeInstanceSampler:
#     """
#     Wiki-framework sampler that returns instance-example groups per anchor.
#     Positive statements: relation does NOT start with 'NOT_' and != subclass_of
#     Negative statements: relation DOES start with 'NOT_'
#     Ontological expansion: subclass_of children expansion for negative classes
#     Returns:
#         z_anchor: (B,D)
#         z_shared_neg: (B,k,D)
#         z_pos_to_u_neg: (B,k,D)
#         z_neg_to_u_pos: (B,k,D)
#         z_shared_pos: (B,k,D)
#     """

#     def __init__(
#         self,
#         k: int = 2,
#         subclass_rel: str = "subclass_of",
#         neg_prefix: str = "NOT_",
#         instance_rel: str = "2",
#         neg_expansion_hops: int = 1,
#     ):
#         self.k = int(k)
#         self.subclass_rel = subclass_rel
#         self.neg_prefix = neg_prefix
#         self.instance_rel = instance_rel
#         self.neg_expansion_hops = int(neg_expansion_hops)

#         self.node_type: str = "node"
#         self.num_nodes: int = 0
#         self._is_instance: Optional[Tensor] = None

#         self.anchors: List[int] = []
#         self.pos_classes: List[List[int]] = []
#         self.neg_classes_direct: List[List[int]] = []
#         self.neg_classes_expanded: List[List[int]] = []

#         self.pool_shared_pos: List[Tensor] = []
#         self.pool_shared_neg: List[Tensor] = []
#         self.pool_pos_to_u_neg: List[Tensor] = []
#         self.pool_neg_to_u_pos: List[Tensor] = []

#         self._global2local_cpu: Optional[Tensor] = None
#         self._in_batch_mask_cpu: Optional[Tensor] = None

#     @staticmethod
#     def _find_edge_key(g: HeteroData, rel: str) -> Optional[Tuple[str, str, str]]:
#         for et in g.edge_types:
#             if et[1] == rel:
#                 return et
#         return None

#     @staticmethod
#     def _infer_num_nodes(g: HeteroData, node_type: str) -> int:
#         nt = g[node_type]
#         if hasattr(nt, "num_nodes") and nt.num_nodes is not None:
#             return int(nt.num_nodes)
#         max_id = -1
#         for eidx in g.edge_index_dict.values():
#             if eidx.numel() > 0:
#                 max_id = max(max_id, int(eidx.max().item()))
#         return max_id + 1 if max_id >= 0 else 0

#     def _as_inst_class(self, a: int, b: int) -> Tuple[Optional[int], Optional[int]]:
#         if self._is_instance is None:
#             return a, b
#         ia = bool(self._is_instance[a])
#         ib = bool(self._is_instance[b])
#         if ia and not ib:
#             return a, b
#         if ib and not ia:
#             return b, a
#         return None, None

#     @staticmethod
#     def _unique_cpu(x: List[int]) -> Tensor:
#         if not x:
#             return torch.empty(0, dtype=torch.long)
#         return torch.unique(torch.tensor(x, dtype=torch.long))

#     @staticmethod
#     def _collect_instances(
#         classes: List[int],
#         class2inst: Dict[int, Tensor],
#         exclude: Optional[int] = None,
#     ) -> Tensor:
#         if not classes:
#             return torch.empty(0, dtype=torch.long)
#         parts = [class2inst[c] for c in classes if c in class2inst and class2inst[c].numel() > 0]
#         if not parts:
#             return torch.empty(0, dtype=torch.long)
#         out = torch.unique(torch.cat(parts, dim=0))
#         if exclude is not None and out.numel() > 0:
#             out = out[out != int(exclude)]
#         return out

#     def _expand_negs(self, direct: List[int], subclass_predecessors: Dict[int, Tensor]) -> List[int]:
#         if not direct:
#             return []
#         expanded = set(int(x) for x in direct)
#         frontier = set(expanded)
#         for _ in range(max(self.neg_expansion_hops, 0)):
#             nxt: set[int] = set()
#             for c in list(frontier):
#                 children = subclass_predecessors.get(c)
#                 if children is None or children.numel() == 0:
#                     continue
#                 for ch in children.tolist():
#                     if ch not in expanded:
#                         expanded.add(ch)
#                         nxt.add(ch)
#             frontier = nxt
#             if not frontier:
#                 break
#         return list(expanded)

#     def prepare_global(self, full_g: HeteroData) -> None:
#         if len(full_g.node_types) != 1:
#             raise ValueError("NegativeStatementSampler assumes a single node type.")
#         self.node_type = full_g.node_types[0]
#         N = self._infer_num_nodes(full_g, self.node_type)
#         self.num_nodes = N

#         instance_nodes: set[int] = set()
#         if self.instance_rel is not None:
#             inst_key = self._find_edge_key(full_g, self.instance_rel)
#             if inst_key is not None and "edge_index" in full_g[inst_key]:
#                 inst_src = full_g[inst_key].edge_index[0].tolist()
#                 instance_nodes.update(int(x) for x in inst_src if 0 <= int(x) < N)

#         if instance_nodes:
#             is_instance = torch.zeros(N, dtype=torch.bool)
#             is_instance[list(instance_nodes)] = True
#             self._is_instance = is_instance
#         else:
#             self._is_instance = None

#         self.pos_classes = [[] for _ in range(N)]
#         self.neg_classes_direct = [[] for _ in range(N)]

#         subclass_predecessors: Dict[int, Tensor] = {}
#         sub_key = self._find_edge_key(full_g, self.subclass_rel)
#         if sub_key is not None and "edge_index" in full_g[sub_key]:
#             sub_src, sub_dst = full_g[sub_key].edge_index
#             tmp: Dict[int, List[int]] = {}
#             for child, parent in zip(sub_src.tolist(), sub_dst.tolist()):
#                 if 0 <= child < N and 0 <= parent < N:
#                     tmp.setdefault(int(parent), []).append(int(child))
#             subclass_predecessors = {p: self._unique_cpu(ch) for p, ch in tmp.items()}

#         class2pos_tmp: Dict[int, List[int]] = {}
#         class2neg_tmp: Dict[int, List[int]] = {}
#         anchor_sources: set[int] = set(instance_nodes)

#         for (_, rel, _), eidx in full_g.edge_index_dict.items():
#             if eidx.numel() == 0 or rel == self.subclass_rel:
#                 continue

#             is_neg_rel = rel.startswith(self.neg_prefix)
#             src_list = eidx[0].tolist()
#             dst_list = eidx[1].tolist()

#             for a, b in zip(src_list, dst_list):
#                 if not (0 <= a < N and 0 <= b < N):
#                     continue
#                 inst, cls = self._as_inst_class(int(a), int(b))
#                 if inst is None or cls is None:
#                     continue
#                 anchor_sources.add(inst)
#                 if is_neg_rel:
#                     self.neg_classes_direct[inst].append(cls)
#                     class2neg_tmp.setdefault(cls, []).append(inst)
#                 else:
#                     self.pos_classes[inst].append(cls)
#                     class2pos_tmp.setdefault(cls, []).append(inst)

#         self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))

#         self.neg_classes_expanded = [[] for _ in range(N)]
#         for u in self.anchors:
#             self.neg_classes_expanded[u] = self._expand_negs(self.neg_classes_direct[u], subclass_predecessors)

#         class2pos = {c: self._unique_cpu(vs) for c, vs in class2pos_tmp.items()}
#         class2neg = {c: self._unique_cpu(vs) for c, vs in class2neg_tmp.items()}

#         empty = torch.empty(0, dtype=torch.long)
#         self.pool_shared_pos = [empty for _ in range(N)]
#         self.pool_shared_neg = [empty for _ in range(N)]
#         self.pool_pos_to_u_neg = [empty for _ in range(N)]
#         self.pool_neg_to_u_pos = [empty for _ in range(N)]

#         for u in self.anchors:
#             pos_cls = self.pos_classes[u]
#             neg_cls = self.neg_classes_expanded[u]

#             self.pool_shared_pos[u] = self._collect_instances(pos_cls, class2pos, exclude=u)
#             self.pool_shared_neg[u] = self._collect_instances(neg_cls, class2neg, exclude=u)
#             self.pool_pos_to_u_neg[u] = self._collect_instances(neg_cls, class2pos, exclude=u)
#             self.pool_neg_to_u_pos[u] = self._collect_instances(pos_cls, class2neg, exclude=u)

#         self._global2local_cpu = None
#         self._in_batch_mask_cpu = None

#     def prepare_batch(self, batch: HeteroData, pos_index: Optional[Tensor] = None) -> None:
#         return

#     def _sample_k_locals(
#         self,
#         pool_globals: Tensor,
#         *,
#         in_batch: Tensor,
#         global2local: Tensor,
#         fallback_local: int,
#         k: int,
#         device: torch.device,
#     ) -> Tensor:
#         if pool_globals.numel() == 0:
#             return torch.full((k,), fallback_local, dtype=torch.long, device=device)

#         pool_in_batch = pool_globals[in_batch[pool_globals]]
#         if pool_in_batch.numel() == 0:
#             return torch.full((k,), fallback_local, dtype=torch.long, device=device)

#         if pool_in_batch.numel() >= k:
#             chosen = pool_in_batch[torch.randperm(pool_in_batch.numel())[:k]]
#         else:
#             chosen = pool_in_batch[torch.randint(0, pool_in_batch.numel(), (k,), dtype=torch.long)]

#         chosen_locals = global2local[chosen]
#         bad = chosen_locals < 0
#         if bad.any():
#             chosen_locals[bad] = fallback_local
#         return chosen_locals.to(device)

#     def get_contrastive_samples(
#         self,
#         z: Tensor,
#         anchor_nodes: Optional[Tensor] = None,
#         n_id: Optional[Tensor] = None,
#     ):
#         device = z.device
#         B_rows, D = z.shape
#         k = self.k

#         global_for_row = torch.arange(B_rows) if n_id is None else n_id.detach().long().cpu()

#         if self._global2local_cpu is None or self._global2local_cpu.numel() < self.num_nodes:
#             self._global2local_cpu = torch.full((self.num_nodes,), -1, dtype=torch.long)
#         global2local = self._global2local_cpu
#         global2local.fill_(-1)
#         global2local[global_for_row] = torch.arange(global_for_row.numel(), dtype=torch.long)

#         if self._in_batch_mask_cpu is None or self._in_batch_mask_cpu.numel() < self.num_nodes:
#             self._in_batch_mask_cpu = torch.zeros(self.num_nodes, dtype=torch.bool)
#         in_batch = self._in_batch_mask_cpu
#         in_batch.zero_()
#         in_batch[global_for_row] = True

#         if anchor_nodes is not None and anchor_nodes.numel() > 0:
#             anchor_globals = global_for_row[anchor_nodes.detach().long().cpu().unique()]
#         else:
#             anchor_globals = torch.tensor(self.anchors, dtype=torch.long)

#         anchor_globals = anchor_globals[in_batch[anchor_globals]]

#         anchors_local_list, shneg_idx, pos2neg_idx, neg2pos_idx, shpos_idx = [], [], [], [], []

#         for u in anchor_globals.tolist():
#             u_local = int(global2local[u].item())
#             if u_local < 0:
#                 continue

#             # Skip anchors with no examples at all
#             if (self.pool_shared_pos[u].numel() == 0 and self.pool_shared_neg[u].numel() == 0 and
#                 self.pool_pos_to_u_neg[u].numel() == 0 and self.pool_neg_to_u_pos[u].numel() == 0):
#                 continue

#             anchors_local_list.append(u_local)
#             shneg_idx.append(self._sample_k_locals(self.pool_shared_neg[u], in_batch=in_batch, global2local=global2local,
#                                                   fallback_local=u_local, k=k, device=device).unsqueeze(0))
#             pos2neg_idx.append(self._sample_k_locals(self.pool_pos_to_u_neg[u], in_batch=in_batch, global2local=global2local,
#                                                     fallback_local=u_local, k=k, device=device).unsqueeze(0))
#             neg2pos_idx.append(self._sample_k_locals(self.pool_neg_to_u_pos[u], in_batch=in_batch, global2local=global2local,
#                                                     fallback_local=u_local, k=k, device=device).unsqueeze(0))
#             shpos_idx.append(self._sample_k_locals(self.pool_shared_pos[u], in_batch=in_batch, global2local=global2local,
#                                                   fallback_local=u_local, k=k, device=device).unsqueeze(0))

#         if not anchors_local_list:
#             empty = torch.empty(0, D, device=device)
#             empty_k = torch.empty(0, k, D, device=device)
#             return empty, empty_k, empty_k, empty_k, empty_k

#         anchors_local_t = torch.tensor(anchors_local_list, dtype=torch.long, device=device)
#         shneg_t = torch.cat(shneg_idx, dim=0)
#         pos2neg_t = torch.cat(pos2neg_idx, dim=0)
#         neg2pos_t = torch.cat(neg2pos_idx, dim=0)
#         shpos_t = torch.cat(shpos_idx, dim=0)

#         return (
#             z[anchors_local_t],
#             z[shneg_t],
#             z[pos2neg_t],
#             z[neg2pos_t],
#             z[shpos_t],
#         )
