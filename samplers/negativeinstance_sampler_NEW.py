import random
import time
import os
import hashlib
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from collections import defaultdict

class NegativeInstanceSampler_NEW:
    """  Wiki-framework sampler that returns instance-example groups per anchor.
    Positive statements: relation does NOT start with 'NOT_' and != subclass_of
    Negative statements: relation DOES start with 'NOT_'
    Ontological expansion: subclass_of children expansion for negative classes  """
    def __init__(self, k: int = 2, subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_", instance_rel: str = "2", neg_expansion_hops: int = 3,
        max_pool_size: int = 100, cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None, max_contrastive_anchors: int = 0):
        self.k = int(k)
        self.subclass_rel = subclass_rel
        self.neg_prefix = neg_prefix
        self.instance_rel = instance_rel
        self.neg_expansion_hops = int(neg_expansion_hops)
        self.max_pool_size = int(max_pool_size)
        self.max_class_pool_size = int(max_pool_size)
        self.cache_dir = cache_dir
        self.cache_key = cache_key
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

        self.pool_shared_pos_by_pred = None
        self.pool_shared_neg_by_pred = None
        self.pool_pos_to_u_neg_by_pred = None
        self.pool_neg_to_u_pos_by_pred = None
        self.pool_shared_neg_any = None
        self.preds_seen = None
        self.pred_order_cache = None
        self.anchors_valid = None
        self._pre_shneg = None
        self._pre_pos2neg = None
        self._pre_neg2pos = None
        self._pre_shpos = None
        self._pre_ready = False
        self._pre_anchor_valid_mask = None
        self.primary_pred = instance_rel  # "instance_of"
        self.cls_edge_suffix = "__cls"
        self.prefer_inst_only = None
        self.inst_pred_key = None
        self._dbg_ctr = 0
        self._dbg_every = 20
        self.max_contrastive_anchors = int(max(0, int(max_contrastive_anchors)))

    def print_pool_stats(self, prefix: str = "[NegativeInstanceSampler_NEW]") -> None:
        if not self.anchors:
            print(f"{prefix} No anchors; skipping pool stats.")
            return

        def _safe_lens_list(lst: List[List[int]]) -> Optional[List[int]]:
            if not lst:
                return None
            max_u = max(self.anchors) if self.anchors else -1
            if max_u >= len(lst):
                return None
            return [len(lst[u]) for u in self.anchors]

        def _safe_lens_tensor(lst: List[Tensor]) -> Optional[List[int]]:
            if not lst:
                return None
            max_u = max(self.anchors) if self.anchors else -1
            if max_u >= len(lst):
                return None
            return [int(lst[u].numel()) for u in self.anchors]

        def _pool_size_from_pred_dict(dd_list) -> Optional[List[int]]:
            if dd_list is None or len(dd_list) == 0:
                return None
            max_u = max(self.anchors) if self.anchors else -1
            if max_u >= len(dd_list):
                return None
            sizes: List[int] = []
            for u in self.anchors:
                dd = dd_list[u]
                parts = [t for t in dd.values() if t is not None and t.numel() > 0]
                if not parts:
                    sizes.append(0)
                    continue
                sizes.append(int(torch.unique(torch.cat(parts, dim=0)).numel()))
            return sizes

        def _empty_count(dd_list) -> Optional[int]:
            if dd_list is None or len(dd_list) == 0:
                return None
            max_u = max(self.anchors) if self.anchors else -1
            if max_u >= len(dd_list):
                return None
            def _has_any(dd) -> bool:
                for t in dd.values():
                    if t is not None and t.numel() > 0:
                        return True
                return False
            return sum(1 for u in self.anchors if not _has_any(dd_list[u]))

        def _summ(name: str, vals: List[int]) -> None:
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

        vals = _safe_lens_list(self.pos_classes)
        if vals is not None:
            _summ("pos_classes_per_anchor", vals)
        vals = _safe_lens_list(self.neg_classes_direct)
        if vals is not None:
            _summ("neg_classes_direct_per_anchor", vals)
        vals = _safe_lens_list(self.neg_classes_expanded)
        if vals is not None:
            _summ("neg_classes_expanded_per_anchor", vals)

        # Prefer typed pools if present
        vals = _pool_size_from_pred_dict(getattr(self, "pool_shared_pos_by_pred", None))
        if vals is None:
            vals = _safe_lens_tensor(getattr(self, "pool_shared_pos", []))
        if vals is not None:
            _summ("pool_shared_pos_size", vals)

        vals = _pool_size_from_pred_dict(getattr(self, "pool_shared_neg_by_pred", None))
        if vals is None:
            vals = _safe_lens_tensor(getattr(self, "pool_shared_neg", []))
        if vals is not None:
            _summ("pool_shared_neg_size", vals)

        vals = _pool_size_from_pred_dict(getattr(self, "pool_pos_to_u_neg_by_pred", None))
        if vals is None:
            vals = _safe_lens_tensor(getattr(self, "pool_pos_to_u_neg", []))
        if vals is not None:
            _summ("pool_pos_to_u_neg_size", vals)

        vals = _pool_size_from_pred_dict(getattr(self, "pool_neg_to_u_pos_by_pred", None))
        if vals is None:
            vals = _safe_lens_tensor(getattr(self, "pool_neg_to_u_pos", []))
        if vals is not None:
            _summ("pool_neg_to_u_pos_size", vals)
        empty_neg2pos = _empty_count(getattr(self, "pool_neg_to_u_pos_by_pred", None))
        if empty_neg2pos is not None:
            print(f"{prefix} pool_neg_to_u_pos_empty_anchors: {empty_neg2pos}")

    def _cache_path(self) -> Optional[str]:
        if not self.cache_dir or not self.cache_key:
            return None
        key = f"{self.cache_key}|k={self.k}|neg_hops={self.neg_expansion_hops}|max_pool={self.max_pool_size}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"neg_sampler_{h}.pt")

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if path is None or not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})
        if meta.get("num_nodes") is None:
            return False
        # Basic compatibility check
        if meta.get("subclass_rel") != self.subclass_rel:
            return False
        if meta.get("neg_prefix") != self.neg_prefix:
            return False
        if meta.get("instance_rel") != self.instance_rel:
            return False
        if meta.get("neg_expansion_hops") != self.neg_expansion_hops:
            return False
        if meta.get("max_pool_size") != self.max_pool_size:
            return False

        self.num_nodes = int(meta["num_nodes"])
        self.node_type = meta.get("node_type", "node")
        self._is_instance = payload.get("is_instance", None)
        self.anchors = payload.get("anchors", [])
        self.anchors_valid = payload.get("anchors_valid", None)
        self.preds_seen = payload.get("preds_seen", None)
        self.pred_order_cache = payload.get("pred_order_cache", None)
        self.pool_shared_neg_any = payload.get("pool_shared_neg_any", None)

        # Rebuild defaultdict-backed structures
        empty = torch.empty(0, dtype=torch.long)
        def _to_dd(list_of_dicts):
            out = []
            for d in list_of_dicts:
                dd = defaultdict(lambda: empty)
                dd.update(d)
                out.append(dd)
            return out

        self.pool_shared_pos_by_pred = _to_dd(payload.get("pool_shared_pos_by_pred", []))
        self.pool_shared_neg_by_pred = _to_dd(payload.get("pool_shared_neg_by_pred", []))
        self.pool_pos_to_u_neg_by_pred = _to_dd(payload.get("pool_pos_to_u_neg_by_pred", []))
        self.pool_neg_to_u_pos_by_pred = _to_dd(payload.get("pool_neg_to_u_pos_by_pred", []))

        self._g2l_cpu = torch.empty(self.num_nodes, dtype=torch.long)
        self._stamp_cpu = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._cur_stamp = 0
        self._compute_prefer_inst_only()
        print(f"[NegativeInstanceSampler_NEW] loaded cache: {path}", flush=True)
        return True

    def _save_cache(self) -> None:
        path = self._cache_path()
        if path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "meta": {
                "num_nodes": int(self.num_nodes),
                "node_type": self.node_type,
                "subclass_rel": self.subclass_rel,
                "neg_prefix": self.neg_prefix,
                "instance_rel": self.instance_rel,
                "neg_expansion_hops": int(self.neg_expansion_hops),
                "max_pool_size": int(self.max_pool_size),
            },
            "is_instance": self._is_instance,
            "anchors": self.anchors,
            "anchors_valid": self.anchors_valid,
            "preds_seen": self.preds_seen,
            "pred_order_cache": self.pred_order_cache,
            "pool_shared_neg_any": self.pool_shared_neg_any,
            "pool_shared_pos_by_pred": [dict(d) for d in self.pool_shared_pos_by_pred],
            "pool_shared_neg_by_pred": [dict(d) for d in self.pool_shared_neg_by_pred],
            "pool_pos_to_u_neg_by_pred": [dict(d) for d in self.pool_pos_to_u_neg_by_pred],
            "pool_neg_to_u_pos_by_pred": [dict(d) for d in self.pool_neg_to_u_pos_by_pred],
        }
        torch.save(payload, path)
        print(f"[NegativeInstanceSampler_NEW] saved cache: {path}", flush=True)


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


    def _base_pred(self, rel: str) -> str:
        rel = str(rel)
        if rel.startswith(self.neg_prefix):
            rel = rel[len(self.neg_prefix):]
        if rel.endswith(self.cls_edge_suffix):
            rel = rel[: -len(self.cls_edge_suffix)]
        return rel

    def _compute_prefer_inst_only(self) -> None:
        if not self.anchors:
            self.prefer_inst_only = []
            self.inst_pred_key = []
            return
        if self.pool_shared_pos_by_pred is None:
            return
        N = int(self.num_nodes)
        self.prefer_inst_only = [False for _ in range(N)]
        self.inst_pred_key = [None for _ in range(N)]

        def _non_empty(dd, key: str) -> bool:
            if dd is None:
                return False
            t = dd.get(key, None)
            return t is not None and t.numel() > 0

        cand_keys = [str(self.primary_pred), "instance_of", "P31"]
        for u in self.anchors:
            key = None
            for k in cand_keys:
                if (
                    _non_empty(self.pool_shared_pos_by_pred[u], k)
                    or _non_empty(self.pool_shared_neg_by_pred[u], k)
                    or _non_empty(self.pool_pos_to_u_neg_by_pred[u], k)
                    or _non_empty(self.pool_neg_to_u_pos_by_pred[u], k)
                ):
                    key = k
                    break
            if key is None:
                continue
            pos_ok = _non_empty(self.pool_shared_pos_by_pred[u], key) and _non_empty(self.pool_shared_neg_by_pred[u], key)
            neg_ok = _non_empty(self.pool_pos_to_u_neg_by_pred[u], key) or _non_empty(self.pool_neg_to_u_pos_by_pred[u], key)
            if pos_ok and neg_ok:
                self.prefer_inst_only[u] = True
                self.inst_pred_key[u] = key


    def prepare_global(self, full_g: HeteroData) -> None:
        if self._load_cache():
            return
        t0 = time.time()
        if len(full_g.node_types) != 1:
            raise ValueError("NegativeInstanceSampler assumes a single node type.")
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N
        print(f"[NegativeInstanceSampler_NEW] prepare_global: start (num_nodes={N})")
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
        # print(f"[NegativeInstanceSampler_NEW] prepare_global: instance_nodes={len(instance_nodes)}")

        # self.pos_classes = [[] for _ in range(N)]
        # self.neg_classes_direct = [[] for _ in range(N)]

        subclass_predecessors: Dict[int, Tensor] = {}
        sub_key = self._find_edge_key(full_g, self.subclass_rel)
        if sub_key is not None and "edge_index" in full_g[sub_key]:
            sub_src, sub_dst = full_g[sub_key].edge_index
            tmp: Dict[int, List[int]] = {}
            for child, parent in zip(sub_src.tolist(), sub_dst.tolist()):
                if 0 <= child < N and 0 <= parent < N:
                    tmp.setdefault(int(parent), []).append(int(child))
            subclass_predecessors = {p: self._unique_cpu(ch) for p, ch in tmp.items()}
        # print(f"[NegativeInstanceSampler_NEW] prepare_global: subclass_predecessors={len(subclass_predecessors)}")

        # Sparse anchor->pred->classes maps: avoid allocating Python dicts for every node id.
        # This preserves sampling behavior while drastically reducing prepare_global overhead.
        pos_by_pred: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        neg_by_pred_direct: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        class2pos_by_pred_tmp = defaultdict(lambda: defaultdict(list))
        class2neg_by_pred_tmp = defaultdict(lambda: defaultdict(list))
        anchor_sources: set[int] = set(instance_nodes)

        t_scan = time.time()
        for (_, rel, _), eidx in full_g.edge_index_dict.items():
            if eidx.numel() == 0 or rel == self.subclass_rel: continue
            rel = str(rel)
            is_neg_rel = rel.startswith(self.neg_prefix)
            pred = self._base_pred(rel)
            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            for a, b in zip(src_list, dst_list):
                if not (0 <= a < N and 0 <= b < N): continue
                inst, cls = self._as_inst_class(int(a), int(b))
                if inst is None: continue
                anchor_sources.add(inst)
                if is_neg_rel:
                    neg_by_pred_direct[inst][pred].append(cls)
                    class2neg_by_pred_tmp[pred][cls].append(inst)
                else:
                    pos_by_pred[inst][pred].append(cls)
                    class2pos_by_pred_tmp[pred][cls].append(inst)
        # print(f"[NegativeInstanceSampler_NEW] prepare_global: edge scan took {time.time() - t_scan:.2f}s")
        self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))

        neg_by_pred_expanded: Dict[int, Dict[str, List[int]]] = defaultdict(dict)
        t_expand = time.time()
        for u in self.anchors:
            for pred, cls_list in neg_by_pred_direct.get(u, {}).items():
                neg_by_pred_expanded[u][pred] = self._expand_negs(cls_list, subclass_predecessors)
        # print(f"[NegativeInstanceSampler_NEW] prepare_global: neg expansion took {time.time() - t_expand:.2f}s")

        class2pos_by_pred = {pred: {c: self._cap_tensor(self._unique_cpu(vs), self.max_class_pool_size)
                                    for c, vs in cls_map.items()}
            for pred, cls_map in class2pos_by_pred_tmp.items()}
        class2neg_by_pred = {pred: {c: self._cap_tensor(self._unique_cpu(vs), self.max_class_pool_size)
                                    for c, vs in cls_map.items()}
            for pred, cls_map in class2neg_by_pred_tmp.items()}

        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_pos_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_shared_neg_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_pos_to_u_neg_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_neg_to_u_pos_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_shared_neg_any = [empty for _ in range(N)]
        self.preds_seen = [set() for _ in range(N)]
        self.pred_order_cache = [[] for _ in range(N)]

        t_pools = time.time()
        for u in self.anchors:
            pos_u = pos_by_pred.get(u, {})
            neg_u = neg_by_pred_expanded.get(u, {})
            preds_u = set(pos_u.keys()) | set(neg_u.keys())
            self.preds_seen[u] = preds_u
            self.pred_order_cache[u] = self._pred_priority(u) if preds_u else []
            neg_union_parts: List[Tensor] = []
            for pred in preds_u:
                pos_cls = pos_u.get(pred, [])
                neg_cls = neg_u.get(pred, [])

                if pos_cls:
                    self.pool_shared_pos_by_pred[u][pred] = self._cap_pool(self._collect_instances_fast(
                        pos_cls, class2pos_by_pred.get(pred, {}), exclude=u))
                if neg_cls:
                    self.pool_shared_neg_by_pred[u][pred] = self._cap_pool(self._collect_instances_fast(
                        neg_cls, class2neg_by_pred.get(pred, {}), exclude=u))
                    self.pool_pos_to_u_neg_by_pred[u][pred] = self._cap_pool(self._collect_instances_fast(
                        neg_cls, class2pos_by_pred.get(pred, {}), exclude=u))
                    if self.pool_shared_neg_by_pred[u][pred].numel() > 0:
                        neg_union_parts.append(self.pool_shared_neg_by_pred[u][pred])
                if pos_cls:
                    self.pool_neg_to_u_pos_by_pred[u][pred] = self._cap_pool(self._collect_instances_fast(
                        pos_cls, class2neg_by_pred.get(pred, {}), exclude=u))
            if neg_union_parts:
                self.pool_shared_neg_any[u] = self._cap_pool(torch.unique(torch.cat(neg_union_parts, dim=0)))
        # print(f"[NegativeInstanceSampler_NEW] prepare_global: pool build took {time.time() - t_pools:.2f}s")
        self.anchors_valid = [u for u in self.anchors if self._has_any_pool(u)]
        self._g2l_cpu = torch.empty(self.num_nodes, dtype=torch.long)
        self._stamp_cpu = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._cur_stamp = 0
        # print(f"[NegativeInstanceSampler_NEW] prepare_global: done in {time.time() - t0:.2f}s")
        self._save_cache()
        self._compute_prefer_inst_only()
        self._pre_ready = False

    def prepare_epoch(self) -> None:
        if self.num_nodes <= 0 or self.anchors_valid is None:
            return
        t0 = time.time()
        N = int(self.num_nodes)
        k = int(self.k)
        # Preallocate with self indices as fallback
        base = torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(1, k)
        self._pre_shneg = base.clone()
        self._pre_pos2neg = base.clone()
        self._pre_neg2pos = base.clone()
        self._pre_shpos = base.clone()
        self._pre_anchor_valid_mask = torch.zeros(N, dtype=torch.bool)

        for u in self.anchors_valid:
            pred_order = self.pred_order_cache[u] if self.pred_order_cache is not None else self._pred_priority(u)
            if self.prefer_inst_only is not None and u < len(self.prefer_inst_only) and self.prefer_inst_only[u]:
                inst_key = self.inst_pred_key[u] if self.inst_pred_key is not None else None
                if inst_key:
                    pred_order = [inst_key]
            shpos_pool = self.pool_shared_pos_by_pred[u]
            shneg_pool = self.pool_shared_neg_by_pred[u]
            if self.prefer_inst_only is not None and u < len(self.prefer_inst_only) and self.prefer_inst_only[u]:
                inst_key = self.inst_pred_key[u] if self.inst_pred_key is not None else None
                if inst_key:
                    shpos_t = shpos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                    shneg_t = shneg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                    shpos_pool = {inst_key: shpos_t}
                    shneg_pool = {inst_key: shneg_t}
            has_shpos = self._has_any_pool_map(shpos_pool)
            has_shneg = self._has_any_pool_map(shneg_pool)
            # Old behavior: sample shared_neg/shared_pos independently
            # self._pre_shneg[u] = self._sample_k_priority_cpu(
            #     self.pool_shared_neg_by_pred[u], pred_order, stamp=0,
            #     fallback_global=u, fallback_local=u, k=k, full_graph_mode=True
            # )
            if not has_shpos and has_shneg:
                shpos_pool = shneg_pool
                has_shpos = True
            if not has_shneg and has_shpos:
                shneg_pool = shpos_pool
                has_shneg = True
            if has_shneg:
                self._pre_shneg[u] = self._sample_k_priority_cpu(
                    shneg_pool, pred_order, stamp=0,
                    fallback_global=u, fallback_local=u, k=k, full_graph_mode=True
                )
            pos2neg_pool = self.pool_pos_to_u_neg_by_pred[u]
            neg2pos_pool = self.pool_neg_to_u_pos_by_pred[u]
            has_pos2neg = self._has_any_pool_map(pos2neg_pool)
            has_neg2pos = self._has_any_pool_map(neg2pos_pool)
            if self.prefer_inst_only is not None and u < len(self.prefer_inst_only) and self.prefer_inst_only[u]:
                inst_key = self.inst_pred_key[u] if self.inst_pred_key is not None else None
                if inst_key:
                    pos2neg_t = pos2neg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                    neg2pos_t = neg2pos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                    pos2neg_pool = {inst_key: pos2neg_t}
                    neg2pos_pool = {inst_key: neg2pos_t}
                    has_pos2neg = pos2neg_t.numel() > 0
                    has_neg2pos = neg2pos_t.numel() > 0
            # Old fallback (shared_neg_any) disabled: use only available neg pool
            # if not self._has_any_pool_map(pos2neg_pool):
            #     pos2neg_pool = {"__any__": self.pool_shared_neg_any[u]}
            # if not self._has_any_pool_map(neg2pos_pool):
            #     neg2pos_pool = {"__any__": self.pool_shared_neg_any[u]}
            if not has_pos2neg and has_neg2pos:
                pos2neg_pool = neg2pos_pool
            if not has_neg2pos and has_pos2neg:
                neg2pos_pool = pos2neg_pool
            if (not has_pos2neg and not has_neg2pos) or (not has_shpos and not has_shneg):
                continue
            self._pre_anchor_valid_mask[u] = True
            self._pre_pos2neg[u] = self._sample_k_priority_cpu(
                pos2neg_pool, pred_order, stamp=0,
                fallback_global=u, fallback_local=u, k=k, full_graph_mode=True
            )
            self._pre_neg2pos[u] = self._sample_k_priority_cpu(
                neg2pos_pool, pred_order, stamp=0,
                fallback_global=u, fallback_local=u, k=k, full_graph_mode=True
            )
            if has_shpos:
                self._pre_shpos[u] = self._sample_k_priority_cpu(
                    shpos_pool, pred_order, stamp=0,
                    fallback_global=u, fallback_local=u, k=k, full_graph_mode=True
                )

        self._pre_ready = True
        # print(f"[NegativeInstanceSampler_NEW] prepare_epoch: pre-sampled in {time.time() - t0:.2f}s", flush=True)

        if self.anchors:
            def _has_any_pool_by_pred(pool_map: dict) -> bool:
                for t in pool_map.values():
                    if t is not None and t.numel() > 0:
                        return True
                return False

            empty_shpos = sum(1 for u in self.anchors if not _has_any_pool_by_pred(self.pool_shared_pos_by_pred[u]))
            empty_shneg = sum(1 for u in self.anchors if not _has_any_pool_by_pred(self.pool_shared_neg_by_pred[u]))
            empty_pos2neg = sum(1 for u in self.anchors if not _has_any_pool_by_pred(self.pool_pos_to_u_neg_by_pred[u]))
            empty_neg2pos = sum(1 for u in self.anchors if not _has_any_pool_by_pred(self.pool_neg_to_u_pos_by_pred[u]))
            # print(
            #     "[NegativeInstanceSampler_NEW] anchors without pool neighbors (global pools): "
            #     f"shared_pos={empty_shpos}/{len(self.anchors)}, "
            #     f"shared_neg={empty_shneg}/{len(self.anchors)}, "
            #     f"pos->u_neg={empty_pos2neg}/{len(self.anchors)}, "
            #     f"neg->u_pos={empty_neg2pos}/{len(self.anchors)}"
            # )


    def prepare_batch(self, batch: HeteroData, pos_index: Optional[Tensor] = None) -> None:
        return

    def _pred_priority(self, u_g: int) -> list[str]:
        # Deterministic ordering (set -> sorted list), prefer instance_of-like preds first.
        preds = sorted(self.preds_seen[u_g]) if self.preds_seen is not None else []
        preferred: List[str] = []
        # Primary pred (instance_rel) first if present
        if self.primary_pred in preds:
            preferred.append(self.primary_pred)
            preds.remove(self.primary_pred)
        # Also prioritize common instance_of aliases if present
        for alias in ("instance_of", "P31"):
            if alias in preds and alias not in preferred:
                preferred.append(alias)
                preds.remove(alias)
        return preferred + preds
    # def _pred_priority(self, u_g: int) -> list[str]:
    #     preds = list(self.preds_seen[u_g]) if self.preds_seen is not None else []
    #     if self.primary_pred in preds:
    #         preds.remove(self.primary_pred)
    #         return [self.primary_pred] + preds
    #     return preds


    def _sample_k_priority_cpu(self, pools_by_pred: dict, pred_order: list[str], *, stamp: int,
        fallback_global: int, fallback_local: int, k: int, full_graph_mode: bool) -> torch.Tensor:
        chosen_globals: list[int] = []
        chosen_set: set[int] = set()

        def _filter_in_batch(globals_: torch.Tensor) -> torch.Tensor:
            if globals_.numel() == 0:
                return globals_
            if full_graph_mode:
                return globals_
            # neighbor-mode filter (globals only)
            mask = (self._stamp_cpu[globals_] == stamp)
            return globals_[mask]

        # take from predicates in priority order
        for pred in pred_order:
            pool_g = pools_by_pred.get(pred, None)
            if pool_g is None or pool_g.numel() == 0:
                continue
            pool_g = _filter_in_batch(pool_g)
            if pool_g.numel() == 0:
                continue

            if chosen_set:
                chosen_t = torch.tensor(list(chosen_set), dtype=pool_g.dtype)
                pool_g = pool_g[~torch.isin(pool_g, chosen_t)]
                # mask = torch.tensor([int(x) not in chosen_set for x in pool_g.tolist()], dtype=torch.bool)
                # pool_g = pool_g[mask]
                if pool_g.numel() == 0: continue

            need = k - len(chosen_globals)
            if need <= 0:
                break

            idx = self._choose_indices(int(pool_g.numel()), need)
            for x in pool_g[idx].tolist():
                x = int(x)
                if x not in chosen_set:
                    chosen_set.add(x)
                    chosen_globals.append(x)
                    if len(chosen_globals) >= k:
                        break

        # fallback: union across predicates (still global ids)
        if len(chosen_globals) < k:
            all_parts = []
            for pred in pred_order:
                p = pools_by_pred.get(pred, None)
                if p is None or p.numel() == 0:
                    continue
                all_parts.append(_filter_in_batch(p))
            if all_parts:
                union = torch.unique(torch.cat(all_parts, dim=0))
                if union.numel() > 0:
                    if chosen_set:
                        chosen_t = torch.tensor(list(chosen_set), dtype=union.dtype)
                        union = union[~torch.isin(union, chosen_t)]
                        # mask = torch.tensor([int(x) not in chosen_set for x in union.tolist()], dtype=torch.bool)
                        # union = union[mask]
                    if union.numel() > 0:
                        need = k - len(chosen_globals)
                        idx = self._choose_indices(int(union.numel()), need)
                        chosen_globals.extend([int(x) for x in union[idx].tolist()])

        # pad with GLOBAL fallback
        if len(chosen_globals) < k:
            chosen_globals.extend([int(fallback_global)] * (k - len(chosen_globals)))

        chosen_globals_t = torch.tensor(chosen_globals[:k], dtype=torch.long)

        if full_graph_mode:
            return chosen_globals_t  # local == global

        # neighbor-mode: map global -> local
        chosen_local = self._g2l_cpu[chosen_globals_t]
        bad = (self._stamp_cpu[chosen_globals_t] != stamp) | (chosen_local < 0)
        if bad.any():
            chosen_local = chosen_local.clone()
            chosen_local[bad] = int(fallback_local)
        return chosen_local.long()


    def _has_any_pool(self, u: int) -> bool:
        for d in (
            self.pool_shared_pos_by_pred[u],
            self.pool_shared_neg_by_pred[u],
            self.pool_pos_to_u_neg_by_pred[u],
            self.pool_neg_to_u_pos_by_pred[u],
        ):
            for t in d.values():
                if t is not None and t.numel() > 0:
                    return True
        return False

    @staticmethod
    def _has_any_pool_map(pool_map: dict) -> bool:
        for t in pool_map.values():
            if t is not None and t.numel() > 0:
                return True
        return False

    def _needs_fallback2_neg(self, u: int) -> bool:
        """Return True if neg pools would fall back to self (no negatives anywhere)."""
        if (
            self.pool_pos_to_u_neg_by_pred is None
            or self.pool_neg_to_u_pos_by_pred is None
            or u >= len(self.pool_pos_to_u_neg_by_pred)
            or u >= len(self.pool_neg_to_u_pos_by_pred)
        ):
            return True
        shared_any = False
        if self.pool_shared_neg_any is not None and u < len(self.pool_shared_neg_any):
            shared_any = self.pool_shared_neg_any[u].numel() > 0

        pos2neg_pool = self.pool_pos_to_u_neg_by_pred[u]
        neg2pos_pool = self.pool_neg_to_u_pos_by_pred[u]
        needs_pos2neg = (not self._has_any_pool_map(pos2neg_pool)) and (not shared_any)
        needs_neg2pos = (not self._has_any_pool_map(neg2pos_pool)) and (not shared_any)
        return needs_pos2neg or needs_neg2pos

    def get_contrastive_samples(self, z: Tensor, anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None):
        """   Returns: (z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos)
        with shapes:
            z_anchor: (B,D)
            each other: (B,k,D)
        where B = number of anchors that had non-empty pools and were valid.
        """
        device = z.device
        B_rows, D = z.shape
        k = self.k
        t_start = time.time()

        if n_id is None:
            t_select = time.time()
            if anchor_nodes is None or anchor_nodes.numel() == 0:
                anchor_globals_cpu = torch.tensor(self.anchors_valid or self.anchors, dtype=torch.long)
            else: anchor_globals_cpu = torch.unique(anchor_nodes.detach().long().cpu())
            anchor_globals_cpu = anchor_globals_cpu[(anchor_globals_cpu >= 0) & (anchor_globals_cpu < self.num_nodes)]
            if self.max_contrastive_anchors > 0 and anchor_globals_cpu.numel() > self.max_contrastive_anchors:
                perm = torch.randperm(anchor_globals_cpu.numel())[: self.max_contrastive_anchors]
                anchor_globals_cpu = anchor_globals_cpu[perm]
            t_select = time.time() - t_select

            anchors_cpu: List[int] = []
            shneg_cpu, pos2neg_cpu, neg2pos_cpu, shpos_cpu = [], [], [], []

            t_sample = time.time()
            fallback_pos2neg = 0
            fallback_neg2pos = 0
            still_empty_pos2neg = 0
            still_empty_neg2pos = 0
            for u in anchor_globals_cpu.tolist():
                if not self._has_any_pool(u):
                    continue
                # Old skip when neg pools empty (fallback2) disabled
                # if self._needs_fallback2_neg(u):
                #     continue
                pred_order = self.pred_order_cache[u] if self.pred_order_cache is not None else self._pred_priority(u)
                if self.prefer_inst_only is not None and u < len(self.prefer_inst_only) and self.prefer_inst_only[u]:
                    inst_key = self.inst_pred_key[u] if self.inst_pred_key is not None else None
                    if inst_key:
                        pred_order = [inst_key]

                pos2neg_pool = self.pool_pos_to_u_neg_by_pred[u]
                neg2pos_pool = self.pool_neg_to_u_pos_by_pred[u]
                has_pos2neg = self._has_any_pool_map(pos2neg_pool)
                has_neg2pos = self._has_any_pool_map(neg2pos_pool)
                if self.prefer_inst_only is not None and u < len(self.prefer_inst_only) and self.prefer_inst_only[u]:
                    inst_key = self.inst_pred_key[u] if self.inst_pred_key is not None else None
                    if inst_key:
                        pos2neg_t = pos2neg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        neg2pos_t = neg2pos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        pos2neg_pool = {inst_key: pos2neg_t}
                        neg2pos_pool = {inst_key: neg2pos_t}
                        has_pos2neg = pos2neg_t.numel() > 0
                        has_neg2pos = neg2pos_t.numel() > 0
                if not has_pos2neg and not has_neg2pos:
                    continue
                # Old fallback (shared_neg_any) disabled: use only available neg pool
                # if not self._has_any_pool_map(pos2neg_pool):
                #     pos2neg_pool = {"__any__": self.pool_shared_neg_any[u]}
                #     fallback_pos2neg += 1
                #     if self.pool_shared_neg_any[u].numel() == 0:
                #         still_empty_pos2neg += 1
                # if not self._has_any_pool_map(neg2pos_pool):
                #     neg2pos_pool = {"__any__": self.pool_shared_neg_any[u]}
                #     fallback_neg2pos += 1
                #     if self.pool_shared_neg_any[u].numel() == 0:
                #         still_empty_neg2pos += 1
                if not has_pos2neg and has_neg2pos:
                    pos2neg_pool = neg2pos_pool
                if not has_neg2pos and has_pos2neg:
                    neg2pos_pool = pos2neg_pool
                shpos_pool = self.pool_shared_pos_by_pred[u]
                shneg_pool = self.pool_shared_neg_by_pred[u]
                if self.prefer_inst_only is not None and u < len(self.prefer_inst_only) and self.prefer_inst_only[u]:
                    inst_key = self.inst_pred_key[u] if self.inst_pred_key is not None else None
                    if inst_key:
                        shpos_t = shpos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        shneg_t = shneg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        shpos_pool = {inst_key: shpos_t}
                        shneg_pool = {inst_key: shneg_t}
                has_shpos = self._has_any_pool_map(shpos_pool)
                has_shneg = self._has_any_pool_map(shneg_pool)
                # Old behavior: sample shared_neg/shared_pos independently
                # shneg_cpu.append(self._sample_k_priority_cpu(self.pool_shared_neg_by_pred[u], pred_order, stamp=0,
                #                             fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                if not has_shpos and not has_shneg:
                    continue
                if not has_shpos and has_shneg:
                    shpos_pool = shneg_pool
                    has_shpos = True
                if not has_shneg and has_shpos:
                    shneg_pool = shpos_pool
                    has_shneg = True
                anchors_cpu.append(u)
                if has_shneg:
                    shneg_cpu.append(self._sample_k_priority_cpu(shneg_pool, pred_order, stamp=0,
                                                fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                pos2neg_cpu.append(self._sample_k_priority_cpu(pos2neg_pool, pred_order, stamp=0,
                                            fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                neg2pos_cpu.append(self._sample_k_priority_cpu(neg2pos_pool, pred_order, stamp=0,
                                            fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                if has_shpos:
                    shpos_cpu.append(self._sample_k_priority_cpu(shpos_pool, pred_order, stamp=0,
                                                fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
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
                # print(
                #     "[NegativeInstanceSampler_NEW] get_contrastive_samples (full): "
                #     f"select={t_select:.3f}s sample={t_sample:.3f}s concat={t_concat:.3f}s total={total:.3f}s "
                #     f"anchors_in={len(anchor_globals_cpu)} anchors_out={len(anchors_cpu)} "
                #     f"fallback_pos2neg={fallback_pos2neg} still_empty_pos2neg={still_empty_pos2neg} "
                #     f"fallback_neg2pos={fallback_neg2pos} still_empty_neg2pos={still_empty_neg2pos}"
                # )
            return z[anchors], z[shneg], z[pos2neg], z[neg2pos], z[shpos]

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
            anchor_globals_cpu = torch.tensor(self.anchors_valid or self.anchors, dtype=torch.long)
            in_batch = (self._stamp_cpu[anchor_globals_cpu] == stamp)
            anchor_globals_cpu = anchor_globals_cpu[in_batch]
            anchor_locals_cpu = self._g2l_cpu[anchor_globals_cpu]
        else:
            # In your code, anchor_nodes are LOCAL indices when n_id is not None.
            anchor_locals_cpu = torch.unique(anchor_nodes.detach().long().cpu())
            # safety filter
            anchor_locals_cpu = anchor_locals_cpu[(anchor_locals_cpu >= 0) & (anchor_locals_cpu < n_id_cpu.numel())]
            if self.max_contrastive_anchors > 0 and anchor_locals_cpu.numel() > self.max_contrastive_anchors:
                perm = torch.randperm(anchor_locals_cpu.numel())[: self.max_contrastive_anchors]
                anchor_locals_cpu = anchor_locals_cpu[perm]
            anchor_globals_cpu = n_id_cpu[anchor_locals_cpu]
        t_select = time.time() - t_select

        anchors_local_cpu: List[int] = []
        shneg_cpu, pos2neg_cpu, neg2pos_cpu, shpos_cpu = [], [], [], []

        t_sample = time.time()
        fallback_pos2neg = 0
        fallback_neg2pos = 0
        still_empty_pos2neg = 0
        still_empty_neg2pos = 0

        # Filter anchors with pools.
        # Fast path: when epoch pre-sampling is ready, reuse a precomputed per-anchor
        # validity mask and avoid per-batch dictionary scans.
        anchor_globals_list = []
        anchor_locals_list = []
        if self._pre_ready and self._pre_anchor_valid_mask is not None:
            valid_mask = self._pre_anchor_valid_mask
            valid_n = int(valid_mask.numel())
            safe = (
                (anchor_locals_cpu >= 0)
                & (anchor_globals_cpu >= 0)
                & (anchor_globals_cpu < valid_n)
            )
            if safe.any():
                g_ok = anchor_globals_cpu[safe]
                l_ok = anchor_locals_cpu[safe]
                keep = valid_mask[g_ok]
                if keep.any():
                    anchor_globals_list = g_ok[keep].tolist()
                    anchor_locals_list = l_ok[keep].tolist()
        else:
            for u_g, u_l in zip(anchor_globals_cpu.tolist(), anchor_locals_cpu.tolist()):
                if u_l < 0:
                    continue
                if not self._has_any_pool(u_g):
                    continue
                # Old skip when neg pools empty (fallback2) disabled
                # if self._needs_fallback2_neg(u_g):
                #     continue
                pos2neg_pool = self.pool_pos_to_u_neg_by_pred[u_g]
                neg2pos_pool = self.pool_neg_to_u_pos_by_pred[u_g]
                has_pos2neg = self._has_any_pool_map(pos2neg_pool)
                has_neg2pos = self._has_any_pool_map(neg2pos_pool)
                if self.prefer_inst_only is not None and u_g < len(self.prefer_inst_only) and self.prefer_inst_only[u_g]:
                    inst_key = self.inst_pred_key[u_g] if self.inst_pred_key is not None else None
                    if inst_key:
                        pos2neg_t = pos2neg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        neg2pos_t = neg2pos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        pos2neg_pool = {inst_key: pos2neg_t}
                        neg2pos_pool = {inst_key: neg2pos_t}
                        has_pos2neg = pos2neg_t.numel() > 0
                        has_neg2pos = neg2pos_t.numel() > 0
                if not has_pos2neg and not has_neg2pos:
                    continue
                shpos_pool = self.pool_shared_pos_by_pred[u_g]
                shneg_pool = self.pool_shared_neg_by_pred[u_g]
                if self.prefer_inst_only is not None and u_g < len(self.prefer_inst_only) and self.prefer_inst_only[u_g]:
                    inst_key = self.inst_pred_key[u_g] if self.inst_pred_key is not None else None
                    if inst_key:
                        shpos_t = shpos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        shneg_t = shneg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                        shpos_pool = {inst_key: shpos_t}
                        shneg_pool = {inst_key: shneg_t}
                has_shpos = self._has_any_pool_map(shpos_pool)
                has_shneg = self._has_any_pool_map(shneg_pool)
                if not has_shpos and not has_shneg:
                    continue
                anchor_globals_list.append(u_g)
                anchor_locals_list.append(u_l)

        if anchor_globals_list:
            anchors_local_cpu = anchor_locals_list
            g = torch.tensor(anchor_globals_list, dtype=torch.long)
            l = torch.tensor(anchor_locals_list, dtype=torch.long)

            if self._pre_ready and self._pre_shneg is not None:
                shneg_g = self._pre_shneg[g]
                pos2neg_g = self._pre_pos2neg[g]
                neg2pos_g = self._pre_neg2pos[g]
                shpos_g = self._pre_shpos[g]

                def _map_to_local(g_ids: torch.Tensor) -> torch.Tensor:
                    loc = self._g2l_cpu[g_ids]
                    bad = (self._stamp_cpu[g_ids] != stamp) | (loc < 0)
                    if bad.any():
                        loc = loc.clone()
                        loc[bad] = l.unsqueeze(1).expand_as(loc)[bad]
                    return loc

                shneg_cpu = [_map_to_local(shneg_g)]
                pos2neg_cpu = [_map_to_local(pos2neg_g)]
                neg2pos_cpu = [_map_to_local(neg2pos_g)]
                shpos_cpu = [_map_to_local(shpos_g)]
            else:
                for u_g, u_l in zip(anchor_globals_list, anchor_locals_list):
                    pred_order = self.pred_order_cache[u_g] if self.pred_order_cache is not None else self._pred_priority(u_g)
                    if self.prefer_inst_only is not None and u_g < len(self.prefer_inst_only) and self.prefer_inst_only[u_g]:
                        inst_key = self.inst_pred_key[u_g] if self.inst_pred_key is not None else None
                        if inst_key:
                            pred_order = [inst_key]
                    pos2neg_pool = self.pool_pos_to_u_neg_by_pred[u_g]
                    neg2pos_pool = self.pool_neg_to_u_pos_by_pred[u_g]
                    has_pos2neg = self._has_any_pool_map(pos2neg_pool)
                    has_neg2pos = self._has_any_pool_map(neg2pos_pool)
                    if self.prefer_inst_only is not None and u_g < len(self.prefer_inst_only) and self.prefer_inst_only[u_g]:
                        inst_key = self.inst_pred_key[u_g] if self.inst_pred_key is not None else None
                        if inst_key:
                            pos2neg_t = pos2neg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                            neg2pos_t = neg2pos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                            pos2neg_pool = {inst_key: pos2neg_t}
                            neg2pos_pool = {inst_key: neg2pos_t}
                            has_pos2neg = pos2neg_t.numel() > 0
                            has_neg2pos = neg2pos_t.numel() > 0
                    if not has_pos2neg and not has_neg2pos:
                        continue
                    shpos_pool = self.pool_shared_pos_by_pred[u_g]
                    shneg_pool = self.pool_shared_neg_by_pred[u_g]
                    if self.prefer_inst_only is not None and u_g < len(self.prefer_inst_only) and self.prefer_inst_only[u_g]:
                        inst_key = self.inst_pred_key[u_g] if self.inst_pred_key is not None else None
                        if inst_key:
                            shpos_t = shpos_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                            shneg_t = shneg_pool.get(inst_key, torch.empty(0, dtype=torch.long))
                            shpos_pool = {inst_key: shpos_t}
                            shneg_pool = {inst_key: shneg_t}
                    has_shpos = self._has_any_pool_map(shpos_pool)
                    has_shneg = self._has_any_pool_map(shneg_pool)
                    if not has_shpos and not has_shneg:
                        continue
                    # Old fallback (shared_neg_any) disabled: use only available neg pool
                    # if not self._has_any_pool_map(pos2neg_pool):
                    #     pos2neg_pool = {"__any__": self.pool_shared_neg_any[u_g]}
                    #     fallback_pos2neg += 1
                    #     if self.pool_shared_neg_any[u_g].numel() == 0:
                    #         still_empty_pos2neg += 1
                    # if not self._has_any_pool_map(neg2pos_pool):
                    #     neg2pos_pool = {"__any__": self.pool_shared_neg_any[u_g]}
                    #     fallback_neg2pos += 1
                    #     if self.pool_shared_neg_any[u_g].numel() == 0:
                    #         still_empty_neg2pos += 1
                    if not has_pos2neg and has_neg2pos:
                        pos2neg_pool = neg2pos_pool
                    if not has_neg2pos and has_pos2neg:
                        neg2pos_pool = pos2neg_pool
                    # Old behavior: sample shared_neg/shared_pos independently
                    # shneg_cpu.append(self._sample_k_priority_cpu(self.pool_shared_neg_by_pred[u_g], pred_order, stamp=stamp,
                    #                             fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
                    if not has_shpos and has_shneg:
                        shpos_pool = shneg_pool
                        has_shpos = True
                    if not has_shneg and has_shpos:
                        shneg_pool = shpos_pool
                        has_shneg = True
                    if has_shneg:
                        shneg_cpu.append(self._sample_k_priority_cpu(shneg_pool, pred_order, stamp=stamp,
                                                    fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
                    pos2neg_cpu.append(self._sample_k_priority_cpu(pos2neg_pool, pred_order, stamp=stamp,
                                                fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
                    neg2pos_cpu.append(self._sample_k_priority_cpu(neg2pos_pool, pred_order, stamp=stamp,
                                                fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
                    if has_shpos:
                        shpos_cpu.append(self._sample_k_priority_cpu(shpos_pool, pred_order, stamp=stamp,
                                                    fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
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
            # print(
            #     "[NegativeInstanceSampler_NEW] get_contrastive_samples (neighbor): "
            #     f"select={t_select:.3f}s sample={t_sample:.3f}s concat={t_concat:.3f}s total={total:.3f}s "
            #     f"anchors_in={len(anchor_globals_cpu)} anchors_out={len(anchors_local_cpu)} n_id={len(n_id_cpu)} "
            #     f"fallback_pos2neg={fallback_pos2neg} still_empty_pos2neg={still_empty_pos2neg} "
            #     f"fallback_neg2pos={fallback_neg2pos} still_empty_neg2pos={still_empty_neg2pos}"
            # )
        return z[anchors_local], z[shneg], z[pos2neg], z[neg2pos], z[shpos]
