import random
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
        neg_prefix: str = "NOT_", instance_rel: str = "2",neg_expansion_hops: int = 1):
        self.k = int(k)
        self.subclass_rel = subclass_rel
        self.neg_prefix = neg_prefix
        self.instance_rel = instance_rel
        self.neg_expansion_hops = int(neg_expansion_hops)
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
        self.preds_seen = None
        self.primary_pred = instance_rel  # "instance_of"


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
        return None, None

    @staticmethod
    def _unique_cpu(x: List[int]) -> Tensor:
        if not x: return torch.empty(0, dtype=torch.long)
        return torch.unique(torch.tensor(x, dtype=torch.long))

    @staticmethod
    def _collect_instances(classes: List[int], class2inst: Dict[int, Tensor],
        exclude: Optional[int] = None) -> Tensor:
        if not classes: return torch.empty(0, dtype=torch.long)
        parts = [class2inst[c] for c in classes if c in class2inst and class2inst[c].numel() > 0]
        if not parts:
            return torch.empty(0, dtype=torch.long)
        out = torch.unique(torch.cat(parts, dim=0))
        if exclude is not None and out.numel() > 0:
            out = out[out != int(exclude)]
        return out

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
        return rel[len(self.neg_prefix):] if rel.startswith(self.neg_prefix) else rel


    def prepare_global(self, full_g: HeteroData) -> None:
        if len(full_g.node_types) != 1:
            raise ValueError("NegativeInstanceSampler assumes a single node type.")
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N
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

        pos_by_pred = [defaultdict(list) for _ in range(N)]
        neg_by_pred_direct = [defaultdict(list) for _ in range(N)]
        class2pos_by_pred_tmp = defaultdict(lambda: defaultdict(list))
        class2neg_by_pred_tmp = defaultdict(lambda: defaultdict(list))
        anchor_sources: set[int] = set(instance_nodes)

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
        self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))

        neg_by_pred_expanded = [defaultdict(list) for _ in range(N)]
        for u in self.anchors:
            for pred, cls_list in neg_by_pred_direct[u].items():
                neg_by_pred_expanded[u][pred] = self._expand_negs(cls_list, subclass_predecessors)

        class2pos_by_pred = {pred: {c: self._unique_cpu(vs) for c, vs in cls_map.items()}
            for pred, cls_map in class2pos_by_pred_tmp.items()}
        class2neg_by_pred = {pred: {c: self._unique_cpu(vs) for c, vs in cls_map.items()}
            for pred, cls_map in class2neg_by_pred_tmp.items()}

        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_pos_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_shared_neg_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_pos_to_u_neg_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.pool_neg_to_u_pos_by_pred = [defaultdict(lambda: empty) for _ in range(N)]
        self.preds_seen = [set() for _ in range(N)]

        for u in self.anchors:
            preds_u = set(pos_by_pred[u].keys()) | set(neg_by_pred_expanded[u].keys())
            self.preds_seen[u] = preds_u
            for pred in preds_u:
                pos_cls = pos_by_pred[u].get(pred, [])
                neg_cls = neg_by_pred_expanded[u].get(pred, [])

                self.pool_shared_pos_by_pred[u][pred] = self._collect_instances(
                    pos_cls, class2pos_by_pred.get(pred, {}), exclude=u)
                self.pool_shared_neg_by_pred[u][pred] = self._collect_instances(
                    neg_cls, class2neg_by_pred.get(pred, {}), exclude=u)
                self.pool_pos_to_u_neg_by_pred[u][pred] = self._collect_instances(
                    neg_cls, class2pos_by_pred.get(pred, {}), exclude=u)
                self.pool_neg_to_u_pos_by_pred[u][pred] = self._collect_instances(
                    pos_cls, class2neg_by_pred.get(pred, {}), exclude=u)
        self._g2l_cpu = torch.empty(self.num_nodes, dtype=torch.long)
        self._stamp_cpu = torch.zeros(self.num_nodes, dtype=torch.int32)
        self._cur_stamp = 0


    def prepare_batch(self, batch: HeteroData, pos_index: Optional[Tensor] = None) -> None:
        return

    def _pred_priority(self, u_g: int) -> list[str]:
        # Deterministic ordering (set -> sorted list)
        preds = sorted(self.preds_seen[u_g]) if self.preds_seen is not None else []
        if self.primary_pred in preds:
            preds.remove(self.primary_pred)
            return [self.primary_pred] + preds
        return preds
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

        if n_id is None:
            if anchor_nodes is None or anchor_nodes.numel() == 0:
                anchor_globals_cpu = torch.tensor(self.anchors, dtype=torch.long)
            else: anchor_globals_cpu = torch.unique(anchor_nodes.detach().long().cpu())
            anchor_globals_cpu = anchor_globals_cpu[(anchor_globals_cpu >= 0) & (anchor_globals_cpu < self.num_nodes)]

            anchors_cpu: List[int] = []
            shneg_cpu, pos2neg_cpu, neg2pos_cpu, shpos_cpu = [], [], [], []

            for u in anchor_globals_cpu.tolist():
                
                if not self._has_any_pool(u): continue
                pred_order = self._pred_priority(u)

                anchors_cpu.append(u)
                shneg_cpu.append(self._sample_k_priority_cpu(self.pool_shared_neg_by_pred[u], pred_order, stamp=0,
                                            fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                pos2neg_cpu.append(self._sample_k_priority_cpu(self.pool_pos_to_u_neg_by_pred[u], pred_order, stamp=0,
                                            fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                neg2pos_cpu.append(self._sample_k_priority_cpu(self.pool_neg_to_u_pos_by_pred[u], pred_order, stamp=0,
                                            fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))
                shpos_cpu.append(self._sample_k_priority_cpu(self.pool_shared_pos_by_pred[u], pred_order, stamp=0,
                                            fallback_global=u, fallback_local=u, k=k, full_graph_mode=True).unsqueeze(0))

            if not anchors_cpu:
                empty = torch.empty(0, D, device=device)
                empty_k = torch.empty(0, k, D, device=device)
                return empty, empty_k, empty_k, empty_k, empty_k

            anchors = torch.tensor(anchors_cpu, dtype=torch.long, device=device)
            shneg = torch.cat(shneg_cpu, dim=0).to(device)
            pos2neg = torch.cat(pos2neg_cpu, dim=0).to(device)
            neg2pos = torch.cat(neg2pos_cpu, dim=0).to(device)
            shpos = torch.cat(shpos_cpu, dim=0).to(device)
            return z[anchors], z[shneg], z[pos2neg], z[neg2pos], z[shpos]

        assert self._stamp_cpu is not None and self._g2l_cpu is not None

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

        anchors_local_cpu: List[int] = []
        shneg_cpu, pos2neg_cpu, neg2pos_cpu, shpos_cpu = [], [], [], []

        for u_g, u_l in zip(anchor_globals_cpu.tolist(), anchor_locals_cpu.tolist()):
            if u_l < 0: continue
            if not self._has_any_pool(u_g):  continue

            anchors_local_cpu.append(u_l)
            pred_order = self._pred_priority(u_g)
            shneg_cpu.append(self._sample_k_priority_cpu(self.pool_shared_neg_by_pred[u_g], pred_order, stamp=stamp,
                                        fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
            pos2neg_cpu.append(self._sample_k_priority_cpu(self.pool_pos_to_u_neg_by_pred[u_g], pred_order, stamp=stamp,
                                        fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
            neg2pos_cpu.append(self._sample_k_priority_cpu(self.pool_neg_to_u_pos_by_pred[u_g], pred_order, stamp=stamp,
                                        fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))
            shpos_cpu.append(self._sample_k_priority_cpu(self.pool_shared_pos_by_pred[u_g], pred_order, stamp=stamp,
                                        fallback_global=u_g, fallback_local=u_l, k=k, full_graph_mode=False).unsqueeze(0))

        if not anchors_local_cpu:
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty_k, empty_k, empty_k, empty_k

        anchors_local = torch.tensor(anchors_local_cpu, dtype=torch.long, device=device)
        shneg = torch.cat(shneg_cpu, dim=0).to(device)
        pos2neg = torch.cat(pos2neg_cpu, dim=0).to(device)
        neg2pos = torch.cat(neg2pos_cpu, dim=0).to(device)
        shpos = torch.cat(shpos_cpu, dim=0).to(device)
        return z[anchors_local], z[shneg], z[pos2neg], z[neg2pos], z[shpos]
