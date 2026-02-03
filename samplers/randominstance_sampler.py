import os, random, time, torch
from typing import Dict, Iterable, List, Optional, Tuple, Union
from torch_geometric.data import HeteroData
from .negativeinstance_sampler import NegativeInstanceSampler


ExternalStmt = Union[Tuple[int, int], Tuple[int, str, int], Tuple[int, int, int]]
class RandomInstanceSampler(NegativeInstanceSampler):
    """
    Wiki framework: random corruption-based instance sampler (TYPE-AWARE, FAST).

    Differences vs NegativeInstanceSampler:
      - Anchor negatives are NOT taken from graph.
      - For each anchor u and each base relation type r, we corrupt u's POS targets under r to
        synthetic NEG targets under the SAME base relation type r (no ontology expansion).
    Candidate retrieval is still based on the graph's real NEG edges:
      - shared_neg and neg_to_u_pos are populated using instances that have NEG statements
        of the SAME base relation type to those synthetic classes.
    Uses batch-local key->local-instance maps (no O(num_nodes) masks per batch).
    Statement polarity rules (Wiki):
      - POS: rel does NOT start with neg_prefix and rel != subclass_rel
      - NEG: rel DOES start with neg_prefix
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
            subclass_rel=subclass_rel,
            neg_prefix=neg_prefix,
            instance_rel=instance_rel,
            neg_expansion_hops=0,
        )
        self.seed = seed
        self.external_negs = external_negs or []
        self.max_corrupt_per_anchor = max_corrupt_per_anchor

        # typed-key machinery
        self._stride: int = 0
        self._base_rel2id: Dict[str, int] = {}
        self._universe_by_rid: Dict[int, Tuple[int, ...]] = {}
        self._global_universe: Tuple[int, ...] = ()

        # per-node typed positive keys, plus per-node per-rel class sets for corruption
        self._pos_keys: List[List[int]] = []
        self._pos_by_rel: List[Dict[int, List[int]]] = []
        self._neg_keys_synth: List[List[int]] = []

        # Optional: external neg statements indexed by instance (for batch augmentation)
        self._ext_neg_by_inst: Dict[int, List[int]] = {}

        # Batch caches
        self._batch_global_for_row: Optional[torch.Tensor] = None   # CPU [B_rows]
        self._batch_pos_map: Optional[Dict[int, torch.Tensor]] = None  # key(int) -> local inst ids (CPU)
        self._batch_neg_map: Optional[Dict[int, torch.Tensor]] = None  # key(int) -> local inst ids (CPU)

        # Timing diagnostics
        self._timing = True
        self._timing_every = int(os.environ.get("RANDOM_SAMPLER_TIMING_EVERY", "50"))
        self._timing_batch_idx = 0
        self._timing_sample_idx = 0

        print("Using RandomInstanceSampler")

    # -----------------------
    # typed-key helpers
    # -----------------------
    def _base_rel(self, rel: str) -> str:
        r = str(rel)
        if r.startswith(self.neg_prefix):
            return r[len(self.neg_prefix) :]
        if self.neg_prefix in r:
            return r.replace(self.neg_prefix, "", 1)
        return r

    def _rel_id(self, base_rel: str) -> int:
        base_rel = str(base_rel)
        rid = self._base_rel2id.get(base_rel)
        if rid is None:
            rid = len(self._base_rel2id)
            self._base_rel2id[base_rel] = rid
        return rid

    def _mk_key(self, rid: int, cls: int) -> int:
        return int(rid) * int(self._stride) + int(cls)

    def _parse_external_item(self, item: ExternalStmt) -> Tuple[int, str, int]:
        if len(item) == 2:
            a, b = item  # type: ignore
            return int(a), "__external__", int(b)
        if len(item) == 3:
            a, rel, b = item  # type: ignore
            return int(a), str(rel), int(b)
        raise ValueError(f"Invalid external statement entry: {item}")

    # -----------------------
    # corruption
    # -----------------------
    def _corrupt_for_rel(self, rid: int, pos_set: set[int], n_draw: int) -> List[int]:
        uni = self._universe_by_rid.get(rid, ())
        if not uni:
            uni = self._global_universe
        if not uni:
            return []

        if self.max_corrupt_per_anchor is not None:
            n_draw = min(n_draw, int(self.max_corrupt_per_anchor))
        n_draw = max(1, int(n_draw))

        M = len(uni)
        if not pos_set:
            if n_draw <= M:
                return random.sample(list(uni), n_draw)
            return [uni[random.randrange(M)] for _ in range(n_draw)]

        out: set[int] = set()
        attempts = 0
        max_attempts = max(200, 50 + 20 * n_draw)
        while len(out) < n_draw and attempts < max_attempts:
            c = uni[random.randrange(M)]
            attempts += 1
            if int(c) in pos_set:
                continue
            out.add(int(c))
        return list(out)

    # -----------------------
    # global prep
    # -----------------------
    def prepare_global(self, full_g: HeteroData) -> None:
        t0 = time.perf_counter()
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        if len(full_g.node_types) != 1:
            raise ValueError("RandomInstanceSampler assumes a single node type.")
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N
        self._stride = N
        self._base_rel2id = {}

        # identify instances
        instance_nodes: set[int] = set()
        if self.instance_rel is not None:
            inst_key = self._find_edge_key(full_g, self.instance_rel)
            if inst_key is not None and "edge_index" in full_g[inst_key]:
                inst_src = full_g[inst_key].edge_index[0].detach().cpu().tolist()
                instance_nodes.update(int(x) for x in inst_src if 0 <= int(x) < N)

        if instance_nodes:
            is_instance = torch.zeros(N, dtype=torch.bool)
            is_instance[list(instance_nodes)] = True
            self._is_instance = is_instance
        else:
            self._is_instance = None

        # collect positive statements per node, per base-rel (typed)
        t_edges = time.perf_counter()
        self._pos_keys = [[] for _ in range(N)]
        self._pos_by_rel = [dict() for _ in range(N)]
        anchor_sources: set[int] = set(instance_nodes) if instance_nodes else set()

        # build universes per base-rel from POS statements
        uni_tmp: Dict[int, set[int]] = {}

        for (_, rel, _), eidx in full_g.edge_index_dict.items():
            if eidx.numel() == 0 or rel == self.subclass_rel:
                continue

            rel_s = str(rel)
            is_neg = rel_s.startswith(self.neg_prefix)
            is_pos = (not is_neg) and (rel_s != self.subclass_rel)
            if not (is_neg or is_pos):
                continue

            base_rel = self._base_rel(rel_s)
            rid = self._rel_id(base_rel)

            src = eidx[0].detach().cpu().tolist()
            dst = eidx[1].detach().cpu().tolist()
            for a, b in zip(src, dst):
                if not (0 <= a < N and 0 <= b < N):
                    continue
                inst, cls = self._as_inst_class(int(a), int(b))
                if inst is None or cls is None:
                    continue

                anchor_sources.add(inst)
                if is_pos:
                    key = self._mk_key(rid, cls)
                    self._pos_keys[inst].append(key)
                    self._pos_by_rel[inst].setdefault(rid, []).append(int(cls))
                    uni_tmp.setdefault(rid, set()).add(int(cls))

        self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))
        t_edges_done = time.perf_counter()

        # finalize universes
        self._universe_by_rid = {rid: tuple(sorted(list(s))) for rid, s in uni_tmp.items()}
        all_classes = set()
        for s in uni_tmp.values():
            all_classes |= s
        self._global_universe = tuple(sorted(list(all_classes)))

        # build synthetic negatives per node, per rel
        t_corrupt = time.perf_counter()
        self._neg_keys_synth = [[] for _ in range(N)]
        for u in self.anchors:
            # corrupt per rel
            for rid, cls_list in self._pos_by_rel[u].items():
                pos_set = set(int(c) for c in cls_list)
                n_draw = len(pos_set) if len(pos_set) > 0 else 1
                neg_classes = self._corrupt_for_rel(rid, pos_set, n_draw)
                for c in neg_classes:
                    self._neg_keys_synth[u].append(self._mk_key(rid, int(c)))
        t_corrupt_done = time.perf_counter()

        # Optional external neg statements (typed) indexed by instance for batch augmentation
        t_ext = time.perf_counter()
        self._ext_neg_by_inst = {}
        warned_pairs = False
        for item in self.external_negs:
            a, rel, b = self._parse_external_item(item)
            if rel == "__external__":
                warned_pairs = True
            if not (0 <= a < N and 0 <= b < N):
                continue
            inst, cls = self._as_inst_class(int(a), int(b))
            if inst is None or cls is None:
                continue
            base_rel = self._base_rel(rel)
            rid = self._rel_id(base_rel)
            key = self._mk_key(rid, cls)
            self._ext_neg_by_inst.setdefault(int(inst), []).append(int(key))
        t_ext_done = time.perf_counter()

        if warned_pairs and self.external_negs:
            print(
                "[RandomInstanceSampler] WARNING: external_negs provided as (src,dst) pairs. "
                "They will be treated as rel='__external__' and will NOT enforce matching against graph relation types. "
                "Provide triples (src, rel, dst) for full type-aware behavior."
            )

        t1 = time.perf_counter()
        if self._timing:
            print(
                "[RandomInstanceSampler][timing][prepare_global] total="
                f"{t1 - t0:.3f}s edges={t_edges_done - t_edges:.3f}s "
                f"corrupt={t_corrupt_done - t_corrupt:.3f}s ext={t_ext_done - t_ext:.3f}s "
                f"anchors={len(self.anchors)} nodes={N}"
            )

        # reset batch caches
        self._batch_global_for_row = None
        self._batch_pos_map = None
        self._batch_neg_map = None

    # -----------------------
    # batch-local map build (fast)
    # -----------------------
    def prepare_batch(self, batch: HeteroData) -> None:
        t0 = time.perf_counter()
        ntype = self.node_type or "node"
        if ntype not in batch.node_types:
            ntype = batch.node_types[0]

        if "n_id" in batch[ntype]:
            global_for_row = batch[ntype].n_id.detach().long().cpu()
        else:
            global_for_row = torch.arange(batch[ntype].num_nodes, dtype=torch.long)

        self._batch_global_for_row = global_for_row

        pos_acc: Dict[int, List[int]] = {}
        neg_acc: Dict[int, List[int]] = {}

        # iterate batch edges
        t_edges = time.perf_counter()
        for key, eidx in batch.edge_index_dict.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            s, rel, d = key
            if s != ntype or d != ntype:
                continue

            rel_s = str(rel)
            if rel_s == self.subclass_rel:
                continue

            is_neg = rel_s.startswith(self.neg_prefix)
            is_pos = (not is_neg) and (rel_s != self.subclass_rel)
            if not (is_neg or is_pos):
                continue

            base_rel = self._base_rel(rel_s)
            rid = self._rel_id(base_rel)  # safe even if unseen
            src_l = eidx[0].detach().long().cpu()
            dst_l = eidx[1].detach().long().cpu()
            src_g = global_for_row[src_l]
            dst_g = global_for_row[dst_l]

            # orient inst/class
            if getattr(self, "_is_instance", None) is not None:
                src_is = self._is_instance[src_g]
                dst_is = self._is_instance[dst_g]
                ok = src_is ^ dst_is
                if ok.sum().item() == 0:
                    continue
                src_l = src_l[ok]
                dst_l = dst_l[ok]
                src_g = src_g[ok]
                dst_g = dst_g[ok]
                src_is = src_is[ok]
                inst_l = torch.where(src_is, src_l, dst_l)
                cls_g = torch.where(src_is, dst_g, src_g)
            else:
                inst_l = src_l
                cls_g = dst_g

            acc = neg_acc if is_neg else pos_acc
            for c, il in zip(cls_g.tolist(), inst_l.tolist()):
                k_int = self._mk_key(rid, int(c))
                acc.setdefault(int(k_int), []).append(int(il))
        t_edges_done = time.perf_counter()

        # add external neg statements for instances in batch (cheap)
        t_ext = time.perf_counter()
        if self._ext_neg_by_inst:
            # map global->local for batch rows
            # (build a tiny dict, avoids O(num_nodes) tensors)
            g2l = {int(g): int(i) for i, g in enumerate(global_for_row.tolist())}
            for inst_g, keys in self._ext_neg_by_inst.items():
                il = g2l.get(int(inst_g))
                if il is None:
                    continue
                for k_int in keys:
                    neg_acc.setdefault(int(k_int), []).append(int(il))
        t_ext_done = time.perf_counter()

        # unique tensors
        t_unique = time.perf_counter()
        self._batch_pos_map = {k: torch.unique(torch.tensor(v, dtype=torch.long)) for k, v in pos_acc.items()}
        self._batch_neg_map = {k: torch.unique(torch.tensor(v, dtype=torch.long)) for k, v in neg_acc.items()}
        t_unique_done = time.perf_counter()

        self._timing_batch_idx += 1
        if self._timing and (self._timing_batch_idx % self._timing_every == 0):
            print(
                "[RandomInstanceSampler][timing][prepare_batch] "
                f"total={t_unique_done - t0:.3f}s edges={t_edges_done - t_edges:.3f}s "
                f"ext={t_ext_done - t_ext:.3f}s unique={t_unique_done - t_unique:.3f}s "
                f"batch_nodes={int(global_for_row.numel())}"
            )

    # -----------------------
    # sampling (fast, type-aware)
    # -----------------------
    @staticmethod
    def _sample_k(cands: torch.Tensor, anchor_local: int, k: int, device: torch.device) -> torch.Tensor:
        """
        Fallback policy:
          - if empty -> all anchor
          - if <k -> take all unique then pad with anchor
          - if >=k -> sample without replacement
        """
        if cands.numel() == 0:
            return torch.full((k,), anchor_local, dtype=torch.long, device=device)

        # unique not strictly necessary if maps are unique; keep safe
        cands = torch.unique(cands)
        if cands.numel() >= k:
            idx = torch.randperm(cands.numel())[:k]
            out = cands[idx]
        else:
            pad = torch.full((k - cands.numel(),), anchor_local, dtype=torch.long)
            out = torch.cat([cands, pad], dim=0)
        return out.to(device)

    @staticmethod
    def _union_from_map(keys: List[int], m: Dict[int, torch.Tensor]) -> torch.Tensor:
        parts = [m[k] for k in keys if k in m and m[k].numel() > 0]
        if not parts:
            return torch.empty(0, dtype=torch.long)
        if len(parts) == 1:
            return parts[0]
        return torch.unique(torch.cat(parts, dim=0))

    def get_contrastive_samples(
        self,
        z: torch.Tensor,
        anchor_nodes: Optional[torch.Tensor] = None,
        n_id: Optional[torch.Tensor] = None,
        anchor_nodes_are_unique: Optional[bool] = None,
        **_: object,
    ):
        t0 = time.perf_counter()
        device = z.device
        B_rows, D = z.shape
        k = self.k

        if self._batch_global_for_row is None or self._batch_pos_map is None or self._batch_neg_map is None:
            # If user forgot to call prepare_batch, fall back to base (slower, and not type-aware)
            return super().get_contrastive_samples(z, anchor_nodes=anchor_nodes, n_id=n_id)

        global_for_row = self._batch_global_for_row
        pos_map = self._batch_pos_map
        neg_map = self._batch_neg_map

        if anchor_nodes is None or anchor_nodes.numel() == 0:
            anchor_locals = torch.arange(B_rows, dtype=torch.long)
        else: anchor_locals = anchor_nodes.detach().long().cpu().unique()
        anchors_local_list: List[int] = []
        shneg_idx, pos2neg_idx, neg2pos_idx, shpos_idx = [], [], [], []

        for u_local in anchor_locals.tolist():
            if u_local < 0 or u_local >= B_rows:
                continue
            u_global = int(global_for_row[u_local].item())

            pos_keys = self._pos_keys[u_global] if u_global < len(self._pos_keys) else []
            neg_keys = self._neg_keys_synth[u_global] if u_global < len(self._neg_keys_synth) else []
            if not pos_keys and not neg_keys:
                continue

            cand_shpos = self._union_from_map(pos_keys, pos_map)
            cand_shneg = self._union_from_map(neg_keys, neg_map)
            cand_pos2neg = self._union_from_map(neg_keys, pos_map)
            cand_neg2pos = self._union_from_map(pos_keys, neg_map)

            # exclude anchor
            if cand_shpos.numel() > 0:
                cand_shpos = cand_shpos[cand_shpos != u_local]
            if cand_shneg.numel() > 0:
                cand_shneg = cand_shneg[cand_shneg != u_local]
            if cand_pos2neg.numel() > 0:
                cand_pos2neg = cand_pos2neg[cand_pos2neg != u_local]
            if cand_neg2pos.numel() > 0:
                cand_neg2pos = cand_neg2pos[cand_neg2pos != u_local]

            # skip anchors with no examples in any group
            if (cand_shpos.numel() == 0 and cand_shneg.numel() == 0 and
                cand_pos2neg.numel() == 0 and cand_neg2pos.numel() == 0):
                continue

            anchors_local_list.append(int(u_local))
            shpos_idx.append(self._sample_k(cand_shpos, u_local, k, device).unsqueeze(0))
            shneg_idx.append(self._sample_k(cand_shneg, u_local, k, device).unsqueeze(0))
            pos2neg_idx.append(self._sample_k(cand_pos2neg, u_local, k, device).unsqueeze(0))
            neg2pos_idx.append(self._sample_k(cand_neg2pos, u_local, k, device).unsqueeze(0))
        t_loop_done = time.perf_counter()

        if not anchors_local_list:
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty_k, empty_k, empty_k, empty_k

        anchors_local_t = torch.tensor(anchors_local_list, dtype=torch.long, device=device)
        shneg_t = torch.cat(shneg_idx, dim=0)
        pos2neg_t = torch.cat(pos2neg_idx, dim=0)
        neg2pos_t = torch.cat(neg2pos_idx, dim=0)
        shpos_t = torch.cat(shpos_idx, dim=0)
        t_end = time.perf_counter()
        self._timing_sample_idx += 1
        if self._timing and (self._timing_sample_idx % self._timing_every == 0):
            print(
                "[RandomInstanceSampler][timing][get_contrastive_samples] "
                f"total={t_end - t0:.3f}s loop={t_loop_done - t0:.3f}s "
                f"anchors={int(anchors_local_t.numel())} B_rows={int(B_rows)}"
            )

        return (z[anchors_local_t], z[shneg_t],
            z[pos2neg_t], z[neg2pos_t], z[shpos_t])
