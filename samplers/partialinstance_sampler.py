from typing import List, Optional, Tuple, Union, Any, Dict
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from .negativeinstance_sampler import NegativeInstanceSampler

ExternalStmt = Union[Tuple[int, int], Tuple[int, str, int], Tuple[int, int, int]]

class PartialInstanceSampler(NegativeInstanceSampler):
    """
    Wiki framework: partial statement sampler (TYPE-AWARE).

    Same instance-example behavior as NegativeStatementSampler, but replaces ONE polarity
    of statement edges with an external list.
    Type-aware behavior:
      - Pools are constructed using keys (base_rel, class) rather than class alone.
      - This ensures examples match the SAME statement type (base relation), possibly with
        different polarity, e.g.:
          - shared_pos: A has POS base_rel=T to C and B has POS base_rel=T to C
          - pos_to_u_neg: A has NEG base_rel=T to C and B has POS base_rel=T to C
          - etc.
    External input:
      - Recommended: List of triples (src, rel, dst) using the relation name (e.g., "P31" or "NOT_P31")
      - Backward compat: (src, dst) pairs are accepted but treated as rel="__external__"
        (this loses type-awareness relative to graph relations).
    Polarity replacement:
      - edges_are_negative=True  : external edges are treated as NEG statements (graph NEG ignored)
      - edges_are_negative=False : external edges are treated as POS statements (graph POS ignored)
    """

    def __init__(self, k: int = 1, go_etype: str = "subclass_of",
        neg_edges: Optional[List[ExternalStmt]] = None,
        edges_are_negative: bool = True, instance_rel: str = "2",
        neg_prefix: str = "NOT_", neg_expansion_hops: int = 1):
        super().__init__(k=k, subclass_rel=go_etype,
            neg_prefix=neg_prefix,
            instance_rel=instance_rel,
            neg_expansion_hops=neg_expansion_hops)
        self.external_edges = neg_edges or []
        self.edges_are_negative = bool(edges_are_negative)
        print("In PartialInstanceSampler")
        # typed-key machinery
        self._stride: int = 0
        self._base_rel2id: Dict[str, int] = {}
        self.seed = 42

    # -----------------------
    # typed-key helpers
    # -----------------------

    def _is_neg_rel(self, rel: str) -> bool:
        """Return True iff this edge relation encodes a NEG statement."""
        return str(rel).startswith(self.neg_prefix)

    def _is_pos_rel(self, rel: str) -> bool:
        """Return True iff this edge relation encodes a POS statement."""
        r = str(rel)
        if r == self.subclass_rel:
            return False
        return not self._is_neg_rel(r)

    @staticmethod
    def _group_unique_by_key(keys: Tensor, vals: Tensor) -> Dict[int, Tensor]:
        """Group vals by keys, returning {key: unique(vals)} as CPU tensors.

        This utility is used by Partial/Random instance samplers to build:
          - parent -> children maps (for subclass expansion), and
          - key -> instances pools (for contrastive sampling).

        Args:
            keys: 1D tensor of integer keys (CPU or GPU).
            vals: 1D tensor of integer values aligned with keys (CPU or GPU).

        Returns:
            dict mapping int(key) -> 1D torch.long CPU tensor of unique vals for that key.
        """
        if keys.numel() == 0:
            return {}
        if keys.numel() != vals.numel():
            raise ValueError(f"keys and vals must have same length, got {keys.numel()} vs {vals.numel()}")
        k = keys.detach().to(dtype=torch.long, device='cpu')
        v = vals.detach().to(dtype=torch.long, device='cpu')
        order = torch.argsort(k)
        k = k[order]
        v = v[order]
        uniq, counts = torch.unique_consecutive(k, return_counts=True)
        out: Dict[int, Tensor] = {}
        start = 0
        for key_i, cnt_i in zip(uniq.tolist(), counts.tolist()):
            sl = v[start:start + cnt_i]
            out[int(key_i)] = torch.unique(sl)
            start += cnt_i
        return out

    def _expand_neg_classes(
        self,
        direct_classes: List[int],
        parent_to_children: Dict[int, Tensor],
    ) -> List[int]:
        """Alias used by typed-key samplers; expands classes via subclass_of."""
        return self._expand_negs(direct_classes, parent_to_children)


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

    def _mk_key(self, rel_id: int, cls: int) -> int:
        return int(rel_id) * int(self._stride) + int(cls)

    def _decode_key(self, key: int) -> Tuple[int, int]:
        rel_id = int(key) // int(self._stride)
        cls = int(key) % int(self._stride)
        return rel_id, cls

    def _expand_neg_keys(
        self,
        neg_keys_direct: List[int],
        parent_to_children: Dict[int, torch.Tensor],
    ) -> List[int]:
        """Expand negatives by subclass_of on the CLASS part, preserving relation id."""
        if not neg_keys_direct:
            return []

        # group direct negs by rel_id
        by_rel: Dict[int, List[int]] = {}
        for key in neg_keys_direct:
            rid, cls = self._decode_key(int(key))
            by_rel.setdefault(rid, []).append(int(cls))

        out_keys: List[int] = []
        for rid, direct_classes in by_rel.items():
            expanded_classes = self._expand_neg_classes(direct_classes, parent_to_children)
            for c in expanded_classes:
                out_keys.append(self._mk_key(rid, int(c)))

        # unique
        return list(set(out_keys))

    def _parse_external_item(self, item: ExternalStmt) -> Tuple[int, str, int]:
        """
        Returns (a, rel, b). If item has no rel, uses "__external__".
        """
        if len(item) == 2:
            a, b = item  # type: ignore
            # This keeps backward compatibility but loses type-awareness vs graph relations:
            return int(a), "__external__", int(b)
        if len(item) == 3:
            a, rel, b = item  # type: ignore
            return int(a), str(rel), int(b)
        raise ValueError(f"Invalid external edge entry: {item}")

    # -----------------------
    # global build
    # -----------------------
    def prepare_global(self, full_g: HeteroData) -> None:
        # Seed behavior same as base
        if self.seed is not None:
            import random
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        if len(full_g.node_types) != 1:
            raise ValueError(
                "PartialStatementSampler assumes a single node type; "
                f"got node types: {full_g.node_types}"
            )
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N
        self._stride = N  # critical for typed-key encoding uniqueness
        self._base_rel2id = {}

        # Identify instances
        instance_nodes: set[int] = set()
        if self.instance_rel is not None:
            inst_key = self._find_edge_key(full_g, self.instance_rel)
            if inst_key is not None and "edge_index" in full_g[inst_key] and full_g[inst_key].edge_index.numel() > 0:
                src = full_g[inst_key].edge_index[0].detach().cpu().tolist()
                instance_nodes.update(int(x) for x in src if 0 <= int(x) < N)

        if instance_nodes:
            is_instance = torch.zeros(N, dtype=torch.bool)
            is_instance[torch.tensor(sorted(instance_nodes), dtype=torch.long)] = True
            self._is_instance = is_instance
        else:
            self._is_instance = None

        # subclass mapping parent -> children (CPU tensors)
        parent_to_children: Dict[int, torch.Tensor] = {}
        sub_key = self._find_edge_key(full_g, self.subclass_rel)
        if sub_key is not None and "edge_index" in full_g[sub_key] and full_g[sub_key].edge_index.numel() > 0:
            child, parent = full_g[sub_key].edge_index
            child = child.detach().cpu()
            parent = parent.detach().cpu()
            parent_to_children = self._group_unique_by_key(parent, child)

        # accumulators over typed keys
        pos_key_lists: List[List[int]] = [[] for _ in range(N)]
        neg_key_lists: List[List[int]] = [[] for _ in range(N)]
        pos_key_list: List[int] = []
        pos_val_list: List[int] = []
        neg_key_list: List[int] = []
        neg_val_list: List[int] = []

        anchor_sources: set[int] = set(instance_nodes) if instance_nodes else set()

        # 1) Read from graph, skipping replaced polarity
        for (_, rel, _), eidx in full_g.edge_index_dict.items():
            if eidx.numel() == 0 or rel == self.subclass_rel:
                continue

            is_neg = self._is_neg_rel(rel)
            is_pos = self._is_pos_rel(rel)
            if not (is_neg or is_pos):
                continue

            if self.edges_are_negative and is_neg:
                continue
            if (not self.edges_are_negative) and is_pos:
                continue

            base_rel = self._base_rel(rel)
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
                key = self._mk_key(rid, cls)

                if is_neg:
                    neg_key_lists[inst].append(key)
                    neg_key_list.append(key)
                    neg_val_list.append(inst)
                else:
                    pos_key_lists[inst].append(key)
                    pos_key_list.append(key)
                    pos_val_list.append(inst)

        # 2) Add external edges into replaced polarity (typed)
        warned_pairs = False
        for item in self.external_edges:
            a, rel, b = self._parse_external_item(item)
            if rel == "__external__":
                warned_pairs = True

            if not (0 <= a < N and 0 <= b < N):
                continue
            inst, cls = self._as_inst_class(int(a), int(b))
            if inst is None or cls is None:
                continue

            anchor_sources.add(inst)

            base_rel = self._base_rel(rel)
            rid = self._rel_id(base_rel)
            key = self._mk_key(rid, cls)

            if self.edges_are_negative:
                neg_key_lists[inst].append(key)
                neg_key_list.append(key)
                neg_val_list.append(inst)
            else:
                pos_key_lists[inst].append(key)
                pos_key_list.append(key)
                pos_val_list.append(inst)

        if warned_pairs:
            print(
                "[PartialStatementSampler] WARNING: external edges provided as (src,dst) pairs. "
                "They will be treated as rel='__external__' and will NOT enforce matching against graph relation types. "
                "Provide triples (src, rel, dst) for full type-aware behavior."
            )

        # anchors
        if instance_nodes:
            self.anchors = sorted(anchor_sources)
        else:
            has_pos = [i for i in range(N) if len(pos_key_lists[i]) > 0]
            self.anchors = has_pos if has_pos else list(range(N))

        # finalize per-instance keys (store into base fields for compatibility)
        self.pos_classes = [list(set(xs)) for xs in pos_key_lists]
        self.neg_classes_direct = [list(set(xs)) for xs in neg_key_lists]

        # expand negatives (typed)
        self.neg_classes_expanded = [[] for _ in range(N)]
        for u in self.anchors:
            self.neg_classes_expanded[u] = self._expand_neg_keys(self.neg_classes_direct[u], parent_to_children)

        # build key->instances tensors (reuse base grouping util)
        self.class2pos_instances = (
            self._group_unique_by_key(torch.tensor(pos_key_list, dtype=torch.long),
                                      torch.tensor(pos_val_list, dtype=torch.long))
            if pos_key_list else {}
        )
        self.class2neg_instances = (
            self._group_unique_by_key(torch.tensor(neg_key_list, dtype=torch.long),
                                      torch.tensor(neg_val_list, dtype=torch.long))
            if neg_key_list else {}
        )

        # pools
        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_pos = [empty for _ in range(N)]
        self.pool_shared_neg = [empty for _ in range(N)]
        self.pool_pos_to_u_neg = [empty for _ in range(N)]
        self.pool_neg_to_u_pos = [empty for _ in range(N)]

        for u in self.anchors:
            pos_keys = self.pos_classes[u]
            neg_keys = self.neg_classes_expanded[u]
            self.pool_shared_pos[u] = self._collect_instances(pos_keys, self.class2pos_instances, exclude=u)
            self.pool_shared_neg[u] = self._collect_instances(neg_keys, self.class2neg_instances, exclude=u)
            self.pool_pos_to_u_neg[u] = self._collect_instances(neg_keys, self.class2pos_instances, exclude=u)
            self.pool_neg_to_u_pos[u] = self._collect_instances(pos_keys, self.class2neg_instances, exclude=u)

        self._global2local_cpu = None
        self._in_batch_mask_cpu = None
