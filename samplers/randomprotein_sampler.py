import torch
from typing import Dict, List, Optional, Set, Tuple
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class RandomProteinSampler:
    """
    Random protein sampler.

    Behaviour:
      - For each anchor protein u, gets real POS statement GO classes.
      - Generates NEG GO classes for u by random sampling excluding u's pos annotations.
      - NO ontological expansion is performed for these generated negative statements.
      - Uses:
          * shared_pos: proteins with POS statements to u's POS GO classes.
          * neg_to_u_pos: proteins with NEG statements to u's POS GO classes.
          * shared_neg: proteins with NEG statements to u's generated NEG GO classes.
          * pos_to_u_neg: proteins with POS statements to u's generated NEG GO classes.

    Fallback behaviour (stable shapes, ~0-grad when padded with anchor):
      - If group pool is empty -> pads with anchor itself (k times).
      - If 0 < pool_size < k -> uses each candidate once (shuffled) then pads with anchor.
      - If pool_size >= k -> samples k unique candidates (no replacement).
    """

    def __init__(
        self,
        k: int = 1,
        anchor_etype: str = "PPI",
        pos_etype: str = "pos_statement",
        neg_etype: str = "neg_statement",
        go_etype: str = "go",
        neg_go_hops: int = 2,  # kept for compatibility, not used (no expansion)
        max_corrupt_go: Optional[int] = None,
    ):
        self.k = int(k)
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.go_etype = go_etype
        self.neg_go_hops = int(neg_go_hops)  # unused (no expansion)
        self.max_corrupt_go = None if max_corrupt_go is None else int(max_corrupt_go)

        self.device = torch.device("cpu")

        # Node type for proteins (resolved in prepare_global)
        self.protein_ntype: Optional[str] = None

        # Global
        self.N: int = 0
        self.pos_go_by_protein: List[List[int]] = []
        self.pos_go_set_by_protein: List[Optional[Set[int]]] = []
        self.go_universe: Tensor = torch.empty(0, dtype=torch.long)  # CPU GO ids

        # GO -> proteins (CPU, unique)
        self.pos_prots_by_go: Dict[int, Tensor] = {}
        self.neg_prots_by_go: Dict[int, Tensor] = {}

        # Lazy caches for pools that depend only on anchor's POS GO terms
        self._cache_shared_pos: List[Optional[Tensor]] = []
        self._cache_neg_to_u_pos: List[Optional[Tensor]] = []

        # Batch pools (CPU tensors)
        self.batch_anchors_local: List[int] = []
        self.Nb: int = 0
        self.batch_pool_shared_neg: List[Tensor] = []
        self.batch_pool_pos_to_u_neg: List[Tensor] = []
        self.batch_pool_neg_to_u_pos: List[Tensor] = []
        self.batch_pool_shared_pos: List[Tensor] = []

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _unique_1d(x: Tensor) -> Tensor:
        # sorted=False avoids extra overhead; order isn't important for our sampling.
        return torch.unique(x, sorted=False)

    @staticmethod
    def _dedup_list(xs: List[int]) -> List[int]:
        seen = set()
        out: List[int] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _get_pos_set(self, u: int) -> Set[int]:
        s = self.pos_go_set_by_protein[u]
        return s if s is not None else set()

    def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
        """
        pool: 1D CPU tensor
        returns: (k,) tensor on `device`

        Fallback rules:
          - empty pool -> [fallback]*k
          - 0 < L < k -> shuffle all L once, then pad with fallback
          - L >= k -> sample k unique
        """
        L = int(pool.numel())
        if L == 0:
            return torch.full((k,), int(fallback), dtype=torch.long, device=device)

        if L >= k:
            idx = torch.randperm(L, device=pool.device)[:k]
            return pool[idx].to(device)

        perm = torch.randperm(L, device=pool.device)
        out = pool[perm]  # (L,)
        pad = torch.full((k - L,), int(fallback), dtype=torch.long, device=pool.device)
        return torch.cat([out, pad], dim=0).to(device)

    def _corrupt_go_terms(self, pos_go: Set[int], n_draw: int) -> List[int]:
        """
        Fast corruption sampler:
          - draws n_draw distinct GO ids from go_universe excluding pos_go
          - avoids torch.randperm(|GO|) which is costly when |GO| is large.
        """
        n_draw = int(n_draw)
        if n_draw <= 0:
            return []
        U = int(self.go_universe.numel())
        if U == 0 or len(pos_go) >= U:
            return []

        # If n_draw is large relative to U, fallback to randperm
        if n_draw > max(64, U // 4):
            perm = torch.randperm(U)
            out: List[int] = []
            for idx in perm.tolist():
                g = int(self.go_universe[idx].item())
                if g in pos_go:
                    continue
                out.append(g)
                if len(out) >= n_draw:
                    break
            return out

        out_set: Set[int] = set()
        # rejection sampling in chunks
        while len(out_set) < n_draw:
            need = n_draw - len(out_set)
            m = min(max(need * 8, 64), 2048)  # chunk size
            idx = torch.randint(0, U, (m,), device=self.go_universe.device)
            gs = self.go_universe[idx].tolist()
            for g in gs:
                gi = int(g)
                if gi in pos_go:
                    continue
                out_set.add(gi)
                if len(out_set) >= n_draw:
                    break

            # Safety break (should rarely happen unless exclusion is huge)
            if m == 2048 and len(out_set) == 0:
                break

        return list(out_set)

    @staticmethod
    def _cat_unique(tensors: List[Tensor]) -> Tensor:
        """
        Concatenate CPU 1D tensors and unique them. Returns CPU 1D.
        """
        if not tensors:
            return torch.empty(0, dtype=torch.long)
        if len(tensors) == 1:
            return tensors[0]
        return torch.unique(torch.cat(tensors, dim=0), sorted=False)

    @staticmethod
    def _group_unique_by_key(key: Tensor, val: Tensor) -> Dict[int, Tensor]:
        """
        Build dict: key_value -> unique(val) for that key.
        Groups by sorting key (loops over number of unique keys, not over edges).
        key and val must be CPU 1D tensors of same length.
        """
        out: Dict[int, Tensor] = {}
        if key.numel() == 0:
            return out

        perm = torch.argsort(key)
        key_s = key[perm]
        val_s = val[perm]

        # boundaries where key changes
        diff = key_s[1:] != key_s[:-1]
        cuts = torch.nonzero(diff, as_tuple=False).flatten().tolist()
        # segment starts/ends
        starts = [0] + [c + 1 for c in cuts]
        ends = [c + 1 for c in cuts] + [int(key_s.numel())]

        for s, e in zip(starts, ends):
            k = int(key_s[s].item())
            seg = val_s[s:e]
            out[k] = torch.unique(seg, sorted=False)

        return out

    def _build_pool_from_go_list(self, go_list: List[int], table: Dict[int, Tensor], exclude_u: int) -> Tensor:
        """
        Union proteins across GO terms in go_list, unique, remove exclude_u.
        Returns CPU tensor.
        """
        tensors = [table[go] for go in go_list if go in table]
        pool = self._cat_unique(tensors)
        if pool.numel() > 0:
            pool = pool[pool != exclude_u]
        return pool

    def _get_or_build_shared_pos_pool(self, u: int) -> Tensor:
        cached = self._cache_shared_pos[u]
        if cached is not None:
            return cached
        go_list = self.pos_go_by_protein[u]
        pool = self._build_pool_from_go_list(go_list, self.pos_prots_by_go, exclude_u=u)
        self._cache_shared_pos[u] = pool
        return pool

    def _get_or_build_neg_to_u_pos_pool(self, u: int) -> Tensor:
        cached = self._cache_neg_to_u_pos[u]
        if cached is not None:
            return cached
        go_list = self.pos_go_by_protein[u]
        pool = self._build_pool_from_go_list(go_list, self.neg_prots_by_go, exclude_u=u)
        self._cache_neg_to_u_pos[u] = pool
        return pool

    # -------------------------
    # Public API used by Trainer
    # -------------------------
    def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None):
        """
        Build global lookup structures from the full graph.
        `negatives` ignored (compat signature).
        """

        # Determine protein node type from POS statement edge type
        try:
            pos_key = _find_key_by_rel(full_g, self.pos_etype)
            self.protein_ntype = pos_key[0]
        except Exception:
            self.protein_ntype = next(iter(full_g.node_types))
            pos_key = None

        assert self.protein_ntype is not None
        self.N = int(_num_nodes_of(full_g, self.protein_ntype))
        N = self.N

        pos_go_by_protein: List[List[int]] = [[] for _ in range(N)]
        pos_go_set_by_protein: List[Optional[Set[int]]] = [None for _ in range(N)]
        go_nodes: Set[int] = set()

        # --- POS edges ---
        pos_prots_by_go: Dict[int, Tensor] = {}
        if pos_key is not None and "edge_index" in full_g[pos_key] and full_g[pos_key].edge_index.numel() > 0:
            src_p, dst_p = full_g[pos_key].edge_index
            src_p = src_p.detach().cpu()
            dst_p = dst_p.detach().cpu()

            # protein -> unique GO list (group by src)
            prot_to_go = self._group_unique_by_key(src_p, dst_p)
            for u, gos in prot_to_go.items():
                glist = gos.tolist()
                pos_go_by_protein[u] = glist
                pos_go_set_by_protein[u] = set(glist)

            # GO -> unique proteins (group by dst)
            pos_prots_by_go = self._group_unique_by_key(dst_p, src_p)

            # Update GO universe
            go_nodes.update(torch.unique(dst_p, sorted=False).tolist())

        # --- NEG edges ---
        neg_prots_by_go: Dict[int, Tensor] = {}
        try:
            neg_key = _find_key_by_rel(full_g, self.neg_etype)
        except Exception:
            neg_key = None

        if neg_key is not None and "edge_index" in full_g[neg_key] and full_g[neg_key].edge_index.numel() > 0:
            src_n, dst_n = full_g[neg_key].edge_index
            src_n = src_n.detach().cpu()
            dst_n = dst_n.detach().cpu()

            # GO -> unique proteins (group by dst)
            neg_prots_by_go = self._group_unique_by_key(dst_n, src_n)

            go_nodes.update(torch.unique(dst_n, sorted=False).tolist())

        # --- GO graph (only to widen universe; no expansion) ---
        try:
            go_key = _find_key_by_rel(full_g, self.go_etype)
        except Exception:
            go_key = None

        if go_key is not None and "edge_index" in full_g[go_key] and full_g[go_key].edge_index.numel() > 0:
            go_src, go_dst = full_g[go_key].edge_index
            go_src = go_src.detach().cpu()
            go_dst = go_dst.detach().cpu()
            go_nodes.update(torch.unique(go_src, sorted=False).tolist())
            go_nodes.update(torch.unique(go_dst, sorted=False).tolist())

        self.pos_go_by_protein = pos_go_by_protein
        self.pos_go_set_by_protein = pos_go_set_by_protein
        self.pos_prots_by_go = pos_prots_by_go
        self.neg_prots_by_go = neg_prots_by_go

        self.go_universe = (
            torch.tensor(sorted(go_nodes), dtype=torch.long)
            if go_nodes
            else torch.empty(0, dtype=torch.long)
        )

        # reset lazy caches
        self._cache_shared_pos = [None for _ in range(N)]
        self._cache_neg_to_u_pos = [None for _ in range(N)]

    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """
        Prepare per-anchor candidate pools for the current minibatch.
        Assumes local==global ids in your loader.
        """
        # device bookkeeping
        if len(batch.edge_types) > 0:
            some_key = next(iter(batch.edge_types))
            self.device = (
                batch[some_key].edge_index.device
                if "edge_index" in batch[some_key]
                else torch.device("cpu")
            )
        else:
            self.device = torch.device("cpu")

        # anchors from current BCE batch if provided, else from PPI edges
        if pos_edge_index is not None and pos_edge_index.numel() > 0:
            anchors_t = torch.unique(pos_edge_index.reshape(-1).to("cpu"), sorted=False)
        else:
            try:
                anchor_key = _find_key_by_rel(batch, self.anchor_etype)
                if "edge_index" in batch[anchor_key] and batch[anchor_key].edge_index.numel() > 0:
                    anchors_t = torch.unique(batch[anchor_key].edge_index.reshape(-1).to("cpu"), sorted=False)
                else:
                    anchors_t = torch.empty(0, dtype=torch.long)
            except Exception:
                anchors_t = torch.empty(0, dtype=torch.long)

        assert self.protein_ntype is not None
        zN = int(_num_nodes_of(batch, self.protein_ntype))
        anchors = [int(u) for u in anchors_t.tolist() if 0 <= int(u) < zN]

        self.batch_anchors_local = anchors
        self.Nb = len(anchors)

        self.batch_pool_shared_neg = []
        self.batch_pool_pos_to_u_neg = []
        self.batch_pool_neg_to_u_pos = []
        self.batch_pool_shared_pos = []

        for u in anchors:
            pos_go_set = self._get_pos_set(u)

            # number of corrupted neg GO terms to draw (default: same as #pos terms)
            n_draw = len(pos_go_set)
            if self.max_corrupt_go is not None:
                n_draw = min(n_draw, self.max_corrupt_go)

            corrupt = self._corrupt_go_terms(pos_go_set, n_draw)
            # IMPORTANT: no expansion for generated negatives
            neg_go_list = corrupt  # already distinct

            # shared_neg = proteins with NEG to neg_go_list
            neg_tensors = [self.neg_prots_by_go[g] for g in neg_go_list if g in self.neg_prots_by_go]
            cand_neg = self._cat_unique(neg_tensors)
            if cand_neg.numel() > 0:
                cand_neg = cand_neg[cand_neg != u]
            self.batch_pool_shared_neg.append(cand_neg)

            # pos_to_u_neg = proteins with POS to neg_go_list
            pos_tensors = [self.pos_prots_by_go[g] for g in neg_go_list if g in self.pos_prots_by_go]
            cand_pos_to_neg = self._cat_unique(pos_tensors)
            if cand_pos_to_neg.numel() > 0:
                cand_pos_to_neg = cand_pos_to_neg[cand_pos_to_neg != u]
            self.batch_pool_pos_to_u_neg.append(cand_pos_to_neg)

            # These two are constant wrt random corruption => use lazy caches:
            self.batch_pool_neg_to_u_pos.append(self._get_or_build_neg_to_u_pos_pool(u))
            self.batch_pool_shared_pos.append(self._get_or_build_shared_pos_pool(u))

    def sample(self) -> Tensor:
        """Trainer compatibility: this sampler doesn't sample explicit edges."""
        return torch.empty(2, 0, dtype=torch.long, device=self.device)

    def get_contrastive_samples(
        self, z: Tensor, neg_statement_index: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = z.device
        k = self.k

        if self.Nb == 0:
            D = int(z.size(-1))
            empty1 = torch.empty(0, D, device=device)
            emptyk = torch.empty(0, k, D, device=device)
            return empty1, emptyk, emptyk, emptyk, emptyk

        anchors = self.batch_anchors_local
        anchors_t = torch.tensor(anchors, dtype=torch.long, device=device)
        z_anchor = z[anchors_t]  # (Nb, D)

        idx_shared_neg = torch.empty((self.Nb, k), dtype=torch.long, device=device)
        idx_pos_to_u_neg = torch.empty((self.Nb, k), dtype=torch.long, device=device)
        idx_neg_to_u_pos = torch.empty((self.Nb, k), dtype=torch.long, device=device)
        idx_shared_pos = torch.empty((self.Nb, k), dtype=torch.long, device=device)

        for i, u in enumerate(anchors):
            fallback = int(u)
            idx_shared_neg[i] = self._sample_k_tensor(self.batch_pool_shared_neg[i], k, fallback, device)
            idx_pos_to_u_neg[i] = self._sample_k_tensor(self.batch_pool_pos_to_u_neg[i], k, fallback, device)
            idx_neg_to_u_pos[i] = self._sample_k_tensor(self.batch_pool_neg_to_u_pos[i], k, fallback, device)
            idx_shared_pos[i] = self._sample_k_tensor(self.batch_pool_shared_pos[i], k, fallback, device)

        z_shared_neg = z[idx_shared_neg]    
        z_pos_to_u_neg = z[idx_pos_to_u_neg]
        z_neg_to_u_pos = z[idx_neg_to_u_pos]
        z_shared_pos = z[idx_shared_pos]    

        return z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos


# import torch
# from typing import Dict, List, Optional, Set, Tuple
# from torch import Tensor
# from torch_geometric.data import HeteroData
# from .utils import _find_key_by_rel, _num_nodes_of

# class RandomProteinSampler:
#     """
#     Random protein sampler.

#     Behaviour:
#       - For each anchor protein u, gets real POS statement GO classes.
#       - Generates NEG GO classes for u by random sampling excluding u's pos annotations.
#       - NO ontological expansion is performed for these generated negative statements.
#       - Uses:
#           * shared_pos: proteins with POS statements to u's POS GO classes.
#           * neg_to_u_pos: proteins with NEG statements to u's POS GO classes.
#           * shared_neg: proteins with NEG statements to u's generated NEG GO classes.
#           * pos_to_u_neg: proteins with POS statements to u's generated NEG GO classes.

#     Fallback behaviour (stable shapes, ~0-grad when padded with anchor):
#       - If group pool is empty -> pads with anchor itself (k times).
#       - If 0 < pool_size < k -> uses each candidate once (shuffled) then pads with anchor.
#       - If pool_size >= k -> samples k unique candidates (no replacement).
#     """

#     def __init__(
#         self,
#         k: int = 1,
#         anchor_etype: str = "PPI",
#         pos_etype: str = "pos_statement",
#         neg_etype: str = "neg_statement",
#         go_etype: str = "go",
#         neg_go_hops: int = 2,  # kept for compatibility, not used (no expansion)
#         max_corrupt_go: Optional[int] = None,
#     ):
#         self.k = int(k)
#         self.anchor_etype = anchor_etype
#         self.pos_etype = pos_etype
#         self.neg_etype = neg_etype
#         self.go_etype = go_etype
#         self.neg_go_hops = int(neg_go_hops)  # unused now
#         self.max_corrupt_go = None if max_corrupt_go is None else int(max_corrupt_go)

#         self.device = torch.device("cpu")

#         # Global
#         self.N: int = 0
#         self.pos_go_by_protein: List[List[int]] = []                 # list-of-lists (unique)
#         self.pos_go_set_by_protein: List[Optional[Set[int]]] = []    # cached sets (None => empty)
#         self.go_universe: Tensor = torch.empty(0, dtype=torch.long)  # CPU universe of GO node ids

#         # GO -> proteins tensors (CPU, unique)
#         self.pos_prots_by_go: Dict[int, Tensor] = {}
#         self.neg_prots_by_go: Dict[int, Tensor] = {}

#         # Batch
#         self.batch_anchors_local: List[int] = []
#         self.Nb: int = 0
#         self.batch_pool_shared_neg: List[Tensor] = []      # CPU 1D tensors
#         self.batch_pool_pos_to_u_neg: List[Tensor] = []
#         self.batch_pool_neg_to_u_pos: List[Tensor] = []
#         self.batch_pool_shared_pos: List[Tensor] = []

#     # -------------------------
#     # Small helpers
#     # -------------------------
#     @staticmethod
#     def _dedup_list(xs: List[int]) -> List[int]:
#         seen = set()
#         out: List[int] = []
#         for x in xs:
#             if x not in seen:
#                 seen.add(x)
#                 out.append(x)
#         return out

#     def _get_pos_set(self, u: int) -> Set[int]:
#         s = self.pos_go_set_by_protein[u]
#         return s if s is not None else set()

#     def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
#         """
#         pool: 1D CPU tensor
#         returns: (k,) tensor on `device`

#         Fallback rules:
#           - empty pool -> [fallback]*k
#           - 0 < L < k -> shuffle all L once, then pad with fallback
#           - L >= k -> sample k unique
#         """
#         L = int(pool.numel())
#         if L == 0:
#             return torch.full((k,), int(fallback), dtype=torch.long, device=device)

#         if L >= k:
#             idx = torch.randperm(L, device=pool.device)[:k]
#             return pool[idx].to(device)

#         perm = torch.randperm(L, device=pool.device)
#         out = pool[perm]  # (L,)
#         pad = torch.full((k - L,), int(fallback), dtype=torch.long, device=pool.device)
#         return torch.cat([out, pad], dim=0).to(device)

#     def _corrupt_go_terms(self, pos_go: Set[int], n_draw: int) -> List[int]:
#         """
#         Draw n_draw distinct GO ids from self.go_universe excluding pos_go.
#         """
#         n_draw = int(n_draw)
#         if n_draw <= 0:
#             return []
#         U = int(self.go_universe.numel())
#         if U == 0 or len(pos_go) >= U:
#             return []

#         perm = torch.randperm(U)
#         out: List[int] = []
#         for idx in perm.tolist():
#             g = int(self.go_universe[idx].item())
#             if g in pos_go:
#                 continue
#             out.append(g)
#             if len(out) >= n_draw:
#                 break
#         return out

#     @staticmethod
#     def _unique_cat_tensors(tensors: List[Tensor]) -> Tensor:
#         """
#         Concatenate CPU 1D tensors and unique them. Returns CPU 1D.
#         """
#         if not tensors:
#             return torch.empty(0, dtype=torch.long)
#         if len(tensors) == 1:
#             return tensors[0]
#         return torch.unique(torch.cat(tensors, dim=0))

#     # -------------------------
#     # Public API used by Trainer
#     # -------------------------
#     def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None):
#         """
#         Build global lookup structures from the full graph.
#         `negatives` ignored (compat signature).
#         """
#         try:
#             pos_key = _find_key_by_rel(full_g, self.pos_etype)
#             node_type = pos_key[0]
#         except Exception:
#             node_type = next(iter(full_g.node_types))
#             pos_key = None

#         self.N = int(_num_nodes_of(full_g, node_type))
#         N = self.N

#         # POS statements: protein -> GO, and GO -> proteins
#         pos_go_by_protein: List[List[int]] = [[] for _ in range(N)]
#         pos_prots_by_go_list: Dict[int, List[int]] = {}
#         go_nodes: Set[int] = set()

#         if pos_key is not None and "edge_index" in full_g[pos_key] and full_g[pos_key].edge_index.numel() > 0:
#             src_p, dst_p = full_g[pos_key].edge_index
#             for u, go in zip(src_p.tolist(), dst_p.tolist()):
#                 pos_go_by_protein[u].append(go)
#                 pos_prots_by_go_list.setdefault(go, []).append(u)
#                 go_nodes.add(go)

#         # NEG statements: GO -> proteins
#         neg_prots_by_go_list: Dict[int, List[int]] = {}
#         try:
#             neg_key = _find_key_by_rel(full_g, self.neg_etype)
#         except Exception:
#             neg_key = None

#         if neg_key is not None and "edge_index" in full_g[neg_key] and full_g[neg_key].edge_index.numel() > 0:
#             src_n, dst_n = full_g[neg_key].edge_index
#             for u, go in zip(src_n.tolist(), dst_n.tolist()):
#                 neg_prots_by_go_list.setdefault(go, []).append(u)
#                 go_nodes.add(go)

#         # Optional: include GO nodes from the GO graph edges to widen corruption universe,
#         # but we DO NOT compute/store predecessors or do expansion.
#         try:
#             go_key = _find_key_by_rel(full_g, self.go_etype)
#         except Exception:
#             go_key = None

#         if go_key is not None and "edge_index" in full_g[go_key] and full_g[go_key].edge_index.numel() > 0:
#             go_src, go_dst = full_g[go_key].edge_index
#             go_nodes.update(go_src.tolist())
#             go_nodes.update(go_dst.tolist())

#         # Dedup + cache sets per protein
#         pos_go_set_by_protein: List[Optional[Set[int]]] = [None for _ in range(N)]
#         for u in range(N):
#             if pos_go_by_protein[u]:
#                 uniq = self._dedup_list(pos_go_by_protein[u])
#                 pos_go_by_protein[u] = uniq
#                 pos_go_set_by_protein[u] = set(uniq)

#         # Convert GO->protein lists to unique CPU tensors
#         pos_prots_by_go: Dict[int, Tensor] = {}
#         for go, ps in pos_prots_by_go_list.items():
#             if ps:
#                 pos_prots_by_go[go] = torch.tensor(self._dedup_list(ps), dtype=torch.long)

#         neg_prots_by_go: Dict[int, Tensor] = {}
#         for go, ps in neg_prots_by_go_list.items():
#             if ps:
#                 neg_prots_by_go[go] = torch.tensor(self._dedup_list(ps), dtype=torch.long)

#         self.pos_go_by_protein = pos_go_by_protein
#         self.pos_go_set_by_protein = pos_go_set_by_protein
#         self.pos_prots_by_go = pos_prots_by_go
#         self.neg_prots_by_go = neg_prots_by_go

#         self.go_universe = (
#             torch.tensor(sorted(go_nodes), dtype=torch.long)
#             if go_nodes
#             else torch.empty(0, dtype=torch.long)
#         )

#     def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
#         """
#         Prepare per-anchor candidate pools for the current minibatch.
#         Assumes local==global ids in your loader.
#         """
#         some_key = next(iter(batch.edge_types))
#         self.device = batch[some_key].edge_index.device if batch[some_key].edge_index.is_cuda else torch.device("cpu")

#         # anchors from current BCE batch if provided, else from PPI edges
#         if pos_edge_index is not None and pos_edge_index.numel() > 0:
#             anchors_t = torch.unique(pos_edge_index.reshape(-1)).to("cpu")
#         else:
#             try:
#                 anchor_key = _find_key_by_rel(batch, self.anchor_etype)
#                 if "edge_index" in batch[anchor_key] and batch[anchor_key].edge_index.numel() > 0:
#                     anchors_t = torch.unique(batch[anchor_key].edge_index.reshape(-1)).to("cpu")
#                 else:
#                     anchors_t = torch.empty(0, dtype=torch.long)
#             except Exception:
#                 anchors_t = torch.empty(0, dtype=torch.long)

#         zN = int(getattr(batch["node"], "num_nodes", 0) or _num_nodes_of(batch, "node"))
#         anchors = [int(u) for u in anchors_t.tolist() if 0 <= int(u) < zN]

#         self.batch_anchors_local = anchors
#         self.Nb = len(anchors)

#         self.batch_pool_shared_neg = []
#         self.batch_pool_pos_to_u_neg = []
#         self.batch_pool_neg_to_u_pos = []
#         self.batch_pool_shared_pos = []

#         for u in anchors:
#             pos_go_set = self._get_pos_set(u)
#             pos_go_list = self.pos_go_by_protein[u] if 0 <= u < len(self.pos_go_by_protein) else []

#             # number of corrupted neg GO terms to draw (default: same as #pos terms)
#             n_draw = len(pos_go_set)
#             if self.max_corrupt_go is not None:
#                 n_draw = min(n_draw, self.max_corrupt_go)

#             corrupt = self._corrupt_go_terms(pos_go_set, n_draw)

#             # IMPORTANT: no expansion for generated negatives
#             neg_go_set = set(corrupt)

#             # shared_neg = proteins with NEG to neg_go_set
#             neg_tensors = [self.neg_prots_by_go[go] for go in neg_go_set if go in self.neg_prots_by_go]
#             cand_neg = self._unique_cat_tensors(neg_tensors)
#             if cand_neg.numel() > 0:
#                 cand_neg = cand_neg[cand_neg != u]
#             self.batch_pool_shared_neg.append(cand_neg)

#             # pos_to_u_neg = proteins with POS to neg_go_set
#             pos_tensors = [self.pos_prots_by_go[go] for go in neg_go_set if go in self.pos_prots_by_go]
#             cand_pos_to_neg = self._unique_cat_tensors(pos_tensors)
#             if cand_pos_to_neg.numel() > 0:
#                 cand_pos_to_neg = cand_pos_to_neg[cand_pos_to_neg != u]
#             self.batch_pool_pos_to_u_neg.append(cand_pos_to_neg)

#             # neg_to_u_pos = proteins with NEG to pos_go_list (real positives)
#             neg_to_pos_tensors = [self.neg_prots_by_go[go] for go in pos_go_list if go in self.neg_prots_by_go]
#             cand_neg_to_pos = self._unique_cat_tensors(neg_to_pos_tensors)
#             if cand_neg_to_pos.numel() > 0:
#                 cand_neg_to_pos = cand_neg_to_pos[cand_neg_to_pos != u]
#             self.batch_pool_neg_to_u_pos.append(cand_neg_to_pos)

#             # shared_pos = proteins with POS to pos_go_list (real positives)
#             shared_pos_tensors = [self.pos_prots_by_go[go] for go in pos_go_list if go in self.pos_prots_by_go]
#             cand_shared_pos = self._unique_cat_tensors(shared_pos_tensors)
#             if cand_shared_pos.numel() > 0:
#                 cand_shared_pos = cand_shared_pos[cand_shared_pos != u]
#             self.batch_pool_shared_pos.append(cand_shared_pos)

#     def sample(self) -> Tensor:
#         """Trainer compatibility: this sampler doesn't sample explicit edges."""
#         return torch.empty(2, 0, dtype=torch.long, device=self.device)

#     def get_contrastive_samples(
#         self, z: Tensor, neg_statement_index: Optional[Tensor] = None
#     ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#         device = z.device
#         k = self.k

#         if self.Nb == 0:
#             D = int(z.size(-1))
#             empty1 = torch.empty(0, D, device=device)
#             emptyk = torch.empty(0, k, D, device=device)
#             return empty1, emptyk, emptyk, emptyk, emptyk

#         anchors = self.batch_anchors_local
#         anchors_t = torch.tensor(anchors, dtype=torch.long, device=device)
#         z_anchor = z[anchors_t]  # (Nb, D)

#         idx_shared_neg = torch.empty((self.Nb, k), dtype=torch.long, device=device)
#         idx_pos_to_u_neg = torch.empty((self.Nb, k), dtype=torch.long, device=device)
#         idx_neg_to_u_pos = torch.empty((self.Nb, k), dtype=torch.long, device=device)
#         idx_shared_pos = torch.empty((self.Nb, k), dtype=torch.long, device=device)

#         for i, u in enumerate(anchors):
#             fallback = int(u)
#             idx_shared_neg[i] = self._sample_k_tensor(self.batch_pool_shared_neg[i], k, fallback, device)
#             idx_pos_to_u_neg[i] = self._sample_k_tensor(self.batch_pool_pos_to_u_neg[i], k, fallback, device)
#             idx_neg_to_u_pos[i] = self._sample_k_tensor(self.batch_pool_neg_to_u_pos[i], k, fallback, device)
#             idx_shared_pos[i] = self._sample_k_tensor(self.batch_pool_shared_pos[i], k, fallback, device)

#         z_shared_neg = z[idx_shared_neg]         # (Nb, k, D)
#         z_pos_to_u_neg = z[idx_pos_to_u_neg]     # (Nb, k, D)
#         z_neg_to_u_pos = z[idx_neg_to_u_pos]     # (Nb, k, D)
#         z_shared_pos = z[idx_shared_pos]         # (Nb, k, D)

#         return z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos
