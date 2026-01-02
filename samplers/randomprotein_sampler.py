import random
from typing import Dict, List, Optional, Set

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from .utils import _find_key_by_rel, _num_nodes_of


class RandomProteinSampler:
    """
    Random negative-statement sampler (SHGCN variant).

    Goal (matches the original framework's RandomProteinSampler logic):
      1) For each anchor protein u (endpoints of `anchor_etype` edges in the minibatch),
         read its real POS statement classes (GO terms) from `pos_etype`.
      2) Generate / corrupt NEG statement classes for u by randomly sampling GO terms from a
         global GO universe while EXCLUDING u's POS statement classes. (No GO expansion.)
      3) Using u's POS classes and generated NEG classes, build protein pools:

         - shared_pos:     proteins with POS statement to any GO in u.POS
         - neg_to_u_pos:   proteins with NEG statement to any GO in u.POS
         - shared_neg:     proteins with NEG statement to any GO in u.NEG(random)
         - pos_to_u_neg:   proteins with POS statement to any GO in u.NEG(random)

      4) Sampling fallback matches the original RandomProteinSampler:
         if a pool is empty / too small, pad with the anchor itself.

    Output (SHGCN / dual-view friendly):
      - `sample_batch()` returns indices compatible with dual_prot_contrastive.ProteinContrastiveLoss:
          anchors:           (B,)
          pos_same_neg:      (B,)  sampled from shared_neg (fallback=anchor)
          pos_same_pos:      (B,)  sampled from shared_pos (fallback=anchor)
          neg_pos_to_my_neg: (B,K) sampled from pos_to_u_neg (fallback=anchor)
          neg_neg_to_my_pos: (B,K) sampled from neg_to_u_pos (fallback=anchor)

      - `sample()` returns a local edge_index (2, B*k_stmt) for corrupted NEG statement edges
        (anchor -> corrupted GO term), which can be optionally injected into the batch graph
        under `neg_etype` to build the negative-view encoder input.
    """

    def __init__(
        self,
        k_neg: int = 2,
        k_stmt: Optional[int] = None,
        anchor_etype: str = "PPI",
        pos_etype: str = "pos_statement",
        neg_etype: str = "neg_statement",
        go_etype: str = "link",
        max_corrupt_go: Optional[int] = None,
    ):
        self.k_neg = int(k_neg)
        self.k_stmt = int(k_stmt) if k_stmt is not None else int(k_neg)
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.go_etype = go_etype
        self.max_corrupt_go = max_corrupt_go

        # Global lookups (CPU)
        self.protein_ntype: str = "node"
        self.N: int = 0
        self.pos_go_set_by_protein: List[Set[int]] = []
        self.pos_prots_by_go: Dict[int, Tensor] = {}
        self.neg_prots_by_go: Dict[int, Tensor] = {}
        self.go_universe: Tensor = torch.empty(0, dtype=torch.long)  # CPU

        # Batch state
        self.device = torch.device("cpu")
        self.batch_anchors_local: List[int] = []
        self.Nb: int = 0
        self.batch_corrupt_go: List[List[int]] = []  # per-anchor corrupted GO ids (local==global)

        # pools (CPU tensors)
        self.pool_shared_neg: List[Tensor] = []
        self.pool_pos_to_u_neg: List[Tensor] = []
        self.pool_neg_to_u_pos: List[Tensor] = []
        self.pool_shared_pos: List[Tensor] = []

    # -------------------------
    # Helpers (global, CPU)
    # -------------------------
    @staticmethod
    def _cat_unique(tensors: List[Tensor]) -> Tensor:
        if not tensors:
            return torch.empty(0, dtype=torch.long)
        if len(tensors) == 1:
            return tensors[0]
        return torch.unique(torch.cat(tensors, dim=0), sorted=False)

    @staticmethod
    def _group_unique_by_key(key: Tensor, val: Tensor) -> Dict[int, Tensor]:
        """
        Build dict: key_value -> unique(val) (CPU tensors), grouped by sorting key.
        """
        if key.numel() == 0:
            return {}
        key = key.detach().cpu()
        val = val.detach().cpu()
        order = torch.argsort(key)
        key_s = key[order]
        val_s = val[order]
        uniq, counts = torch.unique_consecutive(key_s, return_counts=True)
        out: Dict[int, Tensor] = {}
        start = 0
        for k, c in zip(uniq.tolist(), counts.tolist()):
            chunk = val_s[start : start + c]
            out[int(k)] = torch.unique(chunk, sorted=False)
            start += c
        return out

    def _corrupt_go_terms(self, pos_go: Set[int], n_draw: int) -> List[int]:
        """
        Sample n_draw DISTINCT GO ids from self.go_universe excluding pos_go.
        Uses a fast rejection sampler (mirrors original RandomProteinSampler).
        """
        n_draw = int(n_draw)
        if n_draw <= 0:
            return []
        U = int(self.go_universe.numel())
        if U == 0 or len(pos_go) >= U:
            return []

        # If n_draw large, fallback to randperm
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
            m = min(max(need * 8, 64), 2048)
            idx = torch.randint(0, U, (m,), device=self.go_universe.device)
            gs = self.go_universe[idx].tolist()
            for g in gs:
                gi = int(g)
                if gi in pos_go:
                    continue
                out_set.add(gi)
                if len(out_set) >= n_draw:
                    break
            # safety break
            if m == 2048 and len(out_set) == 0:
                break
        return list(out_set)

    def _build_pool_from_go_list(self, go_list: List[int], table: Dict[int, Tensor], exclude_u: int) -> Tensor:
        """
        Union proteins across GO terms in go_list, unique, remove exclude_u.
        Returns CPU tensor.
        """
        tensors = [table[g] for g in go_list if g in table]
        pool = self._cat_unique(tensors)
        if pool.numel() > 0:
            pool = pool[pool != exclude_u]
        return pool

    def _sample_one(self, pool: Tensor, fallback: int) -> int:
        """Sample 1 element from CPU pool (uniform) or return fallback if empty."""
        L = int(pool.numel())
        if L <= 0:
            return int(fallback)
        idx = int(torch.randint(0, L, (1,), device=pool.device).item())
        return int(pool[idx].item())

    def _sample_k(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
        """
        pool: 1D CPU tensor
        returns: (k,) tensor on `device` with fallback padding (anchor itself).
        """
        k = int(k)
        L = int(pool.numel())
        if L == 0:
            return torch.full((k,), int(fallback), dtype=torch.long, device=device)
        if L >= k:
            perm = torch.randperm(L)[:k]
            return pool[perm].to(device)
        # 0 < L < k: shuffle all once then pad
        perm = torch.randperm(L)
        out = pool[perm]
        pad = torch.full((k - L,), int(fallback), dtype=torch.long)
        return torch.cat([out, pad], dim=0).to(device)

    # -------------------------
    # Public API
    # -------------------------
    def prepare_global(self, full_g: HeteroData):
        """Build global lookup tables from the full graph (CPU)."""
        # identify node type and count
        pos_key = _find_key_by_rel(full_g, self.pos_etype)
        self.protein_ntype = pos_key[0]
        self.N = _num_nodes_of(full_g, self.protein_ntype)

        # POS statements: proteins -> GO
        pos_go_set_by_protein: List[Set[int]] = [set() for _ in range(self.N)]
        pos_prots_by_go: Dict[int, Tensor] = {}
        go_nodes: Set[int] = set()

        if "edge_index" in full_g[pos_key] and full_g[pos_key].edge_index.numel() > 0:
            src_pos, dst_pos = full_g[pos_key].edge_index
            src_pos = src_pos.detach().cpu()
            dst_pos = dst_pos.detach().cpu()
            go_nodes.update(torch.unique(dst_pos, sorted=False).tolist())

            # per protein: pos set
            for u, g in zip(src_pos.tolist(), dst_pos.tolist()):
                pos_go_set_by_protein[int(u)].add(int(g))

            # per GO: proteins with POS
            pos_prots_by_go = self._group_unique_by_key(dst_pos, src_pos)

        # NEG statements: proteins -> GO
        neg_prots_by_go: Dict[int, Tensor] = {}
        neg_key = _find_key_by_rel(full_g, self.neg_etype)
        if "edge_index" in full_g[neg_key] and full_g[neg_key].edge_index.numel() > 0:
            src_neg, dst_neg = full_g[neg_key].edge_index
            src_neg = src_neg.detach().cpu()
            dst_neg = dst_neg.detach().cpu()
            go_nodes.update(torch.unique(dst_neg, sorted=False).tolist())

            neg_prots_by_go = self._group_unique_by_key(dst_neg, src_neg)

        # optional GO graph nodes (for a richer universe)
        go_key = None
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

        self.pos_go_set_by_protein = pos_go_set_by_protein
        self.pos_prots_by_go = pos_prots_by_go
        self.neg_prots_by_go = neg_prots_by_go
        self.go_universe = (
            torch.tensor(sorted(go_nodes), dtype=torch.long)
            if go_nodes
            else torch.empty(0, dtype=torch.long)
        )

    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """
        Prepare per-anchor corruption + pools for the current minibatch.
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

        # anchors from PPI pos edges (recommended), else from adjacency
        if pos_edge_index is None:
            anchor_key = _find_key_by_rel(batch, self.anchor_etype)
            src_a, dst_a = batch[anchor_key].edge_index
            pos_edge_index = torch.stack([src_a, dst_a], dim=0)

        src, dst = pos_edge_index
        anchors = torch.unique(torch.cat([src, dst], dim=0)).tolist()
        self.batch_anchors_local = [int(a) for a in anchors]
        self.Nb = len(self.batch_anchors_local)

        self.batch_corrupt_go = []
        self.pool_shared_neg = []
        self.pool_pos_to_u_neg = []
        self.pool_neg_to_u_pos = []
        self.pool_shared_pos = []

        for u in self.batch_anchors_local:
            pos_go_set = self.pos_go_set_by_protein[u] if u < len(self.pos_go_set_by_protein) else set()

            n_draw = len(pos_go_set)
            if self.max_corrupt_go is not None:
                n_draw = min(n_draw, int(self.max_corrupt_go))

            corrupt = self._corrupt_go_terms(pos_go_set, n_draw)

            # If we couldn't draw enough (e.g., tiny universe), that's fine.
            self.batch_corrupt_go.append(corrupt)

            # pools derived from corrupted NEG and real POS
            shared_neg = self._build_pool_from_go_list(corrupt, self.neg_prots_by_go, exclude_u=u)
            pos_to_u_neg = self._build_pool_from_go_list(corrupt, self.pos_prots_by_go, exclude_u=u)

            pos_go_list = list(pos_go_set)
            neg_to_u_pos = self._build_pool_from_go_list(pos_go_list, self.neg_prots_by_go, exclude_u=u)
            shared_pos = self._build_pool_from_go_list(pos_go_list, self.pos_prots_by_go, exclude_u=u)

            self.pool_shared_neg.append(shared_neg)
            self.pool_pos_to_u_neg.append(pos_to_u_neg)
            self.pool_neg_to_u_pos.append(neg_to_u_pos)
            self.pool_shared_pos.append(shared_pos)

    def sample(self) -> Tensor:
        """
        Sample corrupted NEG statement edges (anchor -> GO) for the current batch.
        Returns local edge_index (2, B*k_stmt) on the batch device.
        """
        device = self.device
        if self.Nb == 0:
            return torch.empty(2, 0, dtype=torch.long, device=device)

        src_list: List[int] = []
        dst_list: List[int] = []
        for u, corrupt in zip(self.batch_anchors_local, self.batch_corrupt_go):
            # If no corrupted candidates, try to draw a few directly from the universe excluding pos.
            cands = corrupt
            if not cands and self.go_universe.numel() > 0:
                pos_go_set = self.pos_go_set_by_protein[u] if u < len(self.pos_go_set_by_protein) else set()
                cands = self._corrupt_go_terms(pos_go_set, max(1, self.k_stmt))

            if not cands:
                continue

            chosen = random.choices(cands, k=self.k_stmt)
            src_list.extend([u] * self.k_stmt)
            dst_list.extend(chosen)

        if not src_list:
            return torch.empty(2, 0, dtype=torch.long, device=device)
        return torch.tensor([src_list, dst_list], dtype=torch.long, device=device)

    def sample_batch(self) -> Dict[str, Tensor]:
        """
        Return indices for dual-view protein contrastive loss.
        Uses fallback=anchor for any empty/short pool (mirrors RandomProteinSampler fallback).
        """
        device = self.device
        B = self.Nb
        if B == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return {
                "anchors": empty,
                "pos_same_neg": empty,
                "pos_same_pos": empty,
                "neg_pos_to_my_neg": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
                "neg_neg_to_my_pos": torch.empty(0, self.k_neg, dtype=torch.long, device=device),
            }

        anchors_t = torch.tensor(self.batch_anchors_local, dtype=torch.long, device=device)

        pos_same_neg = torch.empty(B, dtype=torch.long, device=device)
        pos_same_pos = torch.empty(B, dtype=torch.long, device=device)
        neg_pos_to_my_neg = torch.empty((B, self.k_neg), dtype=torch.long, device=device)
        neg_neg_to_my_pos = torch.empty((B, self.k_neg), dtype=torch.long, device=device)

        for i, u in enumerate(self.batch_anchors_local):
            fallback = int(u)
            pos_same_neg[i] = self._sample_one(self.pool_shared_neg[i], fallback)
            pos_same_pos[i] = self._sample_one(self.pool_shared_pos[i], fallback)
            neg_pos_to_my_neg[i] = self._sample_k(self.pool_pos_to_u_neg[i], self.k_neg, fallback, device)
            neg_neg_to_my_pos[i] = self._sample_k(self.pool_neg_to_u_pos[i], self.k_neg, fallback, device)

        return {
            "anchors": anchors_t,
            "pos_same_neg": pos_same_neg,
            "pos_same_pos": pos_same_pos,
            "neg_pos_to_my_neg": neg_pos_to_my_neg,
            "neg_neg_to_my_pos": neg_neg_to_my_pos,
        }

    def get_dual_contrastive_samples(
        self, z_pos: Tensor, z_neg: Tensor
    ):
        """
        Convenience: return the *views* (embeddings) required for a dual-view contrastive loss.

        Returns:
          anchor_neg_view: (B,D)          z_neg[anchors]
          pos_same_neg:    (B,D)          z_neg[pos_same_neg]
          neg_for_neg:     (B,K,D)        z_pos[neg_pos_to_my_neg]
          anchor_pos_view: (B,D)          z_pos[anchors]
          pos_same_pos:    (B,D)          z_pos[pos_same_pos]
          neg_for_pos:     (B,K,D)        z_neg[neg_neg_to_my_pos]
        """
        samples = self.sample_batch()
        anchors = samples["anchors"]
        if anchors.numel() == 0:
            D = int(z_pos.size(-1))
            empty = z_pos.new_empty((0, D))
            emptyk = z_pos.new_empty((0, self.k_neg, D))
            return empty, empty, emptyk, empty, empty, emptyk

        pos_same_neg = samples["pos_same_neg"]
        pos_same_pos = samples["pos_same_pos"]
        neg_pos_to_my_neg = samples["neg_pos_to_my_neg"]
        neg_neg_to_my_pos = samples["neg_neg_to_my_pos"]

        a_neg = z_neg[anchors]
        p_neg = z_neg[pos_same_neg]
        n_for_neg = z_pos[neg_pos_to_my_neg]  # (B,K,D)

        a_pos = z_pos[anchors]
        p_pos = z_pos[pos_same_pos]
        n_for_pos = z_neg[neg_neg_to_my_pos]  # (B,K,D)

        return a_neg, p_neg, n_for_neg, a_pos, p_pos, n_for_pos
