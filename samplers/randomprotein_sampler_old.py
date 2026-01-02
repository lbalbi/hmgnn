import torch
from typing import Dict, List, Optional, Set, Tuple
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class RandomProteinSampler:
    """   Random protein sampler. Intended behaviour:
      - For each anchor protein u, gets real POS statement GO classes.
      - Generates NEG GO classes for u by random sampling random excluding u's pos annotations.
      - Performs ontological expansion on the synthesized NEG set.
      - Use:
          * shared_pos: proteins with POS statements to u's POS GO classes.
          * neg_to_u_pos: proteins with NEG statements to u's POS GO classes.
          * shared_neg: proteins with NEG statements to u's synthesized/expanded NEG GO classes.
          * pos_to_u_neg: proteins with POS statements to u's synthesized/expanded NEG GO classes.
    Fallback behaviour:
      - If group pool is empty -> pads with the anchor itself (no grad change).
      - If 0 < pool_size < k -> use all candidates once, then pad with anchor """

    def __init__(self, k: int = 1, anchor_etype: str = "PPI", pos_etype: str = "pos_statement",
        neg_etype: str = "neg_statement", go_etype: str = "go", neg_go_hops: int = 2,
        max_corrupt_go: Optional[int] = None):
        self.k = int(k)
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.go_etype = go_etype
        self.neg_go_hops = int(neg_go_hops)
        self.max_corrupt_go = None if max_corrupt_go is None else int(max_corrupt_go)
        self.device = torch.device("cpu")
        self.N: int = 0
        self.proteins_global: Set[int] = set()
        self.pos_go_by_protein: List[List[int]] = []
        self.proteins_with_pos_to_go: Dict[int, List[int]] = {}
        self.proteins_with_neg_to_go: Dict[int, List[int]] = {}
        self.predecessors: Dict[int, List[int]] = {}
        self.go_universe: Tensor = torch.empty(0, dtype=torch.long)
        self.batch_anchors_global: List[int] = []
        self.batch_anchors_local: List[int] = []
        self.Nb: int = 0
        self.batch_pool_shared_neg: List[Tensor] = []  
        self.batch_pool_pos_to_u_neg: List[Tensor] = []
        self.batch_pool_neg_to_u_pos: List[Tensor] = []
        self.batch_pool_shared_pos: List[Tensor] = []  

    @staticmethod
    def _dedup_list(xs: List[int]) -> List[int]:
        seen = set()
        out: List[int] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _expand_go_set(self, seeds: Set[int], hops: int) -> Set[int]:
        out = set(seeds)
        frontier = set(seeds)
        for _ in range(max(0, int(hops))):
            nxt: Set[int] = set()
            for go in frontier:
                nxt.update(self.predecessors.get(go, []))
            nxt -= out
            if not nxt: break
            out |= nxt
            frontier = nxt
        return out

    def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
        """Sample up to k unique indices from a 1D CPU tensor pool; pad with fallback to keep shape stable."""
        L = int(pool.numel())
        if L == 0: return torch.full((k,), int(fallback), dtype=torch.long, device=device)
        if L >= k:
            idx = torch.randperm(L)[:k]
            return pool[idx].to(device)
        perm = torch.randperm(L)
        out = pool[perm]
        pad = torch.full((k - L,), int(fallback), dtype=torch.long)
        return torch.cat([out, pad], dim=0).to(device)

    def _corrupt_go_terms(self, pos_go: Set[int], n_draw: int) -> Set[int]:
        """Sample n_draw GO nodes from the GO universe excluding pos_go."""
        n_draw = int(n_draw)
        if n_draw <= 0 or self.go_universe.numel() == 0: return set()
        if len(pos_go) >= int(self.go_universe.numel()): return set()
        out: Set[int] = set()
        max_tries = max(1000, n_draw * 50)
        tries = 0
        U = int(self.go_universe.numel())

        while len(out) < n_draw and tries < max_tries:
            idx = int(torch.randint(0, U, (1,)).item())
            go = int(self.go_universe[idx].item())
            tries += 1
            if go in pos_go: continue
            out.add(go)

        if len(out) < n_draw:
            cand = [int(g) for g in self.go_universe.tolist() if int(g) not in pos_go]
            if cand:
                need = n_draw - len(out)
                perm = torch.randperm(len(cand))[: min(need, len(cand))]
                for j in perm.tolist():
                    out.add(cand[j])
        return out


    def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None):
        """Build global lookup structures from the full graph. `negatives` is ignored (compat signature)."""
        try:
            pos_key = _find_key_by_rel(full_g, self.pos_etype)
            node_type = pos_key[0]
        except Exception:
            node_type = next(iter(full_g.node_types))
            pos_key = None
        self.N = int(_num_nodes_of(full_g, node_type))
        N = self.N
        proteins: Set[int] = set()
        try:
            anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
            if "edge_index" in full_g[anchor_key] and full_g[anchor_key].edge_index.numel() > 0:
                a_src, a_dst = full_g[anchor_key].edge_index
                proteins |= set(a_src.tolist())
                proteins |= set(a_dst.tolist())
        except Exception: proteins = set(range(N))
        self.proteins_global = proteins
        pos_go_by_protein: List[List[int]] = [[] for _ in range(N)]
        proteins_with_pos_to_go: Dict[int, List[int]] = {}
        go_nodes: Set[int] = set()

        if pos_key is not None and "edge_index" in full_g[pos_key] and full_g[pos_key].edge_index.numel() > 0:
            src_p, dst_p = full_g[pos_key].edge_index
            for u, go in zip(src_p.tolist(), dst_p.tolist()):
                pos_go_by_protein[u].append(go)
                proteins_with_pos_to_go.setdefault(go, []).append(u)
                go_nodes.add(go)

        proteins_with_neg_to_go: Dict[int, List[int]] = {}
        try: neg_key = _find_key_by_rel(full_g, self.neg_etype)
        except Exception: neg_key = None

        if neg_key is not None and "edge_index" in full_g[neg_key] and full_g[neg_key].edge_index.numel() > 0:
            src_n, dst_n = full_g[neg_key].edge_index
            for u, go in zip(src_n.tolist(), dst_n.tolist()):
                proteins_with_neg_to_go.setdefault(go, []).append(u)
                go_nodes.add(go)

        predecessors: Dict[int, List[int]] = {}
        try: go_key = _find_key_by_rel(full_g, self.go_etype)
        except Exception: go_key = None
        if go_key is not None and "edge_index" in full_g[go_key] and full_g[go_key].edge_index.numel() > 0:
            go_src, go_dst = full_g[go_key].edge_index
            for s, d in zip(go_src.tolist(), go_dst.tolist()):
                predecessors.setdefault(d, []).append(s)
                go_nodes.add(s)
                go_nodes.add(d)

        for u in range(N):
            pos_go_by_protein[u] = self._dedup_list(pos_go_by_protein[u])
        for go, ps in list(proteins_with_pos_to_go.items()):
            proteins_with_pos_to_go[go] = self._dedup_list(ps)
        for go, ps in list(proteins_with_neg_to_go.items()):
            proteins_with_neg_to_go[go] = self._dedup_list(ps)
        self.pos_go_by_protein = pos_go_by_protein
        self.proteins_with_pos_to_go = proteins_with_pos_to_go
        self.proteins_with_neg_to_go = proteins_with_neg_to_go
        self.predecessors = predecessors
        self.go_universe = torch.tensor(sorted(go_nodes), dtype=torch.long) if go_nodes else torch.empty(0, dtype=torch.long)

    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """Prepare per-anchor candidate pools for the current minibatch. Assumes local==global ids in your loader."""
        some_key = next(iter(batch.edge_types))
        self.device = (batch[some_key].edge_index.device
            if batch[some_key].edge_index.is_cuda else torch.device("cpu"))
        if pos_edge_index is not None and pos_edge_index.numel() > 0:
            anchors_t = torch.unique(pos_edge_index.reshape(-1)).to("cpu")
        else:
            try:
                anchor_key = _find_key_by_rel(batch, self.anchor_etype)
                if "edge_index" in batch[anchor_key] and batch[anchor_key].edge_index.numel() > 0:
                    anchors_t = torch.unique(batch[anchor_key].edge_index.reshape(-1)).to("cpu")
                else: anchors_t = torch.empty(0, dtype=torch.long)
            except Exception: anchors_t = torch.empty(0, dtype=torch.long)
        zN = int(getattr(batch["node"], "num_nodes", 0) or _num_nodes_of(batch, "node"))
        anchors = [int(u) for u in anchors_t.tolist() if 0 <= int(u) < zN]

        self.batch_anchors_global = anchors
        self.batch_anchors_local = anchors
        self.Nb = len(anchors)
        self.batch_pool_shared_neg = []
        self.batch_pool_pos_to_u_neg = []
        self.batch_pool_neg_to_u_pos = []
        self.batch_pool_shared_pos = []

        for u in anchors:
            pos_go = set(self.pos_go_by_protein[u]) if (0 <= u < len(self.pos_go_by_protein)) else set()
            n_draw = len(pos_go)
            if self.max_corrupt_go is not None: n_draw = min(n_draw, self.max_corrupt_go)
            corrupt_neg = self._corrupt_go_terms(pos_go, n_draw)
            ext_neg = self._expand_go_set(corrupt_neg, self.neg_go_hops) if corrupt_neg else set()
            cand: Set[int] = set()
            for go in ext_neg:
                cand.update(self.proteins_with_neg_to_go.get(go, []))
            cand.discard(u)
            self.batch_pool_shared_neg.append(
                torch.tensor(list(cand), dtype=torch.long) if cand else torch.empty(0, dtype=torch.long))
            cand = set()
            for go in ext_neg:
                cand.update(self.proteins_with_pos_to_go.get(go, []))
            cand.discard(u)
            self.batch_pool_pos_to_u_neg.append(
                torch.tensor(list(cand), dtype=torch.long) if cand else torch.empty(0, dtype=torch.long))
            cand = set()
            for go in pos_go:
                cand.update(self.proteins_with_neg_to_go.get(go, []))
            cand.discard(u)
            self.batch_pool_neg_to_u_pos.append(
                torch.tensor(list(cand), dtype=torch.long) if cand else torch.empty(0, dtype=torch.long))
            cand = set()
            for go in pos_go:
                cand.update(self.proteins_with_pos_to_go.get(go, []))
            cand.discard(u)
            self.batch_pool_shared_pos.append(
                torch.tensor(list(cand), dtype=torch.long) if cand else torch.empty(0, dtype=torch.long))

    def sample(self) -> Tensor:
        """Trainer compatibility: this sampler doesn't sample explicit edges."""
        return torch.empty(2, 0, dtype=torch.long, device=self.device)

    def get_contrastive_samples(self, z: Tensor, neg_statement_index: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = z.device
        k = self.k
        if self.Nb == 0:
            D = int(z.size(-1))
            empty1 = torch.empty(0, D, device=device)
            emptyk = torch.empty(0, k, D, device=device)
            return empty1, emptyk, emptyk, emptyk, emptyk

        anchors = self.batch_anchors_local
        z_anchor = z[torch.tensor(anchors, dtype=torch.long, device=device)]
        shared_neg_rows: List[Tensor] = []
        pos_to_u_neg_rows: List[Tensor] = []
        neg_to_u_pos_rows: List[Tensor] = []
        shared_pos_rows: List[Tensor] = []

        for i, u in enumerate(anchors):
            fallback = int(u)
            shared_neg_idx = self._sample_k_tensor(self.batch_pool_shared_neg[i], k, fallback, device)
            pos_to_u_neg_idx = self._sample_k_tensor(self.batch_pool_pos_to_u_neg[i], k, fallback, device)
            neg_to_u_pos_idx = self._sample_k_tensor(self.batch_pool_neg_to_u_pos[i], k, fallback, device)
            shared_pos_idx = self._sample_k_tensor(self.batch_pool_shared_pos[i], k, fallback, device)

            shared_neg_rows.append(z[shared_neg_idx])
            pos_to_u_neg_rows.append(z[pos_to_u_neg_idx])
            neg_to_u_pos_rows.append(z[neg_to_u_pos_idx])
            shared_pos_rows.append(z[shared_pos_idx])
        z_shared_neg = torch.stack(shared_neg_rows, dim=0)
        z_pos_to_u_neg = torch.stack(pos_to_u_neg_rows, dim=0)
        z_neg_to_u_pos = torch.stack(neg_to_u_pos_rows, dim=0)
        z_shared_pos = torch.stack(shared_pos_rows, dim=0)
        return z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos