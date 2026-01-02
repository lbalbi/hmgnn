import random, torch
from typing import Dict, List, Optional, Set, Tuple
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class PartialProteinSampler:
    """Protein-protein statement sampler where ONE statement relation may be provided externally.
    This sampler is designed to behave like `NegativeProteinSampler`:
      - anchors are proteins (endpoints of anchor_etype edges in the current minibatch)
      - for each anchor protein u, samples proteins in four groups (size k):
          1) shared_neg: proteins sharing ≥1 (extended) negative GO statement with u
          2) pos_to_u_neg: proteins with POS statement to a GO in u's (extended) NEG set
          3) neg_to_u_pos: proteins with NEG statement to a GO in u's POS set
          4) shared_pos: proteins sharing ≥1 POS statement with u
    Partial mode wiring:
        - nstatement: prepare_global(full_g)  [graph has pos_statement, external has neg_statement]
        - pstatement: prepare_global(full_g, pos_etype="neg_statement", neg_etype="pos_statement")
                      [graph has neg_statement, external has pos_statement]
    """

    def __init__(self, k: int = 1, go_etype: str = "link", anchor_etype: str = "PPI",
        neg_edges: Optional[List[Tuple[int, int]]] = None, pos_rel: str = "pos_statement",
        neg_rel: str = "neg_statement", neg_go_hops: int = 2):
        self.k = int(k)
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_rel = pos_rel
        self.neg_rel = neg_rel
        self.neg_go_hops = int(neg_go_hops)
        self.state_edges: List[Tuple[int, int]] = list(neg_edges or [])
        self.device = torch.device("cpu")
        self.proteins_global: Set[int] = set()
        self.anchors_global: List[int] = []
        self.pos_go_by_protein: List[List[int]] = []
        self.neg_go_by_protein: List[List[int]] = []
        self.proteins_with_pos_to_go: Dict[int, List[int]] = {}
        self.proteins_with_neg_to_go: Dict[int, List[int]] = {}
        self.predecessors: Dict[int, List[int]] = {}
        self.ext_neg_go: List[Set[int]] = []
        self.pool_shared_neg: List[Tensor] = []
        self.pool_pos_to_u_neg: List[Tensor] = []
        self.pool_neg_to_u_pos: List[Tensor] = []
        self.pool_shared_pos: List[Tensor] = []
        self.batch_anchors_global: List[int] = []
        self.batch_anchors_local: List[int] = []
        self.Nb: int = 0

    def _expand_go_set(self, seeds: Set[int], hops: int) -> Set[int]:
        out = set(seeds)
        frontier = set(seeds)
        for _ in range(max(0, int(hops))):
            nxt: Set[int] = set()
            for go in frontier:
                nxt.update(self.predecessors.get(go, []))
            nxt -= out
            if not nxt:
                break
            out |= nxt
            frontier = nxt
        return out

    @staticmethod
    def _dedup_list(xs: List[int]) -> List[int]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out


    def _get_rel_edges_from_graph(self, g: HeteroData, rel: str) -> Tuple[List[int], List[int]]:
        """Return (src_list, dst_list) for a relation if present, else ([], [])."""
        try: key = _find_key_by_rel(g, rel)
        except Exception: return [], []
        if key is None: return [], []
        if "edge_index" not in g[key]: return [], []
        ei = g[key].edge_index
        if ei is None or ei.numel() == 0: return [], []
        src, dst = ei
        return src.tolist(), dst.tolist()


    # def _get_rel_edges_from_graph(self, g: HeteroData, rel: str) -> Tuple[List[int], List[int]]:
    #     """Return (src_list, dst_list) for a relation if present, else ([], [])."""
    #     try:
    #         key = _find_key_by_rel(g, rel)
    #     except Exception:
    #         return [], []
    #     if "edge_index" not in g[key]:
    #         return [], []
    #     ei = g[key].edge_index
    #     if ei is None or ei.numel() == 0:
    #         return [], []
    #     src, dst = ei
    #     return src.tolist(), dst.tolist()

    def prepare_global(self, full_g: HeteroData, pos_etype: str = "pos_statement",
        neg_etype: str = "neg_statement", negatives: Optional[Tuple[Tensor, Tensor]] = None):
        """Build global sampling pools.
        full_g: HeteroData graph (after statement edge removal).
        pos_etype: Relation name present in the graph.
        neg_etype: Relation name removed from the graph and supplied via `self.state_edges`.
        negatives: Optional extra edge_index (src, dst) whose endpoints should be considered anchors too.
        """
        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_anchor, dst_anchor = full_g[anchor_key].edge_index
        anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        if negatives is not None:
            src_na, dst_na = negatives
            anchors |= set(src_na.tolist()) | set(dst_na.tolist())
        self.anchors_global = sorted(anchors)
        node_type = anchor_key[0]
        N = _num_nodes_of(full_g, node_type)
        # Fetch statement edges from graph (semantic pos/neg)
        pos_src, pos_dst = self._get_rel_edges_from_graph(full_g, self.pos_rel)
        neg_src, neg_dst = self._get_rel_edges_from_graph(full_g, self.neg_rel)
        # Override the removed relation with external edges (if provided)
        if self.state_edges:
            ext_src, ext_dst = zip(*self.state_edges) if self.state_edges else ([], [])
            ext_src = list(ext_src)
            ext_dst = list(ext_dst)

            if neg_etype == self.pos_rel: pos_src, pos_dst = ext_src, ext_dst
            elif neg_etype == self.neg_rel: neg_src, neg_dst = ext_src, ext_dst
            else:
                raise ValueError(
                    f"PartialStatementSampler: expected removed relation to be '{self.pos_rel}' or '{self.neg_rel}', "
                    f"got neg_etype='{neg_etype}'.")
        # Reset caches
        self.proteins_global = set()
        self.pos_go_by_protein = [[] for _ in range(N)]
        self.neg_go_by_protein = [[] for _ in range(N)]
        self.ext_neg_go = [set() for _ in range(N)]
        self.proteins_with_pos_to_go = {}
        self.proteins_with_neg_to_go = {}
        # Populate statement maps
        for p, go in zip(pos_src, pos_dst):
            if 0 <= p < N and 0 <= go < N:
                self.pos_go_by_protein[p].append(go)
                self.proteins_with_pos_to_go.setdefault(go, []).append(p)
                self.proteins_global.add(p)
        for p, go in zip(neg_src, neg_dst):
            if 0 <= p < N and 0 <= go < N:
                self.neg_go_by_protein[p].append(go)
                self.proteins_with_neg_to_go.setdefault(go, []).append(p)
                self.proteins_global.add(p)

        self.proteins_global |= set(self.anchors_global)
        # GO hierarchy predecessors map
        self.predecessors = {}
        go_key = _find_key_by_rel(full_g, self.go_etype)
        if go_key is not None and "edge_index" in full_g[go_key] and full_g[go_key].edge_index is not None:
            go_src, go_dst = full_g[go_key].edge_index
            for s, d in zip(go_src.tolist(), go_dst.tolist()):
                self.predecessors.setdefault(d, []).append(s)
        # Extend negative GO set for each protein
        for p in self.proteins_global:
            direct = set(self.neg_go_by_protein[p])
            if not direct:
                self.ext_neg_go[p] = set()
                continue
            self.ext_neg_go[p] = self._expand_go_set(direct, hops=self.neg_go_hops)
        self._build_pool_tensors(num_nodes=N)


    def _build_pool_tensors(self, num_nodes: int) -> None:
        """Build cached candidate pools per protein (CPU tensors)."""
        self.pool_shared_neg = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        self.pool_pos_to_u_neg = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        self.pool_neg_to_u_pos = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        self.pool_shared_pos = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        # Dedup lists to keep sampling stable
        for go, ps in list(self.proteins_with_pos_to_go.items()):
            self.proteins_with_pos_to_go[go] = self._dedup_list(ps)
        for go, ps in list(self.proteins_with_neg_to_go.items()):
            self.proteins_with_neg_to_go[go] = self._dedup_list(ps)
        for p in self.proteins_global:
            # shared_neg + pos_to_u_neg from extended NEG set
            if self.ext_neg_go[p]:
                cand = set()
                for go in self.ext_neg_go[p]:
                    cand.update(self.proteins_with_neg_to_go.get(go, []))
                cand.discard(p)
                if cand: self.pool_shared_neg[p] = torch.tensor(list(cand), dtype=torch.long)
                cand = set()
                for go in self.ext_neg_go[p]:
                    cand.update(self.proteins_with_pos_to_go.get(go, []))
                cand.discard(p)
                if cand: self.pool_pos_to_u_neg[p] = torch.tensor(list(cand), dtype=torch.long)

            # neg_to_u_pos + shared_pos from POS set
            if self.pos_go_by_protein[p]:
                pos_go = set(self.pos_go_by_protein[p])
                cand = set()
                for go in pos_go:
                    cand.update(self.proteins_with_neg_to_go.get(go, []))
                cand.discard(p)
                if cand: self.pool_neg_to_u_pos[p] = torch.tensor(list(cand), dtype=torch.long)
                cand = set()
                for go in pos_go:
                    cand.update(self.proteins_with_pos_to_go.get(go, []))
                cand.discard(p)
                if cand: self.pool_shared_pos[p] = torch.tensor(list(cand), dtype=torch.long)

    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """Prepare batch anchor list from the *current* positive (BCE) edges."""
        some_key = next(iter(batch.edge_types))
        self.device = (batch[some_key].edge_index.device
            if batch[some_key].edge_index.is_cuda else torch.device("cpu"))
        if pos_edge_index is not None and pos_edge_index.numel() > 0:
            anchors_t = torch.unique(pos_edge_index.reshape(-1)).to("cpu")
        else:
            try:
                anchor_key = _find_key_by_rel(batch, self.anchor_etype)
                
                if anchor_key is None: anchors_t = torch.empty(0, dtype=torch.long)

                elif "edge_index" in batch[anchor_key] and batch[anchor_key].edge_index.numel() > 0:
                    anchors_t = torch.unique(batch[anchor_key].edge_index.reshape(-1)).to("cpu")
                else: anchors_t = torch.empty(0, dtype=torch.long)
            except Exception: anchors_t = torch.empty(0, dtype=torch.long)

        zN = int(getattr(batch["node"], "num_nodes", 0) or _num_nodes_of(batch, "node"))
        anchors = [int(u) for u in anchors_t.tolist()
            if 0 <= int(u) < zN and int(u) in self.proteins_global]
        self.batch_anchors_global = anchors
        self.batch_anchors_local = anchors  # local ids == global ids in your loader
        self.Nb = len(anchors)


    def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
        """Sample up to k unique indices from pool; pad with fallback to keep shape stable.

        - If pool is empty: returns [fallback] * k
        - If 0 < len(pool) < k: returns all pool entries once (shuffled) + [fallback] * (k-len(pool))
        - If len(pool) >= k: returns k unique samples (no replacement)
        """
        L = int(pool.numel())
        if L == 0:
            return torch.full((k,), int(fallback), dtype=torch.long, device=device)

        if L >= k:
            idx = torch.randperm(L)[:k]
            out = pool[idx]
            return out.to(device)

        # 0 < L < k: use all candidates once, then pad with fallback
        perm = torch.randperm(L)
        out = pool[perm]  # all unique candidates, shuffled
        pad = torch.full((k - L,), int(fallback), dtype=torch.long)  # CPU
        out = torch.cat([out, pad], dim=0)
        return out.to(device)


    # def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
    #     """Sample k indices from a 1D CPU tensor pool and return a 1D tensor on `device`."""
    #     if pool.numel() == 0:
    #         return torch.full((k,), int(fallback), dtype=torch.long, device=device)
    #     L = int(pool.numel())
    #     if L >= k:
    #         idx = torch.randperm(L)[:k]
    #     else:
    #         idx = torch.randint(0, L, (k,))
    #     return pool[idx].to(device)

    def get_contrastive_samples(
        self, z: Tensor, neg_statement_index: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return tensors for ProteinContrastiveLoss (5-argument form)."""
        device = z.device
        k = self.k

        anchors_local: List[int] = []
        shared_neg_rows: List[Tensor] = []
        pos_to_u_neg_rows: List[Tensor] = []
        neg_to_u_pos_rows: List[Tensor] = []
        shared_pos_rows: List[Tensor] = []

        for u_local, u_global in zip(self.batch_anchors_local, self.batch_anchors_global):
            if u_global not in self.proteins_global:
                continue
            anchors_local.append(u_local)
            shared_neg_rows.append(
                self._sample_k_tensor(self.pool_shared_neg[u_global], k=k, fallback=u_global, device=device).unsqueeze(0)
            )
            pos_to_u_neg_rows.append(
                self._sample_k_tensor(self.pool_pos_to_u_neg[u_global], k=k, fallback=u_global, device=device).unsqueeze(0)
            )
            neg_to_u_pos_rows.append(
                self._sample_k_tensor(self.pool_neg_to_u_pos[u_global], k=k, fallback=u_global, device=device).unsqueeze(0)
            )
            shared_pos_rows.append(
                self._sample_k_tensor(self.pool_shared_pos[u_global], k=k, fallback=u_global, device=device).unsqueeze(0)
            )

        if not anchors_local:
            D = z.size(-1)
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty_k, empty_k, empty_k, empty_k

        anchors_local_t = torch.tensor(anchors_local, dtype=torch.long, device=device)
        shneg_t = torch.cat(shared_neg_rows, dim=0)
        pos2neg_t = torch.cat(pos_to_u_neg_rows, dim=0)
        neg2pos_t = torch.cat(neg_to_u_pos_rows, dim=0)
        shpos_t = torch.cat(shared_pos_rows, dim=0)

        z_anchor = z[anchors_local_t]
        z_shared_neg = z[shneg_t]
        z_pos_to_u_neg = z[pos2neg_t]
        z_neg_to_u_pos = z[neg2pos_t]
        z_shared_pos = z[shpos_t]
        return z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos
