import random
from typing import Dict, List, Optional, Set, Tuple
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class NegativeProteinSampler:
    """Protein-protein statement-based sampler for contrastive learning.
    Anchors are proteins (nodes that appear in the anchor relation, e.g. PPI).
    Statement edges are assumed to be:
      - (protein) --pos_statement--> (GO class)
      - (protein) --neg_statement--> (GO class)
    For each anchor protein u, samples *proteins* in four groups (each size k):
      1) shared_neg: proteins sharing ≥1 (extended) negative GO statement with u
      2) pos_to_u_neg: proteins with ≥1 POS statement to a GO class in u's (extended) NEG set
      3) neg_to_u_pos: proteins with ≥1 NEG statement to a GO class in u's POS set
      4) shared_pos: proteins sharing ≥1 POS statement with u
    Ontology extension:
      - u's NEG GO classes are extended with up to 2 hops over the GO hierarchy relation `go_etype`.
        (We follow the same direction used by the previous implementation: build
         predecessors[d] = [s] for edges s->d, and expand by repeatedly applying predecessors.)

      - If a group has no candidates for an anchor, fall back to sampling the anchor itself
        for that group (so that shapes are stable; that term contributes ~0 gradient)
    """

    def __init__(self, k: int = 1, go_etype: str = "link", anchor_etype: str = "PPI",
        pos_etype: str = "pos_statement", neg_etype: str = "neg_statement", neg_go_hops: int = 2):
        self.k = int(k)
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.neg_go_hops = int(neg_go_hops)
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


    def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None):

        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_anchor, dst_anchor = full_g[anchor_key].edge_index
        anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        if negatives is not None:
            src_na, dst_na = negatives
            anchors |= set(src_na.tolist()) | set(dst_na.tolist())
        self.anchors_global = sorted(anchors)

        node_type = anchor_key[0]
        N = _num_nodes_of(full_g, node_type)
        self.pos_go_by_protein = [[] for _ in range(N)]
        self.neg_go_by_protein = [[] for _ in range(N)]
        self.ext_neg_go = [set() for _ in range(N)]
        empty = torch.empty(0, dtype=torch.long)
        self.pool_shared_neg = [empty for _ in range(N)]
        self.pool_pos_to_u_neg = [empty for _ in range(N)]
        self.pool_neg_to_u_pos = [empty for _ in range(N)]
        self.pool_shared_pos = [empty for _ in range(N)]

        pos_key = _find_key_by_rel(full_g, self.pos_etype)
        if "edge_index" in full_g[pos_key]:
            src_pos, dst_pos = full_g[pos_key].edge_index
            for p, go in zip(src_pos.tolist(), dst_pos.tolist()):
                self.pos_go_by_protein[p].append(go)
                self.proteins_with_pos_to_go.setdefault(go, []).append(p)
            self.proteins_global |= set(src_pos.tolist())

        neg_key = _find_key_by_rel(full_g, self.neg_etype)
        if "edge_index" in full_g[neg_key]:
            src_neg, dst_neg = full_g[neg_key].edge_index
            for p, go in zip(src_neg.tolist(), dst_neg.tolist()):
                self.neg_go_by_protein[p].append(go)
                self.proteins_with_neg_to_go.setdefault(go, []).append(p)
            self.proteins_global |= set(src_neg.tolist())
        self.proteins_global |= set(self.anchors_global)

        self.predecessors = {}
        go_key = _find_key_by_rel(full_g, self.go_etype)
        if "edge_index" in full_g[go_key]:
            go_src, go_dst = full_g[go_key].edge_index
            for s, d in zip(go_src.tolist(), go_dst.tolist()):
                self.predecessors.setdefault(d, []).append(s)

        for p in self.proteins_global:
            direct = set(self.neg_go_by_protein[p])
            if not direct:
                self.ext_neg_go[p] = set()
                continue
            self.ext_neg_go[p] = self._expand_go_set(direct, hops=self.neg_go_hops)
        self._build_pool_tensors(num_nodes=N)


    def _build_pool_tensors(self, num_nodes: int) -> None:
        """Builds cached candidate pools per protein. Runs once in prepare_global.
        """
        self.pool_shared_neg = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        self.pool_pos_to_u_neg = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        self.pool_neg_to_u_pos = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]
        self.pool_shared_pos = [torch.empty(0, dtype=torch.long) for _ in range(num_nodes)]

        for m in (self.proteins_with_pos_to_go, self.proteins_with_neg_to_go):
            for go, ps in list(m.items()):
                seen = set()
                dedup = []
                for p in ps:
                    if p not in seen:
                        seen.add(p)
                        dedup.append(p)
                m[go] = dedup

        for p in self.proteins_global:
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

    def _expand_go_set(self, seeds: Set[int], hops: int) -> Set[int]:
        """Expand seeds by repeatedly applying predecessors up to `hops` times."""
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


    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """Identifies which anchor proteins participate in the *current* minibatch.
        Batches anchors from the *current positive edges* (pos_edge_index),
        falling back to the batch's anchor edge_index if provided.
        """
        some_key = next(iter(batch.edge_types))
        self.device = batch[some_key].edge_index.device if batch[some_key].edge_index.is_cuda else torch.device("cpu")

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
        anchors = [int(u) for u in anchors_t.tolist() if 0 <= int(u) < zN and int(u) in self.proteins_global]
        self.batch_anchors_global = anchors
        self.batch_anchors_local = anchors
        self.Nb = len(anchors)


    # def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
    #     """Sample k indices from a 1D CPU tensor pool and return a 1D tensor on `device`."""
    #     if pool.numel() == 0:
    #         return torch.full((k,), int(fallback), dtype=torch.long, device=device)
    #     L = int(pool.numel())
    #     if L >= k: idx = torch.randperm(L)[:k]
    #     else: idx = torch.randint(0, L, (k,))
    #     return pool[idx].to(device)

    # def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
    #     """Sample up to k unique indices from pool; pad with fallback to keep shape stable.
    #     - If pool is empty: returns [fallback] * k
    #     - If 0 < len(pool) < k: returns all pool entries once (shuffled) + [fallback] * (k-len(pool))
    #     - If len(pool) >= k: returns k unique samples (no replacement)
    #     """
    #     L = int(pool.numel())
    #     if L == 0:
    #         return torch.full((k,), int(fallback), dtype=torch.long, device=device)

    #     if L >= k:
    #         idx = torch.randperm(L)[:k]
    #         return pool[idx].to(device)

    #     # 0 < L < k: use all candidates once, then pad with fallback
    #     perm = torch.randperm(L)
    #     out = pool[perm]  # all unique candidates, shuffled
    #     pad = torch.full((k - L,), int(fallback), dtype=torch.long)  # CPU
    #     out = torch.cat([out, pad], dim=0)
    #     return out.to(device)

    def _sample_k_tensor(self, pool: Tensor, k: int, fallback: int, device: torch.device) -> Tensor:
        L = int(pool.numel())
        if L == 0:
            return torch.full((k,), int(fallback), dtype=torch.long, device=device)
        if k == 1:
            j = torch.randint(0, L, (1,))
            return pool[j].to(device)
        if L < k:
            perm = torch.randperm(L)
            out = pool[perm]
            pad = torch.full((k - L,), int(fallback), dtype=torch.long)
            return torch.cat([out, pad], dim=0).to(device)
        need = k
        drawn = torch.empty(0, dtype=torch.long)
        while drawn.numel() < k:
            draw = torch.randint(0, L, (min(L, 4 * need),), dtype=torch.long)
            drawn = torch.unique(torch.cat([drawn, draw]))
            need = k - drawn.numel()

        idx = drawn[:k]
        return pool[idx].to(device)


    def get_contrastive_samples(self, z: Tensor, neg_statement_index: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return anchor + 4 protein groups for the contrastive loss."""

        device = z.device
        k = self.k
        anchors_local: List[int] = []
        shared_neg_rows: List[Tensor] = []
        pos_to_u_neg_rows: List[Tensor] = []
        neg_to_u_pos_rows: List[Tensor] = []
        shared_pos_rows: List[Tensor] = []

        for u_local, u_global in zip(self.batch_anchors_local, self.batch_anchors_global):
            if u_global not in self.proteins_global: continue
            anchors_local.append(u_local)
            shared_neg_rows.append(self._sample_k_tensor(self.pool_shared_neg[u_global], k=k, fallback=u_global, device=device).unsqueeze(0))
            pos_to_u_neg_rows.append(self._sample_k_tensor(self.pool_pos_to_u_neg[u_global], k=k, fallback=u_global, device=device).unsqueeze(0))
            neg_to_u_pos_rows.append(self._sample_k_tensor(self.pool_neg_to_u_pos[u_global], k=k, fallback=u_global, device=device).unsqueeze(0))
            shared_pos_rows.append(self._sample_k_tensor(self.pool_shared_pos[u_global], k=k, fallback=u_global, device=device).unsqueeze(0))

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