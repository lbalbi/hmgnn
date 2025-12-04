import torch, random
from typing import Optional, Tuple, List, Dict
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class NegativeStatementSampler:
    """ Draws k negatives and 1 positive for each anchor node in the 'anchor_etype' relation (e.g., 'PPI') using:
      - direct neg_statement neighbors
      - 2-hop negatives via GO hierarchy
      - If an anchor has >= k negatives: sample k negatives.
      - If an anchor has 0 < k' < k negatives: use only those negatives (sample with replacement
        so we still output k negatives, but all from its true pool).
      - If an anchor has 0 negatives: exclude it from contrastive loss and record it in
        'anchors_without_negatives.txt'.
    """

    def __init__(self,k: int = 1, go_etype: str = "link",anchor_etype: str = "PPI",
        pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        self.k = k
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.anchors_global: List[int] = []
        self.pos_global: List[List[int]] = []
        self.direct_global: List[List[int]] = []
        self.two_hop_global: Dict[int, List[int]] = {}
        self.pos_targets: set = set()
        self.neg_pools: Dict[int, List[int]] = {}
        self.pool_sizes: Dict[int, int] = {}
        self.zero_neg_anchors: List[int] = []
        self.batch_anchors_global: List[int] = []
        self.batch_anchors_local: List[int] = []
        self.Nb: int = 0
        self.device = torch.device("cpu")


    def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None):
        """ Precomputes:
          - anchor set (nodes that appear in PPI or given negatives)
          - positive neighbors from pos_statement
          - negative pools per anchor (direct + 2-hop)
          - stats and list of anchors with 0 negatives (written to txt)
        """

        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_anchor, dst_anchor = full_g[anchor_key].edge_index

        if negatives is not None:
            src_na, dst_na = negatives
            anchors = (set(src_anchor.tolist())
                | set(dst_anchor.tolist())
                | set(src_na.tolist())
                | set(dst_na.tolist()))
        else: anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        self.anchors_global = sorted(anchors)

        src_nt, _, dst_nt = anchor_key
        node_type = src_nt
        N = _num_nodes_of(full_g, node_type)

        self.pos_global = [[] for _ in range(N)]
        pos_key = _find_key_by_rel(full_g, self.pos_etype)
        if "edge_index" in full_g[pos_key]:
            src_pos, dst_pos = full_g[pos_key].edge_index
            for u, v in zip(src_pos.tolist(), dst_pos.tolist()):
                self.pos_global[u].append(v)
            self.pos_targets = set(dst_pos.tolist())
        else: self.pos_targets = set()

        self.direct_global = [[] for _ in range(N)]
        neg_key = _find_key_by_rel(full_g, self.neg_etype)
        if "edge_index" in full_g[neg_key]:
            src_neg, dst_neg = full_g[neg_key].edge_index
            for u, v in zip(src_neg.tolist(), dst_neg.tolist()):
                self.direct_global[u].append(v)

        self.two_hop_global = {}
        go_key = _find_key_by_rel(full_g, self.go_etype)
        predecessors: Dict[int, List[int]] = {}
        if "edge_index" in full_g[go_key]:
            go_src, go_dst = full_g[go_key].edge_index
            for s, d in zip(go_src.tolist(), go_dst.tolist()):
                predecessors.setdefault(d, []).append(s)

        for u in range(N):
            if len(self.direct_global[u]) < self.k:
                sup = set()
                for b in self.direct_global[u]:
                    sup.update(predecessors.get(b, []))
                if sup: self.two_hop_global[u] = list(sup)

        # self.neg_pools = {}
        # self.pool_sizes = {}

        # for u in self.anchors_global:
        #     direct = self.direct_global[u]
        #     two_hop = self.two_hop_global.get(u, [])
        #     pool = list({*direct, *two_hop})
        #     self.neg_pools[u] = pool
        #     self.pool_sizes[u] = len(pool)

        # self.zero_neg_anchors = [u for u in self.anchors_global if self.pool_sizes[u] == 0]
        # pool_lens = list(self.pool_sizes.values())
        # if pool_lens:
        #     pool_lens_t = torch.tensor(pool_lens, dtype=torch.float)
        #     print("[NegStmtSampler] Global negative pools: "
        #         f"num_anchors={len(self.anchors_global)}, "
        #         f"min_neg={int(pool_lens_t.min().item())}, "
        #         f"max_neg={int(pool_lens_t.max().item())}, "
        #         f"mean_neg={float(pool_lens_t.mean().item()):.2f}, "
        #         f"median_neg={float(pool_lens_t.median().item()):.2f}",
        #         flush=True)

        #     hist_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300]
        #     counts = [0] * (len(hist_bins) - 1)
        #     for L in pool_lens:
        #         for i in range(len(hist_bins) - 1):
        #             if hist_bins[i] <= L < hist_bins[i + 1]:
        #                 counts[i] += 1
        #                 break
        #     print("[NegStmtSampler] Negative pool size histogram (anchors):", flush=True)
        #     for i in range(len(hist_bins) - 1):
        #         print(f"[{hist_bins[i]:3d}, {hist_bins[i+1]:3d}): {counts[i]} anchors",
        #             flush=True)
        # if self.zero_neg_anchors:
        #     out_path = "anchors_without_negatives.txt"
        #     with open(out_path, "w") as f:
        #         for u in self.zero_neg_anchors:
        #             f.write(str(u) + "\n")
        #     print(f"[NegStmtSampler] Wrote {len(self.zero_neg_anchors)} anchors with 0 negatives to {out_path}",flush=True)


    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """Identify which global anchors appear in this batch subgraph."""
        some_key = next(iter(batch.edge_types))
        if batch[some_key].edge_index.is_cuda: self.device = batch[some_key].edge_index.device
        else: self.device = torch.device("cpu")

        node_type = _find_key_by_rel(batch, self.anchor_etype)[0]
        N_local = _num_nodes_of(batch, node_type)

        self.batch_anchors_global = []
        self.batch_anchors_local = []
        for u in self.anchors_global:
            if 0 <= u < N_local:
                self.batch_anchors_global.append(u)
                self.batch_anchors_local.append(u)
        self.Nb = len(self.batch_anchors_local)


    def sample(self) -> Optional[Tensor]:
        """ Optional helper picks a negative neighbor for each batch anchor.
        Uses the same neg_pools definition (direct + two-hop).
        Returns a 1D tensor of chosen indices, or None if Nb == 0.
        """
        if self.Nb == 0: return None

        picks = []
        for u in self.batch_anchors_global:
            pool = self.neg_pools.get(u, [])
            if pool: picks.append(random.choice(pool))
            else: picks.append(u)
        return torch.tensor(picks, dtype=torch.long, device=self.device)


    def get_contrastive_samples(self,z: Tensor,neg_statement_index: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ Returns:
            z_anchor: (B, D) embeddings of anchors used in contrastive loss
            z_pos:    (B, D) embeddings of positive-statement neighbors (or self if none)
            z_negs:   (B, k, D) embeddings of k negatives per anchor
        Only anchors with at least one negative in their pool are included in B.
        For anchors with < k negatives, we sample with replacement from their true pool.
        """
        device = z.device
        k = self.k
        anchors_local: List[int] = []
        pos_idx_list: List[int] = []
        neg_idx_lists: List[List[int]] = []

        for u_local, u_global in zip(self.batch_anchors_local, self.batch_anchors_global):
            pool = self.neg_pools.get(u_global, [])
            if len(pool) == 0: continue
            anchors_local.append(u_local)

            pos_opts = self.pos_global[u_global]
            if pos_opts: pos_idx = random.choice(pos_opts)
            else: pos_idx = u_global
            pos_idx_list.append(pos_idx)

            # negatives:
            if len(pool) >= k: neg_ids = random.sample(pool, k=k)
            else: neg_ids = random.choices(pool, k=k)
            neg_idx_lists.append(neg_ids)

        if not anchors_local:
            D = z.size(-1)
            empty = torch.empty(0, D, device=device)
            empty_k = torch.empty(0, k, D, device=device)
            return empty, empty, empty_k

        anchors_local_t = torch.tensor(anchors_local, dtype=torch.long, device=device)
        pos_idx_t = torch.tensor(pos_idx_list, dtype=torch.long, device=device)
        neg_idx_t = torch.tensor(neg_idx_lists, dtype=torch.long, device=device)

        z_anchor = z[anchors_local_t]
        z_pos = z[pos_idx_t]
        z_negs = z[neg_idx_t]

        return z_anchor, z_pos, z_negs


# import torch, random
# from typing import Optional, Tuple, List, Dict
# from torch import Tensor
# from torch_geometric.data import HeteroData
# from .utils import _find_key_by_rel, _num_nodes_of


# class NegativeStatementSampler:
#     """ Draws k negatives (and one positive) for each anchor node in the 'anchor_etype' relation
#     (e.g., 'PPI'), and also one negative-statement neighbor as the positive of the negative view.
#     Assumes all relations are between the same node type (e.g., ("node", rel, "node")).
#     """
#     def __init__(self, k: int = 3, go_etype: str = "link", anchor_etype: str = "PPI",
#          pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        
#         self.k = k
#         self.go_etype = go_etype
#         self.anchor_etype = anchor_etype
#         self.pos_etype = pos_etype
#         self.neg_etype = neg_etype
#         self.anchors_global: List[int] = []
#         self.pos_global: List[List[int]] = []
#         self.direct_global: List[List[int]] = []
#         self.two_hop_global: Dict[int, List[int]] = {}
#         self.pos_targets: set = set()
#         self.anchor_to_idx: Dict[int, int] = {}
#         self.pool_matrix_cpu: Optional[Tensor] = None
#         self.pool_max_len: int = 0
#         self.batch_anchors_global: List[int] = []
#         self.batch_anchors_local: List[int] = []
#         self.device = torch.device("cpu")



#     def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None):
#         """ Precomputes global anchor set, positive/negative neighbors, 2-hop backups and padded pool matrix for sampling.
#         Args: full_g: HeteroData; negatives: optional tuple (src_neg, dst_neg) of known negatives to include in anchor set.
#         """
#         anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
#         src_anchor, dst_anchor = full_g[anchor_key].edge_index

#         if negatives is not None:
#             src_na, dst_na = negatives
#             anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist()) | \
#                       set(src_na.tolist()) | set(dst_na.tolist())
#         else: anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
#         self.anchors_global = sorted(anchors)

#         src_nt, _, dst_nt = anchor_key
#         node_type = src_nt
#         N = _num_nodes_of(full_g, node_type)
#         self.pos_global = [[] for _ in range(N)]
#         pos_key = _find_key_by_rel(full_g, self.pos_etype)
#         if "edge_index" in full_g[pos_key]:
#             src_pos, dst_pos = full_g[pos_key].edge_index
#             for u, v in zip(src_pos.tolist(), dst_pos.tolist()):
#                 self.pos_global[u].append(v)
#             self.pos_targets = set(dst_pos.tolist())
#         else: self.pos_targets = set()

#         self.direct_global = [[] for _ in range(N)]
#         neg_key = _find_key_by_rel(full_g, self.neg_etype)
#         if "edge_index" in full_g[neg_key]:
#             src_neg, dst_neg = full_g[neg_key].edge_index
#             for u, v in zip(src_neg.tolist(), dst_neg.tolist()):
#                 self.direct_global[u].append(v)
#         self.two_hop_global = {}
#         go_key = _find_key_by_rel(full_g, self.go_etype)
#         predecessors: Dict[int, List[int]] = {}
#         if "edge_index" in full_g[go_key]:
#             go_src, go_dst = full_g[go_key].edge_index
#             for s, d in zip(go_src.tolist(), go_dst.tolist()):
#                 predecessors.setdefault(d, []).append(s)
#         for u in range(N):
#             if len(self.direct_global[u]) < self.k:
#                 sup = set()
#                 for b in self.direct_global[u]:
#                     sup.update(predecessors.get(b, []))
#                 if sup: self.two_hop_global[u] = list(sup)

#         pools: List[List[int]] = []
#         self.pool_sizes = {}
#         for u in self.anchors_global:
#             direct = self.direct_global[u]
#             if len(direct) >= self.k: pool = direct.copy()
#             else: pool = list({*direct, *self.two_hop_global.get(u, [])})
#             self.pool_sizes[u] = len(pool)
#             if len(pool) < self.k:
#                 excluded = set(self.pos_global[u]) | set(self.direct_global[u])
#                 candidates = list(self.pos_targets - excluded)
#                 needed = self.k - len(pool)
#                 if candidates:
#                     extra = random.sample(candidates, needed) if len(candidates) >= needed \
#                             else random.choices(candidates, k=needed)
#                     pool.extend(extra)
#             pools.append(pool)
#         if len(pools) == 0:
#             self.pool_matrix_cpu = torch.empty(0, 0, dtype=torch.long)
#             self.pool_max_len = 0
#             self.anchor_to_idx = {}
#             return
#         max_len = max(len(p) for p in pools)
#         pool_matrix = torch.empty((len(pools), max_len), dtype=torch.long)
#         for i, p in enumerate(pools):
#             padded = p + [self.anchors_global[i]] * (max_len - len(p))
#             pool_matrix[i] = torch.tensor(padded, dtype=torch.long)
#         self.anchor_to_idx = {u: i for i, u in enumerate(self.anchors_global)}
#         self.pool_matrix_cpu = pool_matrix
#         self.pool_max_len = max_len

#         ## negatives analysis
#         pool_lens = list(self.pool_sizes.values())
#         pool_lens_t = torch.tensor(pool_lens, dtype=torch.float)
#         print("[NegStmtSampler] Global negative pools: "
#             f"num_anchors={len(self.anchors_global)}, "
#             f"min_neg={int(pool_lens_t.min().item())}, "
#             f"max_neg={int(pool_lens_t.max().item())}, "
#             f"mean_neg={float(pool_lens_t.mean().item()):.2f}, "
#             f"median_neg={float(pool_lens_t.median().item()):.2f}",
#             flush=True)
#         hist_bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300]
#         counts = [0] * (len(hist_bins) - 1)
#         for L in pool_lens:
#             for i in range(len(hist_bins) - 1):
#                 if hist_bins[i] <= L < hist_bins[i + 1]:
#                     counts[i] += 1
#                     break
#         print("[NegStmtSampler] Negative pool size histogram (anchors):", flush=True)
#         for i in range(len(hist_bins) - 1):
#             print(f"[{hist_bins[i]:3d}, {hist_bins[i+1]:3d}): {counts[i]} anchors",
#                 flush=True)



#     def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
#         """ Identifies which global anchors appear in this batch subgraph. Local IDs == global IDs.
#         """
#         some_key = next(iter(batch.edge_types))
#         self.device = batch[some_key].edge_index.device if batch[some_key].edge_index.is_cuda else torch.device("cpu")
#         node_type = _find_key_by_rel(batch, self.anchor_etype)[0]
#         N_local = _num_nodes_of(batch, node_type)
#         self.batch_anchors_global = []
#         self.batch_anchors_local = []
#         for u in self.anchors_global:
#             if 0 <= u < N_local:
#                 self.batch_anchors_global.append(u)
#                 self.batch_anchors_local.append(u)
#         self.Nb = len(self.batch_anchors_local)


#     def sample(self) -> Optional[Tensor]:
#         """ Optional helper picks one neg-statement neighbor (or fallback) for each batch anchor.
#         Returns Tensor of chosen indices. Trainers may pass this into get_contrastive_samples(z, neg_statement_index).
#         """
#         if not hasattr(self, "batch_anchors_global") or self.Nb == 0: return None
#         picks = []
#         for u in self.batch_anchors_global:
#             cand = self.direct_global[u]
#             if cand: picks.append(random.choice(cand))
#             else:
#                 if self.pool_matrix_cpu is not None and self.pool_max_len > 0 and u in self.anchor_to_idx:
#                     row = self.pool_matrix_cpu[self.anchor_to_idx[u]]
#                     picks.append(int(row[random.randrange(self.pool_max_len)].item()))
#                 else: picks.append(u)
#         return torch.tensor(picks, dtype=torch.long, device=self.device)



#     def get_contrastive_samples(self, z: Tensor, neg_statement_index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
#         """ Returns: z_anchor: (Nb, D) embeddings of anchors; z_pos: (Nb, D) embeddings of sampled positive-statement neighbors;
#             z_negs: (Nb, k, D) embeddings of sampled negatives from the pool
#         """
#         if self.Nb == 0:
#             D = z.size(-1)
#             empty = torch.empty(0, D, device=z.device)
#             empty_k = torch.empty(0, self.k, D, device=z.device)
#             return empty, empty, empty_k
#         G = self.batch_anchors_global
#         L = self.batch_anchors_local
#         Nb, k = len(G), self.k
#         device = z.device

#         z_anchor, pos_nb = z[L], [] # Anchors
#         for u in G:
#             opts = self.pos_global[u]
#             if opts: pos_nb.append(opts[torch.randint(len(opts), (1,), device=device).item()])
#             else: pos_nb.append(u)
#         pos_nb = torch.tensor(pos_nb, device=device, dtype=torch.long)
#         z_pos = z[pos_nb]

#         if self.pool_matrix_cpu is None or self.pool_max_len == 0:
#             z_negs = z[pos_nb].unsqueeze(1).expand(-1, k, -1)
#             return z_anchor, z_pos, z_negs

#         rows = torch.tensor([self.anchor_to_idx[u] for u in G], device=device, dtype=torch.long)
#         cols = torch.randint(0, self.pool_max_len, (Nb, k), device=device, dtype=torch.long)
#         row_idx = rows.unsqueeze(1).expand(-1, k)
#         pool_mat = self.pool_matrix_cpu.to(device)
#         neg_idx = pool_mat[row_idx, cols]
#         z_negs = z[neg_idx]
#         return z_anchor, z_pos, z_negs
