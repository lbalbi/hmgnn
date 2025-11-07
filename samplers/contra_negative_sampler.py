import torch, random
from typing import Optional, Tuple, List, Dict
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of


class Contra_NegativeStatementSampler:
    """Draws k negatives and k ultras and one positive per anchor. Negatives from neg_statement edges excluding contradictions.
    Ultras from contradiction list. Both neg & ultra pools use the same subclass fallback method """

    def __init__(self, k: Optional[int] = 2, k_ultra: Optional[int] = 2, go_etype: str = "link", anchor_etype: str = "PPI",
             pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        self.k = k
        self.k_ultra = k if k_ultra is None else k_ultra
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.anchors_global: List[int] = []
        self.pos_global: List[List[int]] = []
        self.pos_targets: set = set()
        self.neg_direct_global: List[List[int]] = []
        self.neg_two_hop_global: Dict[int, List[int]] = {}
        self.ultra_direct_global: List[List[int]] = []
        self.ultra_two_hop_global: Dict[int, List[int]] = {}
        self.pool_neg_cpu: Optional[Tensor] = None
        self.pool_ultra_cpu: Optional[Tensor] = None
        self.pool_neg_max_len: int = 0
        self.pool_ultra_max_len: int = 0
        self.anchor_to_idx: Dict[int, int] = {}
        self.batch_anchors_global: List[int] = []
        self.batch_anchors_local: List[int] = []
        self.Nb: int = 0
        self.device = torch.device("cpu")


    @staticmethod
    def _list_to_index_map(src: Tensor, dst: Tensor) -> Dict[int, List[int]]:
        m: Dict[int, List[int]] = {}
        for u, v in zip(src.tolist(), dst.tolist()):
            m.setdefault(u, []).append(v)
        return m

    def _build_pool_matrix(self, pools: List[List[int]], anchors: List[int]):
        lengths = torch.tensor([len(p) for p in pools], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(pools) else 0
        if max_len == 0: return torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long), 0
        mat = torch.empty((len(pools), max_len), dtype=torch.long)
        for i, p in enumerate(pools):
            padded = p + ([p[-1]] * (max_len - len(p)))
            mat[i] = torch.tensor(padded, dtype=torch.long)
        return mat, lengths, max_len


    def prepare_global(self, full_g: HeteroData, negatives: Optional[Tuple[Tensor, Tensor]] = None,
        ultra_negatives: Optional[Tuple[Tensor, Tensor]] = None):

        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_anchor, dst_anchor = full_g[anchor_key].edge_index
        anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        if negatives is not None:
            na_s, na_d = negatives
            anchors |= set(na_s.tolist()) | set(na_d.tolist())
        if ultra_negatives is not None:
            ua_s, ua_d = ultra_negatives
            anchors |= set(ua_s.tolist()) | set(ua_d.tolist())
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

        self.neg_direct_global = [[] for _ in range(N)]
        neg_key = _find_key_by_rel(full_g, self.neg_etype)
        ultra_map: Dict[int, List[int]] = {}
        if ultra_negatives is not None:
            ua_s, ua_d = ultra_negatives
            ultra_map = self._list_to_index_map(ua_s, ua_d)
        if "edge_index" in full_g[neg_key]:
            src_neg, dst_neg = full_g[neg_key].edge_index
            for u, v in zip(src_neg.tolist(), dst_neg.tolist()):
                if v not in set(ultra_map.get(u, [])): self.neg_direct_global[u].append(v) # exclude ultras

        # ultras
        self.ultra_direct_global = [[] for _ in range(N)]
        if ultra_negatives is not None:
            ua_s, ua_d = ultra_negatives
            for u, v in zip(ua_s.tolist(), ua_d.tolist()):
                self.ultra_direct_global[u].append(v)
        # two-hop backup
        go_key = _find_key_by_rel(full_g, self.go_etype)
        subclasses_: Dict[int, List[int]] = {}
        if "edge_index" in full_g[go_key]:
            go_src, go_dst = full_g[go_key].edge_index # go_src = subclass, go_dst = superclass
            for s, d in zip(go_src.tolist(), go_dst.tolist()):
                subclasses_.setdefault(d, []).append(s)

        self.neg_two_hop_global = {}
        self.ultra_two_hop_global = {}
        for u in range(N):
            if len(self.neg_direct_global[u]) < self.k:
                sub = set()
                for b in self.neg_direct_global[u]:
                    sub.update(subclasses_.get(b, []))
                sub -= set(self.ultra_direct_global[u])
                if sub: self.neg_two_hop_global[u] = list(sub)
            if len(self.ultra_direct_global[u]) < self.k_ultra:
                sub_u = set()
                for b in self.ultra_direct_global[u]:
                    sub_u.update(subclasses_.get(b, []))
                sub_u -= set(self.neg_direct_global[u])
                sub_u -= set(self.pos_global[u])
                if sub_u: self.ultra_two_hop_global[u] = list(sub_u)

        # sample pools per anchor for faster sampling later
        neg_pools: List[List[int]] = []
        ultra_pools: List[List[int]] = []
        for u in self.anchors_global:
            # for negs
            direct_n = self.neg_direct_global[u]
            if len(direct_n) >= self.k: neg_pool = direct_n.copy()
            else:
                neg_pool = list({*direct_n, *self.neg_two_hop_global.get(u, [])})
                if len(neg_pool) < self.k:
                    excluded = set(self.pos_global[u]) | set(self.neg_direct_global[u]) | set(self.ultra_direct_global[u])
                    candidates = list(self.pos_targets - excluded)
                    needed = self.k - len(neg_pool)
                    if candidates:
                        extra = random.sample(candidates, needed) if len(candidates) >= needed \
                                else random.choices(candidates, k=needed)
                        neg_pool.extend(extra)
            # for ultras
            direct_u = self.ultra_direct_global[u]
            if len(direct_u) >= self.k_ultra: ultra_pool = direct_u.copy()
            else:
                ultra_pool = list({*direct_u, *self.ultra_two_hop_global.get(u, [])})
                if len(ultra_pool) < self.k_ultra:
                    excluded_u = set(self.pos_global[u]) | set(self.neg_direct_global[u]) | set(self.ultra_direct_global[u])
                    candidates_u = list(self.pos_targets - excluded_u)
                    needed_u = self.k_ultra - len(ultra_pool)
                    if candidates_u:
                        extra_u = random.sample(candidates_u, needed_u) if len(candidates_u) >= needed_u \
                                  else random.choices(candidates_u, k=needed_u)
                        ultra_pool.extend(extra_u)
            neg_pools.append(neg_pool)
            ultra_pools.append(ultra_pool)

        self.anchor_to_idx = {u: i for i, u in enumerate(self.anchors_global)}
        self.pool_neg_cpu, self.pool_neg_max_len = self._build_pool_matrix(neg_pools, self.anchors_global)
        self.pool_ultra_cpu, self.pool_ultra_max_len = self._build_pool_matrix(ultra_pools, self.anchors_global)


    def prepare_batch(self, batch: HeteroData, pos_edge_index: Optional[Tensor] = None):
        """Identifies which global anchors appear in a batch """
        some_key = next(iter(batch.edge_types))
        self.device = batch[some_key].edge_index.device if batch[some_key].edge_index.is_cuda else torch.device("cpu")
        node_type = _find_key_by_rel(batch, self.anchor_etype)[0]
        N_local = _num_nodes_of(batch, node_type)

        self.batch_anchors_global, self.batch_anchors_local = [], []
        for u in self.anchors_global:
            if 0 <= u < N_local:
                self.batch_anchors_global.append(u)
                self.batch_anchors_local.append(u)
        self.Nb = len(self.batch_anchors_local)


    def sample_neg(self) -> Optional[Tensor]:
        if self.Nb == 0: return None
        picks = []
        for u in self.batch_anchors_global:
            cand = self.neg_direct_global[u]
            if cand: picks.append(random.choice(cand))
            else:
                if (self.pool_neg_cpu is not None and self.pool_neg_max_len > 0 and u in self.anchor_to_idx):
                    row = self.pool_neg_cpu[self.anchor_to_idx[u]]
                    picks.append(int(row[random.randrange(self.pool_neg_max_len)].item()))
                else: picks.append(u)
        return torch.tensor(picks, dtype=torch.long, device=self.device)


    def sample_ultra(self) -> Optional[Tensor]:
        if self.Nb == 0: return None
        picks = []
        for u in self.batch_anchors_global:
            cand = self.ultra_direct_global[u]
            if cand: picks.append(random.choice(cand))
            else:
                if (self.pool_ultra_cpu is not None and self.pool_ultra_max_len > 0 and u in self.anchor_to_idx):
                    row = self.pool_ultra_cpu[self.anchor_to_idx[u]]
                    picks.append(int(row[random.randrange(self.pool_ultra_max_len)].item()))
                else: picks.append(u)
        return torch.tensor(picks, dtype=torch.long, device=self.device)



    def get_contrastive_samples( self, z: Tensor, neg_statement_index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns: z_anchor: (Nb, D); z_pos: (Nb, D);  z_negs: (Nb, k, D); z_ultras: (Nb, k_ultra, D) """
        if self.Nb == 0:
            D = z.size(-1)
            empty = torch.empty(0, D, device=z.device)
            empty_k = torch.empty(0, self.k, D, device=z.device)
            empty_ku = torch.empty(0, self.k_ultra, D, device=z.device)
            return empty, empty, empty_k, empty_ku

        G = self.batch_anchors_global
        L = self.batch_anchors_local
        Nb, k, ku = len(G), self.k, self.k_ultra
        device = z.device
        z_anchor = z[L]
        # one random pos neighbor
        pos_nb = []
        for u in G:
            opts = self.pos_global[u]
            if opts: pos_nb.append(opts[torch.randint(len(opts), (1,), device=device).item()])
            else: pos_nb.append(u)
        pos_nb = torch.tensor(pos_nb, device=device, dtype=torch.long)
        z_pos = z[pos_nb]

        # k negatives
        if self.pool_neg_cpu is None or self.pool_neg_max_len == 0: z_negs = z[pos_nb].unsqueeze(1).expand(-1, k, -1)
        else:
            rows = torch.tensor([self.anchor_to_idx[u] for u in G], device=device, dtype=torch.long)
            cols = torch.randint(0, self.pool_neg_max_len, (Nb, k), device=device, dtype=torch.long)
            row_idx = rows.unsqueeze(1).expand(-1, k)
            pool_neg = self.pool_neg_cpu.to(device)
            neg_idx = pool_neg[row_idx, cols]
            z_negs = z[neg_idx]
        # k ultras
        if self.pool_ultra_cpu is None or self.pool_ultra_max_len == 0: z_ultras = z[pos_nb].unsqueeze(1).expand(-1, ku, -1)
        else:
            rows_u = torch.tensor([self.anchor_to_idx[u] for u in G], device=device, dtype=torch.long)
            cols_u = torch.randint(0, self.pool_ultra_max_len, (Nb, ku), device=device, dtype=torch.long)
            row_idx_u = rows_u.unsqueeze(1).expand(-1, ku)
            pool_ultra = self.pool_ultra_cpu.to(device)
            ultra_idx = pool_ultra[row_idx_u, cols_u]
            z_ultras = z[ultra_idx]

        return z_anchor, z_pos, z_negs, z_ultras
