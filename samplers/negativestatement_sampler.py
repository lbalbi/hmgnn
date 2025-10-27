import torch, dgl, random

class NegativeStatementSampler:
    """
    A sampler that draws k negatives (and one positive) for each anchor node in the
    'anchor_etype' edges (e.g. 'PPI'), and also one negative-statement neighbor as
    the positive of the negative view.
    """
    def __init__(self, k: int = 2, go_etype: str = "link", anchor_etype: str = "PPI",
                 pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        self.k = k
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype

    def prepare_global(self, full_g, negatives=None):
        # 1) identify all anchors
        src_anchor, dst_anchor = full_g.edges(etype=self.anchor_etype)
        if negatives is not None:
            src_na, dst_na = negatives
            anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist()) \
                    | set(src_na.tolist())    | set(dst_na.tolist())
        else: anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        self.anchors_global = sorted(anchors)

        # build positive neighbors
        N = full_g.num_nodes()
        self.pos_global = [[] for _ in range(N)]
        src_pos, dst_pos = full_g.edges(etype=self.pos_etype)
        for u, v in zip(src_pos.tolist(), dst_pos.tolist()):
            self.pos_global[u].append(v)
        self.pos_targets = set(dst_pos.tolist())
        # build direct negative-statement neighbors
        self.direct_global = [[] for _ in range(N)]
        src_neg, dst_neg = full_g.edges(etype=self.neg_etype)
        for u, v in zip(src_neg.tolist(), dst_neg.tolist()):
            self.direct_global[u].append(v)
        # two-hop backups for nodes with < k negatives
        self.two_hop_global = {}
        for u in range(N):
            if len(self.direct_global[u]) < self.k:
                sup = set()
                for b in self.direct_global[u]:
                    sup.update(full_g.predecessors(b, etype=self.go_etype).tolist())
                self.two_hop_global[u] = list(sup)
        # ensure every anchor has at least k candidates, sample if needed
        pools = []
        for u in self.anchors_global:
            direct = self.direct_global[u]
            if len(direct) >= self.k: pool = direct.copy()
            else: pool = list({*direct, *self.two_hop_global.get(u, [])})
            if len(pool) < self.k:
                excluded = set(self.pos_global[u]) | set(self.direct_global[u])
                candidates = list(self.pos_targets - excluded)
                needed = self.k - len(pool)
                if candidates:
                    if len(candidates) >= needed: extra = random.sample(candidates, needed)
                    else: extra = random.choices(candidates, k=needed)
                    pool.extend(extra)
            pools.append(pool)

        # build a padded pool matrix for fast GPU sampling
        max_len = max(len(p) for p in pools)
        pool_matrix = torch.empty((len(pools), max_len), dtype=torch.long)
        for i, p in enumerate(pools):
            padded = p + [self.anchors_global[i]] * (max_len - len(p))
            pool_matrix[i] = torch.tensor(padded, dtype=torch.long)
        # map anchor id -> row index in pool_matrix
        self.anchor_to_idx = {u: i for i, u in enumerate(self.anchors_global)}
        self.pool_matrix_cpu = pool_matrix  # keep on CPU
        self.pool_max_len = max_len

    def prepare_batch(self, batch):
        # map local nodes to global IDs
        if dgl.NID in batch.ndata: orig = batch.ndata[dgl.NID].tolist()
        else: orig = list(range(batch.num_nodes()))
        mapping = {g: i for i, g in enumerate(orig)}

        self.batch_anchors_global = []
        self.batch_anchors_local  = []
        for u in self.anchors_global:
            if u in mapping:
                self.batch_anchors_global.append(u)
                self.batch_anchors_local.append(mapping[u])
        self.Nb = len(self.batch_anchors_local)
        self.device = batch.device

    def get_contrastive_samples(self, z: torch.Tensor) -> tuple:

        G = self.batch_anchors_global
        L = self.batch_anchors_local
        Nb, k = len(G), self.k
        device = z.device
        z_anchor = z[L]  # (Nb, D)
        pos_nb = []
        for u in G:
            opts = self.pos_global[u]
            if opts: pos_nb.append(opts[torch.randint(len(opts), (1,), device=device).item()])
            else: pos_nb.append(u)
        pos_nb = torch.tensor(pos_nb, device=device)
        z_pos = z[pos_nb]  # (Nb, D)
        # negative sampling
        rows = torch.tensor([self.anchor_to_idx[u] for u in G], device=device)
        cols = torch.randint(0, self.pool_max_len, (Nb, k), device=device)

        row_idx = rows.unsqueeze(1).expand(-1, k)  # (Nb, k)
        pool_mat = self.pool_matrix_cpu.to(device)
        neg_idx = pool_mat[row_idx, cols]  # (Nb, k)
        z_negs = z[neg_idx]  # (Nb, k, D)
        return z_anchor, z_pos, z_negs
