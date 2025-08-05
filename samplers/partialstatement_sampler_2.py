import torch
import dgl
import random

class PartialStatementSampler:
    """
    A high-performance sampler that uses an external list of negative statements or
    graph-based negative edges, with full vectorization and no Python loops in the hot path.
    """
    def __init__(self, k: int = 2, go_etype: str = "link", neg_edges: list = None):
        self.k = k
        self.go_etype = go_etype
        self.external_neg = neg_edges or []

    def prepare_global(self, full_g, pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        # Number of nodes
        N = full_g.num_nodes()
        # Direct negative edges (external or from graph)
        if self.external_neg:
            src_neg, dst_neg = zip(*self.external_neg)
            src_neg, dst_neg = list(src_neg), list(dst_neg)
        else:
            src_neg, dst_neg = full_g.edges(etype=neg_etype)
            src_neg, dst_neg = src_neg.tolist(), dst_neg.tolist()
        # Positive edges
        src_pos, dst_pos = full_g.edges(etype=pos_etype)
        src_pos, dst_pos = src_pos.tolist(), dst_pos.tolist()
        pos_global = [[] for _ in range(N)]
        for u, v in zip(src_pos, dst_pos): pos_global[u].append(v)
        self.pos_global = pos_global
        self.pos_targets = set(dst_pos)
        # Build direct_global adjacency list
        direct_global = [[] for _ in range(N)]
        for u, v in zip(src_neg, dst_neg):
            direct_global[u].append(v)
        # two-hop backup
        two_hop_global = [[] for _ in range(N)]
        for u in range(N):
            if len(direct_global[u]) < self.k:
                sup = set()
                for b in direct_global[u]:
                    sup.update(full_g.predecessors(b, etype=self.go_etype).tolist())
                two_hop_global[u] = list(sup)
        self.direct_global = direct_global
        self.two_hop_global = two_hop_global
        # Build fixed-size pools for each global node
        pools = []
        for u in range(N):
            if len(direct_global[u]) >= self.k: pool = direct_global[u].copy()
            else: pool = list({*direct_global[u], *two_hop_global[u]})
            # fallback to pos_targets if too small
            if len(pool) < self.k:
                excluded = set(direct_global[u]) | set(pos_global[u])
                candidates = list(self.pos_targets - excluded)
                needed = self.k - len(pool)
                if candidates:
                    if len(candidates) >= needed: pool.extend(random.sample(candidates, needed))
                    else: pool.extend(random.choices(candidates, k=needed))
            pools.append(pool)

        # Pad to max length and build mask
        M = max(len(row) for row in pools)
        pool_mat = torch.full((N, M), fill_value=0, dtype=torch.long)
        mask_mat = torch.zeros((N, M), dtype=torch.bool)
        for u, row in enumerate(pools):
            L = len(row)
            pool_mat[u, :L] = torch.tensor(row, dtype=torch.long)
            mask_mat[u, :L] = True

        self.pool_matrix_cpu = pool_mat    # (N, M)
        self.pool_mask_cpu   = mask_mat    # (N, M)
        self.M = M

    def prepare_batch(self, batch):
        # Map local nodes to global IDs
        if dgl.NID in batch.ndata: orig = batch.ndata[dgl.NID]
        else: orig = torch.arange(batch.num_nodes(), dtype=torch.long)
        self.orig = orig
        # Build mapping global->local
        self.mapping = {g.item(): i for i, g in enumerate(orig)}
        # 2) Gather pool rows and masks onto device
        device = batch.device
        self.batch_pool = self.pool_matrix_cpu[orig].to(device)  # (B, M)
        self.batch_mask = self.pool_mask_cpu[orig].to(device)    # (B, M)
        self.B = batch.num_nodes()
        self.device = device

    def sample(self) -> torch.Tensor:
        # Vectorized multinomial over masked entries
        probs = self.batch_mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, self.k, replacement=True)  # (B, k)
        dst = torch.gather(self.batch_pool, 1, idx)                # (B, k)
        src = torch.arange(self.B, device=self.device).unsqueeze(1).expand(-1, self.k)
        return torch.stack([src, dst], dim=0).view(2, -1)

    def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:
        B, D = z.shape
        z_pos = z
        # Sample one positive neighbor per node
        pos_nb = []
        for i, g in enumerate(self.orig.tolist()):
            opts = self.pos_global[g]
            if opts:
                g2 = random.choice(opts)
                pos_nb.append(self.mapping.get(g2, i))
            else: pos_nb.append(i)
        pos_nb = torch.tensor(pos_nb, device=self.device)
        z_pos_pos = z[pos_nb]
        # Negative view embeddings
        neg_dst = neg_ei[1].view(B, self.k)
        z_pos_neg = z[neg_dst]
        return z_pos, z_pos_pos, z_pos_neg