import dgl, torch, random
class NegativeStatementSampler:
    """
    A sampler that uses negative statements as negative samples
    for each node in the training graph.
    """
    def __init__(self,  k: int = 2, go_etype="GO"):
        self.k = k
        self.go_etype = go_etype


    def prepare_global(self, full_g, pos_etype="pos_statement", neg_etype="neg_statement"):

        N = full_g.num_nodes()
        src_neg, dst_neg = full_g.edges(etype=neg_etype)
        src_pos, dst_pos = full_g.edges(etype=pos_etype)
        direct = [[] for _ in range(N)]
        pos    = [[] for _ in range(N)]
        for u,v in zip(src_neg.tolist(), dst_neg.tolist()): direct[u].append(v)
        for u,v in zip(src_pos.tolist(), dst_pos.tolist()): pos[u].append(v)
        two_hop = {}
        for u in range(N):
            if len(direct[u]) < self.k:
                sup = set()
                for b in direct[u]:
                    sup.update(full_g.predecessors(b, etype=self.go_etype).tolist())
                two_hop[u] = list(sup)
        self.direct_global = direct
        self.two_hop_global = two_hop
        self.pos_global = pos


    def prepare_batch(self, batch):

        if dgl.NID in batch.ndata: orig = batch.ndata[dgl.NID].tolist()
        else: orig = list(range(batch.num_nodes()))
        mapping = {g: i for i, g in enumerate(orig)}

        self.N = len(orig)
        self.device = batch.device
        self.direct, self.pos, self.two_hop = [[] for _ in range(self.N)], [[] for _ in range(self.N)], {}
        for i, g_id in enumerate(orig):
            for g_nb in self.direct_global[g_id]:
                if g_nb in mapping:
                    self.direct[i].append(mapping[g_nb])
            for g_nb in self.pos_global[g_id]:
                if g_nb in mapping:
                    self.pos[i].append(mapping[g_nb])
            if len(self.direct[i]) < self.k:
                sup = set()
                for g_b in self.direct_global[g_id]:
                    for g_pp in self.two_hop_global.get(g_b, []):
                        if g_pp in mapping:
                            sup.add(mapping[g_pp])
                self.two_hop[i] = list(sup)

    # def sample(self) -> torch.Tensor:
    #     neg_src, neg_dst = [], []
    #     for u in range(self.N):
    #         cands = self.direct[u]
    #         if len(cands) < self.k: cands = list({*cands, *self.two_hop.get(u, [])})
    #         if len(cands) >= self.k:  picks = random.sample(cands, self.k)
    #         else: picks = random.choices([v for v in range(self.N) if v != u], k=self.k)
    #         neg_src.extend([u] * self.k)
    #         neg_dst.extend(picks)
    #     return torch.tensor([neg_src, neg_dst], dtype=torch.long, device=self.device)


    def sample(self) -> torch.Tensor:
        N, k = self.N, self.k
        cands = []
        for u in range(N):
            c = self.direct[u]
            if len(c) < k: c = list({*c, *self.two_hop.get(u, [])}) or [u]
            cands.append(c)
        M = max(len(r) for r in cands)
        C = torch.full((N, M), N, dtype=torch.long, device=self.device)
        mask = torch.zeros((N, M), dtype=torch.bool, device=self.device)
        for i, row in enumerate(cands):
            L = len(row)
            C[i, :L] = torch.tensor(row, device=self.device)
            mask[i, :L] = True
        probs = mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, k, replacement=True)
        dst = torch.gather(C, 1, idx)
        src = torch.arange(N, device=self.device).unsqueeze(1).repeat(1, k)
        ei = torch.stack([src, dst], dim=0).view(2, -1)
        return ei


    def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:
        N, D = z.shape
        z_pos = z
        pos_nb = [random.choice(self.pos[u]) if self.pos[u] else u for u in range(N)]
        z_pos_pos = z[pos_nb]
        neg_dst   = neg_ei[1].view(N, self.k)
        z_pos_neg = z[neg_dst]
        # negatives
        z_neg, z_neg_pos, z_neg_neg = z_pos, z_pos_pos, z_pos_neg
        return z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg