import torch, dgl
class NegativeStatementSampler:
    """
    A sampler that draws k negatives (and one positive) for each anchor node in the
    'anchor_etype' edges (e.g. 'PPI'), and also one negative-statement neighbor as
    the positive of the negative view.
    """
    def __init__(self, k: int = 2, go_etype: str = "link",
                 anchor_etype: str = "PPI", # gene_disease_association
                 pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        self.k = k
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype


    def prepare_global(self, full_g, negatives = None):
        """
        Build global structures:
          - self.anchors_global: all unique nodes in the anchor_etype edges
          - self.pos_global: positive neighbors per node
          - self.direct_global: negative-statement neighbors per node
          - self.two_hop_global: 2-hop backups for nodes with < k negatives
        """
        N = full_g.num_nodes()

        src_anchor, dst_anchor = full_g.edges(etype=self.anchor_etype)
        if negatives is not None: 
            src_neg_anchor, dst_neg_anchor = negatives
            anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist()
                ) | set(src_neg_anchor.tolist()) | set(dst_neg_anchor.tolist())
        else: anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        self.anchors_global = sorted(anchors)

        src_pos, dst_pos = full_g.edges(etype=self.pos_etype)
        pos = [[] for _ in range(N)]
        for u, v in zip(src_pos.tolist(), dst_pos.tolist()):
            pos[u].append(v)
        self.pos_targets = set(dst_pos.tolist())

        src_neg, dst_neg = full_g.edges(etype=self.neg_etype)
        direct = [[] for _ in range(N)]
        for u, v in zip(src_neg.tolist(), dst_neg.tolist()):
            direct[u].append(v)

        two_hop = {}
        for u in range(N):
            if len(direct[u]) < self.k:
                sup = set()
                for b in direct[u]:
                    sup.update(full_g.predecessors(b, etype=self.go_etype).tolist())
                two_hop[u] = list(sup)

        self.pos_global = pos
        self.direct_global = direct
        self.two_hop_global = two_hop
        # if there are missing negatives, sample from positive statements
        for u in self.anchors_global:
            if len(self.direct_global[u]) < self.k:
                excluded = set(self.pos_global[u]) | set(self.direct_global[u])
                fallback = list(self.pos_targets - excluded)
                if fallback:
                    if not self.two_hop_global.get(u):
                        self.two_hop_global[u] = fallback


    def prepare_batch(self, batch):
        """
        Given a DGLGraph or block `batch`, figure out which of our
        global anchors actually appear in this batch, and store both
        their global IDs and their batch-local indices.
        """
        if dgl.NID in batch.ndata: orig = batch.ndata[dgl.NID].tolist()   # maps local idx -> global ID
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
        k = self.k
        anchors = self.anchors_global
        device = z.device
        z_anchor = z[anchors]

        pos_nb = []
        for u in anchors:
            options = self.pos_global[u]
            if options:
                idx = torch.randint(0, len(options), (1,))
                pos_nb.append(options[idx])
            else: pos_nb.append(u)  # fallback

        pos_nb = torch.tensor(pos_nb, device=device).squeeze()
        z_pos_pos = z[pos_nb]

        neg_lists = []
        for u in anchors:
            pool = (self.direct_global[u]
                if len(self.direct_global[u]) >= k
                else list({*self.direct_global[u], *self.two_hop_global.get(u, [])}) or [u])
            pool_tensor = torch.tensor(pool, device=device)
            if pool_tensor.size(0) >= k: sampled_idx = torch.randint(0, pool_tensor.size(0), (k,), device=device)
            else: sampled_idx = torch.randint(0, pool_tensor.size(0), (k,), device=device, dtype=torch.long)
            neg_lists.append(pool_tensor[sampled_idx])
        neg_lists = torch.stack(neg_lists)  # shape (n, k)
        z_pos_neg = z[neg_lists]  # shape (n, k, D)

        return z_anchor, z_pos_pos, z_pos_neg



    ## Legacy code, kept for reference
    
    # def get_contrastive_samples(self, z: torch.Tensor) -> tuple:
    #     """
    #     Given the full graph node embedding matrix z (shape [N_all, D]),
    #     returns six tensors for n anchors
    #     """

    #     D = z.size(1)
    #     k = self.k
    #     anchors = self.anchors_global
    #     z_anchor = z[anchors]
    #     pos_nb = [random.choice(self.pos_global[u]) if self.pos_global[u] else u for u in anchors]
    #     # pos_nb_2 = [random.choice(self.pos_global[u]) if self.pos_global[u] else u for u in anchors]  # length n
    #     z_pos_pos = z[pos_nb]  # (n, D)
    #     # z_pos_pos_2 = z[pos_nb_2] # (n, D)
        
    #     neg_lists = []
    #     for u in anchors:
    #         pool = (self.direct_global[u]
    #             if len(self.direct_global[u]) >= k
    #             else list({*self.direct_global[u], *self.two_hop_global.get(u, [])}) or [u])
    #         neg_lists.append(random.choices(pool, k=k))
 
    #     neg_lists = torch.tensor(neg_lists, device=z.device)  # (n, k)
    #     z_pos_neg = z[neg_lists]  # (n, k, D)
    #     # z_neg = z_anchor  # (n, D)
    #     # neg_primary = neg_lists[:, 0].tolist()
    #     # z_neg_pos = z[neg_primary]  # (n, D)
    #     # z_neg_neg = torch.stack([z_pos_pos,z_pos_pos_2], dim=1)  # (n, D)

    #     return z_anchor, z_pos_pos, z_pos_neg #, z_neg, z_neg_pos, z_neg_neg
