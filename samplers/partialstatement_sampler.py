import dgl
import torch
import random

class PartialStatementSampler:
    """
    A sampler that uses an external list of negative statements as negative samples
    for each node in the training graph. Designed for scenarios where only positive
    statement edges exist in the graph itself.
    """
    def __init__(self, k: int = 2, go_etype: str = "link", neg_edges: list = None):
        """
        Args:
            k (int): number of negative samples per positive.
            go_etype (str): edge type to traverse for two-hop expansion.
            neg_edges (list of (src, dst) pairs): external negative_statement edges.
                                                If None, will fall back to graph edges.
        """
        self.k = k
        self.go_etype = go_etype
        self.external_neg = neg_edges or []

    def prepare_global(self, full_g, pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        """
        Build global neighbor structures from the full graph and external negatives.
        """
        N = full_g.num_nodes()

        # Use external list if provided, otherwise read from graph
        if self.external_neg:
            src_neg, dst_neg = zip(*self.external_neg)
            src_neg = list(src_neg)
            dst_neg = list(dst_neg)
        else:
            src_neg, dst_neg = full_g.edges(etype=neg_etype)
            src_neg = src_neg.tolist()
            dst_neg = dst_neg.tolist()

        src_pos, dst_pos = full_g.edges(etype=pos_etype)
        src_pos = src_pos.tolist()
        dst_pos = dst_pos.tolist()

        # Build adjacency lists
        direct = [[] for _ in range(N)]
        pos = [[] for _ in range(N)]
        for u, v in zip(src_neg, dst_neg):
            direct[u].append(v)
        for u, v in zip(src_pos, dst_pos):
            pos[u].append(v)

        # Precompute two-hop expansions for nodes with fewer than k direct negatives
        two_hop = {}
        for u in range(N):
            if len(direct[u]) < self.k:
                sup = set()
                for b in direct[u]:
                    sup.update(full_g.predecessors(b, etype=self.go_etype).tolist())
                two_hop[u] = list(sup)

        # Store global structures
        self.direct_global = direct
        self.two_hop_global = two_hop
        self.pos_global = pos

    def prepare_batch(self, batch):
        """
        Prepare per-batch structures by mapping global indices to local batch indices.
        """
        if dgl.NID in batch.ndata:
            orig = batch.ndata[dgl.NID].tolist()
        else:
            orig = list(range(batch.num_nodes()))
        mapping = {g: i for i, g in enumerate(orig)}

        self.N = len(orig)
        self.device = batch.device
        self.direct = [[] for _ in range(self.N)]
        self.pos = [[] for _ in range(self.N)]
        self.two_hop = {}

        for i, g_id in enumerate(orig):
            # Direct negatives in batch
            for g_nb in self.direct_global[g_id]:
                if g_nb in mapping:
                    self.direct[i].append(mapping[g_nb])
            # Positive neighbors in batch
            for g_nb in self.pos_global[g_id]:
                if g_nb in mapping:
                    self.pos[i].append(mapping[g_nb])
            # Two-hop if needed
            if len(self.direct[i]) < self.k:
                sup = set()
                for g_b in self.direct_global[g_id]:
                    for g_pp in self.two_hop_global.get(g_b, []):
                        if g_pp in mapping:
                            sup.add(mapping[g_pp])
                self.two_hop[i] = list(sup)

    def sample(self) -> torch.Tensor:
        """
        Sample k negative edges for each node in the current batch.

        Returns:
            ei (torch.Tensor): shape (2, N*k) edge index tensor of negative samples.
        """
        N, k = self.N, self.k
        cands = []
        for u in range(N):
            c = self.direct[u]
            if len(c) < k:
                # fallback to two-hop or the node itself
                c = list({*c, *self.two_hop.get(u, [])}) or [u]
            cands.append(c)

        # Create tensor of candidates with padding
        M = max(len(r) for r in cands)
        C = torch.full((N, M), N, dtype=torch.long, device=self.device)
        mask = torch.zeros((N, M), dtype=torch.bool, device=self.device)
        for i, row in enumerate(cands):
            L = len(row)
            C[i, :L] = torch.tensor(row, device=self.device)
            mask[i, :L] = True

        # Sample with probabilities proportional to mask (uniform over valid)
        probs = mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, k, replacement=True)
        dst = torch.gather(C, 1, idx)
        src = torch.arange(N, device=self.device).unsqueeze(1).repeat(1, k)
        ei = torch.stack([src, dst], dim=0).view(2, -1)
        return ei

    def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:
        """
        Generate contrastive positive and negative embeddings.
        """
        N, D = z.shape
        # Positive samples
        z_pos = z
        pos_nb = [random.choice(self.pos[u]) if self.pos[u] else u for u in range(N)]
        z_pos_pos = z[pos_nb]
        # Negative samples from sampled edges
        neg_dst = neg_ei[1].view(N, self.k)
        z_pos_neg = z[neg_dst]
        # Return tuple: (z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg)
        return z_pos, z_pos_pos, z_pos_neg # , z_pos, z_pos_pos, z_pos_neg
