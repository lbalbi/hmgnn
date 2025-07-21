import torch
import random
import dgl

class RandomStatementSampler:
    """
    A sampler that draws negative samples by uniformly sampling over existing "pos_statement" edges,
    ensuring sampled edges don't already exist (including reverse edges),
    and generates dual-view contrastive samples (positive and negative views).
    """
    def __init__(self, k: int = 2, pos_etype: str = "pos_statement"):
        """
        Args:
            k (int): number of negatives per view.
            pos_etype (str): edge type for positive statements.
        """
        self.k = k
        self.pos_etype = pos_etype

    def prepare_global(self, full_g):
        """
        Build global adjacency and reverse adjacency lists from the full graph.
        """
        N = full_g.num_nodes()
        # Extract all positive edges
        src_pos, dst_pos = full_g.edges(etype=self.pos_etype)
        src_pos, dst_pos = src_pos.tolist(), dst_pos.tolist()

        # Forward and reverse adjacency lists
        pos_global = [[] for _ in range(N)]
        rev_pos_global = [[] for _ in range(N)]
        for u, v in zip(src_pos, dst_pos):
            pos_global[u].append(v)
            rev_pos_global[v].append(u)

        # Unique set of all destination nodes for negatives
        global_dst = list(set(dst_pos))

        self.pos_global = pos_global
        self.rev_pos_global = rev_pos_global
        self.global_dst = global_dst

    def prepare_batch(self, batch):
        """
        Map global node IDs to batch-local indices and build per-node candidate lists.
        """
        # Original global IDs of batch nodes
        if dgl.NID in batch.ndata:
            orig = batch.ndata[dgl.NID].tolist()
        else:
            orig = list(range(batch.num_nodes()))
        mapping = {g: i for i, g in enumerate(orig)}

        self.N = len(orig)
        self.device = batch.device

        # Local lists for positives and negatives
        self.pos_local = [[] for _ in range(self.N)]
        self.neg_cands = [[] for _ in range(self.N)]

        # Build lists
        for i, g_id in enumerate(orig):
            # Positive neighbors in batch
            for nb in self.pos_global[g_id]:
                if nb in mapping:
                    self.pos_local[i].append(mapping[nb])
            # Negative candidates: any global_dst not forward or reverse neighbor
            invalid = set(self.pos_global[g_id]) | set(self.rev_pos_global[g_id])
            for v in self.global_dst:
                if v in mapping and v not in invalid:
                    self.neg_cands[i].append(mapping[v])

    def sample(self) -> torch.Tensor:
        """
        Sample k negative edges for each node in the current batch.

        Returns:
            ei (torch.Tensor): shape (2, N*k) edge index tensor of negative samples.
        """
        N, k = self.N, self.k
        cands = self.neg_cands

        # Handle case where some nodes have no candidates: fallback to self-loop
        for i in range(N):
            if not cands[i]:
                cands[i] = [i]

        # Pad candidate lists and create mask
        M = max(len(r) for r in cands)
        C = torch.full((N, M), N, dtype=torch.long, device=self.device)
        mask = torch.zeros((N, M), dtype=torch.bool, device=self.device)
        for i, row in enumerate(cands):
            L = len(row)
            C[i, :L] = torch.tensor(row, device=self.device)
            mask[i, :L] = True

        # Uniform sampling over mask
        probs = mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, k, replacement=True)
        dst = torch.gather(C, 1, idx)
        src = torch.arange(N, device=self.device).unsqueeze(1).repeat(1, k)
        return torch.stack([src.view(-1), dst.view(-1)], dim=0)

    def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:
        """
        Generate dual-view contrastive embeddings:
          - positive view: (z_pos, z_pos_pos, z_pos_neg)
          - negative view: (z_neg, z_neg_pos, z_neg_neg)

        Returns:
            z_pos     (N, D)
            z_pos_pos (N, D)
            z_pos_neg (N, k, D)
            z_neg     (N, k, D)
            z_neg_pos (N, k, D)
            z_neg_neg (N, k, k, D)
        """
        N, D = z.shape
        k = self.k
        # --- Positive view ---
        # Anchor embeddings
        z_pos = z                              # (N, D)
        # One positive neighbor per anchor
        pos_idx = [random.choice(self.pos_local[i]) if self.pos_local[i] else i
                   for i in range(N)]
        z_pos_pos = z[pos_idx]                 # (N, D)
        # k negative neighbors per anchor
        neg_src, neg_dst = neg_ei             # each is (N*k,)
        neg_dst = neg_dst.view(N, k)
        z_pos_neg = z[neg_dst]                 # (N, k, D)

        # --- Negative view ---
        # Anchors are the previously sampled negatives
        neg_src = neg_src.view(N, k)
        z_neg = z[neg_src]                     # (N, k, D)
        # One positive neighbor per negative anchor
        neg_pos_idx = []
        for i in range(N):
            row = []
            for v in neg_src[i].tolist():
                picks = self.pos_local[v] if self.pos_local[v] else [v]
                row.append(random.choice(picks))
            neg_pos_idx.append(row)
        neg_pos_idx = torch.tensor(neg_pos_idx, device=z.device)  # (N, k)
        z_neg_pos = z[neg_pos_idx]               # (N, k, D)

        # k negative neighbors per negative anchor
        neg_neg_idx = []
        for i in range(N):
            rows = []
            for v in neg_src[i].tolist():
                cands = self.neg_cands[v] or [v]
                rows.append(random.choices(cands, k=k))
            neg_neg_idx.append(rows)
        neg_neg_idx = torch.tensor(neg_neg_idx, device=z.device)  # (N, k, k)
        z_neg_neg = z[neg_neg_idx]                # (N, k, k, D)

        return z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg
