import torch
import dgl
import random

import torch
import dgl
import random

class EfficientRandomStatementSampler:
    """
    Efficient random sampler for contrastive learning, fully vectorized and memory-safe.
    """
    def __init__(self, k: int = 2, anchor_etype: str = "PPI",
                 pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        self.k = k
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype

    def prepare_global(self, full_g):
        # Number of global nodes
        N = full_g.num_nodes()
        # 1) Find anchors
        src_a, dst_a = full_g.edges(etype=self.anchor_etype)
        anchors = torch.unique(torch.cat([src_a, dst_a])).tolist()
        self.anchors_global = anchors
        A = len(anchors)

        # 2) Build reverse lookup: global node -> anchor index or -1
        g2a = torch.full((N,), -1, dtype=torch.long)
        for i, u in enumerate(anchors):
            g2a[u] = i
        self.g2a_cpu = g2a  # CPU tensor

        # 3) Build positive neighbor lists
        src_p, dst_p = full_g.edges(etype=self.pos_etype)
        src_p, dst_p = src_p.tolist(), dst_p.tolist()
        pos_global = [[] for _ in range(N)]
        rev_pos   = [[] for _ in range(N)]
        for u, v in zip(src_p, dst_p):
            pos_global[u].append(v)
            rev_pos[v].append(u)
        # Build per-anchor positive pools
        pos_pools = [pos_global[u] if pos_global[u] else [u] for u in anchors]

        # 4) Build negative candidate lists
        global_dst = set(dst_p)
        neg_pools = []
        for u in anchors:
            invalid = set(pos_global[u]) | set(rev_pos[u])
            cand = [v for v in global_dst if v not in invalid]
            if not cand:
                cand = [u]
            neg_pools.append(cand)

        # 5) Pad pos and neg pools to uniform lengths
        M_pos = max(len(r) for r in pos_pools)
        M_neg = max(len(r) for r in neg_pools)
        pos_mat = torch.zeros((A, M_pos), dtype=torch.long)
        pos_mask = torch.zeros((A, M_pos), dtype=torch.bool)
        for i, row in enumerate(pos_pools):
            L = len(row)
            pos_mat[i, :L] = torch.tensor(row, dtype=torch.long)
            pos_mask[i, :L] = True
        neg_mat = torch.zeros((A, M_neg), dtype=torch.long)
        neg_mask = torch.zeros((A, M_neg), dtype=torch.bool)
        for i, row in enumerate(neg_pools):
            L = len(row)
            neg_mat[i, :L] = torch.tensor(row, dtype=torch.long)
            neg_mask[i, :L] = True

        # Store on CPU
        self.pos_mat_cpu = pos_mat
        self.pos_mask_cpu = pos_mask
        self.neg_mat_cpu = neg_mat
        self.neg_mask_cpu = neg_mask
        self.M_pos = M_pos
        self.M_neg = M_neg

    def prepare_batch(self, batch, ppi_edge_index=None):
        """
        Prepare per-batch candidate pools.
        Works with both heterogeneous (R-GCN) and homogeneous (GCN) graphs.
        If `ppi_edge_index` is provided, uses that; otherwise reads edges from the batch by `anchor_etype`.
        """
        # 1) map local -> global IDs
        if dgl.NID in batch.ndata:
            orig = batch.ndata[dgl.NID]
        else:
            orig = torch.arange(batch.num_nodes(), dtype=torch.long)
        self.orig = orig.tolist()  # Python list for fast lookup
        # 2) get anchor edges in this batch
        if ppi_edge_index is not None:
            src, dst = ppi_edge_index
        else:
            # assume heterograph with anchor_etype
            src, dst = batch.edges(etype=self.anchor_etype)
        # unify to CPU list
        src = src.to('cpu'); dst = dst.to('cpu')
        locals_ = torch.unique(torch.cat([src, dst])).tolist()
        # 3) map global anchors to precomputed rows
        g2a = self.g2a_cpu  # CPU tensor
        rows = []
        clean_locals = []
        for loc in locals_:
            g = self.orig[loc]
            ai = int(g2a[g].item()) if 0 <= g < g2a.size(0) else -1
            if ai >= 0:
                rows.append(ai)
                clean_locals.append(loc)
        self.batch_rows = torch.tensor(rows, dtype=torch.long)
        # store local indices on device
        self.batch_locals = torch.tensor(clean_locals, dtype=torch.long, device=batch.device)
        self.B = len(self.batch_locals)
        self.device = batch.device
        # 4) gather CPU pools to GPU
        self.batch_pos_mat  = self.pos_mat_cpu[self.batch_rows].to(self.device)
        self.batch_pos_mask = self.pos_mask_cpu[self.batch_rows].to(self.device)
        self.batch_neg_mat  = self.neg_mat_cpu[self.batch_rows].to(self.device)
        self.batch_neg_mask = self.neg_mask_cpu[self.batch_rows].to(self.device)

    def sample(self) -> torch.Tensor:
        # Vectorized negative sampling
        probs = self.batch_neg_mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, self.k, replacement=True)  # (B, k)
        dst = torch.gather(self.batch_neg_mat, 1, idx)
        src = self.batch_locals.unsqueeze(1).expand(-1, self.k)
        return torch.stack([src, dst], dim=0).view(2, -1)

    def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:
        # z: (batch_size, D), neg_ei: (2, B*k)
        # 1) anchor embeddings
        z_anchor = z[self.batch_locals]
        # 2) positive embeddings via multinomial
        probs = self.batch_pos_mask.float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        idx = torch.multinomial(probs, 1).squeeze(1)  # (B,)
        pos = torch.gather(self.batch_pos_mat.to(self.device), 1, idx.unsqueeze(1)).squeeze(1)
        z_pos = z[pos]
        # 3) negative embeddings
        B, k = self.B, self.k
        neg_dst = neg_ei[1].view(B, k)
        z_negs = z[neg_dst]
        return z_anchor, z_pos, z_negs
