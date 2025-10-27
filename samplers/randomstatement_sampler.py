import random, torch, dgl

class RandomStatementSampler:
    """
    Random sampler for contrastive learning that only
    touches the subset of nodes appearing in `anchor_etype` edges to sample negatives.
    """
    def __init__(self, k: int = 2, anchor_etype: str = "PPI", pos_etype: str = "pos_statement",
                 neg_etype: str = "neg_statement"):
        self.k = k
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype


    def prepare_global(self, full_g):
        """
        Global lookups:
          - anchors_global: all nodes in anchor_etype edges
          - pos_global:     pos_statement neighbors per node
          - neg_global:     candidate negatives per anchor
        """
        N = full_g.num_nodes()
        src_a, dst_a = full_g.edges(etype=self.anchor_etype)
        anchors = set(src_a.tolist()) | set(dst_a.tolist())
        self.anchors_global = sorted(anchors)

        src_p, dst_p = full_g.edges(etype=self.pos_etype)
        pos_global = [[] for _ in range(N)]
        rev_pos_global = [[] for _ in range(N)]
        for u, v in zip(src_p.tolist(), dst_p.tolist()):
            pos_global[u].append(v)
            rev_pos_global[v].append(u)

        global_dst = set(dst_p.tolist())
        neg_global = {}
        for u in self.anchors_global:
            invalid = set(pos_global[u]) | set(rev_pos_global[u])
            neg_global[u] = [v for v in global_dst if v not in invalid]
        self.pos_global = pos_global
        self.neg_global = neg_global


    def prepare_batch(self, batch, ppi_edge_index):
        """
        For this mini‐batch, derive the batch's PPI anchors (local and global IDs) and 
        build per anchor lists of local pos and neg candidates
        """
        if dgl.NID in batch.ndata: orig = batch.ndata[dgl.NID].tolist()
        else: orig = list(range(batch.num_nodes()))
        mapping = {g: i for i, g in enumerate(orig)}

        src, dst = ppi_edge_index
        anchors_local = torch.unique(torch.cat([src, dst])).tolist()
        anchors_global = [orig[i] for i in anchors_local]
        filtered = [(g, loc) for g, loc in zip(anchors_global, anchors_local)
                    if g in self.neg_global]
        self.batch_global = [g for g, _ in filtered]
        self.batch_locals = [loc for _, loc in filtered]
        self.N = len(self.batch_locals)
        self.device = batch.device

        self.pos_local = []
        self.neg_cands = []
        for g, local in zip(self.batch_global, self.batch_locals):
            plc = []
            for v in self.pos_global[g]:
                idx = mapping.get(int(v))
                if idx is not None: plc.append(idx)
            self.pos_local.append(plc)
            ngc = []
            for v in self.neg_global[g]:
                idx = mapping.get(int(v))
                if idx is not None: ngc.append(idx)
            if not ngc: ngc = [local]
            self.neg_cands.append(ngc)


    def sample(self) -> torch.Tensor:
        """
        For each of the N batch anchors, sample k negatives (with replacement),
        and return a (2, N*k) edge index in local coordinates.
        """
        src_list = []
        dst_list = []
        for local_anchor, negs in zip(self.batch_locals, self.neg_cands):
            chosen = random.choices(negs, k=self.k)
            for neg in chosen:
                src_list.append(local_anchor)
                dst_list.append(neg)
        
        idx = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device) # stack into a 2×(N*k) tensor
        return idx


    def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:
        """
        Given z (batch_size, D) and neg_ei (2, N*k), returns:
          - z_anchor  (N,   D)
          - z_pos_pos (N,   D)
          - z_pos_neg (N, k, D)
        """
        N, D = len(self.batch_locals), z.size(1)
        k = self.k
        z_anchor = z[self.batch_locals] # (N, D)
        pos_idx = [random.choice(self.pos_local[i]) if self.pos_local[i]
                else self.batch_locals[i] for i in range(N)]
        z_pos_pos = z[pos_idx] # (N, D)
        neg_dst = neg_ei[1].view(N, k) # (N, k)
        z_pos_neg = z[neg_dst] # (N, k, D)

        return z_anchor, z_pos_pos, z_pos_neg



######### Legacy code, kept for reference #########


# class RandomStatementSampler:
#     """
#     Draws dual‐view contrastive samples only for the nodes participating
#     in a given 'anchor_etype' (e.g. 'PPI') edge type.
#     """
#     def __init__(self, k: int = 2, anchor_etype: str = "PPI",  pos_etype: str = "pos_statement"):

#         self.k = k
#         self.anchor_etype = anchor_etype
#         self.pos_etype = pos_etype

#     def prepare_global(self, full_g):
#         """
#         Build global lookups:
#           - anchors_global: set of nodes in anchor_etype edges
#           - pos_global:     pos_statement neighbors per node
#           - rev_pos_global: reverse pos_statement (for filtering negatives)
#           - global_dst:     all pos_statement destinations
#         """
#         N = full_g.num_nodes()

#         # 1) anchors = all nodes in PPI edges
#         src_a, dst_a = full_g.edges(etype=self.anchor_etype)
#         anchors = set(src_a.tolist()) | set(dst_a.tolist())
#         self.anchors_global = sorted(anchors)

#         # 2) positive statements
#         src_p, dst_p = full_g.edges(etype=self.pos_etype)
#         pos_global = [[] for _ in range(N)]
#         rev_pos_global = [[] for _ in range(N)]
#         for u, v in zip(src_p.tolist(), dst_p.tolist()):
#             pos_global[u].append(v)
#             rev_pos_global[v].append(u)

#         # 3) candidates = all positive destinations
#         self.pos_global = pos_global
#         self.rev_pos_global = rev_pos_global
#         self.global_dst   = sorted(set(dst_p.tolist()))

#     def prepare_batch(self, batch):
#         """
#         From the batch subgraph, pick only those anchors that appear here,
#         remap them to local indices, and build per-anchor pos_local & neg_cands.
#         """
#         # 1) global->local map
#         if dgl.NID in batch.ndata:
#             orig = batch.ndata[dgl.NID].tolist()
#         else:
#             orig = list(range(batch.num_nodes()))
#         mapping = {g:i for i,g in enumerate(orig)}

#         # 2) intersect anchors_global with this batch's nodes
#         self.batch_anchors_global = []
#         self.batch_anchors_local  = []
#         for g in self.anchors_global:
#             if g in mapping:
#                 self.batch_anchors_global.append(g)
#                 self.batch_anchors_local.append(mapping[g])
#         self.N = len(self.batch_anchors_local)
#         self.device = batch.device
#         # 3) build per-anchor pos_local & neg_cands
#         self.pos_local  = [[] for _ in range(self.N)]
#         self.neg_cands  = [[] for _ in range(self.N)]
#         for i, g in enumerate(self.batch_anchors_global):
#             # positive statements (within batch)
#             for nb in self.pos_global[g]:
#                 if nb in mapping:
#                     self.pos_local[i].append(mapping[nb])
#             # negatives = any global_dst not in pos or reverse-pos
#             invalid = set(self.pos_global[g]) | set(self.rev_pos_global[g])
#             for v in self.global_dst:
#                 if v in mapping and v not in invalid:
#                     self.neg_cands[i].append(mapping[v])

#     def sample(self) -> torch.Tensor:
#         """
#         Uniformly sample k negatives for each batch‐anchor (with self‐loop fallback),
#         returning a (2, N*k) edge_index in LOCAL indices.
#         """
#         N, k = self.N, self.k
#         cands = self.neg_cands
#         for i in range(N):
#             if not cands[i]:
#                 cands[i] = [ self.batch_anchors_local[i] ]

#         M = max(len(row) for row in cands)
#         C = torch.full((N, M), N, dtype=torch.long, device=self.device)
#         mask = torch.zeros((N, M), dtype=torch.bool, device=self.device)
#         for i,row in enumerate(cands):
#             L = len(row)
#             C[i, :L] = torch.tensor(row, device=self.device)
#             mask[i, :L] = True

#         # uniform sampling
#         probs = mask.float() / mask.sum(dim=1, keepdim=True)
#         idx   = torch.multinomial(probs, k, replacement=True)
#         dst   = torch.gather(C, 1, idx)
#         src   = torch.tensor(self.batch_anchors_local,
#                              device=self.device).unsqueeze(1).repeat(1, k)
#         ei    = torch.stack([src.view(-1), dst.view(-1)], dim=0)
#         return ei

#     def get_contrastive_samples(self, z:torch.Tensor, neg_ei:torch.Tensor) -> tuple:
#         """
#         Receives node embeddings z of shape (batch_size, D), and neg_ei (2,N*k)
#         """
#         k = self.k
#         D = z.size(1)
#         z_anchor = z[self.batch_anchors_local]     # (N, D)
#         # 2) positive neighbors
#         pos_idx = [random.choice(self.pos_local[i]) if self.pos_local[i] else self.batch_anchors_local[i]
#             for i in range(self.N)]
#         z_pos_pos = z[pos_idx]                     # (N, D)
#         # 3) negative neighbors
#         neg_dst   = neg_ei[1].view(self.N, k)      # (N, k)
#         z_pos_neg = z[neg_dst]                     # (N, k, D)
#         return z_anchor, z_pos_pos, z_pos_neg


# class RandomStatementSampler:
#     """
#     A sampler that draws negative samples by uniformly sampling over existing "pos_statement" edges,
#     ensuring sampled edges don't already exist (including reverse edges),
#     and generates dual-view contrastive samples (positive and negative views).
#     """
#     def __init__(self, k: int = 2, pos_etype: str = "pos_statement"):
#         """
#         Args:
#             k (int): number of negatives per view.
#             pos_etype (str): edge type for positive statements.
#         """
#         self.k = k
#         self.pos_etype = pos_etype

#     def prepare_global(self, full_g):
#         """
#         Build global adjacency and reverse adjacency lists from the full graph.
#         """
#         N = full_g.num_nodes()
#         # Extract all positive edges
#         src_pos, dst_pos = full_g.edges(etype=self.pos_etype)
#         src_pos, dst_pos = src_pos.tolist(), dst_pos.tolist()

#         # Forward and reverse adjacency lists
#         pos_global = [[] for _ in range(N)]
#         rev_pos_global = [[] for _ in range(N)]
#         for u, v in zip(src_pos, dst_pos):
#             pos_global[u].append(v)
#             rev_pos_global[v].append(u)

#         # Unique set of all destination nodes for negatives
#         global_dst = list(set(dst_pos))

#         self.pos_global = pos_global
#         self.rev_pos_global = rev_pos_global
#         self.global_dst = global_dst

#     def prepare_batch(self, batch):
#         """
#         Map global node IDs to batch-local indices and build per-node candidate lists.
#         """
#         # Original global IDs of batch nodes
#         if dgl.NID in batch.ndata:
#             orig = batch.ndata[dgl.NID].tolist()
#         else:
#             orig = list(range(batch.num_nodes()))
#         mapping = {g: i for i, g in enumerate(orig)}

#         self.N = len(orig)
#         self.device = batch.device

#         # Local lists for positives and negatives
#         self.pos_local = [[] for _ in range(self.N)]
#         self.neg_cands = [[] for _ in range(self.N)]

#         # Build lists
#         for i, g_id in enumerate(orig):
#             # Positive neighbors in batch
#             for nb in self.pos_global[g_id]:
#                 if nb in mapping:
#                     self.pos_local[i].append(mapping[nb])
#             # Negative candidates: any global_dst not forward or reverse neighbor
#             invalid = set(self.pos_global[g_id]) | set(self.rev_pos_global[g_id])
#             for v in self.global_dst:
#                 if v in mapping and v not in invalid:
#                     self.neg_cands[i].append(mapping[v])
    

#     def sample(self) -> torch.Tensor:
#         """
#         Sample k negative edges for each node in the current batch.

#         Returns:
#             ei (torch.Tensor): shape (2, N*k) edge index tensor of negative samples.
#         """
#         N, k = self.N, self.k
#         cands = self.neg_cands

#         # Handle case where some nodes have no candidates: fallback to self-loop
#         for i in range(N):
#             if not cands[i]:
#                 cands[i] = [i]

#         # Pad candidate lists and create mask
#         M = max(len(r) for r in cands)
#         C = torch.full((N, M), N, dtype=torch.long, device=self.device)
#         mask = torch.zeros((N, M), dtype=torch.bool, device=self.device)
#         for i, row in enumerate(cands):
#             L = len(row)
#             C[i, :L] = torch.tensor(row, device=self.device)
#             mask[i, :L] = True

#         # Uniform sampling over mask
#         probs = mask.float()
#         probs = probs / probs.sum(dim=1, keepdim=True)
#         idx = torch.multinomial(probs, k, replacement=True)
#         dst = torch.gather(C, 1, idx)
#         src = torch.arange(N, device=self.device).unsqueeze(1).repeat(1, k)
#         return torch.stack([src.view(-1), dst.view(-1)], dim=0)


#     def get_contrastive_samples(self, z: torch.Tensor, neg_ei: torch.Tensor) -> tuple:

#         N, D = z.shape
#         k = self.k
#         z_pos = z
#         pos_nb = [random.choice(self.pos_local[u]) if self.pos_local[u] else u for u in range(N)]
#         z_pos_pos = z[pos_nb]
#         neg_dst = neg_ei[1].view(N, k)         
#         z_pos_neg = z[neg_dst]
#         return z_pos, z_pos_pos, z_pos_neg
