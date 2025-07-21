import torch
import random
import dgl

class NegativeStatementSampler:
    """
    A sampler that draws k negatives (and one positive) for each anchor node in the
    'anchor_etype' edges (e.g. 'PPI'), and also one negative-statement neighbor as
    the positive of the negative view.
    """
    def __init__(self,
                 k: int = 2,
                 go_etype: str = "link",
                 anchor_etype: str = "PPI", # gene_disease_association
                 pos_etype: str = "pos_statement",
                 neg_etype: str = "neg_statement"):
        self.k = k
        self.go_etype = go_etype
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype

    def prepare_global(self, full_g):
        """
        Build global structures:
          - self.anchors_global: all unique nodes in the anchor_etype edges
          - self.pos_global:   positive neighbors per node
          - self.direct_global: negative-statement neighbors per node
          - self.two_hop_global: 2-hop backups for nodes with < k negatives
        """
        N = full_g.num_nodes()

        # --- anchors: all nodes in the PPI edges ---
        src_anchor, dst_anchor = full_g.edges(etype=self.anchor_etype)
        anchors = set(src_anchor.tolist()) | set(dst_anchor.tolist())
        self.anchors_global = sorted(anchors)  # our n anchor nodes

        # --- positive‐statement neighbors ---
        src_pos, dst_pos = full_g.edges(etype=self.pos_etype)
        pos = [[] for _ in range(N)]
        for u, v in zip(src_pos.tolist(), dst_pos.tolist()):
            pos[u].append(v)

        # --- negative‐statement neighbors ---
        src_neg, dst_neg = full_g.edges(etype=self.neg_etype)
        direct = [[] for _ in range(N)]
        for u, v in zip(src_neg.tolist(), dst_neg.tolist()):
            direct[u].append(v)

        # --- two‐hop fallback for nodes with too few direct negatives ---
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

    def prepare_batch(self, batch):
        """
        Given a DGLGraph or block `batch`, figure out which of our
        global anchors actually appear in this batch, and store both
        their global IDs and their batch-local indices.
        """
        # 1) Get the global node IDs for this batch
        if dgl.NID in batch.ndata:
            orig = batch.ndata[dgl.NID].tolist()   # maps local idx -> global ID
        else:
            orig = list(range(batch.num_nodes()))

        # 2) Build a reverse lookup: global ID -> local index
        mapping = {g: i for i, g in enumerate(orig)}

        # 3) Intersect our global anchors with the batch’s nodes
        #    and keep both the global and local lists
        self.batch_anchors_global = []
        self.batch_anchors_local  = []
        for u in self.anchors_global:
            if u in mapping:
                self.batch_anchors_global.append(u)
                self.batch_anchors_local.append(mapping[u])

        # 4) Record how many anchors we have in this batch
        self.Nb = len(self.batch_anchors_local)
        self.device = batch.device
        

    def get_contrastive_samples(self, z: torch.Tensor) -> tuple:
        """
        Given the full-node embedding matrix z (shape [N_all, D]),
        returns six tensors for n anchors
        """
        D = z.size(1)
        k = self.k
        anchors = self.anchors_global         # list of global node‐IDs
        n = len(anchors)
        z_anchor = z[anchors]                 # (n, D)

        # --- positive view ---
        pos_nb = [
            random.choice(self.pos_global[u]) if self.pos_global[u] else u
            for u in anchors
        ]
        pos_nb_2 = [
            random.choice(self.pos_global[u]) if self.pos_global[u] else u
            for u in anchors
        ]                                   # length n
        z_pos_pos = z[pos_nb]               # (n, D)
        z_pos_pos_2 = z[pos_nb_2]             # (n, D)

        neg_lists = []
        for u in anchors:
            pool = (
                self.direct_global[u]
                if len(self.direct_global[u]) >= k
                else list({*self.direct_global[u], *self.two_hop_global.get(u, [])}) or [u]
            )
            neg_lists.append(random.choices(pool, k=k))
        neg_lists = torch.tensor(neg_lists, device=z.device)  # (n, k)
        z_pos_neg = z[neg_lists]                              # (n, k, D)

        # --- negative view ---
        z_neg = z_anchor                                        # (n, D)
        neg_primary = neg_lists[:, 0].tolist()
        z_neg_pos = z[neg_primary]                             # (n, D)
        z_neg_neg = torch.stack([z_pos_pos,z_pos_pos_2], dim=1)# (n, D)

        return z_anchor, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg
