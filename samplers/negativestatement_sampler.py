import dgl, torch, random
class NegativeStatementSampler:
    """
    A sampler that uses negative statements as negative samples
    for each node in the training graph.
    """
    def __init__(self,  k: int = 2):
        self.k = k


    def _get_neg_candidates(self, g: dgl.DGLGraph, nid: int, e_type: str = "neg_statement") -> list:

            direct = g.successors(nid, etype=e_type).tolist()
            if len(direct) < self.k:
                two_hop = []
                for nbr in direct:
                    two_hop.extend(g.successors(nbr, etype=e_type).tolist())
                direct = list({*direct, *two_hop})
            return direct


    def sample(self, g: dgl.DGLGraph) -> torch.Tensor:
            """
            Returns a [2, N_neg] tensor of negative edges of (src, dst) format
            where N_neg = num_nodes * k
            """
            node_ids = g.nodes().tolist()
            neg_src, neg_dst = [], []
            for nid in node_ids:
                candidates = self._get_neg_candidates(g, nid)
                if len(candidates) >= self.k: picked = random.sample(candidates, self.k)
                else:
                    others = [n for n in node_ids if n != nid]
                    picked = random.choices(others, k=self.k)
                for neg in picked:
                    neg_src.append(nid)
                    neg_dst.append(neg)
            edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=g.device)
            return edge_index


    def get_contrastive_samples(self, batch: dgl.DGLGraph, z: torch.Tensor):

        N, D = z.size()
        z_pos = z
        pos_nb = []
        for nid in range(N):
            outs = batch.successors(nid, etype="pos_statement").tolist()
            pos_nb.append(random.choice(outs) if outs else nid)

        z_pos_pos = z[pos_nb]
        neg_ei = self.sample(batch)
        neg_dst = neg_ei[1].view(N, self.k)
        z_pos_neg = z[neg_dst]
        z_neg     = z_pos
        z_neg_pos = z_pos_pos
        z_neg_neg = z_pos_neg

        return z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg