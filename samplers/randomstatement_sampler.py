import random, torch
from typing import List, Tuple, Dict
from torch import Tensor
from torch_geometric.data import HeteroData
from .utils import _find_key_by_rel, _num_nodes_of

class RandomStatementSampler:
    """ Random sampler that only touches the subset of nodes appearing in `anchor_etype` edges to sample negatives.
    """
    def __init__(self, k: int = 2, anchor_etype: str = "PPI", pos_etype: str = "pos_statement", neg_etype: str = "neg_statement"):
        self.k = k
        self.anchor_etype = anchor_etype
        self.pos_etype = pos_etype
        self.neg_etype = neg_etype
        self.anchors_global: List[int] = []
        self.pos_global: List[List[int]] = []
        self.neg_global: Dict[int, List[int]] = {}
        self.batch_global: List[int] = []
        self.batch_locals: List[int] = []
        self.pos_local: List[List[int]] = []
        self.neg_cands: List[List[int]] = []
        self.N: int = 0
        self.device = torch.device("cpu")



    def prepare_global(self, full_g: HeteroData):
        """ Global lookups: anchors_global: all nodes in anchor_etype edges; pos_global: pos_statement neighbors per node
          neg_global: candidate negatives per anchor (global pos targets minus a small invalid set)
        """
        anchor_key = _find_key_by_rel(full_g, self.anchor_etype)
        src_a, dst_a = full_g[anchor_key].edge_index
        anchors = set(src_a.tolist()) | set(dst_a.tolist())
        self.anchors_global = sorted(anchors)

        pos_key = _find_key_by_rel(full_g, self.pos_etype)
        src_nt, _, dst_nt = pos_key
        N = _num_nodes_of(full_g, src_nt)

        pos_global = [[] for _ in range(N)]
        rev_pos_global = [[] for _ in range(N)]
        if "edge_index" in full_g[pos_key]:
            src_p, dst_p = full_g[pos_key].edge_index
            for u, v in zip(src_p.tolist(), dst_p.tolist()):
                pos_global[u].append(v)
                rev_pos_global[v].append(u)
            global_dst = set(dst_p.tolist())
        else: global_dst = set()

        neg_global: Dict[int, List[int]] = {}
        for u in self.anchors_global:
            invalid = set(pos_global[u]) | set(rev_pos_global[u])
            neg_global[u] = [v for v in global_dst if v not in invalid] if global_dst else []
        self.pos_global = pos_global
        self.neg_global = neg_global



    def prepare_batch(self, batch: HeteroData, ppi_edge_index: Tensor):
        """ Derive the batch's PPI anchors (local and global IDs) and builds per-anchor lists of local pos and neg candidates.
        """
        any_key = next(iter(batch.edge_types))
        self.device = batch[any_key].edge_index.device if batch[any_key].edge_index.is_cuda else torch.device("cpu")
        node_type = _find_key_by_rel(batch, self.anchor_etype)[0]
        B = _num_nodes_of(batch, node_type)
        orig = list(range(B))
        mapping: Dict[int, int] = {g: i for i, g in enumerate(orig)}

        src, dst = ppi_edge_index
        anchors_local = torch.unique(torch.cat([src, dst])).tolist()
        anchors_global = [orig[i] for i in anchors_local]
        filtered = [(g, loc) for g, loc in zip(anchors_global, anchors_local) if g in self.neg_global]
        self.batch_global = [g for g, _ in filtered]
        self.batch_locals = [loc for _, loc in filtered]
        self.N = len(self.batch_locals)
        self.pos_local = []
        self.neg_cands = []
        for g, local in zip(self.batch_global, self.batch_locals):
            plc: List[int] = []
            for v in self.pos_global[g]:
                idx = mapping.get(int(v))
                if idx is not None: plc.append(idx)
            self.pos_local.append(plc)
            ngc: List[int] = []
            for v in self.neg_global[g]:
                idx = mapping.get(int(v))
                if idx is not None: ngc.append(idx)
            if not ngc: ngc = [local]
            self.neg_cands.append(ngc)


    def sample(self) -> Tensor:
        """ For each of the N batch anchors, samples k negatives with replacement and returns edge index in local coordinates.
        """
        if self.N == 0: return torch.empty(2, 0, dtype=torch.long, device=self.device)
        src_list: List[int] = []
        dst_list: List[int] = []
        for local_anchor, negs in zip(self.batch_locals, self.neg_cands):
            chosen = random.choices(negs, k=self.k)
            src_list.extend([local_anchor] * self.k)
            dst_list.extend(chosen)
        idx = torch.tensor([src_list, dst_list], dtype=torch.long, device=self.device)
        return idx


    def get_contrastive_samples(self, z: Tensor, neg_ei: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Args: z: (B, D) node embeddings for the batch (local indexing); neg_ei: (2, N*k) negatives from `sample()`
        Returns: z_anchor (N,D); z_pos_pos (N,D); z_pos_neg (N,k,D)
        """
        N = len(self.batch_locals)
        if N == 0:
            D = z.size(-1)
            return (torch.empty(0, D, device=z.device), torch.empty(0, D, device=z.device), torch.empty(0, self.k, D, device=z.device))

        k = self.k
        z_anchor = z[self.batch_locals]
        pos_idx = [random.choice(self.pos_local[i]) if self.pos_local[i] else self.batch_locals[i] for i in range(N)]
        pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=z.device)
        z_pos_pos = z[pos_idx]
        neg_dst = neg_ei[1].view(N, k)
        z_pos_neg = z[neg_dst]
        return z_anchor, z_pos_pos, z_pos_neg
