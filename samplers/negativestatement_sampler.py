import random
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.data import HeteroData


class NegativeStatementSampler:
    """ Ontology-guided negative sampler for contrastive learning on a Wikidata-style KG.
    (Same docstring as before; omitted here for brevity.)
    """

    def __init__(self, k: int = 1, subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_", instance_rel: str = "2"):
        self.k = k
        self.subclass_rel = subclass_rel
        self.neg_prefix = neg_prefix
        self.instance_rel = instance_rel
        self.node_type: str = "node"
        self.num_nodes: int = 0
        self.pos_neighbors: List[List[int]] = []
        self.direct_neg_neighbors: List[List[int]] = []
        self.neg_pool: List[List[int]] = []
        self.anchors: List[int] = []

    def prepare_global(self, full_g: HeteroData) -> None:
        if len(full_g.node_types) != 1:
            raise ValueError("NegativeStatementSampler currently assumes a single node type; "
                             f"got node types: {full_g.node_types}")
        self.node_type = full_g.node_types[0]

        if hasattr(full_g[self.node_type], "num_nodes") and full_g[self.node_type].num_nodes is not None:
            self.num_nodes = int(full_g[self.node_type].num_nodes)
        else:
            max_id = -1
            for (_, _, _), eidx in full_g.edge_index_dict.items():
                max_id = max(max_id, int(eidx.max().item()))
            self.num_nodes = max_id + 1

        N = self.num_nodes
        self.pos_neighbors = [[] for _ in range(N)]
        self.direct_neg_neighbors = [[] for _ in range(N)]

        anchor_sources: set = set()

        if self.instance_rel is not None:
            inst_key = self._find_edge_key(full_g, self.instance_rel)
            if inst_key is not None and "edge_index" in full_g[inst_key]:
                inst_src = full_g[inst_key].edge_index[0].tolist()
                anchor_sources.update(inst_src)

        for (src_nt, rel, dst_nt), eidx in full_g.edge_index_dict.items():
            if src_nt != self.node_type or dst_nt != self.node_type:
                continue
            if rel == self.subclass_rel:
                continue

            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            if self.instance_rel is not None and rel == self.instance_rel:
                continue

            if rel.startswith(self.neg_prefix):
                for u, v in zip(src_list, dst_list):
                    if 0 <= u < N and 0 <= v < N:
                        self.direct_neg_neighbors[u].append(v)
            else:
                for u, v in zip(src_list, dst_list):
                    if 0 <= u < N and 0 <= v < N:
                        self.pos_neighbors[u].append(v)
                        if self.instance_rel is None:
                            anchor_sources.add(u)

        self.anchors = sorted(anchor_sources)

        subclass_predecessors: Dict[int, List[int]] = {}
        subclass_key = self._find_edge_key(full_g, self.subclass_rel)
        if subclass_key is not None and "edge_index" in full_g[subclass_key]:
            sub_src, sub_dst = full_g[subclass_key].edge_index
            for c, p in zip(sub_src.tolist(), sub_dst.tolist()):
                if 0 <= c < N and 0 <= p < N:
                    subclass_predecessors.setdefault(p, []).append(c)

        self.neg_pool = [[] for _ in range(N)]
        for u in range(N):
            pool = set(self.direct_neg_neighbors[u])
            for b in self.direct_neg_neighbors[u]:
                for c in subclass_predecessors.get(b, []):
                    pool.add(c)
            self.neg_pool[u] = list(pool)

        filtered_anchors = [u for u in self.anchors
            if (len(self.pos_neighbors[u]) > 0 and len(self.neg_pool[u]) > 0)]
        if filtered_anchors: self.anchors = filtered_anchors

        self.pos_neighbors_t = [torch.as_tensor(nb, dtype=torch.long)
            if len(nb) > 0 else torch.empty(0, dtype=torch.long)
            for nb in self.pos_neighbors]
        self.neg_pool_t = [torch.as_tensor(nb, dtype=torch.long)
            if len(nb) > 0 else torch.empty(0, dtype=torch.long)
            for nb in self.neg_pool]

        self._global2local_cpu = None
        self._in_batch_mask_cpu = None


    def prepare_batch(self, batch: HeteroData,
        pos_index: Optional[Tensor] = None) -> None:
        return

    def get_contrastive_samples(
        self,
        z: Tensor,
        anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build contrastive triples (z_pos, z_pos_pos, z_pos_neg) with much lower Python
        overhead by using tensor masks and a tensor-based global->local mapping.
        Args:
            z: [N, D] node embeddings. If `n_id` is None, row i is node i (global).
               If `n_id` is provided, row i corresponds to global node n_id[i].
            anchor_nodes: indices into `z` to consider as anchors (row indices).
            n_id: Optional LongTensor[N] of global node IDs for rows of `z`.
        """
        device = z.device
        N, D = z.shape

        if n_id is None: global_for_row = torch.arange(N, dtype=torch.long)
        else: global_for_row = n_id.detach().long().cpu()

        if (getattr(self, "_global2local_cpu", None) is None or
                self._global2local_cpu.numel() < self.num_nodes):
            self._global2local_cpu = torch.full((self.num_nodes,), -1, dtype=torch.long)
        global2local = self._global2local_cpu
        global2local.fill_(-1)
        global2local[global_for_row] = torch.arange(global_for_row.size(0), dtype=torch.long)

        if (getattr(self, "_in_batch_mask_cpu", None) is None or
                self._in_batch_mask_cpu.numel() < self.num_nodes):
            self._in_batch_mask_cpu = torch.zeros(self.num_nodes, dtype=torch.bool)
        in_batch = self._in_batch_mask_cpu
        in_batch.zero_()
        in_batch[global_for_row] = True

        if anchor_nodes is not None and anchor_nodes.numel() > 0:
            anchor_local = anchor_nodes.detach().long().cpu().unique()
            anchor_globals = global_for_row[anchor_local]
        else:
            if self.anchors: anchor_globals = torch.as_tensor(self.anchors, dtype=torch.long)
            else: anchor_globals = torch.arange(self.num_nodes, dtype=torch.long)
        anchor_globals = anchor_globals[in_batch[anchor_globals]]

        if self.anchors and anchor_nodes is not None and anchor_nodes.numel() > 0:
            anchor_filter = torch.as_tensor(self.anchors, dtype=torch.long)
            mask = torch.isin(anchor_globals, anchor_filter)
            if mask.any(): anchor_globals = anchor_globals[mask]

        if anchor_globals.numel() == 0: anchor_globals = global_for_row.clone()
        if anchor_globals.numel() == 0: anchor_globals = torch.tensor([0], dtype=torch.long)

        anchor_local_cpu = global2local[anchor_globals]
        valid = anchor_local_cpu >= 0
        anchor_local_cpu = anchor_local_cpu[valid]
        anchor_globals = anchor_globals[valid]

        if anchor_local_cpu.numel() == 0:
            num_fallback = min(N, max(self.k, 1))
            anchor_local_cpu = torch.arange(num_fallback, dtype=torch.long)
            anchor_globals = global_for_row[anchor_local_cpu]
        B = anchor_local_cpu.numel()

        if not hasattr(self, "pos_neighbors_t"):
            self.pos_neighbors_t = [torch.as_tensor(nb, dtype=torch.long)
                if len(nb) > 0 else torch.empty(0, dtype=torch.long)
                for nb in self.pos_neighbors]

        if not hasattr(self, "neg_pool_t"):
            self.neg_pool_t = [torch.as_tensor(nb, dtype=torch.long)
                if len(nb) > 0 else torch.empty(0, dtype=torch.long)
                for nb in self.neg_pool]

        pos_neighbors_t = self.pos_neighbors_t
        neg_pool_t = self.neg_pool_t

        z_pos_pos_idx_local = torch.empty(B, dtype=torch.long)
        z_pos_neg_idx_local = torch.empty(B, self.k, dtype=torch.long)

        for i, u_global in enumerate(anchor_globals.tolist()):
            u_local = int(anchor_local_cpu[i].item())
            neigh = pos_neighbors_t[u_global]
            if neigh.numel() > 0: neigh_in_batch = neigh[in_batch[neigh]]
            else: neigh_in_batch = neigh

            if neigh_in_batch.numel() > 0:
                j = torch.randint(0, neigh_in_batch.numel(), (1,), dtype=torch.long).item()
                v_pos_global = int(neigh_in_batch[j].item())
                v_pos_local = int(global2local[v_pos_global].item())
                if v_pos_local < 0: v_pos_local = u_local
            else: v_pos_local = u_local

            z_pos_pos_idx_local[i] = v_pos_local
            neg_neigh = neg_pool_t[u_global]
            if neg_neigh.numel() > 0: neg_in_batch = neg_neigh[in_batch[neg_neigh]]
            else: neg_in_batch = neg_neigh

            if neg_in_batch.numel() > 0:
                if neg_in_batch.numel() >= self.k: choice_idx = torch.randperm(neg_in_batch.numel())[:self.k]
                else: choice_idx = torch.randint(0, neg_in_batch.numel(), (self.k,), dtype=torch.long)
                chosen_globals = neg_in_batch[choice_idx]
                chosen_locals = global2local[chosen_globals]
                bad_mask = chosen_locals < 0
                chosen_locals[bad_mask] = u_local
            else:
                if global_for_row.numel() >= self.k:choice_idx = torch.randperm(global_for_row.numel())[:self.k]
                else:choice_idx = torch.randint(0, global_for_row.numel(), (self.k,), dtype=torch.long)
                chosen_locals = choice_idx
            z_pos_neg_idx_local[i] = chosen_locals

        anchor_local = anchor_local_cpu.to(device)
        z_pos = z[anchor_local]
        z_pos_pos = z[z_pos_pos_idx_local.to(device)]
        z_pos_neg = z[z_pos_neg_idx_local.to(device)]
        return z_pos, z_pos_pos, z_pos_neg


    @staticmethod
    def _find_edge_key(g: HeteroData, rel: str, src_ntype: Optional[str] = None,
        dst_ntype: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
        for (s, r, d) in g.edge_types:
            if r != rel:
                continue
            if src_ntype is not None and s != src_ntype:
                continue
            if dst_ntype is not None and d != dst_ntype:
                continue
            return (s, r, d)
        return None
