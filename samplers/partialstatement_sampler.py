import torch, random
from typing import List, Tuple, Optional, Dict
from torch import Tensor
from torch_geometric.data import HeteroData


class PartialStatementSampler:
    """Ontology/GO-guided negative sampler using external statement edges.
    Global structures (built once in `prepare_global`):
      • pos_global[u]   : list of positive neighbors (global node ids)
      • direct_global[u]: list of direct negative neighbors
      • two_hop_global[u]: subclass-based expanded negatives
      • neg_pool[u]     : final negative pool (direct + 2-hop + padding)
      • anchors         : anchor nodes (instances if possible)
    Batch-time:
      `get_contrastive_samples(z, anchor_nodes, n_id)`:
        - z: [B, D] embeddings (local node order for this batch/subgraph)
        - n_id: [B] global node ids for rows in z (or None if full-graph)
        - anchor_nodes: local row indices into z (e.g. nodes in current triple batch)
        -> returns (z_pos, z_pos_pos, z_pos_neg) with shapes:
             z_pos     : [B_anchor, D]
             z_pos_pos : [B_anchor, D]
             z_pos_neg : [B_anchor, k, D]
    """

    def __init__(self, k: int = 1, go_etype: str = "subclass_of",
        neg_edges: Optional[List[Tuple[int, int]]] = None, edges_are_negative: bool = True,
        instance_rel: str = "2", neg_prefix: str = "NOT_"):
        self.k = k
        self.go_etype = go_etype
        self.external_neg = neg_edges or []
        self.edges_are_negative = edges_are_negative
        self.instance_rel = instance_rel
        self.neg_prefix = neg_prefix

        self.pos_global: List[List[int]] = []
        self.direct_global: List[List[int]] = []
        self.two_hop_global: List[List[int]] = []
        self.neg_pool: List[List[int]] = []
        self.pos_targets: set = set()
        self.anchors: List[int] = []
        self.node_type: str = "node"
        self.num_nodes: int = 0

    def prepare_global(self, full_g: HeteroData) -> None:
        """  Scan the full graph and build positive/negative neighbor structures,
        GO / subclass expansion, anchor set, and per-node negative pools.
        """
        if len(full_g.node_types) != 1:
            raise ValueError(
                f"PartialStatementSampler assumes a single node type; got {full_g.node_types}")
        self.node_type = full_g.node_types[0]
        if hasattr(full_g[self.node_type], "num_nodes") and full_g[self.node_type].num_nodes is not None:
            N = int(full_g[self.node_type].num_nodes)
        else:
            max_id = -1
            for (_, _, _), eidx in full_g.edge_index_dict.items():
                if eidx.numel() > 0:
                    max_id = max(max_id, int(eidx.max().item()))
            N = max_id + 1 if max_id >= 0 else 0

        self.num_nodes = N
        self.pos_global = [[] for _ in range(N)]
        self.direct_global = [[] for _ in range(N)]
        self.two_hop_global = [[] for _ in range(N)]
        self.neg_pool = [[] for _ in range(N)]
        self.pos_targets = set()
        predecessors: Dict[int, List[int]] = {}
        anchor_sources: set = set()

        if self.external_neg:
            if self.edges_are_negative:
                for u, v in self.external_neg:
                    if 0 <= u < N and 0 <= v < N:self.direct_global[u].append(v)
            else:
                for u, v in self.external_neg:
                    if 0 <= u < N and 0 <= v < N:
                        self.pos_global[u].append(v)
                        self.pos_targets.add(v)

        for (src_nt, rel, dst_nt), eidx in full_g.edge_index_dict.items():
            if src_nt != self.node_type or dst_nt != self.node_type: continue
            if eidx.numel() == 0: continue
            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            if rel == self.go_etype:
                for s, d in zip(src_list, dst_list):
                    if 0 <= s < N and 0 <= d < N: predecessors.setdefault(d, []).append(s)
                continue

            if self.instance_rel is not None and rel == self.instance_rel:
                for s in src_list:
                    if 0 <= s < N: anchor_sources.add(s)
                continue

            if rel.startswith(self.neg_prefix):
                for u, v in zip(src_list, dst_list):
                    if 0 <= u < N and 0 <= v < N: self.direct_global[u].append(v)
                continue

            for u, v in zip(src_list, dst_list):
                if 0 <= u < N and 0 <= v < N:
                    self.pos_global[u].append(v)
                    self.pos_targets.add(v)

        for u in range(N):
            if len(self.direct_global[u]) < self.k:
                sup = set()
                for b in self.direct_global[u]:
                    for c in predecessors.get(b, []):
                        sup.add(c)
                self.two_hop_global[u] = list(sup)

        for u in range(N):
            if len(self.direct_global[u]) >= self.k: pool = self.direct_global[u].copy()
            else: pool = list({*self.direct_global[u],*self.two_hop_global[u]})

            if len(pool) < self.k:
                excluded = set(self.direct_global[u]) | set(self.pos_global[u])
                candidates = list(self.pos_targets - excluded)
                needed = self.k - len(pool)
                if candidates:
                    if len(candidates) >= needed:pool.extend(random.sample(candidates, needed))
                    else: pool.extend(random.choices(candidates, k=needed))

            if not pool: pool = [u]
            self.neg_pool[u] = pool

        if self.instance_rel is not None and anchor_sources: anchors = sorted(anchor_sources)
        else:
            anchors = [u for u in range(N) if (self.pos_global[u] or self.direct_global[u])]
            if not anchors: anchors = list(range(N))

        filtered = [u for u in anchors if (len(self.pos_global[u]) > 0 and len(self.neg_pool[u]) > 0)]
        self.anchors = filtered if filtered else anchors


    def prepare_batch(self, batch: HeteroData, pos_index: Optional[Tensor] = None) -> None:
        """For compatibility with Train/Train_BestModel. No per-batch state needed."""
        return

    def get_contrastive_samples(self, z: Tensor, anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build contrastive triples (z_pos, z_pos_pos, z_pos_neg).
        Args:
            z: [B, D] node embeddings. If n_id is None, row i is node i (global).
               If n_id is provided, row i corresponds to global node n_id[i].
            anchor_nodes: indices into z (local row indices) to consider as anchors.
            n_id: Optional LongTensor[B] of global node IDs for rows of z.
        """
        device = z.device
        B, D = z.shape

        if n_id is None: global_for_row = torch.arange(B, device=device, dtype=torch.long)
        else: global_for_row = n_id.to(device).long()

        global_list = global_for_row.cpu().tolist()
        global2local = {int(g): i for i, g in enumerate(global_list)}
        all_batch_globals = set(global_list)

        if anchor_nodes is not None and anchor_nodes.numel() > 0:
            anchor_nodes = anchor_nodes.detach().long().to(device)
            row_indices = torch.unique(anchor_nodes).cpu().tolist()
            anchor_globals = [int(global_for_row[i]) for i in row_indices]
        else: anchor_globals = self.anchors if self.anchors else list(range(self.num_nodes))
        anchor_globals = [u for u in anchor_globals if u in all_batch_globals]
        if self.anchors:
            anchor_set = set(self.anchors)
            filtered = [u for u in anchor_globals if u in anchor_set]
            if filtered: anchor_globals = filtered

        if not anchor_globals: anchor_globals = list(all_batch_globals)
        if not anchor_globals: anchor_globals = [0]

        z_pos_list = []
        z_pos_pos_list = []
        z_pos_neg_list = []

        for u in anchor_globals:
            if u not in global2local: continue
            u_local = global2local[u]
            z_pos_list.append(z[u_local].unsqueeze(0))

            pos_candidates_global = [
                v for v in (self.pos_global[u] if u < len(self.pos_global) else [])
                if v in all_batch_globals]

            if pos_candidates_global: v_pos_global = random.choice(pos_candidates_global)
            else: v_pos_global = u
            v_pos_local = global2local.get(v_pos_global, u_local)
            z_pos_pos_list.append(z[v_pos_local].unsqueeze(0))

            neg_candidates_global = [v for v in (self.neg_pool[u] if u < len(self.neg_pool) else [])
                if v in all_batch_globals]
            neg_indices_local: List[int] = []

            if neg_candidates_global:
                if len(neg_candidates_global) >= self.k:
                    chosen_globals = random.sample(neg_candidates_global, self.k)
                else: chosen_globals = random.choices(neg_candidates_global, k=self.k)
                for v in chosen_globals:
                    v_local = global2local.get(v, u_local)
                    neg_indices_local.append(v_local)
            else:
                batch_locals = list(range(B))
                if len(batch_locals) >= self.k: chosen_locals = random.sample(batch_locals, self.k)
                else: chosen_locals = random.choices(batch_locals, k=self.k)
                neg_indices_local = chosen_locals
            neg_idx_tensor = torch.tensor(neg_indices_local, dtype=torch.long, device=device)
            z_neg_for_u = z[neg_idx_tensor]
            z_pos_neg_list.append(z_neg_for_u.unsqueeze(0))

        if not z_pos_list:
            anchor_rows = list(range(min(B, max(self.k, 1))))
            z_pos = z[anchor_rows]
            z_pos_pos = z[anchor_rows]
            z_pos_neg = z[anchor_rows].unsqueeze(1).expand(-1, self.k, -1)
            return z_pos, z_pos_pos, z_pos_neg

        z_pos = torch.cat(z_pos_list, dim=0)
        z_pos_pos = torch.cat(z_pos_pos_list, dim=0)
        z_pos_neg = torch.cat(z_pos_neg_list, dim=0)
        return z_pos, z_pos_pos, z_pos_neg
