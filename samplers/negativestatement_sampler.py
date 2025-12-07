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

        filtered_anchors = [
            u for u in self.anchors
            if (len(self.pos_neighbors[u]) > 0 and len(self.neg_pool[u]) > 0)
        ]
        if filtered_anchors:
            self.anchors = filtered_anchors

    def prepare_batch(self, batch: HeteroData,
        pos_index: Optional[Tensor] = None) -> None:
        # No per-batch state needed; global structures already built.
        return

    def get_contrastive_samples(
        self,
        z: Tensor,
        anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build contrastive triples (z_pos, z_pos_pos, z_pos_neg).

        Args:
            z: [N, D] node embeddings. If `n_id` is None, row i is node i (global).
               If `n_id` is provided, row i corresponds to global node n_id[i].
            anchor_nodes: indices into `z` to consider as anchors (row indices).
            n_id: Optional LongTensor[N] of global node IDs for rows of `z`.
        """
        device = z.device
        N, D = z.shape

        # Map rows in z to global IDs
        if n_id is None:
            global_for_row = torch.arange(N, device=device, dtype=torch.long)
            global2local = {int(i): int(i) for i in range(N)}
        else:
            n_id = n_id.to(device).long()
            global_for_row = n_id
            global2local = {int(g): i for i, g in enumerate(global_for_row.cpu().tolist())}

        all_batch_globals = set(int(g) for g in global_for_row.cpu().tolist())

        # Determine candidate anchors in *global* id space
        if anchor_nodes is not None and anchor_nodes.numel() > 0:
            anchor_nodes = anchor_nodes.detach().long().to(device)
            row_indices = torch.unique(anchor_nodes).cpu().tolist()
            anchor_globals = [int(global_for_row[i]) for i in row_indices]
        else:
            anchor_globals = self.anchors if self.anchors else list(range(self.num_nodes))

        # If we're in subgraph mode, restrict anchors to nodes present in the batch
        anchor_globals = [u for u in anchor_globals if u in all_batch_globals]

        # Optionally restrict to "instance" anchors (self.anchors)
        if self.anchors:
            anchor_set = set(self.anchors)
            filtered = [u for u in anchor_globals if u in anchor_set]
            if filtered:
                anchor_globals = filtered

        if not anchor_globals:
            # Fallback: use all nodes in this batch
            anchor_globals = list(all_batch_globals)

        if not anchor_globals:
            # Degenerate fallback
            anchor_globals = [0]

        z_pos_list = []
        z_pos_pos_list = []
        z_pos_neg_list = []

        # For each anchor (global ID), sample pos/neg neighbors (global IDs),
        # then map to local row indices for z.
        for u in anchor_globals:
            if u not in global2local:
                continue
            u_local = global2local[u]
            z_pos_list.append(z[u_local].unsqueeze(0))

            # --- Positives (filtered to nodes in this batch) ---
            pos_candidates_global = [
                v for v in self.pos_neighbors[u] if v in all_batch_globals
            ]
            if pos_candidates_global:
                v_pos_global = random.choice(pos_candidates_global)
            else:
                v_pos_global = u

            if v_pos_global in global2local:
                v_pos_local = global2local[v_pos_global]
            else:
                v_pos_local = u_local
            z_pos_pos_list.append(z[v_pos_local].unsqueeze(0))

            # --- Negatives (filtered to nodes in this batch) ---
            neg_candidates_global = [
                v for v in self.neg_pool[u] if v in all_batch_globals
            ]
            neg_indices_local: List[int] = []

            if neg_candidates_global:
                if len(neg_candidates_global) >= self.k:
                    chosen_globals = random.sample(neg_candidates_global, self.k)
                else:
                    chosen_globals = random.choices(neg_candidates_global, k=self.k)
                for v in chosen_globals:
                    v_local = global2local.get(v, u_local)
                    neg_indices_local.append(v_local)
            else:
                # Fallback: sample negatives from all nodes in this batch
                batch_locals = list(range(N))
                if len(batch_locals) >= self.k:
                    chosen_locals = random.sample(batch_locals, self.k)
                else:
                    chosen_locals = random.choices(batch_locals, k=self.k)
                neg_indices_local = chosen_locals

            neg_idx_tensor = torch.tensor(
                neg_indices_local, dtype=torch.long, device=device
            )
            z_neg_for_u = z[neg_idx_tensor]  # [k, D]
            z_pos_neg_list.append(z_neg_for_u.unsqueeze(0))  # [1, k, D]

        if not z_pos_list:
            # Extreme fallback: just take first few rows of z
            anchor_rows = list(range(min(N, max(self.k, 1))))
            z_pos = z[anchor_rows]
            z_pos_pos = z[anchor_rows]
            z_pos_neg = z[anchor_rows].unsqueeze(1).expand(-1, self.k, -1)
            return z_pos, z_pos_pos, z_pos_neg

        z_pos = torch.cat(z_pos_list, dim=0)         # [B, D]
        z_pos_pos = torch.cat(z_pos_pos_list, dim=0) # [B, D]
        z_pos_neg = torch.cat(z_pos_neg_list, dim=0) # [B, k, D]
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
