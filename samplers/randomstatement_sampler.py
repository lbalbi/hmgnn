import random
import torch
from typing import List, Tuple, Dict, Optional
from torch import Tensor
from torch_geometric.data import HeteroData


class RandomStatementSampler:
    """  Random sampler over statement edges in a multi-relation KG.
    • Anchors:
        - Sources of relation `instance_rel` (e.g. "2" for "instance of").
        - If no such edges exist, fall back to all nodes with at least one
          positive statement; if still empty, all nodes.
    • Positive statements:
        - Any edge (u, r, v) where:
              r does NOT start with `neg_prefix` and r != subclass_rel
              and r != instance_rel
          Define:
              pos_global[u] = list of v (positive neighbors)
              rev_pos_global[v] = list of u (reverse pos neighbors)
    • Candidate negatives (global):
     - If `external_negs` is given: global_candidates = { v | (u, v) in external_negs }.
     - Otherwise: global_candidates = all positive targets: { v | ∃(u, r, v) positive }.
    • Per-anchor negative pool:
        - For each anchor u:
              invalid(u) = pos_global[u] ∪ rev_pos_global[u] ∪ {u}
              neg_global[u] = { v in global_candidates | v not in invalid(u) }
        - If this set is empty, we fall back to positives or the anchor itself
          at batch time in `get_contrastive_samples`.
    """

    def __init__(self, k: int = 1, instance_rel: str = "2", subclass_rel: str = "subclass_of", 
    neg_prefix: str = "NOT_", external_negs: Optional[List[Tuple[int, int]]] = None):
        self.k = k
        self.instance_rel = instance_rel
        self.subclass_rel = subclass_rel
        self.neg_prefix = neg_prefix
        self.external_negs = external_negs or []
        self.node_type: Optional[str] = None
        self.num_nodes: int = 0
        self.pos_global: List[List[int]] = []
        self.neg_global: List[List[int]] = []
        self.anchors: List[int] = []


    def prepare_global(self, full_g: HeteroData) -> None:
        """  Build global anchors (instance sources), positive neighbors,
        and per-anchor candidate negative sets.
        """
        if len(full_g.node_types) != 1:
            raise ValueError(f"RandomStatementSampler assumes a single node type; "
                f"got {full_g.node_types}")
        self.node_type = full_g.node_types[0]

        if hasattr(full_g[self.node_type], "num_nodes") and full_g[self.node_type].num_nodes is not None:
            N = int(full_g[self.node_type].num_nodes)
        else:
            max_id = -1
            for (_, _, _), eidx in full_g.edge_index_dict.items():
                if eidx.numel() > 0: max_id = max(max_id, int(eidx.max().item()))
            N = max_id + 1 if max_id >= 0 else 0

        self.num_nodes = N
        self.pos_global = [[] for _ in range(N)]
        rev_pos_global: List[List[int]] = [[] for _ in range(N)]
        anchor_sources: set = set()
        global_pos_targets: set = set()

        for (src_nt, rel, dst_nt), eidx in full_g.edge_index_dict.items():
            if src_nt != self.node_type or dst_nt != self.node_type: continue
            if eidx.numel() == 0: continue
            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            if self.instance_rel is not None and rel == self.instance_rel:
                for s in src_list:
                    if 0 <= s < N: anchor_sources.add(s)
                continue
            if rel == self.subclass_rel: continue
            if rel.startswith(self.neg_prefix): continue

            for u, v in zip(src_list, dst_list):
                if 0 <= u < N and 0 <= v < N:
                    self.pos_global[u].append(v)
                    rev_pos_global[v].append(u)
                    global_pos_targets.add(v)

        if self.external_negs:
            global_candidates = {v for (_, v) in self.external_negs if 0 <= v < N}
        else: global_candidates = global_pos_targets

        if self.instance_rel is not None and anchor_sources: anchors = sorted(anchor_sources)
        else:
            anchors = [u for u in range(N) if self.pos_global[u]]
            if not anchors: anchors = list(range(N))
        self.anchors = anchors

        self.neg_global = [[] for _ in range(N)]
        for u in range(N):
            invalid = set(self.pos_global[u])
            if 0 <= u < len(rev_pos_global): invalid |= set(rev_pos_global[u])
            invalid.add(u)
            cand = [v for v in global_candidates if v not in invalid]
            self.neg_global[u] = cand


    def prepare_batch(self, batch: HeteroData) -> None:
        """For compatibility with Train/Train_BestModel. No per-batch state needed."""
        return

    def get_contrastive_samples(self, z: Tensor, anchor_nodes: Optional[Tensor] = None,
        n_id: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """  Build contrastive triples (z_pos, z_pos_pos, z_pos_neg).
        Args:
            z: [B, D] node embeddings. If `n_id` is None, row i is global i.
               If `n_id` is provided, row i corresponds to global node n_id[i].
            anchor_nodes: local row indices into `z` (e.g. unique nodes from triples).
            n_id: Optional LongTensor[B] of global node IDs for rows of `z`.
        Returns:
            z_pos     : [B_anchor, D] anchor embeddings
            z_pos_pos : [B_anchor, D] positive neighbor embeddings
            z_pos_neg : [B_anchor, k, D] negative embeddings
        """
        device = z.device
        B, D = z.shape

        if n_id is None: global_for_row = torch.arange(B, device=device, dtype=torch.long)
        else: global_for_row = n_id.to(device).long()
        global_list = global_for_row.cpu().tolist()
        global2local: Dict[int, int] = {int(g): i for i, g in enumerate(global_list)}
        all_batch_globals = set(global_list)

        if anchor_nodes is not None and anchor_nodes.numel() > 0:
            anchor_nodes = anchor_nodes.detach().long().to(device)
            row_indices = torch.unique(anchor_nodes).cpu().tolist()
            anchor_globals = [int(global_for_row[i]) for i in row_indices]
        else: anchor_globals = self.anchors if self.anchors else list(range(self.num_nodes))

        anchor_globals = [u for u in anchor_globals if u in all_batch_globals]
        if not anchor_globals: anchor_globals = list(all_batch_globals)
        if not anchor_globals: anchor_globals = [0]

        z_pos_list: List[Tensor] = []
        z_pos_pos_list: List[Tensor] = []
        z_pos_neg_list: List[Tensor] = []

        for u in anchor_globals:
            if u not in global2local: continue
            u_local = global2local[u]
            z_pos_list.append(z[u_local].unsqueeze(0))

            pos_candidates_global = [v for v in (self.pos_global[u] if u < len(self.pos_global) else [])
                if v in all_batch_globals]
            if pos_candidates_global: v_pos_global = random.choice(pos_candidates_global)
            else: v_pos_global = u
            v_pos_local = global2local.get(v_pos_global, u_local)
            z_pos_pos_list.append(z[v_pos_local].unsqueeze(0))

            neg_candidates_global = [v for v in (self.neg_global[u] if u < len(self.neg_global) else [])
                if v in all_batch_globals]

            if not neg_candidates_global:
                if pos_candidates_global: neg_candidates_global = pos_candidates_global.copy()
                else: neg_candidates_global = [int(g) for g in all_batch_globals]
            if not neg_candidates_global: neg_candidates_global = [u]

            chosen_globals = random.choices(neg_candidates_global, k=self.k)
            neg_locals: List[int] = [global2local.get(v, u_local) for v in chosen_globals]
            neg_idx_tensor = torch.tensor(neg_locals, dtype=torch.long, device=device)
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
