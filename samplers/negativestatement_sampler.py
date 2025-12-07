import random
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

class NegativeStatementSampler:
    """ Ontology-guided negative sampler for contrastive learning on a Wikidata-style KG.
    Behaviour (Wikidata setting):
    • Anchors = "instances":
        - Nodes that are source nodes in at least one "instance of" (P31) triple
          in the training graph, i.e. edges with relation == `instance_rel`.
    • Positive neighbors of an anchor u:
        - All target nodes v such that there exists a positive training edge
          (u, r, v) with r *not* starting with neg_prefix and r != subclass_rel,
          and optionally r != instance_rel.
    • Negative neighbors of an anchor u (direct):
        - All target nodes w such that there exists a negative training edge
          (u, r, w) with r starting with neg_prefix (e.g. "NOT_3", "NOT_10", ...).
    • Ontology-guided expansion:
        - For each direct negative neighbor b of u, look at all edges
          (c, subclass_rel, b); c is a subclass of b, and is therefore also
          treated as a negative neighbor of u.
        - This yields an expanded negative pool for each anchor u.
    • Contrastive samples:
        - For each anchor u, sample:
            - 1 positive neighbor v⁺ ∈ pos_neighbors[u] (or u itself if none),
            - k negatives v⁻₁,...,v⁻ₖ ∈ neg_pool[u], using replacement if needed.
        - Returns embedding triples (z_pos, z_pos_pos, z_pos_neg) ready for a
          dual contrastive loss.

    The sampler is independent of the specific statement relation types; it
    inspects all edge types in the graph and categorises them by:
        - negative vs positive (using `neg_prefix`),
        - "subclass_of" (ontology),
        - "instance_of" (`instance_rel`, typically the P31 relation index). """

    def __init__(self, k: int = 1, subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_", instance_rel: str = "2"):
        """ Args:
            k: number of negatives per anchor for contrastive loss.
            subclass_rel: relation name for subclass-of edges ("subclass_of").
            neg_prefix: prefix that marks negative statement relations ("NOT_").
            instance_rel: relation name (string) for "instance of" (P31) edges;
            if None, anchors fall back to all sources of positive statement edges."""
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
        """ Scan the full training graph and build:
            - anchors (instances),
            - positive neighbors per anchor,
            - negative neighbors per anchor (direct + ontology expansion). """
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
            if rel == self.subclass_rel: continue
            src_list = eidx[0].tolist()
            dst_list = eidx[1].tolist()

            if self.instance_rel is not None and rel == self.instance_rel:
                continue

            if rel.startswith(self.neg_prefix):
                for u, v in zip(src_list, dst_list):
                    if 0 <= u < N and 0 <= v < N: self.direct_neg_neighbors[u].append(v)
            else:
                for u, v in zip(src_list, dst_list):
                    if 0 <= u < N and 0 <= v < N:
                        self.pos_neighbors[u].append(v)
                        if self.instance_rel is None: anchor_sources.add(u)
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

    def prepare_batch(self, batch: HeteroData,
        pos_index: Optional[Tensor] = None) -> None:
        """ For compatibility with Train/Train_BestModel, which call
        `neg_statement_sampler.prepare_batch(...)` on every batch.

        In this implementation, we assume each batch subgraph preserves the
        full global node set with consistent indexing (which is what your
        Pygloader does), so we don't need to do any per-batch remapping.
        """
        return


    def get_contrastive_samples(self, z: Tensor, anchor_nodes: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """ Build contrastive triples (z_pos, z_pos_pos, z_pos_neg).
        Args:
            z: [N, D] node embeddings for *all* nodes ("node" type), where
               index i corresponds to node i used in prepare_global().
            anchor_nodes: optional LongTensor of node indices to use as anchors
                          (e.g., nodes from the current triple batch). Only those
                          that are in self.anchors (instance anchors) are used.
                          If None or intersection is empty, falls back to all
                          self.anchors (or all nodes if self.anchors is empty).
        """
        device = z.device
        N, D = z.shape

        if anchor_nodes is not None:
            anchor_nodes = anchor_nodes.detach().long()
            if anchor_nodes.numel() > 0:
                batch_nodes = torch.unique(anchor_nodes).cpu().tolist()
                if self.anchors:
                    anchor_set = set(self.anchors)
                    filtered = [u for u in batch_nodes if u in anchor_set]
                else: filtered = batch_nodes

                if filtered: anchors = filtered
                else: anchors = self.anchors if self.anchors else list(range(N))
            else: anchors = self.anchors if self.anchors else list(range(N))
        else: anchors = self.anchors if self.anchors else list(range(N))

        if not anchors: anchors = [0]

        B = len(anchors)
        anchor_tensor = torch.tensor(anchors, device=device, dtype=torch.long)
        z_pos = z[anchor_tensor]  # [B, D]
        pos_indices: List[int] = []
        neg_indices: List[List[int]] = []
        all_nodes_set = set(range(N))

        for u in anchors:
            pos_list = self.pos_neighbors[u]
            neg_list = self.neg_pool[u]
            if pos_list: v_pos = random.choice(pos_list)
            else: v_pos = u
            pos_indices.append(v_pos)

            if not neg_list:
                excluded = set(pos_list)
                excluded.add(u)
                candidates = list(all_nodes_set - excluded)
                if not candidates: negs_u = [u] * self.k
                else:
                    if len(candidates) >= self.k: negs_u = random.sample(candidates, self.k)
                    else: negs_u = random.choices(candidates, k=self.k)
            else:
                if len(neg_list) >= self.k: negs_u = random.sample(neg_list, self.k)
                else: negs_u = random.choices(neg_list, k=self.k)
            neg_indices.append(negs_u)

        pos_idx_tensor = torch.tensor(pos_indices, device=device, dtype=torch.long)
        z_pos_pos = z[pos_idx_tensor]  # [B, D]
        neg_idx_tensor = torch.tensor(neg_indices, device=device, dtype=torch.long)  # [B, k]
        z_pos_neg = z[neg_idx_tensor]  # [B, k, D]

        return z_pos, z_pos_pos, z_pos_neg


    @staticmethod
    def _find_edge_key(g: HeteroData, rel: str, src_ntype: Optional[str] = None,
        dst_ntype: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
        """
        Find the hetero edge key whose relation name matches `rel`.
        If src_ntype / dst_ntype are given, match those too.
        """
        for (s, r, d) in g.edge_types:
            if r != rel: continue
            if src_ntype is not None and s != src_ntype: continue
            if dst_ntype is not None and d != dst_ntype: continue
            return (s, r, d)
        return None
