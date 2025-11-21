import torch
from typing import Optional, Iterator, Tuple, List, Dict
from torch_geometric.data import HeteroData


class Pygloader:
    """
    Splits a heterogeneous graph into train/validation subgraphs based on the source nodes
    of a given edge type (e.g. "PPI"). Allows iteration over mini-batch subgraphs by PPI-edge batches.
    Args:
        graph (HeteroData): Input full heterogeneous graph.
        ppi_rel (str): Relation name for PPI edges (etype[1] in edge_types).
        batch_size (int): Number of PPI edges per batch.
        val_split (float): Fraction of PPI source nodes for validation set.
        device (Optional[torch.device] or str): Device to move subgraphs to.
        shuffle (bool): Shuffle PPI source nodes before splitting.
        seed (Optional[int]): Random seed for reproducibility.
    """
    def __init__( self, graph: HeteroData, ppi_rel: str = "PPI", batch_size: int = 32,
        val_split: float = 0.1, device: Optional[torch.device] = "cpu", shuffle: bool = True,
        seed: Optional[int] = None):
        
        assert isinstance(graph, HeteroData), "graph must be a PyG HeteroData"
        self.graph: HeteroData = graph
        self.ppi_rel = ppi_rel
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if seed is not None: torch.manual_seed(seed)
            
        self.ppi_etype = next((et for et in graph.edge_types if et[1] == ppi_rel), None)
        self._num_nodes: Dict[str, int] = self._infer_all_num_nodes()
        if self.ppi_etype is None:
            self.other_etypes: List[tuple] = list(graph.edge_types)
            self.train_eids = torch.empty(0, dtype=torch.long)
            self.val_eids = torch.empty(0, dtype=torch.long)
            return
        self.other_etypes: List[tuple] = [et for et in graph.edge_types if et != self.ppi_etype]
        self._split_by_source_nodes()
        #self.train_graph = self._create_split_graph(self.train_eids, train=True)
        #self.val_graph = self._create_split_graph(self.val_eids, train=False)


    def _infer_all_num_nodes(self) -> Dict[str, int]:
        """
        Determine number of nodes per node type. Prefer stored `num_nodes` or
        infer from max node id across incident edges.
        """
        num_nodes: Dict[str, int] = {}
        for ntype in self.graph.node_types:
            explicit = getattr(self.graph[ntype], "num_nodes", None)
            if explicit is not None:
                num_nodes[ntype] = int(explicit)
                continue
            max_id = -1
            for (src_t, _, dst_t) in self.graph.edge_types:
                ei = self.graph[(src_t, _, dst_t)].get("edge_index", None)
                if ei is None: continue
                if src_t == ntype and ei.size(1) > 0:
                    max_id = max(max_id, int(ei[0].max().item()))
                if dst_t == ntype and ei.size(1) > 0:
                    max_id = max(max_id, int(ei[1].max().item()))
            num_nodes[ntype] = max_id + 1 if max_id >= 0 else 0
        return num_nodes

    
    def _split_by_source_nodes(self):
        """Get all PPI edges' source nodes and corresponding edge IDs; split by unique sources."""
        edge_index = self.graph[self.ppi_etype].edge_index
        src = edge_index[0]
        num_ppi = src.numel()
        edge_ids = torch.arange(num_ppi, device=src.device)
        unique_src = torch.unique(src)
        if self.shuffle:
            unique_src = unique_src[torch.randperm(len(unique_src), device=unique_src.device)]
        n_val = int(len(unique_src) * self.val_split)
        n_train = len(unique_src) - n_val
        train_src = unique_src[:n_train]
        val_src = unique_src[n_train:n_train + n_val]
        mask_train = torch.isin(src, train_src)
        mask_val = torch.isin(src, val_src)
        self.train_eids = edge_ids[mask_train].cpu()
        self.val_eids = edge_ids[mask_val].cpu()

    
    def _create_split_graph(self, ppi_eids: torch.Tensor, train=True) -> HeteroData:
        """
        Build a subgraph containing specified PPI edges (by edge IDs) and all edges of other types.
        Preserves all original nodes and copies node features if present.
        Also stores original edge IDs in `orig_eid` per relation.
        """
        sub = HeteroData()
        for ntype in self.graph.node_types:
            sub[ntype].num_nodes = self._num_nodes[ntype]
            for key, val in self.graph[ntype].items():
                if key in ("x"): sub[ntype][key] = val

        for et in self.other_etypes:
            store = self.graph[et]
            if "edge_index" not in store: continue
            ei = store.edge_index
            sub[et].edge_index = ei
            sub[et].orig_eid = torch.arange(ei.size(1), dtype=torch.long)

            for key, val in store.items():
                if key in ("edge_index", "orig_eid"): continue
                if hasattr(val, "size") and val.size(0) == ei.size(1):
                    sub[et][key] = val

        ppi_store = self.graph[self.ppi_etype]
        ppi_ei = ppi_store.edge_index[:, ppi_eids]
        if train:
            ppi_ei_rev = ppi_ei.flip(0) # reverse edges
            ppi_ei = torch.cat([ppi_ei, ppi_ei_rev], dim=1)
            sub[self.ppi_etype].edge_index = ppi_ei
            sub[self.ppi_etype].orig_eid = torch.cat([ppi_eids.clone(), ppi_eids.clone()],dim=0) # sub[self.ppi_etype].orig_eid = ppi_eids.clone()

            for key, val in ppi_store.items():
                if key in ("edge_index", "orig_eid"): continue
                if hasattr(val, "size") and val.size(0) == ppi_store.edge_index.size(1):
                    # sub[self.ppi_etype][key] = val[ppi_eids]
                    attr_fwd = val[ppi_eids]
                    attr_full = torch.cat([attr_fwd, attr_fwd], dim=0)
                    sub[self.ppi_etype][key] = attr_full
        else: return sub.to(self.device), ppi_ei
        return sub.to(self.device)



    def _batch_graphs(self, eids: torch.Tensor) -> Iterator[HeteroData]:
        """Yield mini-batch subgraphs for slices of PPI edges."""
        for start in range(0, len(eids), self.batch_size):
            batch_eids = eids[start:start + self.batch_size]
            yield self._create_split_graph(batch_eids)

    def train_batches(self) -> Iterator[HeteroData]:
        """Iterate mini-batch train subgraphs."""
        return self._batch_graphs(self.train_eids)

    def validation_batches(self) -> Iterator[HeteroData]:
        """Iterate mini-batch validation subgraphs."""
        return self._batch_graphs(self.val_eids)

    # def get_split_graphs(self) -> Tuple[HeteroData, HeteroData]:
    #     """Return the full train/val heterographs."""
    #     return self.train_graph, self.val_graph

    def get_relation(self) -> str:
        """Return the relation name used for PPI edges."""
        return self.ppi_rel

    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over mini-batch subgraphs for training."""
        return self.train_batches()

    def __len__(self) -> int:
        """Return the number of mini-batches in the training set."""
        return (len(self.train_eids) + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        return (
            f"Loader(ppi_rel='{self.ppi_rel}', "
            f"train_ppi={len(self.train_eids)}, "
            f"val_ppi={len(self.val_eids)}, "
            f"batch_size={self.batch_size})"
        )
