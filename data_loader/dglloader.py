
import torch
import dgl
from dgl import heterograph
from typing import Optional, Iterator, Tuple, List

class Dglloader:
    """
    Splits a DGLHeteroGraph into train/validation/test subgraphs based on the source nodes
    of a given edge type (e.g. "PPI"). Allows iteration over mini-batch subgraphs by PPI-edge batches.

    Args:
        graph (DGLHeteroGraph): Input full heterogeneous graph.
        ppi_rel (str): Relation name for PPI edges (etype[1] in canonical_etypes).
        batch_size (int): Number of PPI edges per batch.
        val_split (float): Fraction of PPI source nodes for validation set.
        test_split (float): Fraction of PPI source nodes for test set.
        shuffle (bool): Shuffle PPI source nodes before splitting.
        seed (Optional[int]): Random seed for reproducibility.
    """
    def __init__(
        self,
        graph: heterograph,
        ppi_rel: str = "PPI",
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        device : Optional[torch.device] = "cpu",
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        assert isinstance(graph, DGLHeteroGraph), "graph must be a DGLHeteroGraph"
        self.graph = graph
        self.ppi_rel = ppi_rel
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        if seed is not None:
            torch.manual_seed(seed)

        # Identify the canonical edge type for PPI
        self.ppi_etype = next(
            (et for et in graph.canonical_etypes if et[1] == ppi_rel),
            None
        )
        assert self.ppi_etype is not None, f"Relation '{ppi_rel}' not in graph.canonical_etypes"

        # Other edge types remain intact in each split
        self.other_etypes = [et for et in graph.canonical_etypes if et != self.ppi_etype]

        # Split PPI edges by source node
        self._split_by_source_nodes()

        # Create full split subgraphs
        self.train_graph = self._create_split_graph(self.train_eids)
        self.val_graph = self._create_split_graph(self.val_eids)
        self.test_graph = self._create_split_graph(self.test_eids)

    def _split_by_source_nodes(self):
        # Get all PPI edges' source nodes and corresponding edge IDs
        src, _ = self.graph.edges(etype=self.ppi_etype)
        num_ppi = src.size(0)
        edge_ids = torch.arange(num_ppi)

        # Unique source nodes
        unique_src = torch.unique(src)
        if self.shuffle:
            unique_src = unique_src[torch.randperm(len(unique_src))]

        # Split source nodes
        n_test = int(len(unique_src) * self.test_split)
        n_val = int(len(unique_src) * self.val_split)
        n_train = len(unique_src) - n_val - n_test

        train_src = unique_src[:n_train]
        val_src = unique_src[n_train:n_train + n_val]
        test_src = unique_src[n_train + n_val:]

        # Map edges to splits by source membership
        mask_train = torch.isin(src, train_src)
        mask_val = torch.isin(src, val_src)
        mask_test = torch.isin(src, test_src)

        self.train_eids = edge_ids[mask_train]
        self.val_eids = edge_ids[mask_val]
        self.test_eids = edge_ids[mask_test]

    def _create_split_graph(self, ppi_eids: torch.Tensor) -> DGLHeteroGraph:
        """
        Build a subgraph containing specified PPI edges (by edge IDs) and all edges of other types.
        Preserves all original nodes.
        """
        eid_dict = {}
        # Keep all other edges
        for et in self.other_etypes:
            num = self.graph.num_edges(et)
            eid_dict[et] = torch.arange(num)
        # Keep only ppi_eids for the PPI edge type
        eid_dict[self.ppi_etype] = ppi_eids

        # Build edge-induced subgraph, preserving nodes
        return dgl.edge_subgraph(self.graph, eid_dict, preserve_nodes=True)

    def _batch_graphs(self, eids: torch.Tensor) -> Iterator[DGLHeteroGraph]:
        """
        Yield mini-batch subgraphs for slices of PPI edges.
        """
        for start in range(0, len(eids), self.batch_size):
            batch_eids = eids[start:start + self.batch_size]
            yield self._create_split_graph(batch_eids)

    def train_batches(self) -> Iterator[DGLHeteroGraph]:
        """Iterate mini-batch train subgraphs."""
        return self._batch_graphs(self.train_eids)

    def validation_batches(self) -> Iterator[DGLHeteroGraph]:
        """Iterate mini-batch validation subgraphs."""
        return self._batch_graphs(self.val_eids)

    def test_batches(self) -> Iterator[DGLHeteroGraph]:
        """Iterate mini-batch test subgraphs."""
        return self._batch_graphs(self.test_eids)

    def get_split_graphs(self) -> Tuple[DGLHeteroGraph, DGLHeteroGraph, DGLHeteroGraph]:
        """Return the full train/val/test heterographs."""
        return self.train_graph, self.val_graph, self.test_graph

    def __iter__(self) -> Iterator[DGLHeteroGraph]:
        """
        Iterate over mini-batch subgraphs for training.
        """
        return self.train_batches()
    
    def __len__(self) -> int:
        """
        Return the number of mini-batches in the training set.
        """
        return (len(self.train_eids) + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        return (
            f"Loader(ppi_rel='{self.ppi_rel}', "
            f"train_ppi={len(self.train_eids)}, "
            f"val_ppi={len(self.val_eids)}, "
            f"test_ppi={len(self.test_eids)}, "
            f"batch_size={self.batch_size})"
        )
