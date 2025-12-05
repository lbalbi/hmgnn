import os
from typing import List, Dict, Tuple, Union, Optional

import pandas as pd
import torch
from torch_geometric.data import HeteroData


EdgeType = Union[str, int]


class DataLoader:
    """DataLoader for loading heterogeneous graph data from CSV/TXT files.

    It expects each file to have columns: `source_node`, `target_node`, `edge_type`.

    This version supports:
      - Multiple relation types per file (via the `edge_type` column).
      - Separating *training* vs *test* files by filename:
          * files whose name contains "test" (case-insensitive) are treated as test
          * all other files are treated as training
      - Optional extraction/removal of statement edges for samplers.
    """

    def __init__(
        self,
        file_path: str,
        use_pstatement_sampler: bool = False,
        use_nstatement_sampler: bool = False,
        use_rstatement_sampler: bool = False,
    ):
        self.file_path = file_path
        self.use_pstatement_sampler = use_pstatement_sampler
        self.use_nstatement_sampler = use_nstatement_sampler
        self.use_rstatement_sampler = use_rstatement_sampler

        edge_files = self.path_files(file_path)

        # Split into train vs test files by name
        train_files = [
            f for f in edge_files
            if "test" not in os.path.basename(f).lower()
        ]
        test_files = [
            f for f in edge_files
            if "test" in os.path.basename(f).lower()
        ]

        # Graph is built only from training files
        self.data: Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]] = self.load_data(train_files)
        # Test edges (not added to the graph)
        self.test_data: Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]] = (
            self.load_data(test_files) if test_files else {}
        )

        # Extract state_list and optionally remove statement edges from graph data
        self.state_list: List[Tuple[int, int]] = []
        if self.use_pstatement_sampler:
            if "pos_statement" in self.data:
                src, tgt = self.data.pop("pos_statement")
                self.state_list = list(zip(src.tolist(), tgt.tolist()))
        elif self.use_nstatement_sampler:
            if "neg_statement" in self.data:
                src, tgt = self.data.pop("neg_statement")
                self.state_list = list(zip(src.tolist(), tgt.tolist()))
        elif self.use_rstatement_sampler:  # keep neg_statement edges in the graph, but store them as state_list
            if "neg_statement" in self.data:
                src, tgt = self.data["neg_statement"]
                self.state_list = list(zip(src.tolist(), tgt.tolist()))

    # ------------------------------------------------------------------
    # Low-level loading helpers
    # ------------------------------------------------------------------
    @staticmethod
    def path_files(path: str) -> List[str]:
        """Return all regular files in a directory."""
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]

    def load_data(self, edge_files: List[str]) -> Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]]:
        """Load a list of CSV/TXT files and group edges by `edge_type`.
        Returns: Dict[edge_type, (src_tensor, tgt_tensor)]
        """
        if not edge_files: return {}
        dfs = [pd.read_csv(f) for f in edge_files]
        all_edges = pd.concat(dfs, ignore_index=True)

        if not {"source_node", "target_node", "edge_type"}.issubset(all_edges.columns):
            raise ValueError("Input files must contain columns: 'source_node', 'target_node', 'edge_type'. "
                f"Got columns: {list(all_edges.columns)}")

        data_dict: Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_type, group in all_edges.groupby("edge_type"):
            src_nodes = torch.tensor(group["source_node"].values, dtype=torch.long)
            tgt_nodes = torch.tensor(group["target_node"].values, dtype=torch.long)
            data_dict[edge_type] = (src_nodes, tgt_nodes)
        return data_dict


    def make_data_graph(
        self,
        data: Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]],
        orthogonal: bool = False,
        in_dim: int = 128,
    ) -> HeteroData:
        """Create a PyTorch Geometric HeteroData graph from the edge dictionary.
        The graph uses a single node type "node" and multiple edge types of the
        form ("node", edge_type, "node").
        """
        hetero = HeteroData()

        max_id = -1
        for src, tgt in data.values():
            if src.numel():
                max_id = max(max_id, int(src.max().item()))
            if tgt.numel():
                max_id = max(max_id, int(tgt.max().item()))
        num_nodes = max_id + 1 if max_id >= 0 else 0
        hetero["node"].num_nodes = num_nodes

        for edge_type, (src, tgt) in data.items():
            if src.numel() == 0:
                continue
            edge_index = torch.stack([src, tgt], dim=0)
            hetero[("node", edge_type, "node")].edge_index = edge_index

        if num_nodes > 0:
            if orthogonal:
                emb = torch.nn.Embedding(num_nodes, in_dim)
                torch.nn.init.orthogonal_(emb.weight)
                hetero["node"].x = emb.weight
            else: hetero["node"].x = torch.randn(num_nodes, in_dim)
        return hetero


    def get_state_list(self) -> List[Tuple[int, int]]:
        """Return the list of statement edges removed (or referenced) from the graph data."""
        return self.state_list

    def get_data(self) -> Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]]:
        """Training edges (used to build the graph)."""
        return self.data

    def get_test_data(self) -> Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]]:
        """Test edges loaded from 'test' files (e.g., test2id_pos.txt)."""
        return self.test_data

    def get_test_pairs(self, edge_type: EdgeType) -> torch.Tensor:
        """Return test pairs [2, N] for a given relation, or an empty tensor if not present."""
        if edge_type not in self.test_data:
            return torch.empty(2, 0, dtype=torch.long)
        src, tgt = self.test_data[edge_type]
        return torch.stack([src, tgt], dim=0)

    def get_edge_types(self) -> List[EdgeType]:
        return list(self.data.keys())

    def get_negative_edges(self) -> Optional[Dict]:
        return None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: Union[str, int]):
        """Access edges by edge_type. If int is given, use deterministic order of keys."""
        if isinstance(key, str):
            return self.data[key]
        return list(self.data.items())[key][1]
