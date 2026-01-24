import os, torch
from typing import List, Dict, Tuple, Union
import pandas as pd
from torch_geometric.data import HeteroData


class DataLoader:
    """DataLoader for loading heterogeneous graph data from CSV/TSV files.
    Expected format of each edge file: source_node,target_node,edge_type
    All files in `file_path` are loaded EXCEPT:
        - test2id_pos.txt  (NOT included as graph edges)
    """
    def __init__(self, file_path: str,
                 use_pstatement_sampler: bool = False,
                 use_nstatement_sampler: bool = False,
                 use_rstatement_sampler: bool = False):
        self.file_path = file_path
        self.use_pstatement_sampler = use_pstatement_sampler
        self.use_nstatement_sampler = use_nstatement_sampler
        self.use_rstatement_sampler = use_rstatement_sampler

        edge_files = self.path_files(file_path)
        self.data = self.load_data(edge_files)

        self.state_list: List[Tuple[int, int]] = []
        if self.use_pstatement_sampler:
            if "pos_statement" in self.data:
                src, tgt = self.data.pop("pos_statement")
                self.state_list = list(zip(src.tolist(), tgt.tolist()))
        elif self.use_nstatement_sampler:
            if "neg_statement" in self.data:
                src, tgt = self.data.pop("neg_statement")
                self.state_list = list(zip(src.tolist(), tgt.tolist()))
        elif self.use_rstatement_sampler:
            if "neg_statement" in self.data:
                src, tgt = self.data["neg_statement"]
                self.state_list = list(zip(src.tolist(), tgt.tolist()))

    @staticmethod
    def path_files(path: str) -> List[str]:
        """Return list of data files to load as GRAPH edges.
        We explicitly skip test2id_pos.txt so the encoder graph stays "fixed".
        """
        skip_files = {"test2id_pos.txt"}
        files: List[str] = []
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if not os.path.isfile(full):
                continue
            if f in skip_files:
                continue
            if f.endswith(".txt") or f.endswith(".csv") or f.endswith(".tsv"):
                files.append(full)
        return files

    def load_data(self, edge_files: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Load all edge files and group by edge_type."""
        if not edge_files:
            return {}

        dfs = []
        for f in edge_files:
            print(f)
            if f.endswith(".tsv"):
                df = pd.read_csv(f, sep="\t")
            else:
                df = pd.read_csv(f)
            dfs.append(df)

        all_edges = pd.concat(dfs, ignore_index=True)

        required_cols = {"source_node", "target_node", "edge_type"}
        if not required_cols.issubset(all_edges.columns):
            raise ValueError(f"Edge files must contain columns {required_cols}, got {list(all_edges.columns)}")

        data_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_type, group in all_edges.groupby("edge_type"):
            src_nodes = torch.tensor(group["source_node"].values, dtype=torch.long)
            tgt_nodes = torch.tensor(group["target_node"].values, dtype=torch.long)
            data_dict[str(edge_type)] = (src_nodes, tgt_nodes)

        return data_dict

    def make_data_graph(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                        orthogonal: bool = False) -> HeteroData:
        """Creates a PyG HeteroData graph."""
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
                edge_index = torch.empty(2, 0, dtype=torch.long)
            else:
                edge_index = torch.stack([src, tgt], dim=0)
            hetero[("node", str(edge_type), "node")].edge_index = edge_index

        if num_nodes > 0:
            if orthogonal:
                emb = torch.nn.Embedding(num_nodes, 128)
                torch.nn.init.orthogonal_(emb.weight)
                hetero["node"].x = emb.weight
            else:
                hetero["node"].x = torch.randn(num_nodes, 128)
        return hetero

    def get_state_list(self) -> List[Tuple[int, int]]:
        return self.state_list

    def get_data(self):
        return self.data

    def get_edge_types(self):
        return list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, str):
            return self.data[key]
        return list(self.data.items())[key][1]



# class DataLoader:
#     """DataLoader for loading heterogeneous graph data from CSV/TSV files.
#     Expected format of each edge file: source_node,target_node,edge_type
#     All files in `file_path` are loaded EXCEPT:
#         - test2id_pos.txt  (held-out test positives)
#     """
#     def __init__(self,file_path: str, use_pstatement_sampler: bool = False,
#         use_nstatement_sampler: bool = False, use_rstatement_sampler: bool = False):
#         self.file_path = file_path
#         self.use_pstatement_sampler = use_pstatement_sampler
#         self.use_nstatement_sampler = use_nstatement_sampler
#         self.use_rstatement_sampler = use_rstatement_sampler
#         edge_files = self.path_files(file_path)
#         self.data = self.load_data(edge_files)

#         self.state_list: List[Tuple[int, int]] = []
#         if self.use_pstatement_sampler:
#             if "pos_statement" in self.data:
#                 src, tgt = self.data.pop("pos_statement")
#                 self.state_list = list(zip(src.tolist(), tgt.tolist()))
#         elif self.use_nstatement_sampler:
#             if "neg_statement" in self.data:
#                 src, tgt = self.data.pop("neg_statement")
#                 self.state_list = list(zip(src.tolist(), tgt.tolist()))
#         elif self.use_rstatement_sampler:
#             if "neg_statement" in self.data:
#                 src, tgt = self.data["neg_statement"]
#                 self.state_list = list(zip(src.tolist(), tgt.tolist()))

#     @staticmethod
#     def path_files(path: str) -> List[str]:
#         """Return list of data files to load as training edges.
#         We explicitly skip test2id_pos.txt (held-out test positives) so that
#         the training graph is not contaminated with test edges.
#         """
#         skip_files = {"test2id_pos.txt"}
#         files: List[str] = []
#         for f in os.listdir(path):
#             full = os.path.join(path, f)
#             if not os.path.isfile(full): continue
#             if f in skip_files: continue
#             if f.endswith(".txt") or f.endswith(".csv") or f.endswith(".tsv"):
#                 files.append(full)
#         return files

#     def load_data(self, edge_files: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
#         """Load all edge files and group by edge_type.
#         Args: edge_files: list of filenames to read.
#         Returns: Dict where keys are edge_type strings and values are (src_tensor, tgt_tensor).
#         """
#         if not edge_files: return {}
#         dfs = []
#         for f in edge_files:
#             print(f)
#             if f.endswith(".tsv"): df = pd.read_csv(f, sep="\t")
#             else: df = pd.read_csv(f)
#             dfs.append(df)

#         all_edges = pd.concat(dfs, ignore_index=True)

#         required_cols = {"source_node", "target_node", "edge_type"}
#         if not required_cols.issubset(all_edges.columns):
#             raise ValueError(f"Edge files must contain columns {required_cols}, "
#                 f"got {list(all_edges.columns)}")

#         data_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
#         for edge_type, group in all_edges.groupby("edge_type"):
#             src_nodes = torch.tensor(group["source_node"].values, dtype=torch.long)
#             tgt_nodes = torch.tensor(group["target_node"].values, dtype=torch.long)
#             data_dict[str(edge_type)] = (src_nodes, tgt_nodes)

#         return data_dict

#     def make_data_graph(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
#         orthogonal: bool = False) -> HeteroData:
#         """Creates a PyTorch Geometric HeteroData graph from the data dictionary.
#         The graph uses a single node type "node" and multiple edge types given by
#         keys in `data`. Node features are random or orthogonal embeddings.
#         """
#         hetero = HeteroData()

#         max_id = -1
#         for src, tgt in data.values():
#             if src.numel(): max_id = max(max_id, int(src.max().item()))
#             if tgt.numel(): max_id = max(max_id, int(tgt.max().item()))
#         num_nodes = max_id + 1 if max_id >= 0 else 0
#         hetero["node"].num_nodes = num_nodes

#         for edge_type, (src, tgt) in data.items():
#             if src.numel() == 0: edge_index = torch.empty(2, 0, dtype=torch.long)
#             else: edge_index = torch.stack([src, tgt], dim=0)
#             hetero[("node", str(edge_type), "node")].edge_index = edge_index

#         if num_nodes > 0:
#             if orthogonal:
#                 emb = torch.nn.Embedding(num_nodes, 128)
#                 torch.nn.init.orthogonal_(emb.weight)
#                 hetero["node"].x = emb.weight
#             else: hetero["node"].x = torch.randn(num_nodes, 128)
#         return hetero

#     def get_state_list(self) -> List[Tuple[int, int]]:
#         """Return the list of statement edges removed (or referenced) from the graph data."""
#         return self.state_list

#     def get_data(self):
#         return self.data

#     def get_edge_types(self):
#         return list(self.data.keys())

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, key: Union[str, int]):
#         """Access edges by edge_type. If int is given, use deterministic order of keys."""
#         if isinstance(key, str): return self.data[key]
#         return list(self.data.items())[key][1]
