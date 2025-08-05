import pandas as pd
import torch, dgl
from typing import List, Dict, Tuple

class DataLoader:
    """DataLoader for loading heterogeneous graph data from CSV files.
    The output is a dictionary of torch tensors containing the edge types as keys 
    and the pairs of source and target nodes as values."""
    
    def __init__(self, file_path: str,
                 use_pstatement_sampler: bool = False,
                 use_nstatement_sampler: bool = False, use_rstatement_sampler: bool = False):
        
        self.file_path = file_path
        self.use_pstatement_sampler = use_pstatement_sampler
        self.use_nstatement_sampler = use_nstatement_sampler
        self.use_rstatement_sampler = use_rstatement_sampler
        edge_files = self.path_files(file_path)
        self.data = self.load_data(edge_files)

        # Extract state_list and remove statement edges from graph data
        self.state_list: List[Tuple[int, int]] = []
        if self.use_pstatement_sampler:
            if 'pos_statement' in self.data:
                src, tgt = self.data.pop('pos_statement')
                self.state_list = list(zip(src.tolist(), tgt.tolist()))
        elif self.use_nstatement_sampler:
            if 'neg_statement' in self.data:
                src, tgt = self.data.pop('neg_statement')
                self.state_list = list(zip(src.tolist(), tgt.tolist()))
        elif self.use_rstatement_sampler:
            if 'neg_statement' in self.data:
                src, tgt = self.data['neg_statement']
                self.state_list = list(zip(src.tolist(), tgt.tolist()))

    @staticmethod
    def path_files(path):
        import os
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


    def load_data(self, edge_files: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args: edge_files: list of CSV filenames to read.
        Returns: Dict where keys are edge_type strings and values are (src_tensor, tgt_tensor).
        """
        dfs = []
        for f in edge_files:
            df = pd.read_csv(f)
            dfs.append(df)
        all_edges = pd.concat(dfs, ignore_index=True)

        data_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_type, group in all_edges.groupby('edge_type'):
            src_nodes = torch.tensor(group['source_node'].values, dtype=torch.long)
            tgt_nodes = torch.tensor(group['target_node'].values, dtype=torch.long)
            data_dict[edge_type] = (src_nodes, tgt_nodes)
        return data_dict
    

    def make_data_graph(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> dgl.DGLGraph:
        """Create a DGLGraph from the data dictionary."""
        g = dgl.heterograph({
            ("node", edge_type, "node"): (src.tolist(), tgt.tolist())
            for edge_type, (src, tgt) in data.items()
        })
        
        for ntype in g.ntypes:
            num_nodes = g.num_nodes(ntype)
            g.nodes[ntype].data['feat'] = torch.randn(num_nodes, 128)
        return g
    
    
    def get_state_list(self) -> List[Tuple[int, int]]:
        """
        Return the list of statement edges removed from the graph data.
        """
        return self.state_list
    
    
    def get_data(self):
        return self.data
    
    def get_edge_types(self):
        return list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]