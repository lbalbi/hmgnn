import argparse, torch, pandas as pd
from utils import load_config
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLHeteroGraph
from dgl.nn.pytorch import HeteroGraphConv
from typing import List, Tuple, Dict, Optional, Iterator
from collections import defaultdict
import dgl, torch, random


class Logger:

    def __init__(self, name):
        self.name = name
        self.log_file = f"{name}.log"
        self.file = open(self.log_file, 'w')
        self.file.write(f"Results for {name}\n")
        self.file.write("=" * 50 + "\n")

    def log(self, message):
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def is_early_stop(self):
        return self.early_stop
    
    def get_patience(self):
        return self.patience
    
    def get_best_score(self):
        return self.best_score
    
    def get_counter(self):
        return self.counter
    
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
class Metrics:

    @staticmethod
    def accuracy(preds, labels):
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        return correct / total if total > 0 else 0

    @staticmethod
    def f1score(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    @staticmethod
    def precision(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def recall(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def confusion_matrix(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def rocauc_score(preds, labels):
        return roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
    
    def update(self, preds, labels):
        preds = torch.round(preds)
        self.accuracy_value = accuracy_score(labels, preds)
        self.f1_value = f1_score(labels, preds, average='weighted', zero_division=0)
        self.precision_value = precision_score(labels, preds, zero_division=0)
        self.recall_value = recall_score(labels, preds, zero_division=0)
        self.roc_auc_value = roc_auc_score(labels, preds, average='weighted')
        return self.accuracy_value, self.f1_value, self.precision_value, self.recall_value, self.roc_auc_value
    
    def get_names(self):
        return ["accuracy, f1 score, precision, recall, roc auc"]

    
import json
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

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
        graph: dgl.DGLGraph,
        ppi_rel: str = "PPI",
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        device : Optional[torch.device] = "cpu",
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        assert isinstance(graph, dgl.DGLGraph), "graph must be a DGLHeteroGraph"
        self.graph = graph
        self.ppi_rel = ppi_rel
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if seed is not None: torch.manual_seed(seed)

        self.ppi_etype = next(
            (et for et in graph.canonical_etypes if et[1] == ppi_rel),
            None
        )
        assert self.ppi_etype is not None, f"Relation '{ppi_rel}' not in graph.canonical_etypes"

        self.other_etypes = [et for et in graph.canonical_etypes if et != self.ppi_etype]

        self._split_by_source_nodes()
        self.train_graph = self._create_split_graph(self.train_eids)
        self.val_graph = self._create_split_graph(self.val_eids)
        self.test_graph = self._create_split_graph(self.test_eids)


    def _split_by_source_nodes(self): # Get all PPI edges' source nodes and corresponding edge IDs
        src, _ = self.graph.edges(etype=self.ppi_etype)
        num_ppi = src.size(0)
        edge_ids = torch.arange(num_ppi)
        unique_src = torch.unique(src)

        if self.shuffle:  unique_src = unique_src[torch.randperm(len(unique_src))]

        n_test = int(len(unique_src) * self.test_split)
        n_val = int(len(unique_src) * self.val_split)
        n_train = len(unique_src) - n_val - n_test

        train_src = unique_src[:n_train]
        val_src = unique_src[n_train:n_train + n_val]
        test_src = unique_src[n_train + n_val:]

        mask_train = torch.isin(src, train_src)
        mask_val = torch.isin(src, val_src)
        mask_test = torch.isin(src, test_src)

        self.train_eids = edge_ids[mask_train]
        self.val_eids = edge_ids[mask_val]
        self.test_eids = edge_ids[mask_test]


    def _create_split_graph(self, ppi_eids: torch.Tensor) -> dgl.DGLGraph:
        """
        Build a subgraph containing specified PPI edges (by edge IDs) and all edges of other types.
        Preserves all original nodes.
        """
        eid_dict = {}

        for et in self.other_etypes:
            num = self.graph.num_edges(et)
            eid_dict[et] = torch.arange(num)
        eid_dict[self.ppi_etype] = ppi_eids

        # Build edge-induced subgraph, preserving nodes
        return dgl.edge_subgraph(self.graph, eid_dict, relabel_nodes=False, store_ids=True).to(self.device)

    def _batch_graphs(self, eids: torch.Tensor) -> Iterator[dgl.DGLGraph]:
        """
        Yield mini-batch subgraphs for slices of PPI edges.
        """
        for start in range(0, len(eids), self.batch_size):
            batch_eids = eids[start:start + self.batch_size]
            yield self._create_split_graph(batch_eids)

    def train_batches(self) -> Iterator[dgl.DGLGraph]:
        """Iterate mini-batch train subgraphs."""
        return self._batch_graphs(self.train_eids)

    def validation_batches(self) -> Iterator[dgl.DGLGraph]:
        """Iterate mini-batch validation subgraphs."""
        return self._batch_graphs(self.val_eids)

    def test_batches(self) -> Iterator[dgl.DGLGraph]:
        """Iterate mini-batch test subgraphs."""
        return self._batch_graphs(self.test_eids)

    def get_split_graphs(self) -> Tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
        """Return the full train/val/test heterographs."""
        return self.train_graph, self.val_graph, self.test_graph

    def get_relation(self) -> str:
        """
        Return the relation name used for PPI edges.
        """
        return self.ppi_rel

    def __iter__(self) -> Iterator[dgl.DGLGraph]:
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


class DataLoader:
    """DataLoader for loading heterogeneous graph data from CSV files.
    The output is a dictionary of torch tensors containing the edge types as keys 
    and the pairs of source and target nodes as values."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        edge_files = self.path_files(file_path)
        self.data = self.load_data(edge_files)

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
    
    
    def get_data(self):
        return self.data
    
    def get_edge_types(self):
        return list(self.data.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.fc2 = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        e  = self.attn_fc(z2)
        return {"e": F.leaky_relu(e)}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h     = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, g, h):

        if isinstance(h, tuple):
            h_src, h_dst = h
            z_src = self.fc1(h_src)
            z_dst = self.fc2(h_dst)
            g.srcdata["z"] = z_src
            g.dstdata["z"] = z_dst
        else:
            z = self.fc1(h)
            g.ndata["z"] = z

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        if isinstance(h, tuple):
            return g.dstdata.pop("h")
        else:
            return g.ndata.pop("h")


class NegativeStatementSampler:
    """
    A sampler that uses negative statements as negative samples
    for each node in the training graph.
    """
    def __init__(self,  k: int = 2, go_etype="GO"):
        self.k = k
        self.go_etype = go_etype


    def prepare_global(self, full_g, pos_etype="pos_statement", neg_etype="neg_statement"):

        N = full_g.num_nodes()
        src_neg, dst_neg = full_g.edges(etype=neg_etype)
        src_pos, dst_pos = full_g.edges(etype=pos_etype)
        direct = [[] for _ in range(N)]
        pos    = [[] for _ in range(N)]
        for u,v in zip(src_neg.tolist(), dst_neg.tolist()): direct[u].append(v)
        for u,v in zip(src_pos.tolist(), dst_pos.tolist()): pos[u].append(v)
        two_hop = {}
        for u in range(N):
            if len(direct[u]) < self.k:
                sup = set()
                for b in direct[u]:
                    sup.update(full_g.predecessors(b, etype=self.go_etype).tolist())
                two_hop[u] = list(sup)
        self.direct_global = direct
        self.two_hop_global = two_hop
        self.pos_global = pos


    def prepare_batch(self, batch):

        orig = batch.ndata[dgl.NID].tolist()
        mapping = {g:i for i,g in enumerate(orig)}
        self.N = len(orig)
        self.device = batch.device
        self.direct = [[] for _ in range(self.N)]
        self.pos = [[] for _ in range(self.N)]
        self.two_hop = {}
        for i, g_id in enumerate(orig):
            for g_nb in self.direct_global[g_id]:
                if g_nb in mapping:
                    self.direct[i].append(mapping[g_nb])
            for g_nb in self.pos_global[g_id]:
                if g_nb in mapping:
                    self.pos[i].append(mapping[g_nb])
            if len(self.direct[i]) < self.k:
                sup = set()
                for g_b in self.direct_global[g_id]:
                    for g_pp in self.two_hop_global.get(g_b, []):
                        if g_pp in mapping:
                            sup.add(mapping[g_pp])
                self.two_hop[i] = list(sup)


    def sample(self) -> torch.Tensor:

        neg_src, neg_dst = [], []
        for u in range(self.N):
            cands = self.direct[u]
            if len(cands) < self.k: cands = list({*cands, *self.two_hop.get(u, [])})
            if len(cands) >= self.k:  picks = random.sample(cands, self.k)
            else: picks = random.choices([v for v in range(self.N) if v != u], k=self.k)
            neg_src.extend([u] * self.k)
            neg_dst.extend(picks)
        return torch.tensor([neg_src, neg_dst], dtype=torch.long, device=self.device)


    def get_contrastive_samples(self, z: torch.Tensor):

        N, D = z.shape
        z_pos = z
        pos_nb = [random.choice(self.pos[u]) if self.pos[u] else u for u in range(N)]
        z_pos_pos = z[pos_nb]
        neg_ei    = self.sample()
        neg_dst   = neg_ei[1].view(N, self.k)
        z_pos_neg = z[neg_dst]
        # negatives
        z_neg     = z_pos
        z_neg_pos = z_pos_pos
        z_neg_neg = z_pos_neg
        return z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg


class DualContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg) -> torch.Tensor:

        B, D = z_pos.shape
        z_pos = F.normalize(z_pos, dim=1)
        z_pos_pos = F.normalize(z_pos_pos, dim=1)
        z_pos_neg = F.normalize(z_pos_neg, dim=2)
        z_neg = F.normalize(z_neg, dim=1)
        z_neg_pos = F.normalize(z_neg_pos, dim=1)
        z_neg_neg = F.normalize(z_neg_neg, dim=2)

        # POSITIVE REPRESENTATION
        # Similarity between anchor and positive
        sim_pos = torch.sum(z_pos * z_pos_pos, dim=1, keepdim=True) / self.temperature
        # Similarity between anchor and negative
        sim_neg = torch.bmm(z_pos.unsqueeze(1), z_pos_neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits_pos = torch.cat([sim_pos, sim_neg], dim=1)
        labels_pos = torch.zeros(B, dtype=torch.long, device=z_pos.device)
        loss_pos = F.cross_entropy(logits_pos, labels_pos)
        # NEGATIVE REPRESENTATION
        sim_neg_pos = torch.sum(z_neg * z_neg_pos, dim=1, keepdim=True) / self.temperature
        sim_neg_neg = torch.bmm(z_neg.unsqueeze(1), z_neg_neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits_neg = torch.cat([sim_neg_pos, sim_neg_neg], dim=1)
        labels_neg = torch.zeros(B, dtype=torch.long, device=z_neg.device)
        loss_neg = F.cross_entropy(logits_neg, labels_neg)

        total_loss = loss_pos + loss_neg
        return total_loss
    

class HGCN(nn.Module):
    """
    Heterogeneous GCN model for link classification on a specific edge relation (e.g. PPI).

    Args:
        in_feats (Dict[str, int]): Input feature sizes for each node type.
        hidden_dim (int): Hidden embedding dimension.
        out_dim (int): Output dimension for classification (e.g., number of classes).
        n_layers (int): Number of GCN layers.
        ppi_etype (Tuple[str, str, str]): Canonical edge type to classify, e.g. ("protein", "PPI", "protein").
    """
    def __init__(
        self,
        in_feats: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        ppi_etype: Tuple[str, str, str] = ("node", "PPI", "node"),
        n_type: str = "node",
        e_etypes: List[Tuple[str, str, str]] = None,
    ):
        super(HGCN, self).__init__()
        self.ppi_etype = ppi_etype
        self.e_types = e_etypes
        self.n_type = n_type
        self.layers = nn.ModuleList()
        in_dim_maps = [in_feats] + [{etype: hidden_dim for etype in in_feats} for _ in range(n_layers - 1)]
        self.layers = nn.ModuleList([HeteroGraphConv({rel: GCNLayer(in_dim_maps[layer_idx][rel], hidden_dim)
                    for src, rel, dst in e_etypes}, aggregate="mean") for layer_idx in range(n_layers)])

        self.classify = nn.Linear(2 * hidden_dim, out_dim)



    def forward(self, graph: DGLHeteroGraph, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h_dict = {ntype: graph.nodes[ntype].data["feat"] for ntype in graph.ntypes}

        for layer in self.layers:
            h_dict = layer(graph, h_dict)
            h_dict = {nt: F.relu(h) for nt, h in h_dict.items()}

        src_ids, dst_ids = edge_index
        hs = h_dict[self.n_type][src_ids]
        hd = h_dict[self.n_type][dst_ids]
        h_pair = torch.cat([hs, hd], dim=1)
        logits = self.classify(h_pair)
        z = h_dict[self.n_type]
        
        return z, torch.sigmoid(logits)
    
class Train:
    def __init__(self, model, optimizer, epochs, train_loader, val_loader, e_type, log, device, contrastive_weight=0.1):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.e_type = e_type
        self.neg_sampler = NegativeStatementSampler()
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss(temperature=0.5)
        self.alpha = contrastive_weight
        self.log = log
        self.device = device
        self.epochs = epochs
        self.metrics = Metrics()
        self.log.log("Setting, Epoch, " + "".join(name + ", " for name in self.metrics.get_names())[:-2])


    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:

            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            src, dst  = batch.edges(etype=self.e_type)
            edge_index = torch.stack([src, dst], dim=0)
            self.neg_sampler.prepare_batch(batch)
            neg_edge_index = self.neg_sampler.sample()
            labels = torch.cat([torch.ones(edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
            edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
            z, out = self.model(batch, edge_index)
            (z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg) = self.neg_sampler.get_contrastive_samples(z)
            loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg)
            loss = self.loss_fn(out.squeeze(-1), labels)
            loss_ = self.alpha * loss_contrast + loss
            loss_.backward()
            self.optimizer.step()
            num_examples = out.size(0)
            total_loss += loss_.item() * num_examples
            total_examples += num_examples
        if total_examples == 0: return 0
        return total_loss / total_examples
    

    def validate_epoch(self):
        self.model.eval()
        total_loss = total_examples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                src, dst  = batch.edges(etype=self.e_type)
                edge_index = torch.stack([src, dst], dim=0)
                self.neg_sampler.prepare_batch(batch)
                neg_edge_index  = self.neg_sampler.sample()
                edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
                labels = torch.cat([torch.ones(src.size(0), device=self.device),
                    torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
                z, out = self.model(batch, edge_index)
                (z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg) = self.neg_sampler.get_contrastive_samples(z)
                loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg)
                loss = self.loss_fn(out.squeeze(-1), labels)
                loss_ = self.alpha * loss_contrast + loss
                num_examples = out.size(0)
                total_loss += loss_.item() * num_examples
                total_examples += num_examples
            return (total_loss / total_examples), (out, labels)

    def run(self, graph):
        self.neg_sampler.prepare_global(graph)
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            if train_loss != 0: print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}', flush=True)


class Test:
    def __init__(self, model, test_loader, e_type, log, device):
        self.model = model
        self.test_loader = test_loader
        self.e_type = e_type
        self.log = log
        self.device = device
        self.neg_sampler = NegativeStatementSampler()
        self.metrics = Metrics()

    def test_epoch(self):
        self.model.eval()
        total_loss = total_examples = 0

        for batch in self.test_loader:
            batch = batch.to(self.device)
            src, dst  = batch.edges(etype=self.e_type)
            edge_index = torch.stack([src, dst], dim=0)
            self.neg_sampler.prepare_batch(batch)
            neg_edge_index  = self.neg_sampler.sample()
            edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
            labels = torch.cat([torch.ones(src.size(0), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
            with torch.no_grad():
                z, out = self.model(batch, edge_index)
            print(out, labels, flush=True)
            self.metrics.update(out.detach().to("cpu"), labels.to("cpu"))

        return (total_loss / total_examples), out
    
    
    def run(self, graph):
        self.neg_sampler.prepare_global(graph)
        test_loss, out = self.test_epoch()
        acc, f1, precision, recall, roc_auc = self.metrics.update(test_loss, out)
        self.log.log("Test, final", str(acc), str(f1), str(precision), str(recall), str(roc_auc))

        print('Test Results:', flush=True)
        print(f'Loss: {test_loss:.4f}', flush=True)
        print(f'Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f},' \
         'Recall: {recall:.4f}, Roc Auc: {roc_auc:.4f}', flush=True)

        torch.save(self.model.state_dict(), 'model_'+ self.model.__class__.__name__ +'.pth')
        torch.save(out, "predictions_" + self.model.__class__.__name__ + ".pth")
        self.log.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices= ["gcn","gat", "hgcn", "hgat", "bigcn", "bigat"], default="hgcn")
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256*350, help='Batch size for training')
    parser.add_argument('--path', type=str, default="data", help='Folder with data files, defaults to data/ directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = eval(args.model.upper())
    cfg = load_config()
    mcfg = cfg["models"][ModelCls.__name__]
    dl = DataLoader(args.path + "/")
    full_graph = dl.make_data_graph(dl.get_data())
    ppi_etype  = mcfg["ppi_etype"]
    src_all, dst_all = full_graph.edges(etype=ppi_etype)
    pairs = list(zip(src_all.tolist(), dst_all.tolist()))
    n_splits = min(5, len(pairs))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_auc = -1.0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(pairs), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===", flush=True)

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]
        all_etypes = full_graph.canonical_etypes
        full_eids = {}
        for et in all_etypes:
            if et != ppi_etype:
                full_eids[et] = full_graph.edges(form='eid', etype=et)
        train_src, train_dst = zip(*train_pairs)
        val_src, val_dst = zip(*val_pairs)
        train_src = torch.tensor(train_src, dtype=torch.long)
        train_dst = torch.tensor(train_dst, dtype=torch.long)
        val_src = torch.tensor(val_src, dtype=torch.long)
        val_dst = torch.tensor(val_dst, dtype=torch.long)
        train_ppi_eids = full_graph.edge_ids(train_src, train_dst, etype=ppi_etype)
        val_ppi_eids = full_graph.edge_ids(val_src, val_dst, etype=ppi_etype)
        train_eid_map = {}
        val_eid_map   = {}
        for et in all_etypes:
            if et == ppi_etype:
                train_eid_map[et] = train_ppi_eids
                val_eid_map[et]   = val_ppi_eids
            else:
                train_eid_map[et] = full_eids[et]
                val_eid_map[et]   = full_eids[et]
        train_graph = full_graph.edge_subgraph(train_eid_map, relabel_nodes=True, store_ids=True)
        val_graph   = full_graph.edge_subgraph(val_eid_map, relabel_nodes=True, store_ids=True)
        train_loader = Dglloader(train_graph, batch_size=args.batch_size, device=device).train_batches()
        val_loader   = Dglloader(val_graph, batch_size=args.batch_size, device=device).train_batches()

        model = ModelCls(in_feats = mcfg["in_feats"], hidden_dim = mcfg["hidden_dim"],
        out_dim = mcfg["out_dim"], e_etypes = [tuple(e) for e in mcfg["edge_types"]],
        ppi_etype  = ppi_etype).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        log   = Logger(f"{ModelCls.__name__}_fold{fold}")
        trainer = Train(model, optim, args.epochs, train_loader, val_loader,
                        e_type=ppi_etype, log=log, device=device, contrastive_weight=cfg["contrastive_weight"])
        trainer.run(train_graph)
        val_loss, (logits, labels) = trainer.validate_epoch()
        print(logits, labels, flush=True)
        _, _, _, _, auc = trainer.metrics.update(logits.detach().to("cpu"), labels.to("cpu"))
        print(f"Fold {fold} AUC: {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc   = auc
            best_state = model.state_dict()
    print(f"\nBest fold AUC = {best_auc:.4f} – saving model\n", flush=True)

    final_model = ModelCls(in_feats = mcfg["in_feats"], hidden_dim = mcfg["hidden_dim"], out_dim = mcfg["out_dim"],
        e_etypes = [tuple(e) for e in mcfg["edge_types"]], ppi_etype  = ppi_etype).to(device)
    final_model.load_state_dict(best_state)
    final_optim = torch.optim.Adam(final_model.parameters(), lr=cfg["lr"])
    final_log = Logger("final_train")

    full_train_loader = Dglloader(full_graph, batch_size = args.batch_size, device = device).train_batches()
    final_trainer = Train(final_model, final_optim, args.epochs, full_train_loader, [], 
                          e_type=ppi_etype, log=final_log, device=device, contrastive_weight=cfg["contrastive_weight"])
    loss = final_trainer.run()
    print(f"Final training loss: {loss:.4f}", flush=True)

    test_loader = Dglloader(full_graph, batch_size = args.batch_size, device = device).test_batches()
    tester = Test(final_model, loss=torch.nn.BCEWithLogitsLoss(), test_loader=test_loader, log=Logger("final_test"),
                  device=device)
    tester.run(full_graph)


if __name__ == "__main__":
    main()
