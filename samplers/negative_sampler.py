import dgl
import torch
import random

import dgl
import torch
import random

class NegativeSampler:
    """
    Efficient negative sampler for a given edge type in a DGL heterograph.

    Generates 1:1 negative samples stratified by source node distribution,
    avoiding actual positive edges (including inverses), using vectorized ops.
    """
    def __init__(self, graph: dgl.DGLHeteroGraph, edge_type: tuple, oversample_factor: float = 2.0):
        self.graph = graph
        self.edge_type = edge_type

        src, dst = graph.edges(etype=edge_type)
        num_dst = graph.num_nodes(edge_type[2])
        pos_ids = (src * num_dst + dst).tolist()
        inv_ids = (dst * num_dst + src).tolist()
        self.invalid_ids = set(pos_ids) | set(inv_ids)
        self.src_list = src.tolist()
        self.src_tensor = torch.tensor(self.src_list, dtype=torch.long)
        self.num_edges = len(self.src_list)
        self.num_dst = num_dst
        self.oversample = oversample_factor

    def sample(self) -> dgl.DGLHeteroGraph:
        neg_ids = set()
        target_count = self.num_edges
        while len(neg_ids) < target_count:
            remaining = target_count - len(neg_ids)
            M = int(remaining * self.oversample)
            idx = torch.randint(0, len(self.src_tensor), (M,), device=self.src_tensor.device)
            s_cand = self.src_tensor[idx]
            d_cand = torch.randint(0, self.num_dst, (M,), device=s_cand.device)
            cand_ids = (s_cand * self.num_dst + d_cand).tolist()
            for cid in cand_ids:
                if cid in self.invalid_ids or cid in neg_ids:
                    continue
                neg_ids.add(cid)
                if len(neg_ids) >= target_count: break

        neg_ids_list = list(neg_ids)
        neg_src = [cid // self.num_dst for cid in neg_ids_list]
        neg_dst = [cid % self.num_dst for cid in neg_ids_list]
        neg_src_tensor = torch.tensor(neg_src, dtype=torch.long, device=self.src_tensor.device)
        neg_dst_tensor = torch.tensor(neg_dst, dtype=torch.long, device=self.src_tensor.device)
        return dgl.heterograph({self.edge_type: (neg_src_tensor, neg_dst_tensor)})



class NegativeSampler_OLD:
    """
    Given an input DGL HeteroGraph and a given edge type, this class generates
    negative samples for that edge type with a ratio of 1:1. """

    def __init__(self, graph, edge_type):
        self.graph = graph
        self.edge_type = edge_type
        self.negative_edges = None

    def sample(self):
        import dgl
        import torch

        src_nodes, dst_nodes = self.graph.edges(etype=self.edge_type)
        num_edges = len(src_nodes)
        # Randomly sample negative edges
        neg_src_nodes = torch.randint(0, self.graph.num_nodes(self.edge_type[0]), (num_edges,))
        neg_dst_nodes = torch.randint(0, self.graph.num_nodes(self.edge_type[2]), (num_edges,))

        # Create negative edges
        self.negative_edges = (neg_src_nodes, neg_dst_nodes)

        return dgl.heterograph({self.edge_type: self.negative_edges})