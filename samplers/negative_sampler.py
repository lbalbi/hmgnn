import torch
from torch import Tensor
from typing import Tuple, Optional
from torch_geometric.data import HeteroData


class NegativeSampler:
    """ Efficient negative sampler for a given edge type in a PyG HeteroData, 1:1 negative samples for an edge type. 
    Stratifies by positive source node distribution and avoids sampling actual positive edges (and inverse edges).
    Args:
        graph (HeteroData): Input heterogeneous graph.
        edge_type (tuple[str, str, str]): Canonical edge key, e.g. ("node", "PPI", "node").
        oversample_factor (float): Multiplicative factor to oversample candidate pairs per iteration
                                   (improves acceptance rate for dense graphs).
        device (Optional[torch.device] or str): Device to keep internal tensors on (defaults to graph's device).
    """
    def __init__(self, graph: HeteroData, edge_type: tuple, oversample_factor: float = 2.0,
                 device: Optional[torch.device] = None):
        
        self.graph = graph
        self.edge_type = edge_type
        self.oversample = float(oversample_factor)
        ei = graph[edge_type].edge_index
        self.device = ei.device if device is None else (device if isinstance(device, torch.device) else torch.device(device))
        src_type, _, dst_type = edge_type
        src, dst = ei.to(self.device)

        def _num_nodes(nt: str) -> int:
            explicit = getattr(graph[nt], "num_nodes", None)
            if explicit is not None: return int(explicit)
            if "x" in graph[nt] and graph[nt].x is not None: return int(graph[nt].x.size(0))
            max_id = -1
            for (s, _, d) in graph.edge_types:
                ei_nt = graph[(s, _, d)].edge_index
                if s == nt and ei_nt.size(1) > 0: max_id = max(max_id, int(ei_nt[0].max().item()))
                if d == nt and ei_nt.size(1) > 0: max_id = max(max_id, int(ei_nt[1].max().item()))
            return max_id + 1 if max_id >= 0 else 0
        self.num_src = _num_nodes(src_type)
        self.num_dst = _num_nodes(dst_type)

        pos_ids = (src.long() * self.num_dst + dst.long()).tolist()
        if src_type == dst_type:
            inv_ids = (dst.long() * self.num_dst + src.long()).tolist()
            invalid_ids = set(pos_ids) | set(inv_ids)
        else: invalid_ids = set(pos_ids)
        self.invalid_ids = invalid_ids
        self.pos_src = src
        self.pos_dst = dst
        self.num_pos = src.numel()
        self.src_pool = src


    def sample(self) -> Tuple[Tensor, Tensor]:
        """ Returns: neg_src (LongTensor [M]): Negative source node ids, neg_dst (LongTensor [M]): Negative destination node ids.
        Where M == number of positive edges for this edge type.
        """
        target_count = int(self.num_pos)
        if target_count == 0: return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
        neg_ids: set = set()
        while len(neg_ids) < target_count:
            remaining = target_count - len(neg_ids)
            M = max(int(remaining * self.oversample), remaining)
            idx = torch.randint(0, self.src_pool.numel(), (M,), device=self.device)
            s_cand = self.src_pool[idx]
            d_cand = torch.randint(0, self.num_dst, (M,), device=self.device)
            cand_ids = (s_cand.long() * self.num_dst + d_cand.long()).tolist()
            for cid in cand_ids:
                if cid in self.invalid_ids or cid in neg_ids: continue
                neg_ids.add(cid)
                if len(neg_ids) >= target_count: break

        neg_ids_list = list(neg_ids)
        neg_src = torch.tensor([cid // self.num_dst for cid in neg_ids_list], dtype=torch.long, device=self.device)
        neg_dst = torch.tensor([cid %  self.num_dst for cid in neg_ids_list], dtype=torch.long, device=self.device)
        return neg_src, neg_dst

    def sample_edge_index(self) -> Tensor:
        """ Convenience: returns a edge_index tensor for the sampled negatives.
        """
        s, d = self.sample()
        if s.numel() == 0: return torch.empty(2, 0, dtype=torch.long, device=self.device)
        return torch.stack([s, d], dim=0)
