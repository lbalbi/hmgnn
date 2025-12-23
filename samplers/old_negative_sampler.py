import torch
from torch import Tensor
from typing import Tuple, Optional
from torch_geometric.data import HeteroData


class NegativeSampler:
    """ Negative sampler for an edge type, draws negatives by stratifying over training positive sources.
    Forbids sampling any edge that appears in `all_pos_edge_index` (e.g. union of train+val+test PPI edges), 
    including reversed pairs (src_type == dst_type).
    Args:
        graph (HeteroData): Graph that defines the *training* positive edges for this sampler
                            (and node counts). Typically a train-only graph.
        edge_type (tuple[str,str,str]): Canonical edge key, e.g. ("node", "PPI", "node").
        oversample_factor (float): Multiplicative factor for candidate sampling.
        device (torch.device or str, optional): Device for internal tensors.
        all_pos_edge_index (Tensor, optional): [2, N_all] tensor of *all* positive edges
                            (train+val+test) to avoid sampling as negatives.
                            If None, falls back to using graph[edge_type].edge_index
                            (i.e. old behavior). """

    def __init__(self, graph: HeteroData, edge_type: tuple, oversample_factor: float = 2.0,
        device: Optional[torch.device] = None, all_pos_edge_index: Optional[Tensor] = None):

        self.graph = graph
        self.edge_type = edge_type
        self.oversample = float(oversample_factor)
        ei_train = graph[edge_type].edge_index
        self.device = ei_train.device if device is None else (
            device if isinstance(device, torch.device) else torch.device(device))
        src_type, _, dst_type = edge_type


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
        train_src, train_dst = ei_train.to(self.device)
        self.src_pool = train_src
        
        if all_pos_edge_index is not None:
            pos_all = all_pos_edge_index.to(self.device)
            pos_src_all, pos_dst_all = pos_all[0], pos_all[1]
        else: pos_src_all, pos_dst_all = train_src, train_dst
        pos_ids = (pos_src_all.long() * self.num_dst + pos_dst_all.long()).tolist()
        if src_type == dst_type:
            inv_ids = (pos_dst_all.long() * self.num_dst + pos_src_all.long()).tolist()
            invalid_ids = set(pos_ids) | set(inv_ids)
        else: invalid_ids = set(pos_ids)
        self.invalid_ids = invalid_ids


    def sample(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        """ Samples negatives for given positives
        Args: num_samples (int): number of negative edges to sample.
        Returns: neg_src (LongTensor [M]), neg_dst (LongTensor [M])
            where M == num_samples (unless graph is tiny and we exhaust candidates).
        """
        target_count = int(num_samples)
        if target_count <= 0:
            return (torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device))

        neg_ids: set = set()
        while len(neg_ids) < target_count:
            remaining = target_count - len(neg_ids)
            M = max(int(remaining * self.oversample), remaining)
            idx = torch.randint(0, self.src_pool.numel(), (M,), device=self.device)
            s_cand = self.src_pool[idx]
            d_cand = torch.randint(0, self.num_dst, (M,), device=self.device)
            cand_ids = (s_cand.long() * self.num_dst + d_cand.long()).tolist()

            for cid in cand_ids:
                if cid in self.invalid_ids or cid in neg_ids:
                    continue
                neg_ids.add(cid)
                if len(neg_ids) >= target_count:
                    break

        neg_ids_list = list(neg_ids)
        neg_src = torch.tensor([cid // self.num_dst for cid in neg_ids_list], dtype=torch.long, device=self.device)
        neg_dst = torch.tensor([cid % self.num_dst for cid in neg_ids_list],dtype=torch.long,device=self.device)
        return neg_src, neg_dst


    def sample_edge_index(self, num_samples: int) -> Tensor:
        """returns edge_indexfor sampled negatives"""
        s, d = self.sample(num_samples)
        if s.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        return torch.stack([s, d], dim=0)
