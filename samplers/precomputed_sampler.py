import torch
from torch import Tensor
from typing import Tuple, Optional


class PrecomputedNegativeSampler:
    """Negative sampler that draws negatives from a *precomputed* edge list (e.g. edge_type == "neg_PPI").

    Critically, sampling is **stratified by source node**: for each positive edge (s, *), we sample one
    negative edge (s, d_neg) from the precomputed pool where src == s.

    This is what you want when your train/val/test splits are defined by source proteins.

    Args:
        neg_edge_index: [2, E_neg] tensor of precomputed negatives.
        num_nodes: Total number of protein nodes.
        device: Sampling device.
        all_pos_edge_index: Optional [2, E_pos] tensor of all positives to filter out overlaps.
        forbid_reverse: If True, also filters out negatives that are the reverse of any positive.
    """

    def __init__(
        self,
        neg_edge_index: Tensor,
        num_nodes: int,
        device: Optional[torch.device] = None,
        all_pos_edge_index: Optional[Tensor] = None,
        forbid_reverse: bool = True,
    ):
        if neg_edge_index is None:
            raise ValueError("neg_edge_index must be provided for PrecomputedNegativeSampler.")
        if neg_edge_index.dim() != 2 or neg_edge_index.size(0) != 2:
            raise ValueError(f"neg_edge_index must have shape [2, E], got {tuple(neg_edge_index.shape)}")
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be > 0, got {num_nodes}")

        self.num_nodes = int(num_nodes)
        self.device = (
            device
            if isinstance(device, torch.device)
            else (torch.device(device) if device is not None else neg_edge_index.device)
        )

        neg_src = neg_edge_index[0].long().cpu()
        neg_dst = neg_edge_index[1].long().cpu()

        # Filter out overlaps with positives (and optionally reverse overlaps).
        if all_pos_edge_index is not None and all_pos_edge_index.numel() > 0:
            pos = all_pos_edge_index.long().cpu()
            pos_ids = pos[0] * self.num_nodes + pos[1]
            if forbid_reverse:
                pos_ids = torch.cat([pos_ids, pos[1] * self.num_nodes + pos[0]], dim=0)
            pos_ids = torch.unique(pos_ids)

            neg_ids = neg_src * self.num_nodes + neg_dst
            keep = ~torch.isin(neg_ids, pos_ids)
            neg_src = neg_src[keep]
            neg_dst = neg_dst[keep]

        # Deduplicate.
        neg_ids = torch.unique(neg_src * self.num_nodes + neg_dst)
        neg_src = (neg_ids // self.num_nodes).to(torch.long)
        neg_dst = (neg_ids % self.num_nodes).to(torch.long)

        # Sort by source for fast lookup.
        perm = torch.argsort(neg_src)
        self.src_sorted = neg_src[perm].to(self.device)
        self.dst_sorted = neg_dst[perm].to(self.device)
        if self.src_sorted.numel() == 0:
            raise RuntimeError("PrecomputedNegativeSampler received 0 negative edges after filtering.")

        uniq_src, counts = torch.unique_consecutive(self.src_sorted, return_counts=True)
        ptr = torch.zeros(uniq_src.numel() + 1, dtype=torch.long, device=self.device)
        ptr[1:] = torch.cumsum(counts, dim=0)
        self.uniq_src = uniq_src
        self.ptr = ptr

    def _range_for_src(self, s: int) -> Tuple[int, int]:
        """Return [start, end) range in the sorted arrays for a single source node id."""
        s_t = torch.tensor(s, device=self.device, dtype=torch.long)
        i = int(torch.searchsorted(self.uniq_src, s_t).item())
        if i >= int(self.uniq_src.numel()) or int(self.uniq_src[i].item()) != int(s):
            return -1, -1
        start = int(self.ptr[i].item())
        end = int(self.ptr[i + 1].item())
        return start, end

    def sample_like(self, pos_edge_index: Tensor) -> Tensor:
        """Sample exactly one negative per positive edge, matching the positive sources."""
        if pos_edge_index.dim() != 2 or pos_edge_index.size(0) != 2:
            raise ValueError(f"pos_edge_index must have shape [2, N], got {tuple(pos_edge_index.shape)}")
        return self.sample_for_sources(pos_edge_index[0].long().to(self.device))

    def sample_for_sources(self, pos_src: Tensor) -> Tensor:
        """Given a vector of positive sources (length N), sample N negatives from the precomputed pool."""
        pos_src = pos_src.long().to(self.device)
        if pos_src.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)

        uniq, counts = torch.unique(pos_src, return_counts=True)
        sampled_src_parts = []
        sampled_dst_parts = []

        # Loop over sources in the batch; typically small compared to #edges.
        for s, c in zip(uniq.tolist(), counts.tolist()):
            start, end = self._range_for_src(int(s))
            if start < 0:
                raise RuntimeError(
                    f"Precomputed negatives do not contain any edges for source node {s}. "
                    "This breaks source-based splitting; please ensure neg_PPI covers all sources."
                )
            pool_len = end - start
            idx = torch.randint(0, pool_len, (int(c),), device=self.device)
            dst = self.dst_sorted[start + idx]
            sampled_src_parts.append(torch.full((int(c),), int(s), device=self.device, dtype=torch.long))
            sampled_dst_parts.append(dst)

        neg_src = torch.cat(sampled_src_parts, dim=0)
        neg_dst = torch.cat(sampled_dst_parts, dim=0)

        # Shuffle to avoid ordering artifacts across sources.
        perm = torch.randperm(neg_src.numel(), device=self.device)
        neg_src = neg_src[perm]
        neg_dst = neg_dst[perm]
        return torch.stack([neg_src, neg_dst], dim=0)