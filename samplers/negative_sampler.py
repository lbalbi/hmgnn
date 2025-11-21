import dgl
import torch
from typing import Optional, Tuple

class NegativeSampler:
    """
    Negative sampler for a DGL heterograph edge type.

    - Stratifies over *training* positive sources from `graph` for this edge type.
    - Forbids sampling any edge in `all_pos_edges` (e.g. union train+val+test PPIs),
      and for same-type (e.g. node–PPI–node) also forbids reversed pairs.
    - Returns raw (src, dst) tensors instead of a heterograph.
    """

    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        edge_type: tuple,
        oversample_factor: float = 2.0,
        device: Optional[torch.device] = None,
        all_pos_edges: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            graph: DGLHeteroGraph defining *training* positive edges (and node counts).
            edge_type: canonical etype tuple, e.g. ('node', 'PPI', 'node').
            oversample_factor: multiplicative factor for candidate sampling.
            device: device to keep internal tensors on.
            all_pos_edges: optional [2, N_all] tensor of *all* positive edges
                           (train+val+test) in the **global** graph. If None,
                           falls back to using `graph.edges(etype=edge_type)`.
        """
        self.graph = graph
        self.edge_type = edge_type
        self.oversample = float(oversample_factor)

        src_type, _, dst_type = edge_type

        # Training positives: used for source distribution
        src_train, dst_train = graph.edges(etype=edge_type)

        if device is None:
            self.device = src_train.device
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        src_train = src_train.to(self.device)
        dst_train = dst_train.to(self.device)
        self.src_pool = src_train  # stratify over training-positive sources

        self.num_dst = graph.num_nodes(dst_type)

        # Build invalid ID set from all positives (optionally global)
        if all_pos_edges is not None:
            pos_all = all_pos_edges.to(self.device)
            pos_src_all, pos_dst_all = pos_all[0].long(), pos_all[1].long()
        else:
            pos_src_all, pos_dst_all = src_train.long(), dst_train.long()

        pos_ids = (pos_src_all * self.num_dst + pos_dst_all).tolist()
        if src_type == dst_type:
            inv_ids = (pos_dst_all * self.num_dst + pos_src_all).tolist()
            invalid_ids = set(pos_ids) | set(inv_ids)
        else:
            invalid_ids = set(pos_ids)
        self.invalid_ids = invalid_ids

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample `num_samples` negatives (src, dst) for this edge type.

        Returns:
            neg_src: LongTensor [M] on `self.device`
            neg_dst: LongTensor [M] on `self.device`
            where M == num_samples (unless graph is tiny and we exhaust candidates).
        """
        target_count = int(num_samples)
        if target_count <= 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )

        neg_ids: set[int] = set()
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
        neg_src = torch.tensor(
            [cid // self.num_dst for cid in neg_ids_list],
            dtype=torch.long,
            device=self.device,
        )
        neg_dst = torch.tensor(
            [cid % self.num_dst for cid in neg_ids_list],
            dtype=torch.long,
            device=self.device,
        )
        return neg_src, neg_dst
