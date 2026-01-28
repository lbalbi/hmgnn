import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

class FileNegativeSampler:
    """
    File-backed negatives, compatible with trainer_bestmodel.Test_BestModel,
    which calls sampler.sample_for_heads(...).
    """

    def __init__(
        self,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor,
        *,
        num_nodes: int,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.device = device if device is not None else torch.device("cpu")
        self.rng = np.random.RandomState(int(seed))

        nh = neg_heads.detach().cpu().long().numpy() if neg_heads is not None else np.zeros((0,), dtype=np.int64)
        nt = neg_tails.detach().cpu().long().numpy() if neg_tails is not None else np.zeros((0,), dtype=np.int64)

        self.head2tails: Dict[int, List[int]] = defaultdict(list)
        for h, t in zip(nh.tolist(), nt.tolist()):
            self.head2tails[int(h)].append(int(t))

        self.global_tails: List[int] = nt.tolist()

        # deterministic shuffles
        for h in list(self.head2tails.keys()):
            self.rng.shuffle(self.head2tails[h])
        self.rng.shuffle(self.global_tails)

    def _pick_tail(
        self,
        h: int,
        *,
        invalid_ids: Set[int],
    ) -> int:
        """
        Pick a single tail for head h from file negatives:
        - prefer head-specific pool
        - fallback to global pool
        - fallback to random tail if pools empty
        Resamples a few times to avoid invalid_ids.
        """
        pool = self.head2tails.get(h, None)
        if pool is None or len(pool) == 0:
            pool = self.global_tails

        # Try a few times to avoid invalid ids
        if pool and len(pool) > 0:
            for _ in range(20):
                t = pool[self.rng.randint(0, len(pool))]
                if (h * self.num_nodes + t) not in invalid_ids:
                    return int(t)

            # If everything seems invalid, just return something (best effort)
            return int(pool[self.rng.randint(0, len(pool))])

        # Absolute fallback
        return int(self.rng.randint(0, self.num_nodes))

    def sample_for_heads(
        self,
        heads: torch.Tensor,
        num_negs_per_head: torch.Tensor,
        *,
        extra_invalid_ids: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match NegativeSampler API used by Test_BestModel:
          heads: [H]
          num_negs_per_head: [H] (how many negatives for each head)
        Returns:
          neg_src: [sum_k]
          neg_dst: [sum_k]
        """
        if extra_invalid_ids is None:
            extra_invalid_ids = set()

        h_list = heads.detach().cpu().long().tolist()
        k_list = num_negs_per_head.detach().cpu().long().tolist()

        neg_src: List[int] = []
        neg_dst: List[int] = []

        for h, k in zip(h_list, k_list):
            h = int(h)
            k = int(k)
            if k <= 0:
                continue
            for _ in range(k):
                t = self._pick_tail(h, invalid_ids=extra_invalid_ids)
                neg_src.append(h)
                neg_dst.append(int(t))

        if len(neg_src) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )

        return (
            torch.tensor(neg_src, dtype=torch.long, device=self.device),
            torch.tensor(neg_dst, dtype=torch.long, device=self.device),
        )


