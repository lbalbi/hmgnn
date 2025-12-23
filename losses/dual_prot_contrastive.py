import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from samplers import ProteinStatementSampler, PartialProteinSampler


def _cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


class ProteinContrastiveLoss(nn.Module):
    """Dual-view composed contrastive loss using protein↔protein examples.

    Requirements implemented:
      - If the sampler finds fewer than k negatives for an anchor, use only those negatives (no repetition).
      - If an anchor doesn't have at least 1 example for *each* required example type, that anchor is excluded
        by the sampler, so the loss never sees incomplete tuples.

    Loss terms:
      - Pull z_neg(u) towards z_neg(v) where v shares a NEG statement class with u.
        Negatives are z_pos(w) where w has a POS statement to a (expanded) NEG class of u.
      - Pull z_pos(u) towards z_pos(v) where v shares a POS statement class with u.
        Negatives are z_neg(w) where w has a NEG statement to a POS class of u.
    """

    def __init__(self, sampler: ProteinStatementSampler or PartialProteinSampler,
        temperature: float = 0.5,
        w_neg: float = 1.0,
        w_pos: float = 1.0,
    ):
        super().__init__()
        self.sampler = sampler
        self.temperature = float(temperature)
        self.w_neg = float(w_neg)
        self.w_pos = float(w_pos)

    def _info_nce_masked(self, anchor: Tensor, pos: Tensor, neg: Tensor, neg_mask: Tensor) -> Tensor:
        """Masked InfoNCE.

        anchor:   (B, D)
        pos:      (B, D)
        neg:      (B, K, D)  (may include padded rows)
        neg_mask: (B, K) bool, True where neg is valid.

        We set invalid neg logits to a large negative value so they don't affect the denominator.
        """
        B = anchor.size(0)
        if B == 0:
            return anchor.new_tensor(0.0)

        # If any rows have 0 valid negatives, drop them.
        row_ok = neg_mask.any(dim=1)
        if not row_ok.any():
            return anchor.new_tensor(0.0)

        anchor = anchor[row_ok]
        pos = pos[row_ok]
        neg = neg[row_ok]
        neg_mask = neg_mask[row_ok]

        pos_logits = _cosine_sim(anchor, pos).unsqueeze(1) / self.temperature  # (B,1)

        a = F.normalize(anchor, dim=-1).unsqueeze(1)  # (B,1,D)
        n = F.normalize(neg, dim=-1)                  # (B,K,D)
        neg_logits = (a * n).sum(dim=-1) / self.temperature  # (B,K)

        # mask padded negatives
        neg_logits = neg_logits.masked_fill(~neg_mask, -1e9)

        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (B,1+K)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(self, z_pos: Tensor, z_neg: Tensor) -> Tensor:
        samples = self.sampler.sample_batch()
        anchors = samples["anchors"]
        if anchors.numel() == 0:
            return z_pos.new_tensor(0.0)

        # Sampler already filters anchors so pos indices exist and each neg pool has >=1 example.
        pos_same_neg = samples["pos_same_neg"]           # (B,)
        neg_pos_to_my_neg = samples["neg_pos_to_my_neg"] # (B,K) padded with -1

        pos_same_pos = samples["pos_same_pos"]           # (B,)
        neg_neg_to_my_pos = samples["neg_neg_to_my_pos"] # (B,K) padded with -1

        # --- NEG term (anchor neg-view vs pos neg-view; negatives are other proteins pos-view)
        neg_mask = neg_pos_to_my_neg >= 0
        neg_safe = neg_pos_to_my_neg.clamp(min=0)
        a_neg = z_neg[anchors]
        p_neg = z_neg[pos_same_neg]
        n_neg = z_pos[neg_safe]  # (B,K,D)
        loss_neg = self._info_nce_masked(a_neg, p_neg, n_neg, neg_mask)

        # --- POS term (anchor pos-view vs pos pos-view; negatives are other proteins neg-view)
        pos_mask = neg_neg_to_my_pos >= 0
        pos_safe = neg_neg_to_my_pos.clamp(min=0)
        a_pos = z_pos[anchors]
        p_pos = z_pos[pos_same_pos]
        n_pos = z_neg[pos_safe]  # (B,K,D)
        loss_pos = self._info_nce_masked(a_pos, p_pos, n_pos, pos_mask)

        return self.w_neg * loss_neg + self.w_pos * loss_pos
