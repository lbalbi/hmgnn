import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from samplers import ProteinStatementSampler


def _cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


class ProteinContrastiveLoss(nn.Module):
    """
    Dual-view composed contrastive loss using protein-protein examples.

    - Pull z_neg(u) towards z_neg(v) for v from share_neg[u]
    - Pull z_pos(u) towards z_pos(v) for v from share_pos[u]
    - Push z_neg(u) away from z_pos(v) for v from pos_to_my_neg[u]
    - Push z_pos(u) away from z_neg(v) for v from neg_to_my_pos[u]
    """

    def __init__(
        self,
        sampler: ProteinStatementSampler,
        temperature: float = 0.5,
        w_neg: float = 1.0,
        w_pos: float = 1.0,
    ):
        super().__init__()
        self.sampler = sampler
        self.temperature = float(temperature)
        self.w_neg = float(w_neg)
        self.w_pos = float(w_pos)

    def _info_nce(self, anchor: Tensor, pos: Tensor, neg: Tensor) -> Tensor:
        """
        anchor: (B, D)
        pos:    (B, D)
        neg:    (B, K, D)
        """
        B = anchor.size(0)
        if B == 0:
            return anchor.new_tensor(0.0)

        pos_logits = _cosine_sim(anchor, pos).unsqueeze(1) / self.temperature  # (B,1)

        a = F.normalize(anchor, dim=-1).unsqueeze(1)  # (B,1,D)
        n = F.normalize(neg, dim=-1)                  # (B,K,D)
        neg_logits = (a * n).sum(dim=-1) / self.temperature  # (B,K)

        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (B,1+K)
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(self, z_pos: Tensor, z_neg: Tensor) -> Tensor:
        samples = self.sampler.sample_batch()
        anchors = samples["anchors"]
        if anchors.numel() == 0: return z_pos.new_tensor(0.0)

        pos_same_neg = samples["pos_same_neg"]              # (B,)
        neg_pos_to_my_neg = samples["neg_pos_to_my_neg"]    # (B,K)
        mask_neg = (pos_same_neg >= 0) & (neg_pos_to_my_neg >= 0).any(dim=1)
        loss_neg = z_pos.new_tensor(0.0)
        if mask_neg.any():
            a_idx = anchors[mask_neg]
            p_idx = pos_same_neg[mask_neg]
            neg_idx = neg_pos_to_my_neg[mask_neg]
            neg_idx = torch.where(neg_idx < 0, a_idx.unsqueeze(1).expand_as(neg_idx), neg_idx)
            a = z_neg[a_idx]
            p = z_neg[p_idx]
            n = z_pos[neg_idx]  # (B,K,D)
            loss_neg = self._info_nce(a, p, n)

        pos_same_pos = samples["pos_same_pos"]              # (B,)
        neg_neg_to_my_pos = samples["neg_neg_to_my_pos"]    # (B,K)
        mask_pos = (pos_same_pos >= 0) & (neg_neg_to_my_pos >= 0).any(dim=1)
        loss_pos = z_pos.new_tensor(0.0)
        if mask_pos.any():
            a_idx = anchors[mask_pos]
            p_idx = pos_same_pos[mask_pos]
            neg_idx = neg_neg_to_my_pos[mask_pos]
            neg_idx = torch.where(neg_idx < 0, a_idx.unsqueeze(1).expand_as(neg_idx), neg_idx)
            a = z_pos[a_idx]
            p = z_pos[p_idx]
            n = z_neg[neg_idx]  # (B,K,D)
            loss_pos = self._info_nce(a, p, n)
        return self.w_neg * loss_neg + self.w_pos * loss_pos
