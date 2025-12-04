import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ComposedContrastiveLoss_CE(nn.Module):
    """ Dual-view contrastive loss. For each view v in {pos, neg}:
        - pulls z_anchor_v towards z_v_pos
        - pushes z_anchor_v away from z_v_neg (K negatives per anchor)
      Total loss = L_pos + lambda_neg * L_neg
    """

    def __init__(self, sampler, temperature: float = 0.5, lambda_neg: float = 1.0,):
        super().__init__()
        self.sampler = sampler
        self.temperature = temperature
        self.lambda_neg = lambda_neg


    def compute(self, z_anchor: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        """InfoNCE for one view (pos or neg)."""

        if z_anchor.numel() == 0: return z_anchor.new_tensor(0.0, requires_grad=True)
        B, D = z_anchor.shape
        K = z_neg.size(1)
        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=2)

        sim_pos = torch.sum(z_anchor * z_pos, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.bmm(z_anchor.unsqueeze(1), z_neg.transpose(1, 2)).squeeze(1) / self.temperature

        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss


    def forward(self, z_pos_full: torch.Tensor, z_neg_full: torch.Tensor,
        neg_statement_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample inside the loss using the NegativeStatementSampler.
        The sampler returns 6 tensors (same indices applied to both views):
        """

        if self.sampler is None:
            raise RuntimeError("DualContrastiveLoss_CE called without a sampler.")

        (z_anchor_pos, z_pos_pos, z_pos_neg, z_anchor_neg, 
            z_neg_pos, z_neg_neg) = self.sampler.get_dual_contrastive_samples(
            z_pos_full,z_neg_full, neg_statement_index=neg_statement_index)

        loss_pos = self.compute(z_anchor_pos, z_pos_pos, z_pos_neg)
        loss_neg = self.compute(z_anchor_neg, z_neg_pos, z_neg_neg)
        total_loss = loss_pos + self.lambda_neg * loss_neg
        return total_loss