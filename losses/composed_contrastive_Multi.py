import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ComposedContrastiveLoss_Multi(nn.Module):
    """Dual-view contrastive loss.
    Positive view:
        - anchor: z_pos[protein_anchor]
        - positive: z_pos[GO_pos]
        - negatives: z_pos[GO_neg_1..K]
    Negative view (multi-positive if k>1):
        - anchor: z_neg[protein_anchor]
        - positives: z_pos[GO_neg_1..K]
        - negatives: z_pos[GO_pos] (and possibly more if you extend it)
    """

    def __init__(self, sampler, temperature: float = 0.5, lambda_neg: float = 1.0):
        super().__init__()
        self.sampler = sampler
        self.temperature = temperature
        self.lambda_neg = lambda_neg


    def compute_single_pos(self, z_anchor: torch.Tensor, z_pos: torch.Tensor,
        z_neg: torch.Tensor) -> torch.Tensor:
        """ Standard InfoNCE: 1 positive, K negatives """

        if z_anchor.numel() == 0: return z_anchor.new_tensor(0.0, requires_grad=True)

        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=2)

        sim_pos = torch.sum(z_anchor * z_pos, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.bmm(z_anchor.unsqueeze(1), z_neg.transpose(1, 2)).squeeze(1) / self.temperature

        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(z_anchor.size(0), dtype=torch.long, device=z_anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss


    def compute_multi_pos(self, z_anchor: torch.Tensor, z_pos_all: torch.Tensor,
        z_neg_all: torch.Tensor) -> torch.Tensor:
        """ Multi-positive InfoNCE: multiple positives, multiple (k) negatives per anchor """

        if z_anchor.numel() == 0: return z_anchor.new_tensor(0.0, requires_grad=True)

        B, D = z_anchor.shape
        Kp = z_pos_all.size(1)
        Kn = z_neg_all.size(1)
        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos_all = F.normalize(z_pos_all, dim=2)
        z_neg_all = F.normalize(z_neg_all, dim=2)

        sim_pos = torch.einsum("bd,bkd->bk", z_anchor, z_pos_all) / self.temperature
        sim_neg = torch.einsum("bd,bkd->bk", z_anchor, z_neg_all) / self.temperature

        exp_pos = torch.exp(sim_pos)
        exp_neg = torch.exp(sim_neg)
        num = exp_pos.sum(dim=1)                
        denom = num + exp_neg.sum(dim=1)        
        loss = -torch.log(num / (denom + 1e-12))
        return loss.mean()


    def forward(self, z_pos_full: torch.Tensor, z_neg_full: torch.Tensor,
        neg_statement_index: Optional[torch.Tensor] = None) -> torch.Tensor:

        (z_anchor_pos, z_pos_pos, z_pos_neg,
         z_anchor_neg, z_neg_pos, z_neg_neg) = self.sampler.get_dual_contrastive_samples(
            z_pos_full, z_neg_full, neg_statement_index=neg_statement_index)

        loss_pos = self.compute_single_pos(z_anchor_pos, z_pos_pos, z_pos_neg)
        if z_anchor_neg.numel() == 0: loss_neg = z_anchor_neg.new_tensor(0.0, requires_grad=True)
        else:
            z_neg_pos_all = z_pos_neg
            z_neg_neg_all = z_pos_pos.unsqueeze(1)
            loss_neg = self.compute_multi_pos(z_anchor_neg, z_neg_pos_all, z_neg_neg_all)

        total_loss = loss_pos + self.lambda_neg * loss_neg
        return total_loss
