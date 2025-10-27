
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualContrastiveLoss_Margin(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self,z_anchor: torch.Tensor,z_pos: torch.Tensor,
        z_neg: torch.Tensor) -> torch.Tensor:

        diff_pos = z_anchor - z_pos
        d_pos    = diff_pos.norm(p=2, dim=1)
        loss_pos = d_pos.pow(2)

        diff_neg = z_anchor.unsqueeze(1) - z_neg
        d_neg    = diff_neg.norm(p=2, dim=2)
        loss_neg = F.relu(self.margin - d_neg).pow(2).mean(dim=1)

        loss = 0.5 * (loss_pos + loss_neg).mean()
        return loss
