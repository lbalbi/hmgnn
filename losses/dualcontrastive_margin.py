
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DualContrastiveLoss_Margin(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z_anchor: torch.Tensor,   # (N, D)
        z_pos: torch.Tensor,      # (N, D)
        z_neg: torch.Tensor       # (N, k, D)
    ) -> torch.Tensor:

        diff_pos = z_anchor - z_pos                # (N, D)
        d_pos    = diff_pos.norm(p=2, dim=1)       # (N,)
        loss_pos = d_pos.pow(2)                    # (N,)

        diff_neg = z_anchor.unsqueeze(1) - z_neg   # (N, k, D)
        d_neg    = diff_neg.norm(p=2, dim=2)       # (N, k)
        loss_neg = F.relu(self.margin - d_neg).pow(2).mean(dim=1)  # (N,)

        loss = 0.5 * (loss_pos + loss_neg).mean()         # (N,)
        return loss


# class DualContrastiveLoss_Margin(torch.nn.Module):
#     def __init__(self, margin: float = 1.0):
#         super().__init__()
#         self.margin = margin
        
#     def forward(self, z_pos, z_pos_pos, z_pos_neg) -> torch.Tensor:
#         B, D = z_pos.shape

#         def branch_loss(anchor, positives, negatives):
#             d_pos = torch.norm(anchor.unsqueeze(1) - positives, dim=2) # 1) distance to true positive
#             loss_pos = d_pos.pow(2).mean(dim=1) # 2) distances to negative
#             d_neg = torch.norm(anchor.unsqueeze(1) - negatives, dim=2) # hinge loss: max(0, margin - d_neg)^2
#             loss_neg = F.relu(self.margin - d_neg).pow(2).mean(dim=1)
#             return 0.5 * (loss_pos + loss_neg)

#         loss = branch_loss(z_pos, z_pos_pos, z_pos_neg)
#         return loss