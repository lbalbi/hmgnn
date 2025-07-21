import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, z_pos, z_pos_pos, z_pos_neg) -> torch.Tensor:
        B, D = z_pos.shape

        def branch_loss(anchor, positive, negative):
            d_pos = torch.norm(anchor.unsqueeze(1) - positive, dim=2) # 1) distance to true positive
            loss_pos = d_pos.pow(2).mean(dim=1) # 2) distances to negative
            d_neg = torch.norm(anchor.unsqueeze(1) - negative, dim=2) # hinge loss: max(0, margin - d_neg)^2
            loss_neg = F.relu(self.margin - d_neg).pow(2).mean(dim=1)
            return 0.5 * (loss_pos + loss_neg)
        
        loss = branch_loss(z_pos, z_pos_pos, z_pos_neg)
        return loss