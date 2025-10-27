import torch
import torch.nn as nn
import torch.nn.functional as F

class DualContrastiveLoss_CE(torch.nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, z_pos, z_pos_pos, z_pos_neg) -> torch.Tensor:
        B, D = z_pos.shape
        z_pos = F.normalize(z_pos, dim=1)
        z_pos_pos = F.normalize(z_pos_pos, dim=1)
        z_pos_neg = F.normalize(z_pos_neg, dim=2)

        sim_pos = torch.sum(z_pos * z_pos_pos, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.bmm(z_pos.unsqueeze(1), z_pos_neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits_pos = torch.cat([sim_pos, sim_neg], dim=1)
        labels_pos = torch.zeros(B, dtype=torch.long, device=z_pos.device)
        loss_pos = F.cross_entropy(logits_pos, labels_pos)

        total_loss = loss_pos
        return total_loss
