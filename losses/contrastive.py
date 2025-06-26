import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin # margin-based losses use hard cut-off value to separate positive and negative pairs

    def forward(self, z1, z2, labels):

        distances = F.pairwise_distance(z1, z2, p=2)

        positive_loss = labels * distances.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        loss = 0.5 * (positive_loss + negative_loss).mean()
        return loss
