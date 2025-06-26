
import torch
import torch.nn as nn
import torch.nn.functional as F

class DualContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        z_pos,  # positive representations of anchor nodes
        z_pos_pos,  # positive neighbors for pos rep.
        z_pos_neg,  # negative samples for pos rep.
        z_neg,  # negative representations of anchor nodes
        z_neg_pos,  # negative neighbors for neg rep.
        z_neg_neg   # positive samples for neg rep.
    ) -> torch.Tensor:

        B, D = z_pos.shape
        z_pos = F.normalize(z_pos, dim=1)
        z_pos_pos = F.normalize(z_pos_pos, dim=1)
        z_pos_neg = F.normalize(z_pos_neg, dim=2)
        z_neg = F.normalize(z_neg, dim=1)
        z_neg_pos = F.normalize(z_neg_pos, dim=1)
        z_neg_neg = F.normalize(z_neg_neg, dim=2)

        # POSITIVE REPRESENTATION
        # Similarity between anchor and positive
        sim_pos = torch.sum(z_pos * z_pos_pos, dim=1, keepdim=True) / self.temperature
        # Similarity between anchor and negative
        sim_neg = torch.bmm(z_pos.unsqueeze(1), z_pos_neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits_pos = torch.cat([sim_pos, sim_neg], dim=1)
        labels_pos = torch.zeros(B, dtype=torch.long, device=z_pos.device)
        loss_pos = F.cross_entropy(logits_pos, labels_pos)
        # NEGATIVE REPRESENTATION
        sim_neg_pos = torch.sum(z_neg * z_neg_pos, dim=1, keepdim=True) / self.temperature
        sim_neg_neg = torch.bmm(z_neg.unsqueeze(1), z_neg_neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits_neg = torch.cat([sim_neg_pos, sim_neg_neg], dim=1)
        labels_neg = torch.zeros(B, dtype=torch.long, device=z_neg.device)
        loss_neg = F.cross_entropy(logits_neg, labels_neg)

        total_loss = loss_pos + loss_neg
        return total_loss