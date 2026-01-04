import torch
import torch.nn.functional as F


class ContrastiveInstanceLoss(torch.nn.Module):
    """
    Multi-positive InfoNCE-style contrastive loss
    Inputs:
      z_anchor       : (B, D)
      z_shared_neg   : (B, k, D)   positives
      z_pos_to_u_neg : (B, k, D)   negatives
      z_neg_to_u_pos : (B, k, D)   negatives
      z_shared_pos   : (B, k, D)   positives
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z_anchor, z_shared_neg, z_pos_to_u_neg, z_neg_to_u_pos, z_shared_pos):
        if z_anchor.numel() == 0:
            return z_anchor.new_zeros(())

        z_anchor = F.normalize(z_anchor, dim=1)
        z_shared_neg = F.normalize(z_shared_neg, dim=2)
        z_shared_pos = F.normalize(z_shared_pos, dim=2)
        z_pos_to_u_neg = F.normalize(z_pos_to_u_neg, dim=2)
        z_neg_to_u_pos = F.normalize(z_neg_to_u_pos, dim=2)

        pos = torch.cat([z_shared_neg, z_shared_pos], dim=1)      # (B, 2k, D)
        neg = torch.cat([z_pos_to_u_neg, z_neg_to_u_pos], dim=1)  # (B, 2k, D)

        sim_pos = torch.bmm(z_anchor.unsqueeze(1), pos.transpose(1, 2)).squeeze(1) / self.temperature
        sim_neg = torch.bmm(z_anchor.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits = torch.cat([sim_pos, sim_neg], dim=1)

        logp = F.log_softmax(logits, dim=1)
        num_pos = sim_pos.size(1)
        return -(logp[:, :num_pos].mean(dim=1)).mean()
