import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveInstanceLoss(nn.Module):
    """
    Multi-positive InfoNCE-style contrastive loss
    Inputs:
      z_anchor       : (B, D)
      z_shared_neg   : (B, k, D)   positives
      z_pos_to_u_neg : (B, k, D)   negatives
      z_neg_to_u_pos : (B, k, D)   negatives
      z_shared_pos   : (B, k, D)   positives
    """
    def __init__(self, temperature: float = 0.5, learnable_temperature: bool = True,
        min_temperature: float = 1e-4):
        super().__init__()
        self.learnable_temperature = bool(learnable_temperature)
        self.min_temperature = float(min_temperature)
        self.temperature = float(temperature)
        if self.learnable_temperature:
            init_temp = max(float(temperature), self.min_temperature)
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(float(temperature)))

    def _get_temperature(self) -> torch.Tensor:
        if self.learnable_temperature:
            return torch.exp(self.log_temperature).clamp(min=self.min_temperature)
        return self.fixed_temperature

    def get_temperature_value(self) -> float:
        return float(self._get_temperature().detach().cpu().item())

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

        temperature = self._get_temperature()
        sim_pos = torch.bmm(z_anchor.unsqueeze(1), pos.transpose(1, 2)).squeeze(1) / temperature
        sim_neg = torch.bmm(z_anchor.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / temperature
        logits = torch.cat([sim_pos, sim_neg], dim=1)

        logp = F.log_softmax(logits, dim=1)
        num_pos = sim_pos.size(1)
        return -(logp[:, :num_pos].mean(dim=1)).mean()
