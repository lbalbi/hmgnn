import torch
import torch.nn as nn
import torch.nn.functional as F

class ComposedContrastiveLoss_CE(torch.nn.Module):
    """
    Wraps a sampler (e.g. NegativeStatementSampler) and computes InfoNCE-style loss.
    Trainer only passes the full node embeddings (and optional indices), not sampled views.
    """
    def __init__(self, sampler, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.sampler = sampler

    def forward(self, z, neg_stmt_idx=None) -> torch.Tensor:
        """
        Args
        ----
        z : Tensor [N, D]
            Node embeddings.
        neg_stmt_idx : Optional[Tensor]
            Optional indices used by some samplers (random / partial), can be None.
        """
        if self.sampler is None: raise RuntimeError("DualContrastiveLoss_CE was called without a sampler.")

        if neg_stmt_idx is not None:
            z_pos, z_pos_pos, z_pos_neg = self.sampler.get_contrastive_samples(z, neg_stmt_idx)
        else: z_pos, z_pos_pos, z_pos_neg = self.sampler.get_contrastive_samples(z)
        if z_pos.numel() == 0: return z_pos.new_tensor(0.0, requires_grad=True)

        B, D = z_pos.shape
        z_pos = F.normalize(z_pos, dim=1)
        z_pos_pos = F.normalize(z_pos_pos, dim=1)
        z_pos_neg = F.normalize(z_pos_neg, dim=2)

        sim_pos = torch.sum(z_pos * z_pos_pos, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.bmm(z_pos.unsqueeze(1), z_pos_neg.transpose(1, 2)).squeeze(1) / self.temperature

        logits_pos = torch.cat([sim_pos, sim_neg], dim=1)
        labels_pos = torch.zeros(B, dtype=torch.long, device=z_pos.device)
        loss_pos = F.cross_entropy(logits_pos, labels_pos)

        return loss_pos
