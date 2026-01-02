import torch
import torch.nn.functional as F


class ProteinContrastiveLoss(torch.nn.Module):
    """Cross-entropy / InfoNCE-style contrastive loss.
      - (z_anchor, z_shared_neg, z_pos_to_anchor_neg, z_neg_to_anchor_pos, z_shared_pos)
        where each group tensor is (B, k, D)
      - pull toward shared_neg and shared_pos
      - push away from pos_to_anchor_neg and neg_to_anchor_pos
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, *args) -> torch.Tensor:
        if len(args) == 3:
            z_anchor, z_pos, z_negs = args
            if z_anchor.numel() == 0: return z_anchor.new_zeros(())
            B, _ = z_anchor.shape
            z_anchor = F.normalize(z_anchor, dim=1)
            z_pos = F.normalize(z_pos, dim=1)
            z_negs = F.normalize(z_negs, dim=2)
            sim_pos = torch.sum(z_anchor * z_pos, dim=1, keepdim=True) / self.temperature
            sim_neg = torch.bmm(z_anchor.unsqueeze(1), z_negs.transpose(1, 2)).squeeze(1) / self.temperature
            logits = torch.cat([sim_pos, sim_neg], dim=1)
            labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
            return F.cross_entropy(logits, labels)

        if len(args) != 5:
            raise TypeError("DualContrastiveLoss_CE expected 3 or 5 tensors: "
                "(z_anchor, z_pos, z_negs) or "
                "(z_anchor, z_shared_neg, z_pos_to_anchor_neg, z_neg_to_anchor_pos, z_shared_pos)")

        z_anchor, z_shared_neg, z_pos_to_anchor_neg, z_neg_to_anchor_pos, z_shared_pos = args
        if z_anchor.numel() == 0: return z_anchor.new_zeros(())

        B, _ = z_anchor.shape
        z_anchor = F.normalize(z_anchor, dim=1)
        z_shared_neg = F.normalize(z_shared_neg, dim=2)
        z_shared_pos = F.normalize(z_shared_pos, dim=2)
        z_pos_to_anchor_neg = F.normalize(z_pos_to_anchor_neg, dim=2)
        z_neg_to_anchor_pos = F.normalize(z_neg_to_anchor_pos, dim=2)

        pos = torch.cat([z_shared_neg, z_shared_pos], dim=1)
        neg = torch.cat([z_pos_to_anchor_neg, z_neg_to_anchor_pos], dim=1)
        sim_pos = torch.bmm(z_anchor.unsqueeze(1), pos.transpose(1, 2)).squeeze(1) / self.temperature
        sim_neg = torch.bmm(z_anchor.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / self.temperature
        logits = torch.cat([sim_pos, sim_neg], dim=1)

        # Multi-positive CE: maximize probability mass assigned to the positive columns.
        logp = F.log_softmax(logits, dim=1)
        num_pos = sim_pos.size(1)
        loss = -(logp[:, :num_pos].mean(dim=1)).mean()
        return loss
