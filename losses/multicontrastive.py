import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature # temperature because this is a softmax-based loss

    def forward(self, anchor, pos, neg, labels):

        B, D = anchor.shape
        anchor = F.normalize(anchor, dim=1)
        pos = F.normalize(pos, dim=1)
        neg = F.normalize(neg, dim=2)

        sim_pos = torch.sum(anchor * pos, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.bmm(anchor.unsqueeze(1), neg.transpose(1, 2)).squeeze(1) / self.temperature

        logits = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)
