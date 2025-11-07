import torch
import torch.nn.functional as F

class Contradiction_ContrastiveLoss(torch.nn.Module):
    """ InfoNCE-style CE that handles (pos, neg, ultra-neg/neg contradiction) + hinge separates pos–ultra w/ largest cosine sim.
        Args: temperature: similarity temperature (lower = sharper);
              gamma_ultra: subtractive factor on ultra-neg logits inside the CE;
              lambda_pos_ultra: weight for positive-ultra separation term;
              margin_pos_ultra: enforced margin in cosine sim. so pos is farther than ultra-neg. """
    
    def __init__(self, temperature: float = 0.5, gamma_ultra: float = 0.2,
        lambda_pos_ultra: float = 0.3, margin_pos_ultra: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.gamma_ultra = gamma_ultra
        self.lambda_pos_ultra = lambda_pos_ultra
        self.margin_pos_ultra = margin_pos_ultra


    def forward(self, z_anchor, z_pos, z_neg, z_ultra) -> torch.Tensor:
        """ Shapes: z_anchor, z_pos, z_neg, z_ultra """

        B = z_anchor.size(0)
        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        if z_neg.dim() == 2: z_neg = z_neg.unsqueeze(1)
        if z_ultra.dim() == 2: z_ultra = z_ultra.unsqueeze(1)
        z_neg = F.normalize(z_neg, dim=2)
        z_ultra = F.normalize(z_ultra, dim=2)

        sim_pos = torch.sum(z_anchor * z_pos, dim=1, keepdim=True) / self.temperature
        sim_neg = torch.bmm(z_anchor.unsqueeze(1), z_neg.transpose(1, 2)).squeeze(1) / self.temperature
        sim_ultra = torch.bmm(z_anchor.unsqueeze(1), z_ultra.transpose(1, 2)).squeeze(1) / self.temperature

        # Downweighs contradictions (ultra-neg) with gamma_ultra to increase CE penalty
        # subtracting sim_ultra by gamma>1 decreases logits to push them farther away compared to anchor - pos distance
        logits = torch.cat([sim_pos, sim_neg, sim_ultra - self.gamma_ultra], dim=1) # true class (positive) is at index 0
        labels = torch.zeros(B, dtype=torch.long, device=z_anchor.device)
        loss_ce = F.cross_entropy(logits, labels) # softmax classification over logits

        # hinge penalty pushes hardest (highest cosine sim) pos–ultra below margin m_pos_ultra
        sim_pos_ultra = torch.bmm(z_pos.unsqueeze(1), z_ultra.transpose(1,2)).squeeze(1) / self.temperature
        max_pos_ultra, _ = sim_pos_ultra.max(dim=1, keepdim=True)
        loss_pos_ultra = F.relu(max_pos_ultra + self.margin_pos_ultra).mean()

        total = loss_ce  + self.lambda_pos_ultra * loss_pos_ultra
        return total
