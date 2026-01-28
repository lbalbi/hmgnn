import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

def _cosine_sim(a: Tensor, b: Tensor) -> Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

class DualContrastiveInstanceLoss(nn.Module):
    """  Dual-view contrastive loss for SRA-HGCN style models.
    Expected sampler outputs (same 5-tuple as ContrastiveInstanceLoss), but using concatenated embeddings:
      z_anchor : (B, 2D)  where [:D]=pos-view, [D:]=neg-view
      z_shared_neg : (B, k, 2D) positives for the *neg-view* term (we use their neg half)
      z_pos_to_u_neg : (B, k, 2D) negatives for the *neg-view* term (we use their pos half)
      z_neg_to_u_pos : (B, k, 2D) negatives for the *pos-view* term (we use their neg half)
      z_shared_pos : (B, k, 2D) positives for the *pos-view* term (we use their pos half)
    Optional masks:
      If you later update the sampler to PAD with -1 and return validity masks, you can pass
      neg_mask / pos_mask as extra positional args (6th and 7th), shape (B, k) bool.
    """
    def __init__(self, temperature: float = 0.5, w_neg: float = 1.0, w_pos: float = 1.0,
        learnable_temperature: bool = False, min_temperature: float = 1e-4):
        super().__init__()
        self.learnable_temperature = bool(learnable_temperature)
        self.min_temperature = float(min_temperature)
        if self.learnable_temperature:
            init_temp = max(float(temperature), self.min_temperature)
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(float(temperature)))
        self.temperature = float(temperature)
        self.w_neg = float(w_neg)
        self.w_pos = float(w_pos)

    def _get_temperature(self) -> Tensor:
        if self.learnable_temperature:
            return torch.exp(self.log_temperature).clamp(min=self.min_temperature)
        return self.fixed_temperature

    def get_temperature_value(self) -> float:
        return float(self._get_temperature().detach().cpu().item())

    def _info_nce_masked(self, anchor: Tensor, pos: Tensor, neg: Tensor, neg_mask: Optional[Tensor]) -> Tensor:
        """ Masked InfoNCE (single-positive per anchor).
        anchor: (B, D); pos: (B, D); neg: (B, K, D)
        neg_mask: (B, K) bool, True where neg is valid. If None, all valid.
        Invalid neg logits are set to -1e9 so they do not affect the denominator.
        Rows with 0 valid negatives are dropped.  """

        B = anchor.size(0)
        if B == 0: return anchor.new_tensor(0.0)
        if neg_mask is None:
            neg_mask = torch.ones((B, neg.size(1)), dtype=torch.bool, device=anchor.device)
        row_ok = neg_mask.any(dim=1)
        if not row_ok.any(): return anchor.new_tensor(0.0)

        anchor = anchor[row_ok]
        pos = pos[row_ok]
        neg = neg[row_ok]
        neg_mask = neg_mask[row_ok]
        temperature = self._get_temperature()
        pos_logits = _cosine_sim(anchor, pos).unsqueeze(1) / temperature

        a = F.normalize(anchor, dim=-1).unsqueeze(1)
        n = F.normalize(neg, dim=-1)
        neg_logits = (a * n).sum(dim=-1) / temperature
        neg_logits = neg_logits.masked_fill(~neg_mask, -1e9)

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _split_pos_neg(z: Tensor) -> Tuple[Tensor, Tensor]:
        if z.dim() == 1: z = z.unsqueeze(0)
        D2 = z.size(-1)
        D = D2 // 2
        return z[..., :D], z[..., D:]


    def forward(self, z_anchor: Tensor, z_shared_neg: Tensor, z_pos_to_u_neg: Tensor,
        z_neg_to_u_pos: Tensor, z_shared_pos: Tensor, neg_mask: Optional[Tensor] = None, 
        pos_mask: Optional[Tensor] = None) -> Tensor:
        if z_anchor.numel() == 0: return z_anchor.new_zeros(())

        a_pos, a_neg = self._split_pos_neg(z_anchor)
        shneg_pos, shneg_neg = self._split_pos_neg(z_shared_neg)
        pos2neg_pos, pos2neg_neg = self._split_pos_neg(z_pos_to_u_neg)
        neg2pos_pos, neg2pos_neg = self._split_pos_neg(z_neg_to_u_pos)
        shpos_pos, shpos_neg = self._split_pos_neg(z_shared_pos)      

        p_neg = shneg_neg[:, 0, :]
        n_neg = pos2neg_pos
        loss_neg = self._info_nce_masked(a_neg, p_neg, n_neg, neg_mask)
        p_pos = shpos_pos[:, 0, :]
        n_pos = neg2pos_neg
        loss_pos = self._info_nce_masked(a_pos, p_pos, n_pos, pos_mask)
        return self.w_neg * loss_neg + self.w_pos * loss_pos


# class DualContrastiveInstanceLoss(nn.Module):
#     """Dual-view composed contrastive loss using protein↔protein examples.
#     Requirements implemented:
#       - If the sampler finds fewer than k negatives for an anchor, use only those negatives (no repetition).
#       - If an anchor doesn't have at least 1 example for *each* required example type, that anchor is excluded
#         by the sampler, so the loss never sees incomplete tuples.
#     Loss terms:
#       - Pull z_neg(u) towards z_neg(v) where v shares a NEG statement class with u.
#         Negatives are z_pos(w) where w has a POS statement to a (expanded) NEG class of u.
#       - Pull z_pos(u) towards z_pos(v) where v shares a POS statement class with u.
#         Negatives are z_neg(w) where w has a NEG statement to a POS class of u.
#     """

#     def __init__(self, sampler: Optional[NegativeInstanceSampler or PartialInstanceSampler or RandomInstanceSampler],
#         temperature: float = 0.5, w_neg: float = 1.0, w_pos: float = 1.0):
#         super().__init__()
#         self.sampler = sampler
#         self.temperature = float(temperature)
#         self.w_neg = float(w_neg)
#         self.w_pos = float(w_pos)

#     def _info_nce_masked(self, anchor: Tensor, pos: Tensor, neg: Tensor, neg_mask: Tensor) -> Tensor:
#         """Masked InfoNCE.
#         anchor:   (B, D)
#         pos:      (B, D)
#         neg:      (B, K, D)  (may include padded rows)
#         neg_mask: (B, K) bool, True where neg is valid.
#         We set invalid neg logits to a large negative value so they don't affect the denominator.
#         """
#         B = anchor.size(0)
#         if B == 0: return anchor.new_tensor(0.0)

#         # If any rows have 0 valid negatives, drop them.
#         row_ok = neg_mask.any(dim=1)
#         if not row_ok.any(): return anchor.new_tensor(0.0)

#         anchor = anchor[row_ok]
#         pos = pos[row_ok]
#         neg = neg[row_ok]
#         neg_mask = neg_mask[row_ok]
#         pos_logits = _cosine_sim(anchor, pos).unsqueeze(1) / self.temperature  # (B,1)
#         a = F.normalize(anchor, dim=-1).unsqueeze(1)  # (B,1,D)
#         n = F.normalize(neg, dim=-1)                  # (B,K,D)
#         neg_logits = (a * n).sum(dim=-1) / self.temperature  # (B,K)
#         neg_logits = neg_logits.masked_fill(~neg_mask, -1e9)
#         logits = torch.cat([pos_logits, neg_logits], dim=1)  # (B,1+K)
#         labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
#         return F.cross_entropy(logits, labels)


#     def forward(self, z_pos: Tensor, z_neg: Tensor) -> Tensor:
#         samples = self.sampler.sample_batch()
#         anchors = samples["anchors"]
#         if anchors.numel() == 0: return z_pos.new_tensor(0.0)
#         pos_same_neg = samples["pos_same_neg"]           # (B,)
#         neg_pos_to_my_neg = samples["neg_pos_to_my_neg"] # (B,K) padded with -1
#         pos_same_pos = samples["pos_same_pos"]           # (B,)
#         neg_neg_to_my_pos = samples["neg_neg_to_my_pos"] # (B,K) padded with -1
#         neg_mask = neg_pos_to_my_neg >= 0
#         neg_safe = neg_pos_to_my_neg.clamp(min=0)
#         a_neg = z_neg[anchors]
#         p_neg = z_neg[pos_same_neg]
#         n_neg = z_pos[neg_safe]  # (B,K,D)
#         loss_neg = self._info_nce_masked(a_neg, p_neg, n_neg, neg_mask)
        
#         pos_mask = neg_neg_to_my_pos >= 0
#         pos_safe = neg_neg_to_my_pos.clamp(min=0)
#         a_pos = z_pos[anchors]
#         p_pos = z_pos[pos_same_pos]
#         n_pos = z_neg[pos_safe]  # (B,K,D)
#         loss_pos = self._info_nce_masked(a_pos, p_pos, n_pos, pos_mask)

#         return self.w_neg * loss_neg + self.w_pos * loss_pos
