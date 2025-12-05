import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try: from samplers import PartialStatementSampler
except ImportError:  PartialStatementSampler = None


class ComposedContrastiveLoss_Multi(nn.Module):
    """Dual-view contrastive loss
    Positive view:
        - anchor:   z_pos[protein_anchor]
        - positive: z_pos[GO_pos]
        - negatives: z_pos[GO_neg_1..K]

    Negative view (multi-positive if k>1):
        - anchor:   z_neg[protein_anchor]
        - positives: z_pos[GO_neg_1..K]
        - negatives: z_pos[GO_pos] (and possibly more if you extend it)

    When the sampler is a PartialStatementSampler (i.e., neg_statement edges were
    popped out of the graph), we **only** compute the positive-view branch in
    the z_pos space and ignore the negative-view branch.
    """

    def __init__(self, sampler, temperature: float = 0.5, lambda_neg: float = 1.0):
        super().__init__()
        self.sampler = sampler
        self.temperature = temperature
        self.lambda_neg = lambda_neg

    # ----------------------- helpers -----------------------

    def compute_single_pos(
        self,
        z_anchor: torch.Tensor,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Standard InfoNCE: 1 positive, K negatives per anchor.

        z_anchor: [B, D]
        z_pos:    [B, D]
        z_neg:    [B, K, D]
        """
        if z_anchor.numel() == 0:
            return z_anchor.new_tensor(0.0, requires_grad=True)

        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=2)

        # [B, 1]
        sim_pos = torch.sum(z_anchor * z_pos, dim=1, keepdim=True) / self.temperature
        # [B, K]
        sim_neg = torch.bmm(
            z_anchor.unsqueeze(1),  # [B,1,D]
            z_neg.transpose(1, 2),  # [B,D,K]
        ).squeeze(1) / self.temperature

        logits = torch.cat([sim_pos, sim_neg], dim=1)  # [B, 1+K]
        labels = torch.zeros(z_anchor.size(0), dtype=torch.long, device=z_anchor.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def compute_multi_pos(
        self,
        z_anchor: torch.Tensor,
        z_pos_all: torch.Tensor,
        z_neg_all: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-positive InfoNCE:
        z_anchor:  [B, D]
        z_pos_all: [B, Kp, D]  (multiple positives)
        z_neg_all: [B, Kn, D]  (multiple negatives)
        """
        if z_anchor.numel() == 0:
            return z_anchor.new_tensor(0.0, requires_grad=True)

        B, D = z_anchor.shape
        Kp = z_pos_all.size(1)
        Kn = z_neg_all.size(1)

        z_anchor = F.normalize(z_anchor, dim=1)
        z_pos_all = F.normalize(z_pos_all, dim=2)
        z_neg_all = F.normalize(z_neg_all, dim=2)

        # [B, Kp] and [B, Kn]
        sim_pos = torch.einsum("bd,bkd->bk", z_anchor, z_pos_all) / self.temperature
        sim_neg = torch.einsum("bd,bkd->bk", z_anchor, z_neg_all) / self.temperature

        exp_pos = torch.exp(sim_pos)
        exp_neg = torch.exp(sim_neg)

        num = exp_pos.sum(dim=1)                # Σ positives
        denom = num + exp_neg.sum(dim=1)        # Σ positives + Σ negatives

        loss = -torch.log(num / (denom + 1e-12))
        return loss.mean()

    # ------------------------ forward ------------------------

    def forward(
        self,
        z_pos_full: torch.Tensor,
        z_neg_full: torch.Tensor,
        neg_statement_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        z_pos_full: [N_nodes, D]  positive-view embeddings
        z_neg_full: [N_nodes, D]  negative-view embeddings (only meaningful if
                                  neg_statement edges exist, i.e. NOT in
                                  PartialStatementSampler mode).
        neg_statement_index:
            - For NegativeStatementSampler: optional indices for extra constraints.
            - For PartialStatementSampler:   edge_index [2, B*k] used as negatives.
        """

        if self.sampler is None:
            raise RuntimeError("ComposedContrastiveLoss_Multi called without a sampler.")

        # ----------------------------------------------------
        #  CASE 1: PartialStatementSampler -> only POSITIVE branch
        # ----------------------------------------------------
        is_partial = (
            PartialStatementSampler is not None
            and isinstance(self.sampler, PartialStatementSampler)
        )

        if is_partial:
            # Here, neg_statement edges were removed from the graph, so z_neg_full
            # is not a meaningful "negative view". We only use z_pos_full and
            # the external neg list inside the sampler.

            if neg_statement_index is None:
                # No negatives to contrast against -> no contrastive signal.
                return z_pos_full.new_tensor(0.0, requires_grad=True)

            # Original PartialStatementSampler API:
            #   get_contrastive_samples(z: [B,D], neg_ei: [2,B*k])
            # returns: z_anchor, z_pos_pos, z_pos_neg
            z_anchor_pos, z_pos_pos, z_pos_neg = self.sampler.get_contrastive_samples(
                z_pos_full, neg_statement_index
            )

            if z_anchor_pos.numel() == 0:
                return z_pos_full.new_tensor(0.0, requires_grad=True)

            loss_pos = self.compute_single_pos(z_anchor_pos, z_pos_pos, z_pos_neg)
            # No negative-view term here:
            return loss_pos

        # ----------------------------------------------------
        #  CASE 2: "Full" dual-view sampler (NegativeStatementSampler etc.)
        # ----------------------------------------------------
        (z_anchor_pos, z_pos_pos, z_pos_neg,
         z_anchor_neg, z_neg_pos, z_neg_neg) = self.sampler.get_dual_contrastive_samples(
            z_pos_full, z_neg_full, neg_statement_index=neg_statement_index
        )

        # Positive view: standard InfoNCE
        loss_pos = self.compute_single_pos(z_anchor_pos, z_pos_pos, z_pos_neg)

        # Negative view: multi-positive InfoNCE, or 0 if no anchors
        if z_anchor_neg.numel() == 0:
            loss_neg = z_anchor_neg.new_tensor(0.0, requires_grad=True)
        else:
            # For the negative branch we treat GO_neg as positives (z_pos_neg)
            # and GO_pos as negatives (z_pos_pos).
            z_neg_pos_all = z_pos_neg              # [B, K, D] as positives
            z_neg_neg_all = z_pos_pos.unsqueeze(1) # [B, 1, D] as negatives
            loss_neg = self.compute_multi_pos(z_anchor_neg, z_neg_pos_all, z_neg_neg_all)

        total_loss = loss_pos + self.lambda_neg * loss_neg
        return total_loss



# class ComposedContrastiveLoss_Multi(nn.Module):
#     """Dual-view contrastive loss.
#     Positive view:
#         - anchor: z_pos[protein_anchor]
#         - positive: z_pos[GO_pos]
#         - negatives: z_pos[GO_neg_1..K]
#     Negative view (multi-positive if k>1):
#         - anchor: z_neg[protein_anchor]
#         - positives: z_pos[GO_neg_1..K]
#         - negatives: z_pos[GO_pos] (and possibly more if you extend it)
#     """

#     def __init__(self, sampler, temperature: float = 0.5, lambda_neg: float = 1.0):
#         super().__init__()
#         self.sampler = sampler
#         self.temperature = temperature
#         self.lambda_neg = lambda_neg


#     def compute_single_pos(self, z_anchor: torch.Tensor, z_pos: torch.Tensor,
#         z_neg: torch.Tensor) -> torch.Tensor:
#         """ Standard InfoNCE: 1 positive, K negatives """

#         if z_anchor.numel() == 0: return z_anchor.new_tensor(0.0, requires_grad=True)

#         z_anchor = F.normalize(z_anchor, dim=1)
#         z_pos = F.normalize(z_pos, dim=1)
#         z_neg = F.normalize(z_neg, dim=2)

#         sim_pos = torch.sum(z_anchor * z_pos, dim=1, keepdim=True) / self.temperature
#         sim_neg = torch.bmm(z_anchor.unsqueeze(1), z_neg.transpose(1, 2)).squeeze(1) / self.temperature

#         logits = torch.cat([sim_pos, sim_neg], dim=1)
#         labels = torch.zeros(z_anchor.size(0), dtype=torch.long, device=z_anchor.device)
#         loss = F.cross_entropy(logits, labels)
#         return loss


#     def compute_multi_pos(self, z_anchor: torch.Tensor, z_pos_all: torch.Tensor,
#         z_neg_all: torch.Tensor) -> torch.Tensor:
#         """ Multi-positive InfoNCE: multiple positives, multiple (k) negatives per anchor """

#         if z_anchor.numel() == 0: return z_anchor.new_tensor(0.0, requires_grad=True)

#         B, D = z_anchor.shape
#         Kp = z_pos_all.size(1)
#         Kn = z_neg_all.size(1)
#         z_anchor = F.normalize(z_anchor, dim=1)
#         z_pos_all = F.normalize(z_pos_all, dim=2)
#         z_neg_all = F.normalize(z_neg_all, dim=2)

#         sim_pos = torch.einsum("bd,bkd->bk", z_anchor, z_pos_all) / self.temperature
#         sim_neg = torch.einsum("bd,bkd->bk", z_anchor, z_neg_all) / self.temperature

#         exp_pos = torch.exp(sim_pos)
#         exp_neg = torch.exp(sim_neg)
#         num = exp_pos.sum(dim=1)                
#         denom = num + exp_neg.sum(dim=1)        
#         loss = -torch.log(num / (denom + 1e-12))
#         return loss.mean()


#     def forward(self, z_pos_full: torch.Tensor, z_neg_full: torch.Tensor,
#         neg_statement_index: Optional[torch.Tensor] = None) -> torch.Tensor:

#         (z_anchor_pos, z_pos_pos, z_pos_neg,
#          z_anchor_neg, z_neg_pos, z_neg_neg) = self.sampler.get_dual_contrastive_samples(
#             z_pos_full, z_neg_full, neg_statement_index=neg_statement_index)

#         loss_pos = self.compute_single_pos(z_anchor_pos, z_pos_pos, z_pos_neg)
#         if z_anchor_neg.numel() == 0: loss_neg = z_anchor_neg.new_tensor(0.0, requires_grad=True)
#         else:
#             z_neg_pos_all = z_pos_neg
#             z_neg_neg_all = z_pos_pos.unsqueeze(1)
#             loss_neg = self.compute_multi_pos(z_anchor_neg, z_neg_pos_all, z_neg_neg_all)

#         total_loss = loss_pos + self.lambda_neg * loss_neg
#         return total_loss
