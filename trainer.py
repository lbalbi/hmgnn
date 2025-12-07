import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_geometric.data import HeteroData

from utils import Metrics, EarlyStopping
from samplers import NegativeStatementSampler
from losses import DualContrastiveLoss_CE


class Train:
    """  Triple-level trainer for a relation-aware GNN with optional contrastive learning.
    Assumptions about `model`:
      - has attribute `n_type` (string) for the main node type (default "node").
      - has method:
            encode(data: HeteroData) -> Dict[str, Tensor]
        returning node embeddings per node type.
      - has method:
            score_triples(z: Tensor, edge_index: LongTensor[2, B], rel_ids: LongTensor[B])
        returning (logits: Tensor[B], probs: Tensor[B]).

    This trainer:
      - splits triples into train/val either internally (random) or via explicit
        train_idx / val_idx,
      - optimizes BCE on labeled triples,
      - optionally adds a contrastive loss term per batch using a
        NegativeStatementSampler and DualContrastiveLoss_CE.
      - BCE and contrastive losses are *merged* into a single loss per
        forward–backward pass (per batch), so gradients flow through the encoder.
    """

    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor,
        lr: float, epochs: int, device: torch.device, log, batch_size: int = 4096,
        val_ratio: float = 0.1, early_stopping_patience: int = 20,
        train_idx: Optional[torch.Tensor] = None, val_idx: Optional[torch.Tensor] = None,
        contrastive_sampler: Optional[NegativeStatementSampler] = None,
        contrastive_weight: float = 0.1):
        self.model = model.to(device)
        self.graph = graph
        self.heads = heads
        self.rels = rel_ids
        self.tails = tails
        self.labels = labels
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.device = device
        self.log = log
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)

        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = Metrics()
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")

        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = float(contrastive_weight) if contrastive_sampler is not None else 0.0
        self.contrastive_loss_fn = DualContrastiveLoss_CE() if contrastive_sampler is not None else None

        num_triples = self.heads.size(0)
        if train_idx is not None and val_idx is not None:
            self.train_idx = train_idx.to(torch.long)
            self.val_idx = val_idx.to(torch.long)
        else:
            perm = torch.randperm(num_triples)
            split = int(num_triples * (1.0 - self.val_ratio))
            self.train_idx = perm[:split]
            self.val_idx = perm[split:]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _iterate_batches(self, idx: torch.Tensor, train: bool = True
    ) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Iterate over batches of triples indexed by `idx`, compute BCE loss and
        (optionally) contrastive loss, and (optionally) collect outputs and
        labels for metrics.

        `z` is computed inside each batch via `self.model.encode(self.graph)`
        so that gradients flow through the encoder. For training batches, the
        total loss per batch is:
            total_loss = BCE_loss + contrastive_weight * contrastive_loss
        For validation batches, only BCE is computed (no contrastive).
        Returns:
            avg_bce_loss, avg_contrastive_loss, all_probs, all_labels
        """
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        all_probs = []
        all_labels = []

        n_type = getattr(self.model, "n_type", "node")
        if train and self.contrastive_sampler is not None and hasattr(self.contrastive_sampler, "prepare_batch"):
            self.contrastive_sampler.prepare_batch(self.graph)

        if train: self.model.train()
        else: self.model.eval()

        with torch.set_grad_enabled(train):
            for start in range(0, idx.size(0), self.batch_size):
                end = min(start + self.batch_size, idx.size(0))
                b_idx = idx[start:end]

                h = self.heads[b_idx].to(self.device)
                t = self.tails[b_idx].to(self.device)
                r = self.rels[b_idx].to(self.device)
                y = self.labels[b_idx].to(self.device)

                h_dict = self.model.encode(self.graph)
                z = h_dict[n_type]

                edge_index = torch.stack([h, t], dim=0)
                logits, probs = self.model.score_triples(z, edge_index, r)
                bce_loss = self.criterion(logits, y)

                contr_loss_val = 0.0
                total_loss = bce_loss
                if train and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(z)
                    contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                    total_loss = total_loss + self.contrastive_weight * contr_loss
                    contr_loss_val = float(contr_loss.detach().cpu().item())

                if train:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                batch_size_eff = y.size(0)
                total_bce += bce_loss.detach().cpu().item() * batch_size_eff
                total_contr += contr_loss_val * batch_size_eff
                total_examples += batch_size_eff
                all_probs.append(probs.detach().cpu())
                all_labels.append(y.detach().cpu())

        avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
        avg_contr = total_contr / total_examples if total_examples > 0 else 0.0

        if all_probs:
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        else:
            all_probs = None
            all_labels = None
        return avg_bce, avg_contr, all_probs, all_labels

    def run(self):
        """
        Train with a train/val split (either internal or provided) and early stopping
        on validation BCE loss. If a NegativeStatementSampler is given, we add a
        contrastive loss term in each *training batch* using the same embeddings
        as for BCE.
        Returns:
            best_val_loss, best_epoch, best_metrics
        """
        self.graph = self.graph.to(self.device)
        self.heads = self.heads.to(self.device)
        self.rels = self.rels.to(self.device)
        self.tails = self.tails.to(self.device)
        self.labels = self.labels.to(self.device)

        best_val_loss = float("inf")
        best_epoch = -1
        best_metrics = None
        best_state = None

        for epoch in range(1, self.epochs + 1):

            train_bce, train_contr, _, _ = self._iterate_batches(self.train_idx, train=True)
            with torch.no_grad():
                val_bce, _, val_probs, val_labels = self._iterate_batches(
                    self.val_idx, train=False)

            if val_probs is not None and val_labels is not None:
                val_metrics = self.metrics.update(val_probs, val_labels)
            else: val_metrics = None

            if not getattr(self.log, "non_verbose", False):
                msg = (f"Epoch {epoch:03d} | "
                    f"TrainLoss(BCE)={train_bce:.4f} | "
                    f"TrainLoss(Contr)={train_contr:.4f} | "
                    f"ValLoss(BCE)={val_bce:.4f}")
                if val_metrics is not None:
                    msg += " | " + ", ".join(f"{name}={val_metrics[i]:.4f}"
                        for i, name in enumerate(self.metrics.get_names()))
                self.log.log(msg)

            if val_bce < best_val_loss:
                best_val_loss = val_bce
                best_epoch = epoch
                best_metrics = val_metrics
                best_state = self.model.state_dict()
            if self.early_stopping.step(val_bce, self.model):
                if not getattr(self.log, "non_verbose", False):
                    self.log.log(f"Early stopping at epoch {epoch}.")
                break

        if best_state is not None: self.model.load_state_dict(best_state)

        if not getattr(self.log, "non_verbose", False):
            self.log.log(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
            if best_metrics is not None:
                for name, val in zip(self.metrics.get_names(), best_metrics):
                    self.log.log(f"  {name}: {val:.4f}")

        return best_val_loss, best_epoch, best_metrics
