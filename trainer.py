# trainer.py
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch_geometric.data import HeteroData

from utils import Metrics, EarlyStopping
from negativestatement_sampler import NegativeStatementSampler
from dualcontrastive_CE import DualContrastiveLoss_CE


class Train:
    """
    Triple-level trainer for a relation-aware GNN with optional contrastive learning.

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
      - optionally adds a contrastive loss term per epoch using a
        NegativeStatementSampler and DualContrastiveLoss_CE.
    """

    def __init__(
        self,
        model: nn.Module,
        graph: HeteroData,
        heads: torch.Tensor,
        rel_ids: torch.Tensor,
        tails: torch.Tensor,
        labels: torch.Tensor,
        lr: float,
        epochs: int,
        device: torch.device,
        log,
        batch_size: int = 4096,
        val_ratio: float = 0.1,
        early_stopping_patience: int = 20,
        # NEW: explicit indices and contrastive sampler
        train_idx: Optional[torch.Tensor] = None,
        val_idx: Optional[torch.Tensor] = None,
        contrastive_sampler: Optional[NegativeStatementSampler] = None,
        contrastive_weight: float = 0.1,
    ):
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

        # Contrastive setup
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = float(contrastive_weight) if contrastive_sampler is not None else 0.0
        self.contrastive_loss_fn = DualContrastiveLoss_CE() if contrastive_sampler is not None else None

        # Build train/val split over triples
        num_triples = self.heads.size(0)
        if train_idx is not None and val_idx is not None:
            # Use provided indices (e.g., from KFold)
            self.train_idx = train_idx.to(torch.long)
            self.val_idx = val_idx.to(torch.long)
        else:
            # Fallback: random split
            perm = torch.randperm(num_triples)
            split = int(num_triples * (1.0 - self.val_ratio))
            self.train_idx = perm[:split]
            self.val_idx = perm[split:]

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _iterate_batches(
        self,
        idx: torch.Tensor,
        z: torch.Tensor,
        train: bool = True
    ) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Iterate over batches of triples indexed by `idx`, compute BCE loss and
        (optionally) collect outputs and labels for metrics.

        `z` is assumed to be node embeddings for the whole graph (for self.model.n_type).
        If `train=True`, we update the model parameters using BCE only.
        Contrastive loss is handled separately at the epoch level.
        """
        total_loss = 0.0
        total_examples = 0
        all_probs = []
        all_labels = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            for start in range(0, idx.size(0), self.batch_size):
                end = min(start + self.batch_size, idx.size(0))
                b_idx = idx[start:end]

                h = self.heads[b_idx]
                t = self.tails[b_idx]
                r = self.rels[b_idx]
                y = self.labels[b_idx]

                edge_index = torch.stack([h, t], dim=0)
                logits, probs = self.model.score_triples(z, edge_index, r)
                loss = self.criterion(logits, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_examples += y.size(0)
                all_probs.append(probs.detach().cpu())
                all_labels.append(y.detach().cpu())

        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        if all_probs:
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        else:
            all_probs = None
            all_labels = None
        return avg_loss, all_probs, all_labels

    def run(self):
        """
        Train with a train/val split (either internal or provided) and early stopping
        on validation BCE loss. If a NegativeStatementSampler is given, we add a
        contrastive loss term once per epoch.

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

        n_type = getattr(self.model, "n_type", "node")

        for epoch in range(1, self.epochs + 1):
            # ------------------- Classification training -------------------
            # Encode graph once for BCE; detach so encoder is not updated here.
            h_dict = self.model.encode(self.graph)
            z_full = h_dict[n_type]
            z_cls = z_full.detach()

            train_loss, _, _ = self._iterate_batches(self.train_idx, z_cls, train=True)

            # ------------------- Contrastive training ---------------------
            if self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                self.model.train()
                h_dict_c = self.model.encode(self.graph)
                z_c = h_dict_c[n_type]

                z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(z_c)
                contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                total_contr_loss = self.contrastive_weight * contr_loss

                self.optimizer.zero_grad()
                total_contr_loss.backward()
                self.optimizer.step()
            else:
                total_contr_loss = torch.tensor(0.0)

            # ------------------- Validation (BCE only) --------------------
            with torch.no_grad():
                h_dict_val = self.model.encode(self.graph)
                z_val = h_dict_val[n_type]
                val_loss, val_probs, val_labels = self._iterate_batches(
                    self.val_idx, z_val, train=False
                )

            # Metrics on validation
            if val_probs is not None and val_labels is not None:
                val_metrics = self.metrics.update(val_probs, val_labels)
            else:
                val_metrics = None

            if not getattr(self.log, "non_verbose", False):
                msg = (
                    f"Epoch {epoch:03d} | "
                    f"TrainLoss(BCE)={train_loss:.4f} | "
                    f"ValLoss(BCE)={val_loss:.4f}"
                )
                if self.contrastive_sampler is not None:
                    msg += f" | ConstrLoss={float(total_contr_loss):.4f}"
                if val_metrics is not None:
                    msg += " | " + ", ".join(
                        f"{name}={val_metrics[i]:.4f}"
                        for i, name in enumerate(self.metrics.get_names())
                    )
                self.log.log(msg)

            # Early stopping on val_loss (BCE)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = val_metrics
                best_state = self.model.state_dict()
            if self.early_stopping.step(val_loss, self.model):
                if not getattr(self.log, "non_verbose", False):
                    self.log.log(f"Early stopping at epoch {epoch}.")
                break

        # Restore best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if not getattr(self.log, "non_verbose", False):
            self.log.log(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
            if best_metrics is not None:
                for name, val in zip(self.metrics.get_names(), best_metrics):
                    self.log.log(f"  {name}: {val:.4f}")

        return best_val_loss, best_epoch, best_metrics
