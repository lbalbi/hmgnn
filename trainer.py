import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from torch_geometric.data import HeteroData

from utils import Metrics, EarlyStopping
from samplers import NegativeStatementSampler
from losses import DualContrastiveLoss_CE


class Train:
    """
    Triple-level trainer for a relation-aware GNN with optional contrastive learning
    and an internal learning-rate sweep per CV fold.
    Assumptions about `model`:
      - has attribute `n_type` (string) for the main node type (default "node").
      - has method: encode(data: HeteroData) -> Dict[str, Tensor] returning node embeddings per node type.
      - has method: score_triples(z: Tensor, edge_index: LongTensor[2, B], rel_ids: LongTensor[B])
        returning (logits: Tensor[B], probs: Tensor[B]).
      
      • for each LR candidate:
          - resets model to the same initial weights;
          - trains with encode-once-per-epoch + early stopping;
          - tracks the best epoch & val loss for that LR;
      • returns the best LR (lowest val BCE), its epoch, and its metrics
        for that fold.
    """

    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor,
        lr_candidates: List[float], epochs: int, device: torch.device, log,
        batch_size: int = 1024, val_ratio: float = 0.1, early_stopping_patience: int = 20,
        train_idx: Optional[torch.Tensor] = None, val_idx: Optional[torch.Tensor] = None,
        contrastive_sampler: Optional[NegativeStatementSampler] = None,
        contrastive_weight: float = 0.1):
        self.model = model.to(device)
        self.graph = graph
        self.heads = heads
        self.rels = rel_ids
        self.tails = tails
        self.labels = labels
        self.lr_candidates = [float(lr) for lr in lr_candidates]
        self.max_epochs = int(epochs)
        self.device = device
        self.log = log
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.es_patience = int(early_stopping_patience)

        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = Metrics()
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = (
            float(contrastive_weight) if contrastive_sampler is not None else 0.0)
        self.contrastive_loss_fn = (
            DualContrastiveLoss_CE() if contrastive_sampler is not None else None)

        num_triples = self.heads.size(0)
        if train_idx is not None and val_idx is not None:
            self.train_idx = train_idx.to(torch.long)
            self.val_idx = val_idx.to(torch.long)
        else:
            perm = torch.randperm(num_triples)
            split = int(num_triples * (1.0 - self.val_ratio))
            self.train_idx = perm[:split]
            self.val_idx = perm[split:]
        self._init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}


    def _iterate_batches(self, idx: torch.Tensor, z: torch.Tensor, train: bool = True,
    ) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Iterate over batches of triples indexed by `idx`, using a **fixed** embedding
        tensor `z` for the whole epoch.
        Args:
            idx: indices of triples (into self.heads / self.rels / self.tails / self.labels).
            z: node embeddings for the main node type, shape [num_nodes, hidden_dim],
               computed once per epoch via model.encode(self.graph).
            train: whether we're in training mode (controls contrastive usage and
                   whether a combined loss tensor for backprop is returned).
        Returns:
            avg_bce_loss: float
            avg_contrastive_loss: float
            all_probs: Tensor or None (concatenated probabilities for metrics)
            all_labels: Tensor or None (concatenated labels for metrics)
            total_loss_tensor: Tensor or None
                - If train=True: scalar tensor to call backward() on.
                - If train=False: None (no gradient, used for validation).
        """
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        all_probs = []
        all_labels = []

        total_loss_tensor = None
        if train: total_loss_tensor = torch.zeros((), device=self.device)

        num_triples = idx.size(0)
        if num_triples == 0:
            return 0.0, 0.0, None, None, total_loss_tensor

        for start in range(0, num_triples, self.batch_size):
            end = min(start + self.batch_size, num_triples)
            b_idx = idx[start:end]

            h = self.heads[b_idx]
            t = self.tails[b_idx]
            r = self.rels[b_idx]
            y = self.labels[b_idx]

            edge_index = torch.stack([h, t], dim=0)
            logits, probs = self.model.score_triples(z, edge_index, r)
            bce_loss = self.criterion(logits, y)

            loss = bce_loss
            contr_loss_val = 0.0

            if train and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([h, t], dim=0))
                z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes)
                contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())

            batch_size_eff = y.size(0)

            if train:
                weight = batch_size_eff / float(num_triples)
                total_loss_tensor = total_loss_tensor + loss * weight

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
        return avg_bce, avg_contr, all_probs, all_labels, total_loss_tensor


    def run(self):
        """
        Train with a train/val split (either internal or provided), performing a
        learning-rate sweep **inside** this method.

        For each lr in `self.lr_candidates`:
          - reset the model to the initial state;
          - create a fresh optimizer;
          - train with early stopping on the current fold (encode-once-per-epoch).

        Returns:
            overall_best_val_loss, overall_best_epoch, overall_best_metrics, overall_best_lr
        """

        self.graph = self.graph.to(self.device)
        self.heads = self.heads.to(self.device)
        self.rels = self.rels.to(self.device)
        self.tails = self.tails.to(self.device)
        self.labels = self.labels.to(self.device)
        n_type = getattr(self.model, "n_type", "node")
        overall_best_val_loss = float("inf")
        overall_best_epoch = -1
        overall_best_metrics = None
        overall_best_lr = None
        overall_best_state = None

        for lr in self.lr_candidates:
            self.model.load_state_dict(self._init_state)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            lr_early_stopping = EarlyStopping(patience=self.es_patience, mode="min")

            best_val_loss_lr = float("inf")
            best_epoch_lr = -1
            best_metrics_lr = None
            best_state_lr = None

            if not getattr(self.log, "non_verbose", False):
                self.log.log(f"=== Starting LR sweep for lr={lr:.3g} ===")

            for epoch in range(1, self.max_epochs + 1):
                self.model.train()

                if (self.contrastive_sampler is not None
                    and hasattr(self.contrastive_sampler, "prepare_batch")):
                    self.contrastive_sampler.prepare_batch(self.graph)

                optimizer.zero_grad()
                h_dict = self.model.encode(self.graph)
                z_train = h_dict[n_type]

                train_bce, train_contr, _, _, train_total_loss = self._iterate_batches(
                    self.train_idx, z_train, train=True)

                if train_total_loss is None: break

                train_total_loss.backward()
                optimizer.step()
                self.model.eval()
                with torch.no_grad():
                    h_dict_val = self.model.encode(self.graph)
                    z_val = h_dict_val[n_type]
                    val_bce, _, val_probs, val_labels, _ = self._iterate_batches(
                        self.val_idx, z_val, train=False)

                if val_probs is not None and val_labels is not None:
                    val_metrics = self.metrics.update(val_probs, val_labels)
                else: val_metrics = None

                if not getattr(self.log, "non_verbose", False):
                    msg = (f"[lr={lr:.3g}] Epoch {epoch:03d} | "
                        f"TrainLoss(BCE)={train_bce:.4f} | "
                        f"TrainLoss(Contr)={train_contr:.4f} | "
                        f"ValLoss(BCE)={val_bce:.4f}")
                    if val_metrics is not None:
                        msg += " | " + ", ".join(f"{name}={val_metrics[i]:.4f}"
                            for i, name in enumerate(self.metrics.get_names()))
                    self.log.log(msg)

                if val_bce < best_val_loss_lr:
                    best_val_loss_lr = val_bce
                    best_epoch_lr = epoch
                    best_metrics_lr = val_metrics
                    best_state_lr = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

                if lr_early_stopping.step(val_bce, self.model):
                    if not getattr(self.log, "non_verbose", False):
                        self.log.log(
                            f"[lr={lr:.3g}] Early stopping at epoch {epoch} "
                            f"(best val loss so far: {best_val_loss_lr:.4f}).")
                    break
            if best_val_loss_lr < overall_best_val_loss:
                overall_best_val_loss = best_val_loss_lr
                overall_best_epoch = best_epoch_lr
                overall_best_metrics = best_metrics_lr
                overall_best_lr = lr
                overall_best_state = best_state_lr

            if not getattr(self.log, "non_verbose", False):
                self.log.log(f"=== Finished LR={lr:.3g} | best_val_loss={best_val_loss_lr:.4f} "
                    f"at epoch {best_epoch_lr} ===")

        if overall_best_state is not None:
            self.model.load_state_dict(overall_best_state)

        if not getattr(self.log, "non_verbose", False):
            self.log.log(f"[CV Fold] Best overall LR={overall_best_lr:.3g} | "
                f"Best epoch={overall_best_epoch} | "
                f"Best val loss={overall_best_val_loss:.4f}")

        return overall_best_val_loss, overall_best_epoch, overall_best_metrics, overall_best_lr
