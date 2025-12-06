import torch
import torch.nn as nn
from typing import Dict, Optional
from torch_geometric.data import HeteroData

from utils import Metrics
from negative_sampler import NegativeSampler
from negativestatement_sampler import NegativeStatementSampler
from dualcontrastive_CE import DualContrastiveLoss_CE


class Train_BestModel:
    """
    Final training on the full training set for a fixed number of epochs
    (e.g., the best epoch found during validation, or args.epochs).

    Uses triple mini-batching for memory efficiency and optionally
    a contrastive objective via NegativeStatementSampler + DualContrastiveLoss_CE.
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

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Contrastive setup (optional)
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = float(contrastive_weight) if contrastive_sampler is not None else 0.0
        self.contrastive_loss_fn = DualContrastiveLoss_CE() if contrastive_sampler is not None else None

    def _iterate_batches(self, z: torch.Tensor) -> float:
        """
        One epoch: iterate over all training triples in mini-batches, update params
        using BCE only. `z` is assumed to be detached node embeddings so that
        BCE updates only the classifier parameters.
        Returns average BCE loss.
        """
        num_triples = self.heads.size(0)
        perm = torch.randperm(num_triples, device=self.device)

        total_loss = 0.0
        total_examples = 0

        self.model.train()
        for start in range(0, num_triples, self.batch_size):
            end = min(start + self.batch_size, num_triples)
            idx = perm[start:end]

            h = self.heads[idx]
            t = self.tails[idx]
            r = self.rels[idx]
            y = self.labels[idx]

            edge_index = torch.stack([h, t], dim=0)
            logits, _ = self.model.score_triples(z, edge_index, r)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_examples += y.size(0)

        avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
        return avg_loss

    def run(self) -> float:
        """
        Final training loop.

        Per epoch:
          1) Encode graph -> z_full, detach -> z_cls, run BCE mini-batch training.
          2) If contrastive_sampler is provided:
                - re-encode graph,
                - compute contrastive loss on global embeddings,
                - take one optimizer step on contrastive loss.
        Returns:
            last BCE loss (from the last epoch).
        """
        self.graph = self.graph.to(self.device)
        self.heads = self.heads.to(self.device)
        self.rels = self.rels.to(self.device)
        self.tails = self.tails.to(self.device)
        self.labels = self.labels.to(self.device)

        last_bce_loss = 0.0
        n_type = getattr(self.model, "n_type", "node")

        for epoch in range(1, self.epochs + 1):
            # ------------------- BCE classification step -------------------
            # Encode once, then detach so BCE updates only classifier parameters.
            h_dict = self.model.encode(self.graph)
            z_full = h_dict[n_type]
            z_cls = z_full.detach()

            last_bce_loss = self._iterate_batches(z_cls)

            # ------------------- Contrastive step (optional) ---------------
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

            if not getattr(self.log, "non_verbose", False):
                msg = f"[Final Train] Epoch {epoch:03d} | BCE_Loss={last_bce_loss:.4f}"
                if self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    msg += f" | ConstrLoss={float(total_contr_loss):.4f}"
                self.log.log(msg)

        return last_bce_loss


class Test_BestModel:
    """
    Test-time evaluation on a global triple-level test set.

    Positives: provided explicitly as (heads, rel_ids, tails).
    Negatives: sampled per relation using existing NegativeSampler.
    """

    def __init__(
        self,
        model: nn.Module,
        graph: HeteroData,
        test_heads: torch.Tensor,
        test_rels: torch.Tensor,
        test_tails: torch.Tensor,
        id2rel: Dict[int, str],
        neg_samplers: Dict[str, NegativeSampler],
        num_neg_per_pos: int,
        device: torch.device,
        log,
        batch_size: int = 4096,
    ):
        self.model = model.to(device)
        self.graph = graph
        self.test_heads = test_heads
        self.test_rels = test_rels
        self.test_tails = test_tails
        self.id2rel = id2rel
        self.neg_samplers = neg_samplers
        self.num_neg_per_pos = int(num_neg_per_pos)
        self.device = device
        self.log = log
        self.batch_size = int(batch_size)

        self.metrics = Metrics()

    def _score_triples_in_batches(
        self,
        z: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute probabilities for triples in mini-batches to save memory.
        Returns a tensor of shape [N] with probabilities.
        """
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            num = heads.size(0)
            for start in range(0, num, self.batch_size):
                end = min(start + self.batch_size, num)
                h = heads[start:end]
                t = tails[start:end]
                r = rels[start:end]
                edge_index = torch.stack([h, t], dim=0)
                _, probs = self.model.score_triples(z, edge_index, r)
                all_probs.append(probs.detach().cpu())
        return torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)

    def run(self):
        self.graph = self.graph.to(self.device)
        self.test_heads = self.test_heads.to(self.device)
        self.test_rels = self.test_rels.to(self.device)
        self.test_tails = self.test_tails.to(self.device)

        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.encode(self.graph)
            z = h_dict[getattr(self.model, "n_type", "node")]

            # ----------------- Positives -----------------
            pos_probs = self._score_triples_in_batches(
                z, self.test_heads, self.test_rels, self.test_tails
            )
            pos_labels = torch.ones_like(pos_probs)

            # ----------------- Negatives -----------------
            neg_heads_list = []
            neg_rels_list = []
            neg_tails_list = []

            unique_rels, counts = torch.unique(self.test_rels, return_counts=True)
            for rel_id, count in zip(unique_rels.tolist(), counts.tolist()):
                rel_name = self.id2rel[rel_id]
                sampler = self.neg_samplers.get(rel_name, None)
                if sampler is None:
                    continue

                num_to_sample = count * self.num_neg_per_pos
                neg_edge_index = sampler.sample_edge_index(num_to_sample)
                if neg_edge_index.numel() == 0:
                    continue

                h_neg = neg_edge_index[0]
                t_neg = neg_edge_index[1]
                r_neg = torch.full(
                    (h_neg.size(0),), rel_id, dtype=torch.long, device=self.device
                )

                neg_heads_list.append(h_neg)
                neg_tails_list.append(t_neg)
                neg_rels_list.append(r_neg)

            if neg_heads_list:
                neg_heads = torch.cat(neg_heads_list, dim=0)
                neg_tails = torch.cat(neg_tails_list, dim=0)
                neg_rels = torch.cat(neg_rels_list, dim=0)

                neg_probs = self._score_triples_in_batches(
                    z, neg_heads, neg_rels, neg_tails
                )
                neg_labels = torch.zeros_like(neg_probs)

                # Stack positives and negatives and evaluate once
                all_probs = torch.cat([pos_probs, neg_probs], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)
            else:
                all_probs = pos_probs
                all_labels = pos_labels

            metrics = self.metrics.update(all_probs, all_labels)
            names = self.metrics.get_names()

            self.log.log("=== Test metrics (global) ===")
            for name, val in zip(names, metrics):
                self.log.log(f"{name}: {val:.4f}")

        return metrics
