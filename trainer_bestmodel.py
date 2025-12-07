import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from torch_geometric.data import HeteroData

from utils import Metrics
from samplers import NegativeSampler, NegativeStatementSampler
from losses import ContrastiveLoss_CE


class Train_BestModel:
    """Final training on the full training set for a fixed number of epochs
    (e.g., the median epoch found during cross-validation).

    Two modes:
      1) Full-graph mode (default, if loader is None):
         - Encode graph once per epoch.
         - Run triple mini-batches over all triples.
      2) NeighborLoader mode (if loader is provided):
         - For each subgraph:
             • find all triples fully contained in the subgraph;
             • run message passing on that subgraph only;
             • train on those triples with BCE (+ contrastive).

    BCE and contrastive losses are merged into a single loss per epoch so that
    gradients flow through both encoder and classifier jointly.
    """

    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor, lr: float,
        epochs: int, device: torch.device, log, batch_size: int = 1024,
        contrastive_sampler: Optional[NegativeStatementSampler] = None,
        contrastive_weight: float = 0.1,
        loader=None):
        self.model = model.to(device)
        self.graph = graph
        self.heads = heads.clone().long()
        self.rels = rel_ids.clone().long()
        self.tails = tails.clone().long()
        self.labels = labels.clone().float()
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.device = device
        self.log = log
        self.batch_size = int(batch_size)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = (float(contrastive_weight) if contrastive_sampler is not None else 0.0)
        self.contrastive_loss_fn = (ContrastiveLoss_CE() if contrastive_sampler is not None else None)

        # Optional NeighborLoader for subgraph-based final training
        self.loader = loader

        # Precompute node→triple map (by head) for subgraph mode
        if "node" not in self.graph.node_types:
            raise ValueError("Train_BestModel assumes a single node type 'node'.")
        num_nodes = int(self.graph["node"].num_nodes)
        self.num_nodes = num_nodes
        num_triples = self.heads.size(0)

        self.node_to_triples: List[List[int]] = [[] for _ in range(num_nodes)]
        for idx in range(num_triples):
            h = int(self.heads[idx])
            if 0 <= h < num_nodes:
                self.node_to_triples[h].append(idx)

    # -------- Full-graph path (old behaviour) ------------------------
    def _epoch_step_fullgraph(self, z: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        num_triples = self.heads.size(0)
        if num_triples == 0:
            total_loss_tensor = torch.zeros((), device=self.device)
            return 0.0, 0.0, total_loss_tensor

        perm = torch.randperm(num_triples, device=self.device)

        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        total_loss_tensor = torch.zeros((), device=self.device)

        for start in range(0, num_triples, self.batch_size):
            end = min(start + self.batch_size, num_triples)
            idx = perm[start:end]

            h = self.heads[idx]
            t = self.tails[idx]
            r = self.rels[idx]
            y = self.labels[idx]

            edge_index = torch.stack([h, t], dim=0)
            logits, probs = self.model.score_triples(z, edge_index, r)
            bce_loss = self.criterion(logits, y)

            loss = bce_loss
            contr_loss_val = 0.0

            if self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([h, t], dim=0))
                # Full-graph mode: no n_id
                z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=None)
                contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())

            batch_size_eff = y.size(0)
            weight = batch_size_eff / float(num_triples)
            total_loss_tensor = total_loss_tensor + loss * weight

            total_bce += bce_loss.detach().cpu().item() * batch_size_eff
            total_contr += contr_loss_val * batch_size_eff
            total_examples += batch_size_eff

        avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
        avg_contr = total_contr / total_examples if total_examples > 0 else 0.0
        return avg_bce, avg_contr, total_loss_tensor

    # -------- NeighborLoader helpers --------------------------------
    @staticmethod
    def _build_global_to_local(n_id: torch.Tensor) -> Dict[int, int]:
        n_id_list = n_id.cpu().tolist()
        return {int(g): i for i, g in enumerate(n_id_list)}

    def _get_batch_triple_indices(self, batch: HeteroData) -> torch.Tensor:
        """All triples whose head/tail are both inside the subgraph."""
        n_id = batch["node"].n_id.cpu().tolist()
        nodes_set = set(n_id)

        candidate_indices: set = set()
        for u in nodes_set:
            if 0 <= u < self.num_nodes:
                candidate_indices.update(self.node_to_triples[u])

        selected = []
        for idx in candidate_indices:
            t_global = int(self.tails[idx])
            if t_global in nodes_set:
                selected.append(idx)

        if not selected:
            return torch.empty(0, dtype=torch.long)
        selected = sorted(selected)
        return torch.tensor(selected, dtype=torch.long)

    def _build_local_triple_tensors(
        self,
        triple_idx: torch.Tensor,
        n_id: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g2l = self._build_global_to_local(n_id)
        heads_global = self.heads[triple_idx].tolist()
        tails_global = self.tails[triple_idx].tolist()

        h_local = [g2l[int(h)] for h in heads_global]
        t_local = [g2l[int(t)] for t in tails_global]

        h_local_t = torch.tensor(h_local, dtype=torch.long, device=device)
        t_local_t = torch.tensor(t_local, dtype=torch.long, device=device)
        edge_index_local = torch.stack([h_local_t, t_local_t], dim=0)

        rel_ids = self.rels[triple_idx].to(device)
        labels = self.labels[triple_idx].to(device)
        return edge_index_local, rel_ids, labels

    def _epoch_step_neighbors(self, n_type: str) -> Tuple[float, float]:
        """One epoch using NeighborLoader subgraphs."""
        assert self.loader is not None, "NeighborLoader not provided for Train_BestModel."
        self.model.train()
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0

        for batch in self.loader:
            batch = batch.to(self.device)
            triple_idx = self._get_batch_triple_indices(batch)
            if triple_idx.numel() == 0:
                continue

            if (self.contrastive_sampler is not None
                and hasattr(self.contrastive_sampler, "prepare_batch")):
                self.contrastive_sampler.prepare_batch(batch)

            self.optimizer.zero_grad()
            h_dict = self.model.encode(batch)
            z = h_dict[n_type]

            edge_index_local, rel_ids, labels = self._build_local_triple_tensors(
                triple_idx, batch["node"].n_id, self.device)

            logits, probs = self.model.score_triples(z, edge_index_local, rel_ids)
            bce_loss = self.criterion(logits, labels)

            loss = bce_loss
            contr_loss_val = 0.0

            if self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([edge_index_local[0], edge_index_local[1]], dim=0))
                z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id)
                contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())

            loss.backward()
            self.optimizer.step()

            batch_size_eff = labels.size(0)
            total_bce += bce_loss.detach().cpu().item() * batch_size_eff
            total_contr += contr_loss_val * batch_size_eff
            total_examples += batch_size_eff

        avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
        avg_contr = total_contr / total_examples if total_examples > 0 else 0.0
        return avg_bce, avg_contr

    # -------- Main run() --------------------------------------------
    def run(self) -> float:
        """Final training loop."""
        use_neighbor_mode = self.loader is not None
        n_type = getattr(self.model, "n_type", "node")
        last_bce_loss = 0.0
        last_contr_loss = 0.0

        # Move graph & triples only if we need full-graph mode
        self.graph = self.graph.to(self.device)
        if not use_neighbor_mode:
            self.heads = self.heads.to(self.device)
            self.rels = self.rels.to(self.device)
            self.tails = self.tails.to(self.device)
            self.labels = self.labels.to(self.device)

        for epoch in range(1, self.epochs + 1):
            if use_neighbor_mode:
                # Subgraph-based final training
                bce_loss, contr_loss = self._epoch_step_neighbors(n_type)
            else:
                # Original full-graph encode-once-per-epoch training
                self.model.train()

                if (self.contrastive_sampler is not None
                    and hasattr(self.contrastive_sampler, "prepare_batch")):
                    self.contrastive_sampler.prepare_batch(self.graph)

                self.optimizer.zero_grad()
                h_dict = self.model.encode(self.graph)
                z = h_dict[n_type]

                bce_loss, contr_loss, total_loss = self._epoch_step_fullgraph(z)
                total_loss.backward()
                self.optimizer.step()

            last_bce_loss = bce_loss
            last_contr_loss = contr_loss

            if not getattr(self.log, "non_verbose", False):
                msg = f"[Final Train] Epoch {epoch:03d} | BCE_Loss={bce_loss:.4f}"
                if self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    msg += f" | ConstrLoss={contr_loss:.4f}"
                self.log.log(msg)
        return last_bce_loss
