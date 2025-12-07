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
        contrastive_weight: Optional[float] = 0.0, loader=None, use_contrastive: bool = False):
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
        self.use_contrastive = use_contrastive
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = (float(contrastive_weight) if contrastive_sampler is not None else 0.0)
        self.contrastive_loss_fn = (ContrastiveLoss_CE() if contrastive_sampler is not None else None)
        self.loader = loader

        if "node" not in self.graph.node_types:
            raise ValueError("Train_BestModel assumes a single node type 'node'.")
        num_nodes = int(self.graph["node"].num_nodes)
        self.num_nodes = num_nodes
        num_triples = self.heads.size(0)

        self.node_to_triples: List[List[int]] = [[] for _ in range(num_nodes)]
        for idx in range(num_triples):
            h = int(self.heads[idx])
            if 0 <= h < num_nodes: self.node_to_triples[h].append(idx)

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

            if self.use_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([h, t], dim=0))
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
            if 0 <= u < self.num_nodes: candidate_indices.update(self.node_to_triples[u])

        selected = []
        for idx in candidate_indices:
            t_global = int(self.tails[idx])
            if t_global in nodes_set: selected.append(idx)

        if not selected: return torch.empty(0, dtype=torch.long)
        selected = sorted(selected)
        return torch.tensor(selected, dtype=torch.long)

    def _build_local_triple_tensors(self, triple_idx: torch.Tensor, n_id: torch.Tensor,
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            if triple_idx.numel() == 0: continue

            if (self.use_contrastive and self.contrastive_sampler is not None
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

            if self.use_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
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


    def run(self) -> float:
        """Final training loop."""
        use_neighbor_mode = self.loader is not None
        n_type = getattr(self.model, "n_type", "node")
        last_bce_loss = 0.0
        last_contr_loss = 0.0

        self.graph = self.graph.to(self.device)
        if not use_neighbor_mode:
            self.heads = self.heads.to(self.device)
            self.rels = self.rels.to(self.device)
            self.tails = self.tails.to(self.device)
            self.labels = self.labels.to(self.device)

        for epoch in range(1, self.epochs + 1):
            if use_neighbor_mode: bce_loss, contr_loss = self._epoch_step_neighbors(n_type)
            else:
                self.model.train()
                if (self.use_contrastive and self.contrastive_sampler is not None
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
                if self.use_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    msg += f" | ConstrLoss={contr_loss:.4f}"
                self.log.log(msg)
        return last_bce_loss


class Test_BestModel:
    """Test-time evaluation on a global triple-level test set.
       Positives: provided explicitly as (heads, rel_ids, tails).
       Negatives: sampled per relation using existing NegativeSampler.
       - Negative sampling at test time now ALSO excludes *test positives*,
       in addition to whatever training positives `NegativeSampler` already
       avoids. This is done via an extra filtering step per relation.
    """

    def __init__(self, model: nn.Module, graph: HeteroData, test_heads: torch.Tensor,
        test_rels: torch.Tensor, test_tails: torch.Tensor, id2rel: Dict[int, str],
        neg_samplers: Dict[str, NegativeSampler], num_neg_per_pos: int, device: torch.device,
        log, batch_size: int = 4096):
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

        self.num_nodes = int(graph["node"].num_nodes)
        self.test_pos_ids_per_rel: Dict[str, set] = {}
        for rel_id, rel_name in id2rel.items():
            mask = (test_rels == rel_id)
            if not mask.any(): continue
            h = test_heads[mask]
            t = test_tails[mask]
            ids = (h.long() * self.num_nodes + t.long()).tolist()
            self.test_pos_ids_per_rel[rel_name] = set(ids)


    def _score_triples_in_batches(self, z: torch.Tensor, heads: torch.Tensor,
        rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
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


    def _sample_negatives_excluding_test(self, sampler: NegativeSampler, rel_name: str,
        num_to_sample: int) -> torch.Tensor:
        """
        Use NegativeSampler to sample candidate negatives, then filter out any
        edges that coincide with test positives for `rel_name`. Resamples a few
        times if needed to reach the desired count (best effort).
        """
        invalid_ids = self.test_pos_ids_per_rel.get(rel_name, set())
        if not invalid_ids: return sampler.sample_edge_index(num_to_sample)

        collected = []
        remaining = num_to_sample
        max_attempts = 10
        attempts = 0

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            cand_edge_index = sampler.sample_edge_index(int(remaining * 1.5))
            if cand_edge_index.numel() == 0: break

            h = cand_edge_index[0]
            t = cand_edge_index[1]
            ids = (h.long() * self.num_nodes + t.long()).tolist()
            keep_mask_list = [id_ not in invalid_ids for id_ in ids]

            if not any(keep_mask_list): continue

            keep_mask = torch.tensor(keep_mask_list, dtype=torch.bool, device=self.device)
            kept_edges = cand_edge_index[:, keep_mask]
            if kept_edges.numel() == 0: continue

            collected.append(kept_edges)
            remaining = num_to_sample - sum(c.size(1) for c in collected)

        if not collected:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)

        neg_edge_index = torch.cat(collected, dim=1)
        if neg_edge_index.size(1) > num_to_sample: neg_edge_index = neg_edge_index[:, :num_to_sample]
        return neg_edge_index

    def run(self):
        self.graph = self.graph.to(self.device)
        self.test_heads = self.test_heads.to(self.device)
        self.test_rels = self.test_rels.to(self.device)
        self.test_tails = self.test_tails.to(self.device)

        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.encode(self.graph)
            z = h_dict[getattr(self.model, "n_type", "node")]
            pos_probs = self._score_triples_in_batches(
                z, self.test_heads, self.test_rels, self.test_tails
            )
            pos_labels = torch.ones_like(pos_probs)

            neg_heads_list = []
            neg_rels_list = []
            neg_tails_list = []

            unique_rels, counts = torch.unique(self.test_rels, return_counts=True)
            for rel_id, count in zip(unique_rels.tolist(), counts.tolist()):
                rel_name = self.id2rel[rel_id]
                sampler = self.neg_samplers.get(rel_name, None)
                if sampler is None: continue

                num_to_sample = count * self.num_neg_per_pos
                neg_edge_index = self._sample_negatives_excluding_test(
                    sampler, rel_name, num_to_sample)
                if neg_edge_index.numel() == 0: continue

                h_neg = neg_edge_index[0]
                t_neg = neg_edge_index[1]
                r_neg = torch.full(
                    (h_neg.size(0),), rel_id, dtype=torch.long, device=self.device)

                neg_heads_list.append(h_neg)
                neg_tails_list.append(t_neg)
                neg_rels_list.append(r_neg)

            if neg_heads_list:
                neg_heads = torch.cat(neg_heads_list, dim=0)
                neg_tails = torch.cat(neg_tails_list, dim=0)
                neg_rels = torch.cat(neg_rels_list, dim=0)

                neg_probs = self._score_triples_in_batches(
                    z, neg_heads, neg_rels, neg_tails)
                neg_labels = torch.zeros_like(neg_probs)

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
