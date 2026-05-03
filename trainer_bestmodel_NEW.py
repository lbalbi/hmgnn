import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from torch_geometric.data import HeteroData
from utils import EarlyStopping
from samplers import (NegativeSampler, NegativeInstanceSampler_NEW, 
    PartialInstanceSampler, RandomInstanceSampler, TypedInstanceSampler)
from losses import ContrastiveLoss_CE, ContrastiveInstanceLoss, DualContrastiveInstanceLoss
import os
import csv

CLS_EDGE_TYPE = ("node", "cls_link", "node")
ContrastiveSamplerT = Union[
    NegativeInstanceSampler_NEW, PartialInstanceSampler, RandomInstanceSampler, TypedInstanceSampler
]

def atomic_torch_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def maybe_load_neg_cache(path: str, map_location: str = "cpu"):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    print(f"Loading negative sample cache", flush=True)
    return torch.load(path, map_location=map_location)


class Train_BestModel:
    """Final training on the full training set.
    Two modes:
      1) Full-graph mode (default, if loader is None):
         - Encode graph once per epoch.
         - Run triple mini-batches over all triples.
      2) NeighborLoader mode (if loader is provided).
    BCE and contrastive losses are merged into a single loss per epoch so that
    gradients flow through both encoder and classifier jointly.
    """
    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor, lr: float,
        epochs: int, device: torch.device, log, batch_size: int = 1024,
        contrastive_sampler: Optional[ContrastiveSamplerT] = None,
        contrastive_weight: Optional[float] = 0.1, loader=None, no_contrastive: bool = False,
        early_stopping_patience: int = 15, contrastive_temperature: float = 0.5):
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
        self.no_contrastive = no_contrastive
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = (float(contrastive_weight) if contrastive_sampler is not None else 0.0)
        self.contrastive_temperature = float(contrastive_temperature)
        # self.contrastive_loss_fn = (ContrastiveLoss_CE() if contrastive_sampler is not None else None)
        # self.contrastive_loss_fn = (ContrastiveInstanceLoss() if contrastive_sampler is not None else None)
        self.dual_view = bool(getattr(self.model, "dual_view", False))
        if contrastive_sampler is not None:
            self.contrastive_loss_fn = (
                DualContrastiveInstanceLoss(temperature=self.contrastive_temperature)
                if self.dual_view else ContrastiveInstanceLoss(temperature=self.contrastive_temperature)
            )
        else: self.contrastive_loss_fn = None
        self.es_patience = int(early_stopping_patience)
        lr_early_stopping = EarlyStopping(patience=self.es_patience, mode="min")
        self.loader = loader

        if "node" not in self.graph.node_types:
            raise ValueError("Train_BestModel assumes a single node type 'node'.")
        num_nodes = int(self.graph["node"].num_nodes)
        self.num_nodes = num_nodes
        self.node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        num_triples = self.heads.size(0)

        # if "node" not in self.graph.node_types:
        #     raise ValueError("Train_BestModel assumes a single node type 'node'.")
        # num_nodes = int(self.graph["node"].num_nodes)
        # self.num_nodes = num_nodes
        # self.node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        # num_triples = self.heads.size(0)

        # self.node_to_triples: List[List[int]] = [[] for _ in range(num_nodes)]
        # for idx in range(num_triples):
        #     h = int(self.heads[idx])
        #     if 0 <= h < num_nodes: self.node_to_triples[h].append(idx)

    def _get_link_supervision(self, batch: HeteroData):
        if isinstance(batch, HeteroData):
            if CLS_EDGE_TYPE not in batch.edge_types:
                raise RuntimeError(f"Batch missing cls edge store {CLS_EDGE_TYPE}.")
            store = batch[CLS_EDGE_TYPE]
            return store.edge_label_index, getattr(store, "edge_label", None), getattr(store, "input_id", None)
        eli = getattr(batch, "edge_label_index", None)
        if eli is not None:
            return eli, getattr(batch, "edge_label", None), getattr(batch, "input_id", None)
        raise RuntimeError("No edge_label_index found in batch. Are you using LinkNeighborLoader?")


    def _maybe_switch_to_dual(self, h_dict: Dict[str, torch.Tensor], n_type: str) -> None:
        if self.contrastive_sampler is None or self.no_contrastive or self.contrastive_weight <= 0.0:
            return
        if getattr(self, "dual_view", False):
            return
        pos_k = f"{n_type}_pos"
        neg_k = f"{n_type}_neg"
        if pos_k in h_dict and neg_k in h_dict and n_type in h_dict:
            z = h_dict[n_type]
            zp = h_dict[pos_k]
            if z.dim() == 2 and zp.dim() == 2 and z.size(1) == 2 * zp.size(1):
                self.dual_view = True
                self.contrastive_loss_fn = DualContrastiveInstanceLoss(temperature=self.contrastive_temperature)

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

            if edge_index.numel() > 0:
                if int(edge_index.max()) >= int(z.size(0)):
                    raise RuntimeError("edge_index seems to use GLOBAL node ids, but z is LOCAL embeddings. "
                        "You need to map endpoints via batch['node'].n_id -> local.")

            loss = bce_loss
            contr_loss_val = 0.0

            if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([h, t], dim=0))
                samples = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=None)
                contr_loss = self.contrastive_loss_fn(*samples)
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

    # @staticmethod
    # def _build_global_to_local(n_id: torch.Tensor) -> Dict[int, int]:
    #     n_id_list = n_id.cpu().tolist()
    #     return {int(g): i for i, g in enumerate(n_id_list)}


    # def _get_batch_triple_indices(self, batch: HeteroData, subset: str) -> torch.Tensor:
    #     """
    #     Returns the indices of triples in the given subset ('train' or 'val')
    #     whose head and tail are both inside the current NeighborLoader subgraph.
    #     This implementation avoids Python loops by using a boolean node mask.
    #     """
    #     n_id = batch["node"].n_id
    #     if n_id.is_cuda: n_id = n_id.cpu()

    #     if subset == "train": subset_idx = self.train_idx
    #     elif subset == "val": subset_idx = self.val_idx  
    #     else: raise ValueError(f"Unknown subset: {subset}")
    #     node_mask = self.node_mask
    #     node_mask[n_id] = True
    #     heads_sub = self.heads[subset_idx]
    #     tails_sub = self.tails[subset_idx]
    #     in_batch = node_mask[heads_sub] & node_mask[tails_sub]
    #     node_mask[n_id] = False
    #     return subset_idx[in_batch]

    # def _get_batch_triple_indices(self, batch: HeteroData) -> torch.Tensor:
    #     """
    #     Returns the indices of *all* triples (over self.heads/self.tails/self.rels/self.labels)
    #     whose head and tail are both inside the current NeighborLoader subgraph.
    #     This is simpler than in the CV trainer: there is no 'train'/'val' split here,
    #     we are training on the full classification triple set.
    #     """
    #     n_id = batch["node"].n_id
    #     if n_id.is_cuda: n_id = n_id.cpu()
    #     node_mask = self.node_mask
    #     node_mask[n_id] = True
    #     heads_sub = self.heads
    #     tails_sub = self.tails
    #     in_batch = node_mask[heads_sub] & node_mask[tails_sub]
    #     node_mask[n_id] = False
    #     return torch.nonzero(in_batch, as_tuple=False).view(-1)

    # def _build_local_triple_tensors(self, triple_idx: torch.Tensor, n_id: torch.Tensor,
    #     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     g2l = self._build_global_to_local(n_id)
    #     heads_global = self.heads[triple_idx].tolist()
    #     tails_global = self.tails[triple_idx].tolist()

    #     h_local = [g2l[int(h)] for h in heads_global]
    #     t_local = [g2l[int(t)] for t in tails_global]
    #     h_local_t = torch.tensor(h_local, dtype=torch.long, device=device)
    #     t_local_t = torch.tensor(t_local, dtype=torch.long, device=device)
    #     edge_index_local = torch.stack([h_local_t, t_local_t], dim=0)

    #     rel_ids = self.rels[triple_idx].to(device)
    #     labels = self.labels[triple_idx].to(device)
    #     return edge_index_local, rel_ids, labels

    # def _epoch_step_neighbors(self, n_type: str) -> Tuple[float, float]:
    #     """One epoch using NeighborLoader subgraphs."""
    #     assert self.loader is not None, "NeighborLoader not provided for Train_BestModel."
    #     self.model.train()
    #     total_bce = 0.0
    #     total_contr = 0.0
    #     total_examples = 0

    #     for batch in self.loader:
    #         batch = batch.to(self.device)
    #         triple_idx = self._get_batch_triple_indices(batch)
    #         if triple_idx.numel() == 0: continue

    #         if (not self.no_contrastive and self.contrastive_sampler is not None
    #             and hasattr(self.contrastive_sampler, "prepare_batch")):
    #             self.contrastive_sampler.prepare_batch(batch)
    #         self.optimizer.zero_grad()
    #         h_dict = self.model.encode(batch)
    #         self._maybe_switch_to_dual(h_dict, n_type)
    #         z = h_dict[n_type]

    #         edge_index_local, rel_ids, labels = self._build_local_triple_tensors(
    #             triple_idx, batch["node"].n_id, self.device)

    #         logits, probs = self.model.score_triples(z, edge_index_local, rel_ids)
    #         bce_loss = self.criterion(logits, labels)
    #         loss = bce_loss
    #         contr_loss_val = 0.0

    #         if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
    #             batch_nodes = torch.unique(torch.cat([edge_index_local[0], edge_index_local[1]], dim=0))
    #             samples = self.contrastive_sampler.get_contrastive_samples(
    #                 z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id)
    #             contr_loss = self.contrastive_loss_fn(*samples)
    #             loss = loss + self.contrastive_weight * contr_loss
    #             contr_loss_val = float(contr_loss.detach().cpu().item())

    #         loss.backward()
    #         self.optimizer.step()
    #         batch_size_eff = labels.size(0)
    #         total_bce += bce_loss.detach().cpu().item() * batch_size_eff
    #         total_contr += contr_loss_val * batch_size_eff
    #         total_examples += batch_size_eff
    #     avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
    #     avg_contr = total_contr / total_examples if total_examples > 0 else 0.0
    #     return avg_bce, avg_contr

    def _epoch_step_neighbors(self, n_type: str):
        assert self.loader is not None, "LinkNeighborLoader not provided for Train_BestModel."
        self.model.train()
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0

        for batch in self.loader:
            batch = batch.to(self.device)

            edge_label_index, edge_label, input_id = self._get_link_supervision(batch)
            if input_id is None:
                raise RuntimeError("Batch is missing input_id; ensure you're using LinkNeighborLoader.")
            input_id_cpu = input_id.detach().to("cpu").long()

            labels = (edge_label if edge_label is not None else self.labels[input_id_cpu])
            labels = labels.to(self.device).float()
            rel_ids = self.rels[input_id_cpu].to(self.device).long()
            if labels.numel() == 0: continue

            if (not self.no_contrastive and self.contrastive_sampler is not None
                and hasattr(self.contrastive_sampler, "prepare_batch")):
                self.contrastive_sampler.prepare_batch(batch)

            self.optimizer.zero_grad()
            h_dict = self.model.encode(batch)
            self._maybe_switch_to_dual(h_dict, n_type)
            z = h_dict[n_type]

            if edge_label_index.numel() > 0:
                if int(edge_label_index.max()) >= int(z.size(0)):
                    raise RuntimeError(
                            "edge_index seems to use GLOBAL node ids, but z is LOCAL embeddings. "
                            "You need to map endpoints via batch['node'].n_id -> local.")
            assert rel_ids.numel() == labels.numel() == edge_label_index.size(1)

            logits, probs = self.model.score_triples(z, edge_label_index, rel_ids)
            bce_loss = self.criterion(logits, labels)

            loss = bce_loss
            contr_loss_val = 0.0
            if (not self.no_contrastive and self.contrastive_sampler is not None
                and self.contrastive_weight > 0.0):
                # Sampler already deduplicates/filters anchors; avoid redundant unique() here.
                batch_nodes = edge_label_index.view(-1)
                samples = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id
                )
                contr_loss = self.contrastive_loss_fn(*samples)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())

            loss.backward()
            self.optimizer.step()

            bs = labels.size(0)
            total_bce += float(bce_loss.detach().cpu().item()) * bs
            total_contr += float(contr_loss_val) * bs
            total_examples += bs

        avg_bce = total_bce / total_examples if total_examples else 0.0
        avg_contr = total_contr / total_examples if total_examples else 0.0
        return avg_bce, avg_contr


    def run(self) -> float:
        """Final training loop."""
        use_neighbor_mode = self.loader is not None
        n_type = getattr(self.model, "n_type", "node")
        last_bce_loss = 0.0
        last_contr_loss = 0.0

        if not use_neighbor_mode:
            self.graph = self.graph.to(self.device)
            self.heads = self.heads.to(self.device)
            self.rels = self.rels.to(self.device)
            self.tails = self.tails.to(self.device)
            self.labels = self.labels.to(self.device)

        for epoch in range(1, self.epochs + 1):
            if use_neighbor_mode: bce_loss, contr_loss = self._epoch_step_neighbors(n_type)
            else:
                self.model.train()
                if (not self.no_contrastive and self.contrastive_sampler is not None
                    and hasattr(self.contrastive_sampler, "prepare_batch")):
                    self.contrastive_sampler.prepare_batch(self.graph)
                self.optimizer.zero_grad()
                h_dict = self.model.encode(self.graph)
                self._maybe_switch_to_dual(h_dict, n_type)
                z = h_dict[n_type]
                bce_loss, contr_loss, total_loss = self._epoch_step_fullgraph(z)
                total_loss.backward()
                self.optimizer.step()

            last_bce_loss = bce_loss
            last_contr_loss = contr_loss
            if not getattr(self.log, "non_verbose", False):
                msg = f"[Final Train] Epoch {epoch:03d} | BCE_Loss={bce_loss:.4f}"
                if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    msg += f" | ConstrLoss={contr_loss:.4f}"
                self.log.log(msg)
        return last_bce_loss


class LinkPredictionEvaluator:
    """Filtered link-prediction evaluation (MR/MRR/Hits@K) over positive test triples."""

    def __init__(
        self,
        model: nn.Module,
        graph: HeteroData,
        test_heads: torch.Tensor,
        test_rels: torch.Tensor,
        test_tails: torch.Tensor,
        known_true_tails: Dict[Tuple[int, int], set],
        known_true_heads: Dict[Tuple[int, int], set],
        id2rel: Dict[int, str],
        device: torch.device,
        log,
        batch_size: int = 1024,
        save_embeddings_path: Optional[str] = None,
        rankings_save_path: Optional[str] = None,
        rankings_tsv_path: Optional[str] = None,
        metrics_save_path: Optional[str] = None,
        metrics_tsv_path: Optional[str] = None,
        predictions_save_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.graph = graph
        self.test_heads = test_heads.clone().long()
        self.test_rels = test_rels.clone().long()
        self.test_tails = test_tails.clone().long()
        self.known_true_tails = known_true_tails
        self.known_true_heads = known_true_heads
        self.id2rel = id2rel
        self.device = device
        self.log = log
        self.batch_size = int(batch_size)
        self.save_embeddings_path = save_embeddings_path
        self.rankings_save_path = rankings_save_path
        self.rankings_tsv_path = rankings_tsv_path
        self.metrics_save_path = metrics_save_path
        self.metrics_tsv_path = metrics_tsv_path
        self.predictions_save_path = predictions_save_path
        self.num_nodes = int(graph["node"].num_nodes)

    def _score_logits_in_batches(
        self,
        z: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        all_logits: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            num = heads.size(0)
            for start in range(0, num, self.batch_size):
                end = min(start + self.batch_size, num)
                h = heads[start:end]
                t = tails[start:end]
                r = rels[start:end]
                edge_index = torch.stack([h, t], dim=0).to(z.device)
                r = r.to(z.device)
                logits, _ = self.model.score_triples(z, edge_index, r)
                all_logits.append(logits.detach().cpu())
        return torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, dtype=torch.float32)

    def _filtered_tail_rank(self, z: torch.Tensor, h: int, r: int, t: int, all_nodes: torch.Tensor) -> int:
        key = (h, r)
        filtered = self.known_true_tails.get(key, set())
        mask = torch.ones(self.num_nodes, dtype=torch.bool, device=all_nodes.device)
        if filtered:
            idx = torch.tensor(list(filtered), dtype=torch.long, device=all_nodes.device)
            mask[idx] = False
        mask[t] = True
        cand_tails = all_nodes[mask]

        h_vec = torch.full((cand_tails.numel(),), h, dtype=torch.long, device=all_nodes.device)
        r_vec = torch.full((cand_tails.numel(),), r, dtype=torch.long, device=all_nodes.device)
        scores = self._score_logits_in_batches(z, h_vec, r_vec, cand_tails)
        cand_tails_cpu = cand_tails.detach().cpu()
        true_idx = torch.where(cand_tails_cpu == int(t))[0]
        if true_idx.numel() == 0:
            raise RuntimeError("True tail disappeared after filtering; this should not happen.")
        true_score = scores[true_idx[0]]
        return int((scores > true_score).sum().item()) + 1

    def _filtered_head_rank(self, z: torch.Tensor, h: int, r: int, t: int, all_nodes: torch.Tensor) -> int:
        key = (t, r)
        filtered = self.known_true_heads.get(key, set())
        mask = torch.ones(self.num_nodes, dtype=torch.bool, device=all_nodes.device)
        if filtered:
            idx = torch.tensor(list(filtered), dtype=torch.long, device=all_nodes.device)
            mask[idx] = False
        mask[h] = True
        cand_heads = all_nodes[mask]

        t_vec = torch.full((cand_heads.numel(),), t, dtype=torch.long, device=all_nodes.device)
        r_vec = torch.full((cand_heads.numel(),), r, dtype=torch.long, device=all_nodes.device)
        scores = self._score_logits_in_batches(z, cand_heads, r_vec, t_vec)
        cand_heads_cpu = cand_heads.detach().cpu()
        true_idx = torch.where(cand_heads_cpu == int(h))[0]
        if true_idx.numel() == 0:
            raise RuntimeError("True head disappeared after filtering; this should not happen.")
        true_score = scores[true_idx[0]]
        return int((scores > true_score).sum().item()) + 1

    @staticmethod
    def _hits_at(rank: int, k: int) -> float:
        return 1.0 if rank <= k else 0.0

    def run(self):
        self.model = self.model.cpu()
        graph_cpu = self.graph.cpu()
        heads = self.test_heads.cpu()
        rels = self.test_rels.cpu()
        tails = self.test_tails.cpu()

        with torch.inference_mode():
            h_dict = self.model.encode(graph_cpu)
            z = h_dict[getattr(self.model, "n_type", "node")]
            del h_dict

            if self.save_embeddings_path:
                try:
                    torch.save({"embeddings": z.detach().cpu()}, self.save_embeddings_path)
                    print(f"[INFO] Saved test-time embeddings: {self.save_embeddings_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save test-time embeddings: {e}")

            all_nodes = torch.arange(self.num_nodes, dtype=torch.long, device=z.device)
            n = int(heads.numel())
            if n == 0:
                raise RuntimeError("No positive test triples available for link prediction.")
            true_logits = self._score_logits_in_batches(z, heads, rels, tails)
            true_probs = torch.sigmoid(true_logits)

            head_rank_sum = 0.0
            head_rr_sum = 0.0
            head_h1 = 0.0
            head_h3 = 0.0
            head_h10 = 0.0

            tail_rank_sum = 0.0
            tail_rr_sum = 0.0
            tail_h1 = 0.0
            tail_h3 = 0.0
            tail_h10 = 0.0
            head_ranks: List[int] = []
            tail_ranks: List[int] = []

            for i in range(n):
                h = int(heads[i].item())
                r = int(rels[i].item())
                t = int(tails[i].item())

                rank_t = self._filtered_tail_rank(z, h, r, t, all_nodes)
                tail_ranks.append(rank_t)
                tail_rank_sum += rank_t
                tail_rr_sum += 1.0 / float(rank_t)
                tail_h1 += self._hits_at(rank_t, 1)
                tail_h3 += self._hits_at(rank_t, 3)
                tail_h10 += self._hits_at(rank_t, 10)

                rank_h = self._filtered_head_rank(z, h, r, t, all_nodes)
                head_ranks.append(rank_h)
                head_rank_sum += rank_h
                head_rr_sum += 1.0 / float(rank_h)
                head_h1 += self._hits_at(rank_h, 1)
                head_h3 += self._hits_at(rank_h, 3)
                head_h10 += self._hits_at(rank_h, 10)

                if (i + 1) % 100 == 0:
                    print(f"[LP Eval] Processed {i + 1}/{n} queries")

        n_f = float(n)
        tail_mr = tail_rank_sum / n_f
        tail_mrr = tail_rr_sum / n_f
        tail_hits1 = tail_h1 / n_f
        tail_hits3 = tail_h3 / n_f
        tail_hits10 = tail_h10 / n_f

        head_mr = head_rank_sum / n_f
        head_mrr = head_rr_sum / n_f
        head_hits1 = head_h1 / n_f
        head_hits3 = head_h3 / n_f
        head_hits10 = head_h10 / n_f

        mr = 0.5 * (head_mr + tail_mr)
        mrr = 0.5 * (head_mrr + tail_mrr)
        hits1 = 0.5 * (head_hits1 + tail_hits1)
        hits3 = 0.5 * (head_hits3 + tail_hits3)
        hits10 = 0.5 * (head_hits10 + tail_hits10)

        metrics = {
            "tail_mr": tail_mr,
            "tail_mrr": tail_mrr,
            "tail_hits@1": tail_hits1,
            "tail_hits@3": tail_hits3,
            "tail_hits@10": tail_hits10,
            "head_mr": head_mr,
            "head_mrr": head_mrr,
            "head_hits@1": head_hits1,
            "head_hits@3": head_hits3,
            "head_hits@10": head_hits10,
            "mr": mr,
            "mrr": mrr,
            "hits@1": hits1,
            "hits@3": hits3,
            "hits@10": hits10,
        }

        self.log.log("=== Link Prediction (filtered) ===")
        for k, v in metrics.items():
            self.log.log(f"{k}: {v:.6f}")
        self._save_artifacts(heads, rels, tails, true_logits, true_probs, head_ranks, tail_ranks, metrics)
        return metrics

    def _save_artifacts(
        self,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
        true_logits: torch.Tensor,
        true_probs: torch.Tensor,
        head_ranks: List[int],
        tail_ranks: List[int],
        metrics: Dict[str, float],
    ) -> None:
        head_rank_t = torch.tensor(head_ranks, dtype=torch.long)
        tail_rank_t = torch.tensor(tail_ranks, dtype=torch.long)
        labels = torch.ones_like(true_probs, dtype=torch.float32)
        payload = {
            "heads": heads.detach().cpu().long(),
            "rels": rels.detach().cpu().long(),
            "tails": tails.detach().cpu().long(),
            "labels": labels.detach().cpu(),
            "true_logits": true_logits.detach().cpu(),
            "true_probs": true_probs.detach().cpu(),
            "head_ranks": head_rank_t,
            "tail_ranks": tail_rank_t,
            "mean_ranks_per_triple": 0.5 * (head_rank_t.float() + tail_rank_t.float()),
            "reciprocal_ranks_per_triple": 0.5 * (1.0 / head_rank_t.float() + 1.0 / tail_rank_t.float()),
            "metrics": metrics,
        }
        for path in [self.rankings_save_path, self.predictions_save_path]:
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(payload, path)
        if self.metrics_save_path:
            os.makedirs(os.path.dirname(self.metrics_save_path), exist_ok=True)
            torch.save({"metrics": metrics}, self.metrics_save_path)
        if self.metrics_tsv_path:
            os.makedirs(os.path.dirname(self.metrics_tsv_path), exist_ok=True)
            with open(self.metrics_tsv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["metric", "value"])
                for key, value in metrics.items():
                    writer.writerow([key, f"{float(value):.8f}"])
        if self.rankings_tsv_path:
            os.makedirs(os.path.dirname(self.rankings_tsv_path), exist_ok=True)
            with open(self.rankings_tsv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([
                    "query_index",
                    "head",
                    "rel_id",
                    "rel_name",
                    "tail",
                    "label",
                    "true_logit",
                    "true_prob",
                    "head_rank",
                    "tail_rank",
                    "mean_rank",
                    "reciprocal_rank",
                ])
                for i, (h, r, t) in enumerate(zip(heads.tolist(), rels.tolist(), tails.tolist())):
                    hr = int(head_ranks[i])
                    tr = int(tail_ranks[i])
                    writer.writerow([
                        i,
                        int(h),
                        int(r),
                        self.id2rel.get(int(r), str(int(r))),
                        int(t),
                        1,
                        f"{float(true_logits[i]):.8f}",
                        f"{float(true_probs[i]):.8f}",
                        hr,
                        tr,
                        f"{0.5 * (hr + tr):.8f}",
                        f"{0.5 * ((1.0 / hr) + (1.0 / tr)):.8f}",
                    ])
