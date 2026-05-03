import torch
import torch.nn as nn
import os
from typing import Optional, Tuple, List, Dict, Set, Union
from torch_geometric.data import HeteroData
from utils import EarlyStopping
from samplers import (
    NegativeInstanceSampler_NEW, PartialInstanceSampler, RandomInstanceSampler, TypedInstanceSampler
)
from losses import ContrastiveLoss_CE, ContrastiveInstanceLoss, DualContrastiveInstanceLoss

import subprocess
import time

CLS_EDGE_TYPE = ("node", "cls_link", "node")
ContrastiveSamplerT = Union[
    NegativeInstanceSampler_NEW, PartialInstanceSampler, RandomInstanceSampler, TypedInstanceSampler
]

class Train:
    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor, lr_candidates: List[float],
        epochs: int, device: torch.device, log, batch_size: int = 1024, val_ratio: float = 0.1,
        early_stopping_patience: int = 15, train_idx: Optional[torch.Tensor] = None,
        val_idx: Optional[torch.Tensor] = None, contrastive_sampler: Optional[ContrastiveSamplerT] = None,
        contrastive_weight: float = 0.1, train_loader=None, val_loader=None, no_contrastive: bool = False,
        val_lp_eval_every: int = 1, contrastive_temperature: float = 0.5,
        abort_random_bce: bool = True, abort_random_bce_checks: int = 12,
        abort_random_bce_tol: float = 0.003, abort_random_bce_span_tol: float = 0.0015,
        abort_random_bce_warmup: int = 12):

        self.model = model.to(device)
        print(model.__class__.__name__)
        self.graph = graph
        self.device = device
        self.log = log
        self.heads = heads.clone().long()
        self.rels = rel_ids.clone().long()
        self.tails = tails.clone().long()
        self.labels = labels.clone().float()
        self.lr_candidates = [float(lr) for lr in lr_candidates]
        self.max_epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.es_patience = int(early_stopping_patience)
        self.criterion = nn.BCEWithLogitsLoss()
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
        self.train_loader = train_loader
        self.val_loader = val_loader

        num_triples = self.heads.size(0)
        if train_idx is not None and val_idx is not None:
            self.train_idx = train_idx.to(torch.long)
            self.val_idx = val_idx.to(torch.long)
        else:
            perm = torch.randperm(num_triples)
            split = int(num_triples * (1.0 - self.val_ratio))
            self.train_idx = perm[:split]
            self.val_idx = perm[split:]
        # NOTE: train/val index sets were only used by removed legacy paths.
        # Keep no per-fold Python set materialization to reduce CV overhead.

        ## LinkNeighborLoader
        self.train_rels_sub = self.rels[self.train_idx].clone().long()
        self.train_labels_sub = self.labels[self.train_idx].clone().float()
        self.val_rels_sub = self.rels[self.val_idx].clone().long()
        self.val_labels_sub = self.labels[self.val_idx].clone().float()


        self._init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        if "node" not in self.graph.node_types:
            raise ValueError("Train currently assumes a single node type 'node' in the graph.")
        num_nodes = int(self.graph["node"].num_nodes)
        self.num_nodes = num_nodes
        self.node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        self.cls_edge_type = CLS_EDGE_TYPE
        self._lp_eval_graph_on_device = None
        self._lp_epoch_log_path = os.path.join(self.log.dir, f"{self.log.name}_lp_epoch_metrics.csv")
        self._init_lp_epoch_log_file()
        self.val_lp_eval_every = max(1, int(val_lp_eval_every))
        self.abort_random_bce = bool(abort_random_bce)
        self.abort_random_bce_checks = max(2, int(abort_random_bce_checks))
        self.abort_random_bce_tol = float(abort_random_bce_tol)
        self.abort_random_bce_span_tol = float(abort_random_bce_span_tol)
        self.abort_random_bce_warmup = max(1, int(abort_random_bce_warmup))
        self.random_bce_baseline = 0.6931471805599453

        pos_mask_all = self.labels > 0.5
        pos_heads_all = self.heads[pos_mask_all].tolist()
        pos_rels_all = self.rels[pos_mask_all].tolist()
        pos_tails_all = self.tails[pos_mask_all].tolist()
        self.known_true_tails: Dict[Tuple[int, int], Set[int]] = {}
        self.known_true_heads: Dict[Tuple[int, int], Set[int]] = {}
        for h, r, t in zip(pos_heads_all, pos_rels_all, pos_tails_all):
            h_i = int(h)
            r_i = int(r)
            t_i = int(t)
            key_t = (h_i, r_i)
            key_h = (t_i, r_i)
            if key_t not in self.known_true_tails:
                self.known_true_tails[key_t] = set()
            if key_h not in self.known_true_heads:
                self.known_true_heads[key_h] = set()
            self.known_true_tails[key_t].add(t_i)
            self.known_true_heads[key_h].add(h_i)

        val_pos_mask = self.labels[self.val_idx] > 0.5
        self.val_pos_idx = self.val_idx[val_pos_mask]
        self.val_lp_heads = self.heads[self.val_pos_idx].clone().long()
        self.val_lp_rels = self.rels[self.val_pos_idx].clone().long()
        self.val_lp_tails = self.tails[self.val_pos_idx].clone().long()


    # def _get_link_supervision(self, batch: HeteroData):
    #     """
    #     Returns (edge_label_index, edge_label, input_id) from a LinkNeighborLoader batch.
    #     For HeteroData, these live on the edge store corresponding to the edge_type you
    #     passed into LinkNeighborLoader(edge_label_index=(edge_type, ...)).
    #     """
    #     # Hetero: supervision attrs live in an edge store:
    #     if isinstance(batch, HeteroData):
    #         for et in batch.edge_types:
    #             store = batch[et]
    #             eli = getattr(store, "edge_label_index", None)
    #             if eli is not None:
    #                 el = getattr(store, "edge_label", None)
    #                 iid = getattr(store, "input_id", None)
    #                 return eli, el, iid
    #     # Homo fallback (in case you ever switch to Data):
    #     eli = getattr(batch, "edge_label_index", None)
    #     if eli is not None:
    #         return eli, getattr(batch, "edge_label", None), getattr(batch, "input_id", None)
    #     raise RuntimeError("No edge_label_index found in batch. Are you using LinkNeighborLoader?")

    def _get_link_supervision(self, batch: HeteroData):
        if isinstance(batch, HeteroData):
            if self.cls_edge_type not in batch.edge_types:
                raise RuntimeError(f"Batch missing cls edge store {self.cls_edge_type}.")
            store = batch[self.cls_edge_type]
            return store.edge_label_index, getattr(store, "edge_label", None), getattr(store, "input_id", None)
        eli = getattr(batch, "edge_label_index", None)
        if eli is None: raise RuntimeError("No edge_label_index found in batch.")
        return eli, getattr(batch, "edge_label", None), getattr(batch, "input_id", None)


    def _maybe_switch_to_dual(self, h_dict: Dict[str, torch.Tensor], n_type: str) -> None:
        """If the encoder exposes two per-node views (e.g. SRA-HGCN), switch the
        contrastive loss to the dual-view objective. Safe to call every batch."""
        if self.contrastive_sampler is None or self.no_contrastive or self.contrastive_weight <= 0.0:
            return
        if self.dual_view:
            return
        pos_k = f"{n_type}_pos"
        neg_k = f"{n_type}_neg"
        if pos_k in h_dict and neg_k in h_dict and n_type in h_dict:
            z = h_dict[n_type]
            zp = h_dict[pos_k]
            if z.dim() == 2 and zp.dim() == 2 and z.size(1) == 2 * zp.size(1):
                self.dual_view = True
                self.contrastive_loss_fn = DualContrastiveInstanceLoss(temperature=self.contrastive_temperature)

    def _iterate_batches(self, idx: torch.Tensor, z: torch.Tensor,
        train: bool = True) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        total_loss_tensor = None
        if train: total_loss_tensor = torch.zeros((), device=self.device)

        num_triples = idx.size(0)
        if num_triples == 0: return 0.0, 0.0, None, None, total_loss_tensor
        for start in range(0, num_triples, self.batch_size):
            end = min(start + self.batch_size, num_triples)
            b_idx = idx[start:end]
            h = self.heads[b_idx].to(self.device)
            t = self.tails[b_idx].to(self.device)
            r = self.rels[b_idx].to(self.device)
            y = self.labels[b_idx].to(self.device)

            edge_index = torch.stack([h, t], dim=0)
            logits, probs = self.model.score_triples(z, edge_index, r)
            bce_loss = self.criterion(logits, y)
            loss = bce_loss
            contr_loss_val = 0.0

            if train and not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([h, t], dim=0))
                samples = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=None)
                contr_loss = self.contrastive_loss_fn(*samples)
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


    def _get_batch_triple_indices(self, batch: HeteroData, subset: str) -> torch.Tensor:
        """
        Returns the indices of triples in the given subset ('train' or 'val')
        whose head and tail are both inside the current NeighborLoader subgraph.
        This implementation avoids Python loops by using a boolean node mask.
        """
        n_id = batch["node"].n_id
        if n_id.is_cuda: n_id = n_id.cpu()

        if subset == "train": subset_idx = self.train_idx
        elif subset == "val": subset_idx = self.val_idx  
        else: raise ValueError(f"Unknown subset: {subset}")
        node_mask = self.node_mask
        node_mask[n_id] = True
        heads_sub = self.heads[subset_idx]
        tails_sub = self.tails[subset_idx]
        in_batch = node_mask[heads_sub] & node_mask[tails_sub]
        node_mask[n_id] = False
        return subset_idx[in_batch]

    @staticmethod
    def _build_global_to_local(n_id: torch.Tensor) -> Dict[int, int]:
        n_id_list = n_id.cpu().tolist()
        return {int(g): i for i, g in enumerate(n_id_list)}

    def _build_local_triple_tensors(self, triple_idx: torch.Tensor,
        n_id: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # def _train_one_epoch_with_neighbors(self, optimizer, n_type: str) -> Tuple[float, float]:
    #     assert self.train_loader is not None, "NeighborLoader not provided."
    #     self.model.train()
    #     total_bce = 0.0
    #     total_contr = 0.0
    #     total_examples = 0

    #     for batch in self.train_loader:
    #         batch = batch.to(self.device)
    #         triple_idx = self._get_batch_triple_indices(batch, subset="train")
    #         if triple_idx.numel() == 0: continue

    #         if (not self.no_contrastive and self.contrastive_sampler is not None
    #             and hasattr(self.contrastive_sampler, "prepare_batch")):
    #             self.contrastive_sampler.prepare_batch(batch)
    #         optimizer.zero_grad()
    #         h_dict = self.model.encode(batch)
    #         self._maybe_switch_to_dual(h_dict, n_type)
    #         z = h_dict[n_type]
    #         del h_dict
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
    #         optimizer.step()

    #         batch_size_eff = labels.size(0)
    #         total_bce += bce_loss.detach().cpu().item() * batch_size_eff
    #         total_contr += contr_loss_val * batch_size_eff
    #         total_examples += batch_size_eff
    #     avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
    #     avg_contr = total_contr / total_examples if total_examples > 0 else 0.0
    #     return avg_bce, avg_contr
    def _train_one_epoch_with_neighbors(self, optimizer, n_type: str) -> Tuple[float, float]:
        assert self.train_loader is not None, "LinkNeighborLoader not provided."
        self.model.train()
        t0 = time.time()
        t_data = 0.0
        t_step = 0.0
        t_encode = 0.0
        t_score = 0.0
        t_contrastive = 0.0
        t_backward = 0.0
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        total_batches = 0

        if self.contrastive_sampler is not None and hasattr(self.contrastive_sampler, "prepare_epoch"):
            self.contrastive_sampler.prepare_epoch()

        for batch in self.train_loader:
            t_batch = time.time()
            batch = batch.to(self.device)
            t_data += time.time() - t_batch
            t_step_start = time.time()

            edge_label_index, edge_label, input_id = self._get_link_supervision(batch)
            if input_id is None:
                raise RuntimeError("Batch is missing input_id; ensure you're using LinkNeighborLoader. "
                                "input_id is required to recover rel_ids.")  # :contentReference[oaicite:2]{index=2}

            # input_id may be on GPU; index CPU tensors safely:
            input_id_cpu = input_id.detach().to("cpu").long()

            # Labels: prefer what loader provides; fallback to cached subset:
            labels = (edge_label if edge_label is not None else self.train_labels_sub[input_id_cpu])
            labels = labels.to(self.device).float()

            # Relation ids come from your original triple dataset:
            rel_ids = self.train_rels_sub[input_id_cpu].to(self.device).long()

            if labels.numel() == 0:
                continue

            if (not self.no_contrastive and self.contrastive_sampler is not None
                and hasattr(self.contrastive_sampler, "prepare_batch")):
                self.contrastive_sampler.prepare_batch(batch)

            optimizer.zero_grad()

            t = time.time()
            h_dict = self.model.encode(batch)
            self._maybe_switch_to_dual(h_dict, n_type)
            z = h_dict[n_type]
            del h_dict
            t_encode += time.time() - t

            if edge_label_index.numel() > 0:
                if int(edge_label_index.max()) >= int(z.size(0)):
                    raise RuntimeError("edge_label_index seems to use GLOBAL node ids, but z is LOCAL embeddings. "
                        "You need to map endpoints via batch['node'].n_id -> local.")
            assert rel_ids.numel() == labels.numel() == edge_label_index.size(1)

            t = time.time()
            logits, probs = self.model.score_triples(z, edge_label_index, rel_ids)
            bce_loss = self.criterion(logits, labels)
            t_score += time.time() - t

            loss = bce_loss
            contr_loss_val = 0.0

            if (not self.no_contrastive and self.contrastive_sampler is not None
                and self.contrastive_weight > 0.0):
                t = time.time()
                # Sampler already deduplicates/filters anchors; avoid redundant unique() here.
                batch_nodes = edge_label_index.view(-1)
                samples = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id
                )
                contr_loss = self.contrastive_loss_fn(*samples)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())
                t_contrastive += time.time() - t

            t = time.time()
            loss.backward()
            optimizer.step()
            t_backward += time.time() - t

            bs = labels.size(0)
            total_bce += float(bce_loss.detach().cpu().item()) * bs
            total_contr += float(contr_loss_val) * bs
            total_examples += bs
            total_batches += 1
            t_step += time.time() - t_step_start

        avg_bce = total_bce / total_examples if total_examples else 0.0
        avg_contr = total_contr / total_examples if total_examples else 0.0
        print(
            f"[Train] epoch time: total={time.time() - t0:.2f}s "
            f"data={t_data:.2f}s step={t_step:.2f}s batches={total_batches}",
            flush=True,
        )
        if total_batches > 0:
            print(
                "[Train] step breakdown: "
                f"encode={t_encode:.2f}s score={t_score:.2f}s "
                f"contrastive={t_contrastive:.2f}s backward={t_backward:.2f}s",
                flush=True,
            )
        return avg_bce, avg_contr


    # def _eval_with_neighbors(self, n_type:str) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     assert self.val_loader is not None, "NeighborLoader not provided."
    #     self.model.eval()
    #     total_bce = 0.0
    #     total_contr = 0.0
    #     total_examples = 0
    #     all_probs = []
    #     all_labels = []

    #     with torch.no_grad():
    #         for batch in self.val_loader:
    #             batch = batch.to(self.device)
    #             triple_idx = self._get_batch_triple_indices(batch, subset="val")
    #             if triple_idx.numel() == 0: continue

    #             # if (not self.no_contrastive and self.contrastive_sampler is not None
    #             #     and hasattr(self.contrastive_sampler, "prepare_batch")):
    #             #     self.contrastive_sampler.prepare_batch(batch)

    #             h_dict = self.model.encode(batch)
    #             z = h_dict[n_type]

    #             edge_index_local, rel_ids, labels = self._build_local_triple_tensors(
    #                 triple_idx, batch["node"].n_id, self.device)
    #             logits, probs = self.model.score_triples(z, edge_index_local, rel_ids)
    #             bce_loss = self.criterion(logits, labels)
    #             loss = bce_loss
    #             contr_loss_val = 0.0

    #             # if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
    #             #     batch_nodes = torch.unique(torch.cat([edge_index_local[0], edge_index_local[1]], dim=0))
    #             #     z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
    #             #         z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id)
    #             #     contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
    #             #     loss = loss + self.contrastive_weight * contr_loss
    #             #     contr_loss_val = float(contr_loss.detach().cpu().item())

    #             batch_size_eff = labels.size(0)
    #             total_bce += bce_loss.detach().cpu().item() * batch_size_eff
    #             total_contr += contr_loss_val * batch_size_eff
    #             total_examples += batch_size_eff
    #             all_probs.append(probs.detach().cpu())
    #             all_labels.append(labels.detach().cpu())

    #     avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
    #     avg_contr = total_contr / total_examples if total_examples > 0 else 0.0

    #     if all_probs:
    #         all_probs = torch.cat(all_probs, dim=0)
    #         all_labels = torch.cat(all_labels, dim=0)
    #     else:
    #         all_probs = None
    #         all_labels = None
    #     return avg_bce, avg_contr, all_probs, all_labels
    def _eval_with_neighbors(self, n_type: str):
        assert self.val_loader is not None, "LinkNeighborLoader not provided."
        self.model.eval()
        t0 = time.time()
        t_data = 0.0
        t_step = 0.0
        total_bce = 0.0
        total_examples = 0
        total_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                t_batch = time.time()
                batch = batch.to(self.device)
                t_data += time.time() - t_batch
                t_step_start = time.time()

                edge_label_index, edge_label, input_id = self._get_link_supervision(batch)
                if input_id is None:
                    raise RuntimeError("Batch is missing input_id; ensure you're using LinkNeighborLoader.")

                input_id_cpu = input_id.detach().to("cpu").long()

                labels = (edge_label if edge_label is not None else self.val_labels_sub[input_id_cpu])
                labels = labels.to(self.device).float()
                rel_ids = self.val_rels_sub[input_id_cpu].to(self.device).long()

                if labels.numel() == 0:
                    continue

                h_dict = self.model.encode(batch)
                z = h_dict[n_type]

                if edge_label_index.numel() > 0:
                    if int(edge_label_index.max()) >= int(z.size(0)):
                        raise RuntimeError("edge_label_index seems to use GLOBAL node ids, but z is LOCAL embeddings. "
                            "You need to map endpoints via batch['node'].n_id -> local.")
                assert rel_ids.numel() == labels.numel() == edge_label_index.size(1)

                logits, _ = self.model.score_triples(z, edge_label_index, rel_ids)
                bce_loss = self.criterion(logits, labels)

                bs = labels.size(0)
                total_bce += float(bce_loss.detach().cpu().item()) * bs
                total_examples += bs
                total_batches += 1
                t_step += time.time() - t_step_start

        avg_bce = total_bce / total_examples if total_examples else 0.0
        print(
            f"[Val] epoch time: total={time.time() - t0:.2f}s "
            f"data={t_data:.2f}s step={t_step:.2f}s batches={total_batches}",
            flush=True,
        )
        return avg_bce, 0.0, None, None

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

    def _compute_val_lp_metrics(self, n_type: str) -> Optional[Dict[str, float]]:
        n = int(self.val_lp_heads.numel())
        if n == 0:
            return None
        self.model.eval()
        if self._lp_eval_graph_on_device is None:
            self._lp_eval_graph_on_device = self.graph.to(self.device)
        graph_eval = self._lp_eval_graph_on_device
        with torch.inference_mode():
            h_dict = self.model.encode(graph_eval)
            z = h_dict[n_type]
            del h_dict
            all_nodes = torch.arange(self.num_nodes, dtype=torch.long, device=z.device)

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
            heads = self.val_lp_heads
            rels = self.val_lp_rels
            tails = self.val_lp_tails
            for i in range(n):
                h = int(heads[i].item())
                r = int(rels[i].item())
                t = int(tails[i].item())
                rank_t = self._filtered_tail_rank(z, h, r, t, all_nodes)
                tail_rank_sum += rank_t
                tail_rr_sum += 1.0 / float(rank_t)
                tail_h1 += 1.0 if rank_t <= 1 else 0.0
                tail_h3 += 1.0 if rank_t <= 3 else 0.0
                tail_h10 += 1.0 if rank_t <= 10 else 0.0
                rank_h = self._filtered_head_rank(z, h, r, t, all_nodes)
                head_rank_sum += rank_h
                head_rr_sum += 1.0 / float(rank_h)
                head_h1 += 1.0 if rank_h <= 1 else 0.0
                head_h3 += 1.0 if rank_h <= 3 else 0.0
                head_h10 += 1.0 if rank_h <= 10 else 0.0
        n_f = float(n)
        head_mr = head_rank_sum / n_f
        head_mrr = head_rr_sum / n_f
        head_hits1 = head_h1 / n_f
        head_hits3 = head_h3 / n_f
        head_hits10 = head_h10 / n_f
        tail_mr = tail_rank_sum / n_f
        tail_mrr = tail_rr_sum / n_f
        tail_hits1 = tail_h1 / n_f
        tail_hits3 = tail_h3 / n_f
        tail_hits10 = tail_h10 / n_f
        return {
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
            "mr": 0.5 * (head_mr + tail_mr),
            "mrr": 0.5 * (head_mrr + tail_mrr),
            "hits@1": 0.5 * (head_hits1 + tail_hits1),
            "hits@3": 0.5 * (head_hits3 + tail_hits3),
            "hits@10": 0.5 * (head_hits10 + tail_hits10),
        }

    def _init_lp_epoch_log_file(self) -> None:
        header = (
            "lr,epoch,train_bce,train_contrastive,val_bce,val_lp_mrr,val_lp_hits1,"
            "val_lp_hits3,val_lp_hits10,val_score,best_val_score_so_far,is_new_best,early_stop\n"
        )
        with open(self._lp_epoch_log_path, "w", encoding="utf-8") as f:
            f.write(header)

    def _append_lp_epoch_log(
        self,
        lr: float,
        epoch: int,
        train_bce: float,
        train_contr: float,
        val_bce: float,
        val_lp_metrics: Optional[Dict[str, float]],
        val_score: float,
        best_val_score_so_far: float,
        is_new_best: bool,
        early_stop: bool,
    ) -> None:
        def _fmt_opt(v: Optional[float]) -> str:
            return "" if v is None else f"{float(v):.8f}"
        mrr = None if val_lp_metrics is None else val_lp_metrics.get("mrr")
        h1 = None if val_lp_metrics is None else val_lp_metrics.get("hits@1")
        h3 = None if val_lp_metrics is None else val_lp_metrics.get("hits@3")
        h10 = None if val_lp_metrics is None else val_lp_metrics.get("hits@10")
        row = (
            f"{float(lr):.12g},{int(epoch)},{float(train_bce):.8f},{float(train_contr):.8f},"
            f"{float(val_bce):.8f},{_fmt_opt(mrr)},{_fmt_opt(h1)},{_fmt_opt(h3)},{_fmt_opt(h10)},"
            f"{float(val_score):.8f},{float(best_val_score_so_far):.8f},{int(is_new_best)},{int(early_stop)}\n"
        )
        with open(self._lp_epoch_log_path, "a", encoding="utf-8") as f:
            f.write(row)


    def run(self):
        n_type = getattr(self.model, "n_type", "node")
        overall_best_val_score = -float("inf")
        overall_best_val_loss_proxy = float("inf")
        overall_best_epoch = -1
        overall_best_metrics = None
        overall_best_lr = None
        overall_best_state = None
        per_lr_results: Dict[float, Dict[str, object]] = {}

        use_neighbor_mode = (self.train_loader is not None) and (self.val_loader is not None)
        if not getattr(self.log, "non_verbose", False):
            self.log.log(
                f"[CV Fold] Selection mode: LP-only (ValLP every {self.val_lp_eval_every} epochs); "
                "ValLoss(BCE) is logged but not used for early stopping/model selection."
            )

        for lr in self.lr_candidates:
            self.model.load_state_dict(self._init_state)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            lr_early_stopping = EarlyStopping(patience=self.es_patience, mode="max")
            lr_val_bce_hist: List[float] = []
            lr_aborted_random_bce = False

            best_val_score_lr = -float("inf")
            best_val_loss_lr = float("inf")
            best_epoch_lr = -1
            best_metrics_lr = None
            best_state_lr = None
            if not getattr(self.log, "non_verbose", False):
                self.log.log(f"=== Starting LR sweep for lr={lr:.3g} ===")
            
            if not use_neighbor_mode:
                self.graph = self.graph.to(self.device)
                self.heads = self.heads.to(self.device)
                self.rels = self.rels.to(self.device)
                self.tails = self.tails.to(self.device)
                self.labels = self.labels.to(self.device)
            
            for epoch in range(1, self.max_epochs + 1):               
                t_epoch = time.time()
                if use_neighbor_mode: train_bce, train_contr = self._train_one_epoch_with_neighbors(
                        optimizer, n_type=n_type)
                else:
                    self.model.train()
                    if (not self.no_contrastive and self.contrastive_sampler is not None
                        and hasattr(self.contrastive_sampler, "prepare_batch")):
                        self.contrastive_sampler.prepare_batch(self.graph)
                    optimizer.zero_grad()
                    h_dict = self.model.encode(self.graph)
                    self._maybe_switch_to_dual(h_dict, n_type)
                    z_train = h_dict[n_type]

                    train_bce, train_contr, _, _, train_total_loss = self._iterate_batches(
                        self.train_idx, z_train, train=True)
                    if train_total_loss is None: break
                    train_total_loss.backward()
                    optimizer.step()

                self.model.eval()
                if use_neighbor_mode:
                    val_bce, _, _, _ = self._eval_with_neighbors(n_type=n_type)
                else:
                    with torch.no_grad():
                        h_dict_val = self.model.encode(self.graph)
                        z_val = h_dict_val[n_type]
                        val_bce, _, _, _, _ = self._iterate_batches(
                            self.val_idx, z_val, train=False)

                # print("val labels mean:", val_labels.float().mean().item())   # in Train_BestModel
                # print("val unique labels:", torch.unique(val_labels).tolist())
                # p = val_probs.detach().cpu().view(-1)
                # y = val_labels.detach().cpu().view(-1)
                # print("mean prob on val positives:", p[y==1].mean().item())
                # print("mean prob on val negatives:", p[y==0].mean().item())

                should_eval_lp = (epoch % self.val_lp_eval_every == 0)
                val_lp_metrics = self._compute_val_lp_metrics(n_type=n_type) if should_eval_lp else None
                eligible_for_selection = (val_lp_metrics is not None)
                if val_lp_metrics is not None:
                    val_score = float(val_lp_metrics["mrr"])
                    score_desc = f"ValLP(MRR)={val_score:.6f}"
                else:
                    val_score = float("nan")
                    if should_eval_lp:
                        score_desc = "ValLP(MRR)=N/A; LP-only selection skipped this epoch"
                    else:
                        score_desc = (
                            f"ValLP skipped (every {self.val_lp_eval_every} epochs); "
                            "LP-only selection skipped this epoch"
                        )

                if not getattr(self.log, "non_verbose", False):
                    msg = (f"[lr={lr:.3g}] Epoch {epoch:03d} | "
                        f"TrainLoss(BCE)={train_bce:.4f} | "
                        f"TrainLoss(Contr)={train_contr:.4f} | "
                        f"ValLoss(BCE)={val_bce:.4f} | {score_desc}")
                    if val_lp_metrics is not None:
                        msg += (f" | LP Hits@1={val_lp_metrics['hits@1']:.4f}"
                            f", Hits@3={val_lp_metrics['hits@3']:.4f}, Hits@10={val_lp_metrics['hits@10']:.4f}")
                    self.log.log(msg)

                print(
                    f"[Train] Epoch {epoch:03d} total_time={time.time() - t_epoch:.2f}s",
                    flush=True,
                )

                is_new_best = eligible_for_selection and (val_score > best_val_score_lr)
                if is_new_best:
                    best_val_score_lr = val_score
                    best_val_loss_lr = -val_score
                    best_epoch_lr = epoch
                    best_metrics_lr = val_lp_metrics
                    best_state_lr = {k:v.detach().clone() for k,v in self.model.state_dict().items()}

                should_stop = False
                val_bce_f = float(val_bce)
                lr_val_bce_hist.append(val_bce_f)
                if (
                    self.abort_random_bce
                    and epoch >= self.abort_random_bce_warmup
                    and len(lr_val_bce_hist) >= self.abort_random_bce_checks
                ):
                    win = lr_val_bce_hist[-self.abort_random_bce_checks :]
                    win_mean = sum(win) / float(len(win))
                    win_span = max(win) - min(win)
                    if (
                        abs(win_mean - self.random_bce_baseline) <= self.abort_random_bce_tol
                        and win_span <= self.abort_random_bce_span_tol
                    ):
                        lr_aborted_random_bce = True
                        should_stop = True
                        if not getattr(self.log, "non_verbose", False):
                            self.log.log(
                                f"[lr={lr:.3g}] Early-abort LR: ValLoss(BCE) flat near random "
                                f"(mean={win_mean:.6f}, span={win_span:.6f}, window={len(win)})."
                            )

                if (not should_stop) and eligible_for_selection:
                    should_stop = lr_early_stopping.step(val_score, self.model)
                self._append_lp_epoch_log(
                    lr=lr,
                    epoch=epoch,
                    train_bce=train_bce,
                    train_contr=train_contr,
                    val_bce=val_bce,
                    val_lp_metrics=val_lp_metrics,
                    val_score=val_score,
                    best_val_score_so_far=best_val_score_lr,
                    is_new_best=is_new_best,
                    early_stop=should_stop,
                )
                if should_stop:
                    if not getattr(self.log, "non_verbose", False):
                        self.log.log(f"[lr={lr:.3g}] Early stopping at epoch {epoch} "
                            f"(best val LP score so far: {best_val_score_lr:.6f}).")
                    break

            if lr_aborted_random_bce:
                if not getattr(self.log, "non_verbose", False):
                    self.log.log(
                        f"[lr={lr:.3g}] Discarded from LR selection due to random-BCE collapse."
                    )
                best_val_score_lr = -float("inf")
                best_val_loss_lr = float("inf")
                best_epoch_lr = -1
                best_metrics_lr = None
                best_state_lr = None

            per_lr_results[float(lr)] = {
                "best_val_score": float(best_val_score_lr),
                "best_val_loss": float(best_val_loss_lr),
                "best_epoch": int(best_epoch_lr),
                "best_metrics": best_metrics_lr,
                "best_state": best_state_lr,
                "aborted_random_bce": int(lr_aborted_random_bce),
            }

            if best_val_score_lr > overall_best_val_score:
                overall_best_val_score = best_val_score_lr
                overall_best_val_loss_proxy = -best_val_score_lr
                overall_best_epoch = best_epoch_lr
                overall_best_metrics = best_metrics_lr
                overall_best_lr = lr
                overall_best_state = best_state_lr

            if not getattr(self.log, "non_verbose", False):
                self.log.log(f"=== Finished LR={lr:.3g} | best_val_score={best_val_score_lr:.6f} "
                    f"at epoch {best_epoch_lr} ===")

        # if overall_best_state is not None:
        #     self.model.load_state_dict(overall_best_state)
        if not getattr(self.log, "non_verbose", False):
            self.log.log(f"[CV Fold] Best overall LR={overall_best_lr:.3g} | "
                f"Best epoch={overall_best_epoch} | "
                f"Best val LP score={overall_best_val_score:.6f}")

        return overall_best_val_loss_proxy, overall_best_epoch, overall_best_metrics, overall_best_lr, per_lr_results
