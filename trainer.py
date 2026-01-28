import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from torch_geometric.data import HeteroData
from utils import Metrics, EarlyStopping
from samplers import (NegativeInstanceSampler, PartialInstanceSampler, RandomInstanceSampler)
from losses import ContrastiveInstanceLoss, DualContrastiveInstanceLoss

import subprocess

CLS_EDGE_TYPE = ("node", "cls_link", "node")

class Train:
    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor, lr_candidates: List[float],
        epochs: int, device: torch.device, log, batch_size: int = 1024, val_ratio: float = 0.1,
        early_stopping_patience: int = 15, train_idx: Optional[torch.Tensor] = None,
        val_idx: Optional[torch.Tensor] = None, contrastive_sampler: Optional[NegativeInstanceSampler] = None,
        contrastive_weight: float = 0.1, train_loader=None, val_loader=None, no_contrastive: bool = False,
        contrastive_temperature: float = 0.5, learnable_contrastive_temperature: bool = True,
        clone_inputs: bool = False, prune_ratio: float = 0.0, prune_warmup_epochs: int = 0,
        prune_target: Optional[float] = None):

        self.model = model.to(device)
        print(model.__class__.__name__)
        self.graph = graph
        self.device = device
        self.log = log
        def _prep_tensor(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            if clone_inputs:
                return x.clone().to(dtype)
            if x.dtype != dtype:
                return x.to(dtype)
            return x

        self.heads = _prep_tensor(heads, torch.long)
        self.rels = _prep_tensor(rel_ids, torch.long)
        self.tails = _prep_tensor(tails, torch.long)
        self.labels = _prep_tensor(labels, torch.float)
        self.lr_candidates = [float(lr) for lr in lr_candidates]
        self.max_epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.val_ratio = float(val_ratio)
        self.es_patience = int(early_stopping_patience)
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics = Metrics()
        self.no_contrastive = no_contrastive
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = (float(contrastive_weight) if contrastive_sampler is not None else 0.0)
        self.learnable_contrastive_temperature = bool(learnable_contrastive_temperature)
        self.contrastive_temperature_init = float(contrastive_temperature)
        self.clone_inputs = bool(clone_inputs)
        self.prune_ratio = float(prune_ratio)
        self.prune_warmup_epochs = int(prune_warmup_epochs)
        self.prune_target = (float(prune_target) if prune_target is not None else None)

        self.dual_view = bool(getattr(self.model, "dual_view", False))
        if contrastive_sampler is not None:
            self.contrastive_loss_fn = self._make_contrastive_loss_fn(self.dual_view, self.contrastive_temperature_init)
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
        self.train_idx_set = set(self.train_idx.tolist())
        self.val_idx_set = set(self.val_idx.tolist())

        ## LinkNeighborLoader
        self.train_rels_sub = self.rels[self.train_idx].clone().long()
        self.train_labels_sub = self.labels[self.train_idx].clone().float()
        self.val_rels_sub = self.rels[self.val_idx].clone().long()
        self.val_labels_sub = self.labels[self.val_idx].clone().float()


        self._init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self._init_dual_view = self.dual_view
        self._init_contrastive_temperature = self._get_contrastive_temperature_value()
        if "node" not in self.graph.node_types:
            raise ValueError("Train currently assumes a single node type 'node' in the graph.")
        num_nodes = int(self.graph["node"].num_nodes)
        self.num_nodes = num_nodes
        self.node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.node_to_triples: List[List[int]] = [[] for _ in range(num_nodes)]
        for idx in range(num_triples):
            h = int(self.heads[idx])
            if 0 <= h < num_nodes: self.node_to_triples[h].append(idx)

        self.cls_edge_type = CLS_EDGE_TYPE

    def _make_contrastive_loss_fn(self, dual_view: bool, temperature: float):
        if dual_view:
            loss_fn = DualContrastiveInstanceLoss(
                temperature=temperature, learnable_temperature=self.learnable_contrastive_temperature
            )
        else:
            loss_fn = ContrastiveInstanceLoss(
                temperature=temperature, learnable_temperature=self.learnable_contrastive_temperature
            )
        return loss_fn.to(self.device)

    def _get_contrastive_temperature_value(self) -> Optional[float]:
        if self.contrastive_loss_fn is None:
            return None
        if hasattr(self.contrastive_loss_fn, "get_temperature_value"):
            return float(self.contrastive_loss_fn.get_temperature_value())
        if hasattr(self.contrastive_loss_fn, "temperature"):
            return float(self.contrastive_loss_fn.temperature)
        return None

    def _reset_contrastive_state(self) -> None:
        if self.contrastive_loss_fn is None:
            return
        self.dual_view = self._init_dual_view
        init_temp = self._init_contrastive_temperature
        if init_temp is None:
            init_temp = self.contrastive_temperature_init
        self.contrastive_loss_fn = self._make_contrastive_loss_fn(self.dual_view, init_temp)

    def _get_link_supervision(self, batch: HeteroData):
        if isinstance(batch, HeteroData):
            if self.cls_edge_type not in batch.edge_types:
                raise RuntimeError(f"Batch missing cls edge store {self.cls_edge_type}.")
            store = batch[self.cls_edge_type]
            return store.edge_label_index, getattr(store, "edge_label", None), getattr(store, "input_id", None)
        eli = getattr(batch, "edge_label_index", None)
        if eli is None: raise RuntimeError("No edge_label_index found in batch.")
        return eli, getattr(batch, "edge_label", None), getattr(batch, "input_id", None)


    def _maybe_switch_to_dual(self, h_dict: Dict[str, torch.Tensor], n_type: str,
        optimizer: Optional[torch.optim.Optimizer] = None) -> None:
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
                current_temp = self._get_contrastive_temperature_value()
                if current_temp is None:
                    current_temp = self.contrastive_temperature_init
                self.contrastive_loss_fn = self._make_contrastive_loss_fn(True, current_temp)
                if optimizer is not None:
                    new_params = [p for p in self.contrastive_loss_fn.parameters() if p.requires_grad]
                    if new_params:
                        optimizer.add_param_group({"params": new_params})

    def _iterate_batches(self, idx: torch.Tensor, z: torch.Tensor,
        train: bool = True) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        all_probs = []
        all_labels = []
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

    def _train_one_epoch_with_neighbors(self, optimizer, n_type: str) -> Tuple[float, float]:
        assert self.train_loader is not None, "LinkNeighborLoader not provided."
        self.model.train()
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

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

            h_dict = self.model.encode(batch)
            self._maybe_switch_to_dual(h_dict, n_type, optimizer=optimizer)
            z = h_dict[n_type]
            del h_dict

            if edge_label_index.numel() > 0:
                if int(edge_label_index.max()) >= int(z.size(0)):
                    raise RuntimeError("edge_label_index seems to use GLOBAL node ids, but z is LOCAL embeddings. "
                        "You need to map endpoints via batch['node'].n_id -> local.")
            assert rel_ids.numel() == labels.numel() == edge_label_index.size(1)

            logits, probs = self.model.score_triples(z, edge_label_index, rel_ids)
            bce_loss = self.criterion(logits, labels)

            loss = bce_loss
            contr_loss_val = 0.0

            if (not self.no_contrastive and self.contrastive_sampler is not None
                and self.contrastive_weight > 0.0):
                batch_nodes = torch.unique(edge_label_index.view(-1))
                samples = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id
                )
                contr_loss = self.contrastive_loss_fn(*samples)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_bce += float(bce_loss.detach().cpu().item()) * bs
            total_contr += float(contr_loss_val) * bs
            total_examples += bs

        avg_bce = total_bce / total_examples if total_examples else 0.0
        avg_contr = total_contr / total_examples if total_examples else 0.0
        return avg_bce, avg_contr


    def _eval_with_neighbors(self, n_type: str):
        assert self.val_loader is not None, "LinkNeighborLoader not provided."
        self.model.eval()
        total_bce = 0.0
        total_examples = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

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

                logits, probs = self.model.score_triples(z, edge_label_index, rel_ids)
                bce_loss = self.criterion(logits, labels)

                bs = labels.size(0)
                total_bce += float(bce_loss.detach().cpu().item()) * bs
                total_examples += bs
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

        avg_bce = total_bce / total_examples if total_examples else 0.0
        if all_probs:
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        else:
            all_probs, all_labels = None, None
        return avg_bce, 0.0, all_probs, all_labels


    def run(self):
        n_type = getattr(self.model, "n_type", "node")
        overall_best_val_loss = float("inf")
        overall_best_epoch = -1
        overall_best_metrics = None
        overall_best_lr = None
        overall_best_state = None
        per_lr_results: Dict[float, Dict[str, object]] = {}

        use_neighbor_mode = (self.train_loader is not None) and (self.val_loader is not None)

        for lr in self.lr_candidates:
            self.model.load_state_dict(self._init_state)
            self._reset_contrastive_state()
            params = list(self.model.parameters())
            if self.contrastive_loss_fn is not None:
                params.extend([p for p in self.contrastive_loss_fn.parameters() if p.requires_grad])
            optimizer = torch.optim.Adam(params, lr=lr)
            lr_early_stopping = EarlyStopping(patience=self.es_patience, mode="min")

            best_val_loss_lr = float("inf")
            best_epoch_lr = -1
            best_metrics_lr = None
            best_state_lr = None
            best_temperature_lr = None
            if not getattr(self.log, "non_verbose", False):
                self.log.log(f"=== Starting LR sweep for lr={lr:.3g} ===")
            
            if not use_neighbor_mode:
                self.graph = self.graph.to(self.device)
                self.heads = self.heads.to(self.device)
                self.rels = self.rels.to(self.device)
                self.tails = self.tails.to(self.device)
                self.labels = self.labels.to(self.device)
            
            for epoch in range(1, self.max_epochs + 1):               
                if use_neighbor_mode: train_bce, train_contr = self._train_one_epoch_with_neighbors(
                        optimizer, n_type=n_type)
                else:
                    self.model.train()
                    if (not self.no_contrastive and self.contrastive_sampler is not None
                        and hasattr(self.contrastive_sampler, "prepare_batch")):
                        self.contrastive_sampler.prepare_batch(self.graph)
                    optimizer.zero_grad()
                    h_dict = self.model.encode(self.graph)
                    self._maybe_switch_to_dual(h_dict, n_type, optimizer=optimizer)
                    z_train = h_dict[n_type]

                    train_bce, train_contr, _, _, train_total_loss = self._iterate_batches(
                        self.train_idx, z_train, train=True)
                    if train_total_loss is None: break
                    train_total_loss.backward()
                    optimizer.step()

                self.model.eval()
                if use_neighbor_mode:
                    val_bce, _, val_probs, val_labels = self._eval_with_neighbors(n_type=n_type)
                else:
                    with torch.no_grad():
                        h_dict_val = self.model.encode(self.graph)
                        z_val = h_dict_val[n_type]
                        val_bce, _, val_probs, val_labels, _ = self._iterate_batches(
                            self.val_idx, z_val, train=False)

                # print("val labels mean:", val_labels.float().mean().item())   # in Train_BestModel
                # print("val unique labels:", torch.unique(val_labels).tolist())
                # p = val_probs.detach().cpu().view(-1)
                # y = val_labels.detach().cpu().view(-1)
                # print("mean prob on val positives:", p[y==1].mean().item())
                # print("mean prob on val negatives:", p[y==0].mean().item())

                if val_probs is not None and val_labels is not None:
                    val_metrics = self.metrics.update_all(val_probs, val_labels)
                else:
                    val_metrics = None

                if not getattr(self.log, "non_verbose", False):
                    msg = (f"[lr={lr:.3g}] Epoch {epoch:03d} | "
                        f"TrainLoss(BCE)={train_bce:.4f} | "
                        f"TrainLoss(Contr)={train_contr:.4f} | "
                        f"ValLoss(BCE)={val_bce:.4f}")
                    if val_metrics is not None:
                        msg += " | " + ", ".join(f"{name}={val_metrics[i]:.4f}"
                            for i, name in enumerate(self.metrics.get_allnames()))
                    self.log.log(msg)

                if val_bce < best_val_loss_lr:
                    best_val_loss_lr = val_bce
                    best_epoch_lr = epoch
                    best_metrics_lr = val_metrics
                    best_state_lr = {k:v.detach().clone() for k,v in self.model.state_dict().items()}
                    best_temperature_lr = self._get_contrastive_temperature_value()

                if (self.prune_ratio > 0.0 and self.prune_target is not None
                    and self.prune_target != float("inf")
                    and epoch >= max(1, self.prune_warmup_epochs)):
                    if val_bce > self.prune_target * (1.0 + self.prune_ratio):
                        if not getattr(self.log, "non_verbose", False):
                            self.log.log(
                                f"[lr={lr:.3g}] Pruned at epoch {epoch} "
                                f"(val {val_bce:.4f} > target {self.prune_target:.4f} * "
                                f"{1.0 + self.prune_ratio:.2f})"
                            )
                        break

                if lr_early_stopping.step(val_bce, self.model):
                    if not getattr(self.log, "non_verbose", False):
                        self.log.log(f"[lr={lr:.3g}] Early stopping at epoch {epoch} "
                            f"(best val loss so far: {best_val_loss_lr:.4f}).")
                    break

            per_lr_results[float(lr)] = {
                "best_val_loss": float(best_val_loss_lr),
                "best_epoch": int(best_epoch_lr),
                "best_metrics": best_metrics_lr,
                "best_state": best_state_lr,
                "best_temperature": (float(best_temperature_lr)
                    if best_temperature_lr is not None else None),
            }

            if best_val_loss_lr < overall_best_val_loss:
                overall_best_val_loss = best_val_loss_lr
                overall_best_epoch = best_epoch_lr
                overall_best_metrics = best_metrics_lr
                overall_best_lr = lr
                overall_best_state = best_state_lr

            if not getattr(self.log, "non_verbose", False):
                self.log.log(f"=== Finished LR={lr:.3g} | best_val_loss={best_val_loss_lr:.4f} "
                    f"at epoch {best_epoch_lr} ===")

        # if overall_best_state is not None:
        #     self.model.load_state_dict(overall_best_state)
        if not getattr(self.log, "non_verbose", False):
            self.log.log(f"[CV Fold] Best overall LR={overall_best_lr:.3g} | "
                f"Best epoch={overall_best_epoch} | "
                f"Best val loss={overall_best_val_loss:.4f}")

        return overall_best_val_loss, overall_best_epoch, overall_best_metrics, overall_best_lr, per_lr_results
