import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from torch_geometric.data import HeteroData
from utils import Metrics, EarlyStopping
from samplers import NegativeStatementSampler
from losses import ContrastiveLoss_CE


class Train:
    def __init__(self, model: nn.Module, graph: HeteroData, heads: torch.Tensor,
        rel_ids: torch.Tensor, tails: torch.Tensor, labels: torch.Tensor, lr_candidates: List[float],
        epochs: int, device: torch.device, log, batch_size: int = 1024, val_ratio: float = 0.1,
        early_stopping_patience: int = 15, train_idx: Optional[torch.Tensor] = None,
        val_idx: Optional[torch.Tensor] = None, contrastive_sampler: Optional[NegativeStatementSampler] = None,
        contrastive_weight: float = 0.1, train_loader=None, val_loader=None, no_contrastive: bool = False):

        self.model = model.to(device)
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
        self.metrics = Metrics()
        self.no_contrastive = no_contrastive
        self.contrastive_sampler = contrastive_sampler
        self.contrastive_weight = (float(contrastive_weight) if contrastive_sampler is not None else 0.0)
        self.contrastive_loss_fn = (ContrastiveLoss_CE() if contrastive_sampler is not None else None)
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

        self._init_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        if "node" not in self.graph.node_types:
            raise ValueError("Train currently assumes a single node type 'node' in the graph.")
        num_nodes = int(self.graph["node"].num_nodes)
        self.num_nodes = num_nodes
        self.node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.node_to_triples: List[List[int]] = [[] for _ in range(num_nodes)]
        for idx in range(num_triples):
            h = int(self.heads[idx])
            if 0 <= h < num_nodes: self.node_to_triples[h].append(idx)


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
                z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=None)
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
        assert self.train_loader is not None, "NeighborLoader not provided."
        self.model.train()
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            triple_idx = self._get_batch_triple_indices(batch, subset="train")
            if triple_idx.numel() == 0: continue

            if (not self.no_contrastive and self.contrastive_sampler is not None
                and hasattr(self.contrastive_sampler, "prepare_batch")):
                self.contrastive_sampler.prepare_batch(batch)

            optimizer.zero_grad()
            h_dict = self.model.encode(batch)
            z = h_dict[n_type]

            edge_index_local, rel_ids, labels = self._build_local_triple_tensors(
                triple_idx, batch["node"].n_id, self.device)
            logits, probs = self.model.score_triples(z, edge_index_local, rel_ids)
            bce_loss = self.criterion(logits, labels)

            loss = bce_loss
            contr_loss_val = 0.0

            if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                batch_nodes = torch.unique(torch.cat([edge_index_local[0], edge_index_local[1]], dim=0))
                z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
                    z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id)
                contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                loss = loss + self.contrastive_weight * contr_loss
                contr_loss_val = float(contr_loss.detach().cpu().item())
            loss.backward()
            optimizer.step()

            batch_size_eff = labels.size(0)
            total_bce += bce_loss.detach().cpu().item() * batch_size_eff
            total_contr += contr_loss_val * batch_size_eff
            total_examples += batch_size_eff
        avg_bce = total_bce / total_examples if total_examples > 0 else 0.0
        avg_contr = total_contr / total_examples if total_examples > 0 else 0.0
        return avg_bce, avg_contr


    def _eval_with_neighbors(self, n_type:str) -> Tuple[float, float, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self.val_loader is not None, "NeighborLoader not provided."
        self.model.eval()
        total_bce = 0.0
        total_contr = 0.0
        total_examples = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                triple_idx = self._get_batch_triple_indices(batch, subset="val")
                if triple_idx.numel() == 0: continue

                if (not self.no_contrastive and self.contrastive_sampler is not None
                    and hasattr(self.contrastive_sampler, "prepare_batch")):
                    self.contrastive_sampler.prepare_batch(batch)

                h_dict = self.model.encode(batch)
                z = h_dict[n_type]

                edge_index_local, rel_ids, labels = self._build_local_triple_tensors(
                    triple_idx, batch["node"].n_id, self.device)
                logits, probs = self.model.score_triples(z, edge_index_local, rel_ids)
                bce_loss = self.criterion(logits, labels)
                loss = bce_loss
                contr_loss_val = 0.0

                if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    batch_nodes = torch.unique(torch.cat([edge_index_local[0], edge_index_local[1]], dim=0))
                    z_pos, z_pos_pos, z_pos_neg = self.contrastive_sampler.get_contrastive_samples(
                        z, anchor_nodes=batch_nodes, n_id=batch["node"].n_id)
                    contr_loss = self.contrastive_loss_fn(z_pos, z_pos_pos, z_pos_neg)
                    loss = loss + self.contrastive_weight * contr_loss
                    contr_loss_val = float(contr_loss.detach().cpu().item())

                batch_size_eff = labels.size(0)
                total_bce += bce_loss.detach().cpu().item() * batch_size_eff
                total_contr += contr_loss_val * batch_size_eff
                total_examples += batch_size_eff
                all_probs.append(probs.detach().cpu())
                all_labels.append(labels.detach().cpu())

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
        n_type = getattr(self.model, "n_type", "node")
        overall_best_val_loss = float("inf")
        overall_best_epoch = -1
        overall_best_metrics = None
        overall_best_lr = None
        overall_best_state = None

        use_neighbor_mode = (self.train_loader is not None) and (self.val_loader is not None)

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

                if val_probs is not None and val_labels is not None:
                    val_metrics = self.metrics.update(val_probs, val_labels)
                else:
                    val_metrics = None

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
                    best_state_lr = {k:v.detach().clone() for k,v in self.model.state_dict().items()}

                if lr_early_stopping.step(val_bce, self.model):
                    if not getattr(self.log, "non_verbose", False):
                        self.log.log(f"[lr={lr:.3g}] Early stopping at epoch {epoch} "
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

        # if overall_best_state is not None:
        #     self.model.load_state_dict(overall_best_state)

        if not getattr(self.log, "non_verbose", False):
            self.log.log(f"[CV Fold] Best overall LR={overall_best_lr:.3g} | "
                f"Best epoch={overall_best_epoch} | "
                f"Best val loss={overall_best_val_loss:.4f}")
        return overall_best_val_loss, overall_best_epoch, overall_best_metrics, overall_best_lr
