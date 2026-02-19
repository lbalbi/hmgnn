import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from torch_geometric.data import HeteroData
from utils import Metrics, EarlyStopping
from samplers import (NegativeSampler, NegativeInstanceSampler_V2, 
    PartialInstanceSampler, RandomInstanceSampler)
from losses import ContrastiveLoss_CE, ContrastiveInstanceLoss, DualContrastiveInstanceLoss
import os

CLS_EDGE_TYPE = ("node", "cls_link", "node")

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
        contrastive_sampler: Optional[NegativeInstanceSampler_V2] = None,
        contrastive_weight: Optional[float] = 0.1, loader=None, no_contrastive: bool = False,
        early_stopping_patience: int = 15):
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
        # self.contrastive_loss_fn = (ContrastiveLoss_CE() if contrastive_sampler is not None else None)
        # self.contrastive_loss_fn = (ContrastiveInstanceLoss() if contrastive_sampler is not None else None)
        self.dual_view = bool(getattr(self.model, "dual_view", False))
        if contrastive_sampler is not None:
            self.contrastive_loss_fn = DualContrastiveInstanceLoss() if self.dual_view else ContrastiveInstanceLoss()
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
            for et in batch.edge_types:
                store = batch[et]
                eli = getattr(store, "edge_label_index", None)
                if eli is not None:
                    return eli, getattr(store, "edge_label", None), getattr(store, "input_id", None)
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
                self.contrastive_loss_fn = DualContrastiveInstanceLoss()

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
                batch_nodes = torch.unique(edge_label_index.view(-1))
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
                else: val_metrics = None
                if not getattr(self.log, "non_verbose", False):
                    msg = (f"[lr={lr:.3g}] Epoch {epoch:03d} | "
                        f"TrainLoss(BCE)={bce_loss:.4f} | "
                        f"TrainLoss(Contr)={contr_loss:.4f} | "
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
                else:
                    print(f"Training Epoch: {epoch} completed.", flush=True)
                    break

            last_bce_loss = bce_loss
            last_contr_loss = contr_loss
            if not getattr(self.log, "non_verbose", False):
                msg = f"[Final Train] Epoch {epoch:03d} | BCE_Loss={bce_loss:.4f}"
                if not self.no_contrastive and self.contrastive_sampler is not None and self.contrastive_weight > 0.0:
                    msg += f" | ConstrLoss={contr_loss:.4f}"
                self.log.log(msg)
        return last_bce_loss


class Test_BestModel:
    """Test-time evaluation on a global triple-level test set.
       Positives: provided explicitly as (heads, rel_ids, tails).
       Negatives: sampled per relation using existing NegativeSampler.
    """

    def __init__(self, model: nn.Module, graph: HeteroData, test_heads: torch.Tensor,
        test_rels: torch.Tensor, test_tails: torch.Tensor, id2rel: Dict[int, str],
        neg_samplers: Dict[str, NegativeSampler], num_neg_per_pos: int, 
        device: torch.device, log, batch_size: int = 1024,
        neg_cache_path: Optional[str] = None, force_regen_negs: bool = True,
        save_embeddings_path: Optional[str] = None):
        self.model = model.to(device)
        self.graph = graph
        self.test_heads = test_heads
        self.test_rels = test_rels
        self.test_tails = test_tails
        self.id2rel = id2rel
        self.neg_samplers = neg_samplers
        self.num_neg_per_pos = int(num_neg_per_pos)
        self.neg_cache_path = neg_cache_path
        self.force_regen_negs = force_regen_negs
        self.save_embeddings_path = save_embeddings_path
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


    def confusion_matrix_binary(self, probs: torch.Tensor,
                                labels: torch.Tensor,
                                threshold: float = 0.5) -> torch.Tensor:
        """
        Returns confusion matrix:
            [[TN, FP],
            [FN, TP]]
        """
        probs = probs.detach().view(-1).cpu()
        labels = labels.detach().view(-1).cpu().to(torch.long)
        preds = (probs >= threshold).to(torch.long)
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tp = ((preds == 1) & (labels == 1)).sum()
        return torch.stack([torch.stack([tn, fp]), torch.stack([fn, tp])])


    def _score_triples_in_batches(self, z: torch.Tensor, heads: torch.Tensor,
        rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """
        Computes probabilities for triples in mini-batches to save memory.
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
                device = z.device
                edge_index = edge_index.to(device)
                r = r.to(device)
                _, probs = self.model.score_triples(z, edge_index, r)

                all_probs.append(probs.detach().cpu())
        return torch.cat(all_probs, dim=0) if all_probs else torch.empty(0)

    def _sample_negatives_excluding_test(self, sampler: NegativeSampler, rel_name: str,
        num_to_sample: int) -> torch.Tensor:
        """
        Uses NegativeSampler to sample candidate negatives, then filter out any
        edges that coincide with test positives for "rel_name".
        Returns edge_index on the sampler's device.
        """
        invalid_ids = self.test_pos_ids_per_rel.get(rel_name, set())
        if not invalid_ids: return sampler.sample_edge_index(num_to_sample)

        device = sampler.device
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
            keep_mask = torch.tensor(keep_mask_list, dtype=torch.bool, device=device)
            kept_edges = cand_edge_index[:, keep_mask]
            if kept_edges.numel() == 0: continue
            collected.append(kept_edges)
            remaining = num_to_sample - sum(c.size(1) for c in collected)
        if not collected: return torch.empty(2, 0, dtype=torch.long, device=device)

        neg_edge_index = torch.cat(collected, dim=1)
        if neg_edge_index.size(1) > num_to_sample:neg_edge_index = neg_edge_index[:, :num_to_sample]
        return neg_edge_index

    # def run(self):
    #     self.graph = self.graph.to(self.device)
    #     self.test_heads = self.test_heads.to(self.device)
    #     self.test_rels = self.test_rels.to(self.device)
    #     self.test_tails = self.test_tails.to(self.device)
    #     # passed to cpu because of RA-HGCN
    #     self.model = self.model.cpu()
    #     graph_cpu = self.graph.cpu()
        
    #     cached = None
    #     if (self.neg_cache_path is not None) and (not self.force_regen_negs):
    #         cached = maybe_load_neg_cache(self.neg_cache_path, map_location="cpu")

    #     self.model.eval()
    #     # with torch.no_grad():
    #     with torch.inference_mode():
    #         h_dict = self.model.encode(graph_cpu)
    #         z = h_dict[getattr(self.model, "n_type", "node")]
    #         del h_dict
    #         pos_probs = self._score_triples_in_batches(
    #             z, self.test_heads, self.test_rels, self.test_tails)
    #         pos_labels = torch.ones_like(pos_probs)

    #         if cached is not None:
    #             neg_heads = cached["neg_heads"].to(self.device)
    #             neg_rels  = cached["neg_rels"].to(self.device)
    #             neg_tails = cached["neg_tails"].to(self.device)
    #         else:
    #             neg_heads_list,neg_rels_list,neg_tails_list = [], [], []

    #         unique_rels = torch.unique(self.test_rels)
    #         for rel_id in unique_rels.tolist():
    #             rel_name = self.id2rel[rel_id]
    #             sampler = self.neg_samplers.get(rel_name, None)
    #             if sampler is None: continue
    #             # group by head within this relation
    #             mask = (self.test_rels == rel_id)
    #             h_rel = self.test_heads[mask]
    #             if h_rel.numel() == 0:  continue

    #             uniq_h, pos_counts_per_h = torch.unique(h_rel, return_counts=True)
    #             neg_counts_per_h = pos_counts_per_h * self.num_neg_per_pos
    #             extra_invalid = self.test_pos_ids_per_rel.get(rel_name, set())
    #             neg_src, neg_dst = sampler.sample_for_heads(
    #                 uniq_h.to(sampler.device),neg_counts_per_h.to(sampler.device),
    #                 extra_invalid_ids=extra_invalid)

    #             if neg_src.numel() == 0: continue
    #             neg_src = neg_src.to(self.device)
    #             neg_dst = neg_dst.to(self.device)
    #             r_neg = torch.full((neg_src.size(0),), rel_id, dtype=torch.long, device=self.device)

    #             neg_heads_list.append(neg_src)
    #             neg_tails_list.append(neg_dst)
    #             neg_rels_list.append(r_neg)


    #         # unique_rels, counts = torch.unique(self.test_rels, return_counts=True)
    #         # for rel_id, count in zip(unique_rels.tolist(), counts.tolist()):
    #         #     rel_name = self.id2rel[rel_id]
    #         #     sampler = self.neg_samplers.get(rel_name, None)
    #         #     if sampler is None: continue

    #         #     num_to_sample = count * self.num_neg_per_pos
    #         #     neg_edge_index = self._sample_negatives_excluding_test(
    #         #         sampler, rel_name, num_to_sample)

    #         #     if neg_edge_index.numel() == 0: continue
    #         #     neg_edge_index = neg_edge_index.to(self.device)
    #         #     h_neg = neg_edge_index[0]
    #         #     t_neg = neg_edge_index[1]
    #         #     r_neg = torch.full((h_neg.size(0),), rel_id,
    #         #         dtype=torch.long, device=self.device)
    #         #     neg_heads_list.append(h_neg)
    #         #     neg_tails_list.append(t_neg)
    #         #     neg_rels_list.append(r_neg)

    #         if neg_heads_list:
    #             neg_heads = torch.cat(neg_heads_list, dim=0)
    #             neg_tails = torch.cat(neg_tails_list, dim=0)
    #             neg_rels = torch.cat(neg_rels_list, dim=0)
    #             neg_probs = self._score_triples_in_batches(
    #                 z, neg_heads, neg_rels, neg_tails)
    #             neg_labels = torch.zeros_like(neg_probs)
    #             all_probs = torch.cat([pos_probs, neg_probs], dim=0)
    #             all_labels = torch.cat([pos_labels, neg_labels], dim=0)
    #         else:
    #             all_probs = pos_probs
    #             all_labels = pos_labels


    #         metrics = self.metrics.update_all(all_probs, all_labels)
    #         names = self.metrics.get_allnames()
    #         self.log.log("=== Test metrics (global) ===")
    #         for name, val in zip(names, metrics):
    #             self.log.log(f"{name}: {val:.4f}")
    #     return metrics


    def run(self):
        self.graph = self.graph.to(self.device)
        self.test_heads = self.test_heads.to(self.device)
        self.test_rels  = self.test_rels.to(self.device)
        self.test_tails = self.test_tails.to(self.device)
        # RA-HGCN path: encode on CPU
        self.model = self.model.cpu()
        graph_cpu = self.graph.cpu()

        cached = None
        if (self.neg_cache_path is not None) and (not self.force_regen_negs):
            cached = maybe_load_neg_cache(self.neg_cache_path, map_location="cpu")

        with torch.inference_mode():
            h_dict = self.model.encode(graph_cpu)
            z = h_dict[getattr(self.model, "n_type", "node")]
            del h_dict
            if self.save_embeddings_path:
                try:
                    torch.save({"embeddings": z.detach().cpu()}, self.save_embeddings_path)
                    print(f"[INFO] Saved test-time embeddings: {self.save_embeddings_path}")
                except Exception as e: print(f"[WARN] Failed to save test-time embeddings: {e}")

            pos_probs = self._score_triples_in_batches(
                z, self.test_heads, self.test_rels, self.test_tails)
            pos_labels = torch.ones_like(pos_probs)
            if cached is not None:
                neg_heads = cached["neg_heads"].to(self.device)
                neg_rels  = cached["neg_rels"].to(self.device)
                neg_tails = cached["neg_tails"].to(self.device)
            else:
                neg_heads_list, neg_rels_list, neg_tails_list = [], [], []
                unique_rels = torch.unique(self.test_rels)
                for rel_id in unique_rels.tolist():
                    rel_name = self.id2rel[rel_id]
                    sampler = self.neg_samplers.get(rel_name, None)
                    if sampler is None:
                        continue

                    mask = (self.test_rels == rel_id)
                    h_rel = self.test_heads[mask]
                    if h_rel.numel() == 0:
                        continue

                    uniq_h, pos_counts_per_h = torch.unique(h_rel, return_counts=True)
                    neg_counts_per_h = pos_counts_per_h * self.num_neg_per_pos

                    extra_invalid = self.test_pos_ids_per_rel.get(rel_name, set())
                    neg_src, neg_dst = sampler.sample_for_heads(
                        uniq_h.to(sampler.device),
                        neg_counts_per_h.to(sampler.device),
                        extra_invalid_ids=extra_invalid
                    )

                    requested = int(neg_counts_per_h.sum().item())
                    got = int(neg_src.numel())
                    if got < requested:
                        print(f"[WARN] rel={rel_name} requested_negs={requested} got={got} "
                            f"({got/requested:.2%} of target)")

                    if neg_src.numel() == 0: continue
                    neg_src = neg_src.to(self.device)
                    neg_dst = neg_dst.to(self.device)
                    r_neg = torch.full((neg_src.size(0),), rel_id,
                                    dtype=torch.long, device=self.device)

                    neg_heads_list.append(neg_src)
                    neg_tails_list.append(neg_dst)
                    neg_rels_list.append(r_neg)

                if neg_heads_list:
                    neg_heads = torch.cat(neg_heads_list, dim=0)
                    neg_tails = torch.cat(neg_tails_list, dim=0)
                    neg_rels  = torch.cat(neg_rels_list,  dim=0)
                else:
                    neg_heads = torch.empty(0, dtype=torch.long, device=self.device)
                    neg_tails = torch.empty(0, dtype=torch.long, device=self.device)
                    neg_rels  = torch.empty(0, dtype=torch.long, device=self.device)

                # --- save cache
                if self.neg_cache_path is not None:
                    payload = {
                        "neg_heads": neg_heads.detach().cpu(),
                        "neg_rels":  neg_rels.detach().cpu(),
                        "neg_tails": neg_tails.detach().cpu(),
                        "meta": {
                            "num_neg_per_pos": int(self.num_neg_per_pos),
                            "num_nodes": int(self.num_nodes)}}
                    atomic_torch_save(payload, self.neg_cache_path)


            if neg_heads.numel() > 0:
                neg_probs = self._score_triples_in_batches(z, neg_heads, neg_rels, neg_tails)
                neg_labels = torch.zeros_like(neg_probs)
                all_probs  = torch.cat([pos_probs, neg_probs], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)
            else: all_probs, all_labels = pos_probs, pos_labels

            p = all_probs.detach().cpu().view(-1)
            y = all_labels.detach().cpu().view(-1)

            # # --- Confusion matrix (global) ---
            # cm = self.confusion_matrix_binary(all_probs, all_labels, threshold=0.5)
            # tn, fp = cm[0].tolist()
            # fn, tp = cm[1].tolist()
            # self.log.log("=== Confusion Matrix @thr=0.5 ===")
            # self.log.log(f"TN={tn}  FP={fp}")
            # self.log.log(f"FN={fn}  TP={tp}")
            # self.log.log(f"Matrix:\n{cm}")

            metrics = self.metrics.update_all(all_probs, all_labels)
            names = self.metrics.get_allnames()
            self.log.log("=== Test metrics (global) ===")
            for name, val in zip(names, metrics):
                self.log.log(f"{name}: {val:.4f}")
            return metrics
