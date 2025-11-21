import copy
import torch
import dgl
from utils import Metrics, EarlyStopping
from samplers import (
    NegativeStatementSampler,
    PartialStatementSampler,
    NegativeSampler,
    RandomStatementSampler,
)
from losses import DualContrastiveLoss_CE, DualContrastiveLoss_Margin


class Train:
    def __init__(
        self,
        model,
        optimizer,
        epochs,
        train_loader,
        val_loader,
        full_graph,      # NEW: global graph (train+val+test)
        full_cvgraph,    # train-only graph for this CV fold
        e_type,
        log,
        device,
        task,
        lrs=[0.001],
        gda_negs=None,
        pstatement_sampler=False,
        nstatement_sampler=False,
        rstatement_sampler=False,
        contrastive_weight=0.1,
        state_list=None,
        no_contrastive=False,
    ):
        self.device = device
        self.log = log
        self.task = task
        self.epochs = epochs
        self.lrs = lrs
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self._init_state = copy.deepcopy(self.model.state_dict())
        self.train_loader = list(train_loader)
        self.val_loader = list(val_loader)
        self.e_type = e_type  # relation name, e.g. "PPI"
        self.full_graph = full_graph.to(self.device)
        self.full_cvgraph = full_cvgraph.to(self.device)

        self._setup_samplers(
            self.full_cvgraph,
            self.full_graph,
            state_list,
            gda_negs,
            pstatement_sampler,
            nstatement_sampler,
            rstatement_sampler,
            no_contrastive,
        )
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.metrics = Metrics()
        header = "Setting, Epoch, " + ", ".join(self.metrics.get_names())
        self.log.log(header)

    def _setup_samplers(
        self,
        full_cvgraph,
        full_graph,
        state_list,
        gda_negs,
        pstatement_sampler,
        nstatement_sampler,
        rstatement_sampler,
        no_contrastive,
    ):
        self.no_contrastive = no_contrastive
        self.rstatement_sampler = rstatement_sampler
        self.pstatement_sampler = pstatement_sampler
        self.nstatement_sampler = nstatement_sampler

        if rstatement_sampler:
            self.neg_statement_sampler = RandomStatementSampler()
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif nstatement_sampler:
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif pstatement_sampler:
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(
                full_cvgraph, pos_etype="neg_statement", neg_etype="pos_statement"
            )
        elif not no_contrastive:
            self.neg_statement_sampler = NegativeStatementSampler(anchor_etype=self.e_type)
            self.neg_statement_sampler.prepare_global(
                full_cvgraph,
                negatives=gda_negs if self.task == "gda" else None,
            )

        # All PPI positives from the *global* graph
        ppi_etype = next(et for et in full_graph.canonical_etypes if et[1] == self.e_type)
        src_all, dst_all = full_graph.edges(etype=ppi_etype)
        all_pos_edges = torch.stack([src_all, dst_all], dim=0)

        # Negative sampler uses training-only graph for source distribution,
        # but forbids any edge in the global positives.
        self.neg_sampler = NegativeSampler(
            full_cvgraph,
            edge_type=ppi_etype,
            device=self.device,
            all_pos_edges=all_pos_edges,
        )

    def _to_homogeneous(self, batch: dgl.DGLGraph, edge_index: torch.Tensor):
        """Convert heterograph -> homogeneous and remap edge_index."""
        orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()

        for ntype in batch.ntypes:
            num = batch.num_nodes(ntype)
            batch.nodes[ntype].data[dgl.NID] = torch.arange(
                num, device=batch.device, dtype=torch.long
            )

        g_h = dgl.to_homogeneous(batch, ndata=["feat", dgl.NID], store_type=True).to(
            self.device
        )
        g_h = dgl.add_self_loop(g_h)
        homo_nid = g_h.ndata[dgl.NID]
        inv = torch.empty(g_h.num_nodes(), dtype=torch.long, device=self.device)
        inv[homo_nid] = torch.arange(g_h.num_nodes(), device=self.device)
        src = inv[orig_src.to(self.device)]
        dst = inv[orig_dst.to(self.device)]
        mapped_edge_index = torch.stack([src, dst], dim=0)
        return g_h, mapped_edge_index

    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            src, dst = batch.edges(etype=self.e_type)
            pos_index = torch.stack([src, dst], dim=0)

            # Statement-based negatives (contrastive)
            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch, pos_index)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif self.nstatement_sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif not self.no_contrastive:
                self.neg_statement_sampler.prepare_batch(batch)

            # Classification negatives (no label leakage)
            neg_src, neg_dst = self.neg_sampler.sample(pos_index.size(1))
            neg_index = torch.stack([neg_src, neg_dst], dim=0)
            edge_index = torch.cat([pos_index, neg_index], dim=1)
            labels = torch.cat(
                [
                    torch.ones(pos_index.size(1), device=self.device),
                    torch.zeros(neg_index.size(1), device=self.device),
                ],
                dim=0,
            )

            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                g_h, mapped_pairs = self._to_homogeneous(batch, edge_index)
                z, out = self.model(g_h, mapped_pairs)
            else:
                z, out = self.model(batch, edge_index)

            if not self.no_contrastive:
                args = (z, neg_stmt_idx) if "neg_stmt_idx" in locals() else (z,)
                z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                    *args
                )
                loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss = self.alpha * loss_contrast + self.loss_fn(
                    out.squeeze(-1), labels
                )
            else:
                loss = self.loss_fn(out.squeeze(-1), labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * out.size(0)
            total_examples += out.size(0)

        avg_loss = total_loss / total_examples if total_examples else 0.0
        return avg_loss, (out, labels)

    def validate_epoch(self):
        self.model.eval()
        total_loss = total_examples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                src, dst = batch.edges(etype=self.e_type)
                pos_index = torch.stack([src, dst], dim=0)

                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch, pos_index)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif self.nstatement_sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif not self.no_contrastive:
                    self.neg_statement_sampler.prepare_batch(batch)

                neg_src, neg_dst = self.neg_sampler.sample(pos_index.size(1))
                neg_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([pos_index, neg_index], dim=1)
                labels = torch.cat(
                    [
                        torch.ones(pos_index.size(1), device=self.device),
                        torch.zeros(neg_index.size(1), device=self.device),
                    ],
                    dim=0,
                )

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    g_h, mapped_pairs = self._to_homogeneous(batch, edge_index)
                    z, out = self.model(g_h, mapped_pairs)
                else:
                    z, out = self.model(batch, edge_index)

                if not self.no_contrastive:
                    args = (z, neg_stmt_idx) if "neg_stmt_idx" in locals() else (z,)
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                        *args
                    )
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    loss = self.alpha * loss_contrast + self.loss_fn(
                        out.squeeze(-1), labels
                    )
                else:
                    loss = self.loss_fn(out.squeeze(-1), labels)

                total_loss += loss.item() * out.size(0)
                total_examples += out.size(0)

        avg_loss = total_loss / total_examples if total_examples else 0.0
        return avg_loss, (out, labels)

    def run(self):
        best_lr = None
        best_val_loss = float("inf")
        best_metrics = None

        for lr in self.lrs:
            self.model.load_state_dict(self._init_state)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            self.earlystopper = EarlyStopping()
            print(f"\n=== Starting sweep with LR = {lr} ===", flush=True)

            for epoch in range(1, self.epochs + 1):
                train_loss, _ = self.train_epoch()
                val_loss, (out, lbls) = self.validate_epoch()

                print(
                    f"LR={lr} | Epoch {epoch}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
                    flush=True,
                )
                if self.earlystopper.step(val_loss, self.model):
                    print(f" -- Early stopping at epoch {epoch}", flush=True)
                    break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                best_metrics = self.metrics.update(
                    out.detach().to("cpu"), lbls.to("cpu")
                )
                torch.save(
                    self.model.state_dict(),
                    self.log.dir + "model_" + self.model.__class__.__name__ + ".pth",
                )

        print(f"\n*** Best LR = {best_lr}, Val Loss = {best_val_loss:.4f}  ***")
        for name, val in zip(self.metrics.get_names(), best_metrics):
            print(f"{name}: {val:.4f}")
        return best_lr, best_val_loss, best_metrics
