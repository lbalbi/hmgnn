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


class Train_BestModel:
    def __init__(
        self,
        model,
        epochs,
        train_loader,
        val_loader,
        full_cvgraph,
        full_graph,
        e_type,
        log,
        device,
        task,
        lr=0.001,
        mcfg=None,
        cfg=None,
        pstatement_sampler=False,
        nstatement_sampler=False,
        rstatement_sampler=False,
        contrastive_weight=0.1,
        gda_negs=None,
        state_list=None,
        no_contrastive=False,
    ):
        self.model = model.to(device)
        self.best_lr = lr
        self.train_loader = list(train_loader)
        self.val_loader = list(val_loader)
        self.e_type = e_type
        self.rstatement_sampler = rstatement_sampler
        self.pstatement_sampler = pstatement_sampler
        self.nstatement__sampler = nstatement_sampler
        self.no_contrastive = no_contrastive
        self.device = device
        self.full_cvgraph = full_cvgraph.to(device)
        self.full_graph = full_graph.to(device)

        if self.rstatement_sampler:
            self.neg_statement_sampler = RandomStatementSampler()
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif self.nstatement__sampler:
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif self.pstatement_sampler:
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(
                full_cvgraph, pos_etype="neg_statement", neg_etype="pos_statement"
            )
        elif self.no_contrastive:
            pass
        else:
            self.neg_statement_sampler = NegativeStatementSampler(anchor_etype=e_type)
            self.neg_statement_sampler.prepare_global(
                full_cvgraph, negatives=gda_negs if task == "gda" else None
            )

        # Global positives
        ppi_etype = next(et for et in self.full_graph.canonical_etypes if et[1] == e_type)
        src_all, dst_all = self.full_graph.edges(etype=ppi_etype)
        all_pos_edges = torch.stack([src_all, dst_all], dim=0)

        # Negatives from train+val graph; forbid any global positive
        self.neg_sampler = NegativeSampler(
            self.full_cvgraph,
            edge_type=ppi_etype,
            device=self.device,
            all_pos_edges=all_pos_edges,
        )

        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.log = log
        self.task = task
        self.gda_negs = gda_negs
        self.epochs = epochs
        self.metrics = Metrics()
        self.log.log(
            "Setting, Epoch, "
            + "".join(name + ", " for name in self.metrics.get_names())[:-2]
        )

    def _to_homogeneous(self, batch: dgl.DGLGraph, edge_index: torch.Tensor):
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
            pos_edge_index = torch.stack([src, dst], dim=0)

            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch, pos_edge_index)
                neg_statement_index = self.neg_statement_sampler.sample()
            elif self.nstatement__sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_statement_index = self.neg_statement_sampler.sample()
            elif self.no_contrastive:
                pass
            else:
                self.neg_statement_sampler.prepare_batch(batch)

            neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat(
                [
                    torch.ones(pos_edge_index.size(1), device=self.device),
                    torch.zeros(neg_edge_index.size(1), device=self.device),
                ],
                dim=0,
            )

            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                g_h, mapped_pairs = self._to_homogeneous(batch, edge_index)
                z, out = self.model(g_h, mapped_pairs)
            else:
                z, out = self.model(batch, edge_index)

            if not self.no_contrastive:
                if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                        z, neg_statement_index
                    )
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                else:
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                        z
                    )
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
            loss_cls = self.loss_fn(out.squeeze(-1), labels)
            loss_total = self.alpha * loss_contrast + loss_cls if not self.no_contrastive else loss_cls

            loss_total.backward()
            self.optimizer.step()
            num_examples = out.size(0)
            total_loss += loss_total.item() * num_examples
            total_examples += num_examples

        if total_examples == 0:
            return 0, (None, None)
        return (total_loss / total_examples), (out, labels)

    def validate_epoch(self):
        self.model.eval()
        total_loss = total_examples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                src, dst = batch.edges(etype=self.e_type)
                pos_edge_index = torch.stack([src, dst], dim=0)

                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch, pos_edge_index)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.nstatement__sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.no_contrastive:
                    pass
                else:
                    self.neg_statement_sampler.prepare_batch(batch)

                neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                labels = torch.cat(
                    [
                        torch.ones(pos_edge_index.size(1), device=self.device),
                        torch.zeros(neg_edge_index.size(1), device=self.device),
                    ],
                    dim=0,
                )

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    g_h, mapped_pairs = self._to_homogeneous(batch, edge_index)
                    z, out = self.model(g_h, mapped_pairs)
                else:
                    z, out = self.model(batch, edge_index)

                if not self.no_contrastive:
                    if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                            z, neg_statement_index
                        )
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    else:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                            z
                        )
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss_cls = self.loss_fn(out.squeeze(-1), labels)
                loss_total = self.alpha * loss_contrast + loss_cls if not self.no_contrastive else loss_cls
                num_examples = out.size(0)
                total_loss += loss_total.item() * num_examples
                total_examples += num_examples

        if total_examples == 0:
            return 0.0, (out, labels)
        return (total_loss / total_examples), (out, labels)

    def run(self):
        self.model.load_state_dict(
            torch.load(self.log.dir + "model_" + self.model.__class__.__name__ + ".pth")
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.best_lr)
        for epoch in range(self.epochs):
            train_loss, (out, labels) = self.train_epoch()
            if train_loss != 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}", flush=True)
            if self.earlystopper.step(train_loss, self.model):
                print(f" --  Early stopping at epoch {epoch}", flush=True)
                break
        return train_loss, (out, labels)



class Test_BestModel:
    def __init__(
        self,
        model,
        test_loader,          # kept for signature compatibility, not actually used
        e_type,
        test_graph,           # graph used for message passing (train+val PPIs only)
        full_graph,           # global graph with all positives (for invalid negative set)
        log,
        device,
        task,
        gda_negs=None,
        test_edges: torch.Tensor = None,      # [2, N_test]
        test_edge_batch_size: int = None,
    ):
        self.model = model.to(device)
        self.test_loader = list(test_loader)
        self.e_type = e_type
        self.log = log
        self.device = device
        self.task = task
        self.gda_negs = gda_negs
        self.metrics = Metrics()

        self.graph = test_graph.to(device)
        self.full_graph = full_graph.to(device)
        self.test_edges = test_edges
        self.test_edge_batch_size = test_edge_batch_size

        # Build global positives and negative sampler
        ppi_etype = next(et for et in self.full_graph.canonical_etypes if et[1] == e_type)
        src_all, dst_all = self.full_graph.edges(etype=ppi_etype)
        all_pos_edges = torch.stack([src_all, dst_all], dim=0)

        # Negative sampler uses test_graph (train+val edges) for src distribution,
        # but forbids any global positive.
        self.neg_sampler = NegativeSampler(
            self.graph,
            edge_type=ppi_etype,
            device=self.device,
            all_pos_edges=all_pos_edges,
        )

    def _to_homogeneous(self, g: dgl.DGLGraph, edge_index: torch.Tensor):
        orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()
        for ntype in g.ntypes:
            num = g.num_nodes(ntype)
            g.nodes[ntype].data[dgl.NID] = torch.arange(
                num, device=g.device, dtype=torch.long
            )
        g_h = dgl.to_homogeneous(g, ndata=["feat", dgl.NID], store_type=True).to(
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

    def test_epoch(self):
        assert self.test_edges is not None, "test_edges [2, N_test] must be provided."

        self.model.eval()
        all_labels = []
        all_out = []

        pos_edges = self.test_edges.to(self.device)
        num_pos = pos_edges.size(1)
        batch_size = self.test_edge_batch_size or num_pos

        g = self.graph

        with torch.no_grad():
            for start in range(0, num_pos, batch_size):
                end = min(start + batch_size, num_pos)
                pos_edge_index = pos_edges[:, start:end]  # [2, B]

                # Negatives (no leakage)
                neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                labels = torch.cat(
                    [
                        torch.ones(pos_edge_index.size(1), device=self.device),
                        torch.zeros(neg_edge_index.size(1), device=self.device),
                    ],
                    dim=0,
                )

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    g_h, mapped_pairs = self._to_homogeneous(g, edge_index)
                    _, out = self.model(g_h, mapped_pairs)
                else:
                    _, out = self.model(g, edge_index)

                all_labels.append(labels.cpu())
                all_out.append(out.cpu())

        labels = torch.cat(all_labels, dim=0)
        out = torch.cat(all_out, dim=0)
        return labels, out

    def run(self):
        labels, out = self.test_epoch()
        torch.save(
            out,
            self.log.dir + "predictions_" + self.model.__class__.__name__ + ".pth",
        )
        torch.save(
            labels, self.log.dir + "labels_" + self.model.__class__.__name__ + ".pth"
        )

        acc, f1, precision_p, recall_p, precision_n, recall_n, roc_auc = \
            self.metrics.update_all(out.detach().to("cpu"), labels.to("cpu"))
        print("Test Results:", flush=True)
        print(
            f"Accuracy: {acc:.4f}, F1 Score (W): {f1:.4f}, Precision (+): {precision_p:.4f}, "
            f"Recall (+): {recall_p:.4f}, Precision (-): {precision_n:.4f}, "
            f"Recall (-): {recall_n:.4f}, Roc Auc: {roc_auc:.4f}",
            flush=True,
        )
        self.log.log(
            "Test, final,"
            + str(acc)
            + ","
            + str(f1)
            + ","
            + str(precision_p)
            + ","
            + str(recall_p)
            + ","
            + str(precision_n)
            + ","
            + str(recall_n)
            + ","
            + str(roc_auc)
        )
        self.log.close()
