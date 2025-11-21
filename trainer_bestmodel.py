import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from utils import Metrics, EarlyStopping, _get_pos_edge_index, _target_edge_key
from samplers import (
    NegativeStatementSampler,
    PartialStatementSampler,
    NegativeSampler,
    RandomStatementSampler,
)
from losses import DualContrastiveLoss_CE, DualContrastiveLoss_Margin



def _to_homogeneous_pyg(hetero, device):
    """ Converts HeteroData -> homogeneous Data and returns (hom_data, offsets). """
    node_types = list(hetero.x_dict.keys())
    offsets, xs, running = {}, [], 0
    for nt in node_types:
        offsets[nt] = running
        x_nt = hetero.x_dict[nt]
        xs.append(x_nt)
        running += x_nt.size(0)
    hom_x = torch.cat(xs, dim=0) if xs else torch.empty(0, device=device)

    hom_edges = []
    for (src_nt, _, dst_nt), eidx in hetero.edge_index_dict.items():
        remap = eidx.clone()
        remap[0] = remap[0] + offsets[src_nt]
        remap[1] = remap[1] + offsets[dst_nt]
        hom_edges.append(remap)
    if hom_edges: hom_edge_index = torch.cat(hom_edges, dim=1)
    else: hom_edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    hom_edge_index, _ = add_self_loops(hom_edge_index, num_nodes=hom_x.size(0))
    hom_data = Data(x=hom_x, edge_index=hom_edge_index).to(device)
    return hom_data, offsets


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
        val_edges: torch.Tensor = None,
        val_edge_batch_size: int = None,
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
        self.val_edges = val_edges
        self.val_edge_batch_size = val_edge_batch_size

        if self.rstatement_sampler:
            self.neg_statement_sampler = RandomStatementSampler()
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif self.nstatement__sampler:
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif self.pstatement_sampler:
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(full_cvgraph, pos_etype="neg_statement", neg_etype="pos_statement")
        elif self.no_contrastive: pass
        else:
            self.neg_statement_sampler = NegativeStatementSampler(anchor_etype=e_type)
            self.neg_statement_sampler.prepare_global(full_cvgraph, negatives=gda_negs if task == "gda" else None)

        ppi_key = next(et for et in self.full_graph.edge_types if et[1] == e_type)
        all_pos_edge_index = self.full_graph[ppi_key].edge_index

        self.neg_sampler = NegativeSampler(self.full_cvgraph, edge_type=("node", e_type, "node"),
            all_pos_edge_index=all_pos_edge_index)    
        #self.neg_sampler = NegativeSampler(full_cvgraph, edge_type=("node", e_type, "node"),device=self.device)
        
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.log = log
        self.task = task
        self.gda_negs = gda_negs
        self.epochs = epochs
        self.metrics = Metrics()
        self.log.log("Setting, Epoch, "
            + "".join(name + ", " for name in self.metrics.get_names())[:-2])

    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pos_edge_index = _get_pos_edge_index(self.model, self.e_type, batch)
            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch, pos_edge_index)
                neg_statement_index = self.neg_statement_sampler.sample()
            elif self.nstatement__sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_statement_index = self.neg_statement_sampler.sample()
            elif self.no_contrastive: pass
            else: self.neg_statement_sampler.prepare_batch(batch)

            neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([torch.ones(pos_edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)],dim=0)

            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                hom_data, offsets = _to_homogeneous_pyg(batch, self.device)
                _, n_type = _target_edge_key(self.model, self.e_type, batch)
                src = edge_index[0] + offsets[n_type]
                dst = edge_index[1] + offsets[n_type]
                mapped_pairs = torch.stack([src, dst], dim=0)
                z, out = self.model(hom_data, mapped_pairs)
            else: z, out = self.model(batch, edge_index)

            if not self.no_contrastive:
                if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(z, neg_statement_index)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                else:
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(z)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)

            loss_cls = self.loss_fn(out.squeeze(-1), labels)
            loss_total = (self.alpha * loss_contrast + loss_cls if not self.no_contrastive else loss_cls)
            loss_total.backward()
            self.optimizer.step()
            num_examples = out.size(0)
            total_loss += loss_total.item() * num_examples
            total_examples += num_examples
        if total_examples == 0: return 0, (None, None)
        return (total_loss / total_examples), (out, labels)


    def _val_on_graph_and_edges(self, graph, pos_edges: torch.Tensor):
        total_loss = total_examples = 0
        last_out, last_labels = None, None

        pos_edges = pos_edges.to(self.device)
        num_pos = pos_edges.size(1)
        batch_size = self.val_edge_batch_size or num_pos

        with torch.no_grad():
            for start in range(0, num_pos, batch_size):
                end = min(start + batch_size, num_pos)
                pos_edge_index = pos_edges[:, start:end]

                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(graph, pos_edge_index)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.nstatement__sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(graph)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.no_contrastive: pass
                else: self.neg_statement_sampler.prepare_batch(graph)

                neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                labels = torch.cat([torch.ones(pos_edge_index.size(1), device=self.device),
                        torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    hom_data, offsets = _to_homogeneous_pyg(graph, self.device)
                    _, n_type = _target_edge_key(self.model, self.e_type, graph)
                    src = edge_index[0] + offsets[n_type]
                    dst = edge_index[1] + offsets[n_type]
                    mapped_pairs = torch.stack([src, dst], dim=0)
                    z, out = self.model(hom_data, mapped_pairs)
                else: z, out = self.model(graph, edge_index)

                if not self.no_contrastive:
                    if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(z, neg_statement_index)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    else:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                            z
                        )
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss_cls = self.loss_fn(out.squeeze(-1), labels)
                loss_total = (self.alpha * loss_contrast + loss_cls if not self.no_contrastive else loss_cls)
                num_examples = out.size(0)
                total_loss += loss_total.item() * num_examples
                total_examples += num_examples
                last_out, last_labels = out, labels

        if total_examples == 0: return 0.0, (last_out, last_labels)
        return (total_loss / total_examples), (last_out, last_labels)


    def validate_epoch(self):
        if self.val_edges is not None: return self._val_on_graph_and_edges(self.full_cvgraph, self.val_edges)
        self.model.eval()
        total_loss = total_examples = 0
        last_out, last_labels = None, None
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                pos_edge_index = _get_pos_edge_index(self.model, self.e_type, batch)
                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch, pos_edge_index)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.nstatement__sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.no_contrastive: pass
                else: self.neg_statement_sampler.prepare_batch(batch)

                neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                labels = torch.cat([torch.ones(pos_edge_index.size(1), device=self.device),
                        torch.zeros(neg_edge_index.size(1), device=self.device)],dim=0)

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    hom_data, offsets = _to_homogeneous_pyg(batch, self.device)
                    _, n_type = _target_edge_key(self.model, self.e_type, batch)
                    src = edge_index[0] + offsets[n_type]
                    dst = edge_index[1] + offsets[n_type]
                    mapped_pairs = torch.stack([src, dst], dim=0)
                    z, out = self.model(hom_data, mapped_pairs)
                else: z, out = self.model(batch, edge_index)

                if not self.no_contrastive:
                    if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(
                            z, neg_statement_index)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    else:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(z)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss_cls = self.loss_fn(out.squeeze(-1), labels)
                loss_total = (self.alpha * loss_contrast + loss_cls if not self.no_contrastive else loss_cls)
                num_examples = out.size(0)
                total_loss += loss_total.item() * num_examples
                total_examples += num_examples
                last_out, last_labels = out, labels
        return (total_loss / total_examples), (last_out, last_labels)


    def run(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.best_lr)
        for epoch in range(self.epochs):
            train_loss, (out, labels) = self.train_epoch()
            if train_loss != 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}",
                    flush=True)
            if self.earlystopper.step(train_loss, self.model):
                print(f" --  Early stopping at epoch {epoch}", flush=True)
                break
        return train_loss, (out, labels)



class Test_BestModel:
    def __init__(
        self,
        model,
        test_loader,
        e_type,
        test_graph,
        full_graph,
        log,
        device,
        task,
        gda_negs=None,
        test_edges: torch.Tensor = None,
        test_edge_batch_size: int = None,
    ):
        self.model = model.to(device)
        self.test_loader = list(test_loader)
        self.e_type = e_type
        self.log = log
        self.device = device
        self.task = task
        self.gda_negs = gda_negs
        self.neg_sampler = NegativeSampler(full_graph, edge_type=("node", e_type, "node"),device=self.device)
        self.metrics = Metrics()
        self.graph = test_graph.to(device)
        self.test_edges = test_edges
        self.test_edge_batch_size = test_edge_batch_size

    def test_epoch(self):
        assert self.test_edges is not None, "test_edges [2, N_test] must be provided."

        self.model.eval()
        all_labels = []
        all_out = []

        pos_edges = self.test_edges.to(self.device)
        num_pos = pos_edges.size(1)
        batch_size = self.test_edge_batch_size or num_pos

        with torch.no_grad():
            for start in range(0, num_pos, batch_size):
                end = min(start + batch_size, num_pos)
                pos_edge_index = pos_edges[:, start:end]
                neg_src, neg_dst = self.neg_sampler.sample(pos_edge_index.size(1))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                labels = torch.cat([torch.ones(pos_edge_index.size(1), device=self.device),
                        torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
                g = self.graph
                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    hom_data, offsets = _to_homogeneous_pyg(g, self.device)
                    _, n_type = _target_edge_key(self.model, self.e_type, g)
                    src = edge_index[0] + offsets[n_type]
                    dst = edge_index[1] + offsets[n_type]
                    mapped_pairs = torch.stack([src, dst], dim=0)
                    _, out = self.model(hom_data, mapped_pairs)
                else: _, out = self.model(g, edge_index)

                all_labels.append(labels.cpu())
                all_out.append(out.cpu())

        labels = torch.cat(all_labels, dim=0)
        out = torch.cat(all_out, dim=0)
        return labels, out


    def run(self):
        labels, out = self.test_epoch()
        torch.save(out, self.log.dir + "predictions_" + self.model.__class__.__name__ + ".pth")
        torch.save(labels, self.log.dir + "labels_" + self.model.__class__.__name__ + ".pth")
        acc, f1, precision_p, recall_p, precision_n, recall_n, roc_auc = self.metrics.update_all(
            out.detach().to("cpu"), labels.to("cpu"))
        print("Test Results:", flush=True)
        print(f"Accuracy: {acc:.4f}, F1 Score (W): {f1:.4f}, Precision (+): {precision_p:.4f}, Recall (+): {recall_p:.4f}, "
            f"Precision (-): {precision_n:.4f}, Recall (-): {recall_n:.4f}, Roc Auc: {roc_auc:.4f}", flush=True)
        self.log.log("Test, final,"+str(acc)+","+str(f1)+","+str(precision_p)+","+str(recall_p)
            + ","+ str(precision_n)+ ","+ str(recall_n)+ ","+ str(roc_auc))
        self.log.close()
