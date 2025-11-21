import copy, torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from utils import Metrics, EarlyStopping, _get_pos_edge_index
from samplers import (
    NegativeStatementSampler,
    PartialStatementSampler,
    NegativeSampler,
    RandomStatementSampler,
)
from losses import DualContrastiveLoss_CE, DualContrastiveLoss_Margin


class Train:
    def __init__(self, model, epochs, train_loader, val_loader, full_graph,
        full_cvgraph, e_type, log, device, task, lrs=[0.001], gda_negs=None,
        pstatement_sampler=False, nstatement_sampler=False, rstatement_sampler=False,
        contrastive_weight=0.1, state_list=None, no_contrastive=False,
        val_edges: torch.Tensor = None, val_edge_batch_size: int = None):

        self.device = device
        self.log = log
        self.task = task
        self.epochs = epochs
        self.lrs = lrs
        self.model = model.to(self.device)
        self._init_state = copy.deepcopy(self.model.state_dict())
        self.train_loader = list(train_loader)
        self.val_loader = list(val_loader)
        self.e_type = e_type
        self.ppi_etype = getattr(self.model, "ppi_etype", None)
        self.n_type = getattr(self.model, "n_type", "node")
        self.fullcv_graph = full_cvgraph.to(self.device)
        self.val_edges = val_edges
        self.val_edge_batch_size = val_edge_batch_size
        self._setup_samplers(full_cvgraph, full_graph, state_list, gda_negs, pstatement_sampler,
            nstatement_sampler, rstatement_sampler, no_contrastive)
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.metrics = Metrics()
        header = "Setting, Epoch, " + ", ".join(self.metrics.get_names())
        self.log.log(header)



    def _setup_samplers(self, full_cvgraph, full_graph,
        state_list, gda_negs, pstatement_sampler,
        nstatement_sampler, rstatement_sampler, no_contrastive):

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
            self.neg_statement_sampler.prepare_global(full_cvgraph,pos_etype="neg_statement",
                neg_etype="pos_statement")
        elif not self.no_contrastive:
            self.neg_statement_sampler = NegativeStatementSampler(anchor_etype=self.e_type)
            self.neg_statement_sampler.prepare_global(full_cvgraph, negatives=gda_negs if self.task == "gda" else None)
        #self.neg_sampler = NegativeSampler(full_graph, edge_type=("node", self.e_type, "node"), device=self.device)
        ppi_key = next(et for et in full_graph.edge_types if et[1] == self.e_type)
        all_pos_edge_index = full_graph[ppi_key].edge_index
        self.neg_sampler = NegativeSampler(full_cvgraph, edge_type=("node", self.e_type, "node"),
            all_pos_edge_index=all_pos_edge_index)



    def _to_homogeneous_pyg(self, hetero):
        """ Convert HeteroData -> homogeneous Data and return (hom_data, offsets).
        offsets[ntype] = starting global id for that node type in concatenated x. """

        node_types = list(hetero.x_dict.keys())
        offsets = {}
        running = 0
        xs = []
        for nt in node_types:
            offsets[nt] = running
            x_nt = hetero.x_dict[nt]
            xs.append(x_nt)
            running += x_nt.size(0)
        hom_x = torch.cat(xs, dim=0) if xs else torch.empty(0, device=self.device)

        hom_edges = []
        for (src_nt, rel, dst_nt), eidx in hetero.edge_index_dict.items():
            src_off = offsets[src_nt]
            dst_off = offsets[dst_nt]
            remap = eidx.clone()
            remap[0] = remap[0] + src_off
            remap[1] = remap[1] + dst_off
            hom_edges.append(remap)
        hom_edge_index = (torch.cat(hom_edges, dim=1) if hom_edges
            else torch.empty(2, 0, dtype=torch.long, device=self.device))
        hom_edge_index, _ = add_self_loops(hom_edge_index, num_nodes=hom_x.size(0))
        hom_data = Data(x=hom_x, edge_index=hom_edge_index).to(self.device)
        return hom_data, offsets



    def _prepare_pairs_and_labels(self,graph,pos_index: torch.Tensor = None):
        """
        Build edge_index (pos+neg) and labels given:
          - graph: HeteroData to use for samplers / embedding
          - pos_index: [2, N_pos] or None
        If pos_index is None, positives are taken from the graph adjacency
        via _get_pos_edge_index (training). If provided, they are used
        directly (validation/test masked-edge setting).
        """

        if pos_index is None: pos_index = _get_pos_edge_index(self.model, self.e_type, graph)
        pos_index = pos_index.to(self.device)
        neg_src, neg_dst = self.neg_sampler.sample(pos_index.size(1))
        neg_index = torch.stack([neg_src, neg_dst], dim=0)
        edge_index = torch.cat([pos_index, neg_index], dim=1)
        labels = torch.cat([torch.ones(pos_index.size(1), device=self.device),
                torch.zeros(neg_index.size(1), device=self.device),], dim=0)
        return edge_index, labels, pos_index



    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            edge_index_pairs, labels, pos_index = self._prepare_pairs_and_labels(
                batch, pos_index=None)

            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch, pos_index)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif self.nstatement_sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif not self.no_contrastive:
                self.neg_statement_sampler.prepare_batch(batch)

            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                hom_data, offsets = self._to_homogeneous_pyg(batch)
                src = edge_index_pairs[0] + offsets[self.n_type]
                dst = edge_index_pairs[1] + offsets[self.n_type]
                mapped_pairs = torch.stack([src, dst], dim=0)
                z, out = self.model(hom_data, mapped_pairs)
            else: z, out = self.model(batch, edge_index_pairs)

            if not self.no_contrastive:
                args = (z, neg_stmt_idx) if "neg_stmt_idx" in locals() else (z,)
                z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(*args)
                loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss = self.alpha * loss_contrast + self.loss_fn(
                    out.squeeze(-1), labels)
            else:
                loss = self.loss_fn(out.squeeze(-1), labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * out.size(0)
            total_examples += out.size(0)
        avg_loss = total_loss / total_examples if total_examples else 0.0
        return avg_loss, (out, labels)



    def _val_on_graph_and_edges(self, graph, pos_edges: torch.Tensor):
        """
        Runs validation on graph with PPI edges not in the adjacency,
        using an explicit tensor of PPI edges.
        """
        total_loss = total_examples = 0
        last_out, last_labels = None, None
        pos_edges = pos_edges.to(self.device)
        num_pos = pos_edges.size(1)
        batch_size = self.val_edge_batch_size or num_pos

        with torch.no_grad():
            for start in range(0, num_pos, batch_size):
                end = min(start + batch_size, num_pos)
                pos_index = pos_edges[:, start:end]
                edge_index_pairs, labels, pos_index = self._prepare_pairs_and_labels(
                    graph, pos_index=pos_index)

                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(graph, pos_index)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif self.nstatement_sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(graph)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif not self.no_contrastive: self.neg_statement_sampler.prepare_batch(graph)

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    hom_data, offsets = self._to_homogeneous_pyg(graph)
                    src = edge_index_pairs[0] + offsets[self.n_type]
                    dst = edge_index_pairs[1] + offsets[self.n_type]
                    mapped_pairs = torch.stack([src, dst], dim=0)
                    z, out = self.model(hom_data, mapped_pairs)
                else: z, out = self.model(graph, edge_index_pairs)

                if not self.no_contrastive:
                    if self.rstatement_sampler or self.nstatement_sampler or self.pstatement_sampler:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(z, neg_stmt_idx)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    else:
                        z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(z)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    loss = self.alpha * loss_contrast + self.loss_fn(out.squeeze(-1), labels)
                else: loss = self.loss_fn(out.squeeze(-1), labels)

                total_loss += loss.item() * out.size(0)
                total_examples += out.size(0)
                last_out, last_labels = out, labels

        avg_loss = total_loss / total_examples if total_examples else 0.0
        return avg_loss, (last_out, last_labels)


    def validate_epoch(self):
        
        self.model.eval()
        if self.val_edges is not None: return self._val_on_graph_and_edges(self.fullcv_graph, self.val_edges)
        total_loss = total_examples = 0
        last_out, last_labels = None, None

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                edge_index_pairs, labels, pos_index = self._prepare_pairs_and_labels(batch)
                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch, pos_index)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif self.nstatement_sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif not self.no_contrastive:
                    self.neg_statement_sampler.prepare_batch(batch)

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    hom_data, offsets = self._to_homogeneous_pyg(batch)
                    src = edge_index_pairs[0] + offsets[self.n_type]
                    dst = edge_index_pairs[1] + offsets[self.n_type]
                    mapped_pairs = torch.stack([src, dst], dim=0)
                    z, out = self.model(hom_data, mapped_pairs)
                else: z, out = self.model(batch, edge_index_pairs)

                if not self.no_contrastive:
                    args = (z, neg_stmt_idx) if "neg_stmt_idx" in locals() else (z,)
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(*args)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    loss = self.alpha * loss_contrast + self.loss_fn(out.squeeze(-1), labels)
                else: loss = self.loss_fn(out.squeeze(-1), labels)

                total_loss += loss.item() * out.size(0)
                total_examples += out.size(0)
                last_out, last_labels = out, labels

        avg_loss = total_loss / total_examples if total_examples else 0.0
        return avg_loss, (last_out, last_labels)

    def run(self):
        best_lr = None
        best_val_loss = float("inf")
        best_metrics = None

        for lr_ in self.lrs:
            best_lr_epoch = None
            best_lr_val_loss = float("inf")
            best_lr_metrics = None

            self.model.load_state_dict(self._init_state)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_
            self.earlystopper = EarlyStopping()
            print(f"\n=== Starting sweep with LR = {lr_} ===", flush=True)

            for epoch in range(1, self.epochs + 1):
                train_loss, _ = self.train_epoch()
                val_loss, (out, lbls) = self.validate_epoch()

                print(f"LR={lr_} | Epoch {epoch}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)
                if self.earlystopper.step(val_loss, self.model):
                    print(f" -- Early stopping at epoch {epoch}", flush=True)
                    best_epoch = epoch
                    break

            if val_loss < best_lr_val_loss:  # best per LR
                best_lr_val_loss, best_lr_epoch = val_loss, epoch
                best_lr_metrics = self.metrics.update(out.detach().cpu(), lbls.cpu())
                self.log.log(f"Best Metrics: LR={lr_}, {epoch}, " + ", ".join([f"{v:.4f}" for v in best_lr_metrics]))
            if val_loss < best_val_loss:  # best overall
                best_val_loss, best_lr, best_epoch = val_loss, lr_, epoch
                best_metrics = self.metrics.update(out.detach().cpu(), lbls.cpu())
                torch.save(self.model.state_dict(), self.log.dir + "model_" + self.model.__class__.__name__ + ".pth")

        print(f"\n*** Best LR = {best_lr}, Val Loss = {best_val_loss:.4f}  ***")
        for name, val in zip(self.metrics.get_names(), best_metrics):
            print(f"{name}: {val:.4f}")
        return best_lr, best_val_loss, best_metrics, best_epoch
