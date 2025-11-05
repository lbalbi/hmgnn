import copy, torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from utils import Metrics, EarlyStopping, _get_pos_edge_index
from samplers import (NegativeStatementSampler, PartialStatementSampler, NegativeSampler, RandomStatementSampler)
from losses import DualContrastiveLoss_CE, DualContrastiveLoss_Margin


class Train:
    def __init__(
        self, model, epochs, train_loader, val_loader,
        full_cvgraph, e_type, log, device, task, lrs=[0.001],
        gda_negs=None, pstatement_sampler=False,  nstatement_sampler=False,
        rstatement_sampler=False, contrastive_weight=0.1, state_list=None,
        no_contrastive=False):
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
        self._setup_samplers(full_cvgraph, state_list, gda_negs, pstatement_sampler,
            nstatement_sampler, rstatement_sampler, no_contrastive)
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.metrics = Metrics()
        header = "Setting, Epoch, " + ", ".join(self.metrics.get_names())
        self.log.log(header)


    def _setup_samplers(self, full_cvgraph, state_list, gda_negs, pstatement_sampler,
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
            self.neg_statement_sampler.prepare_global(full_cvgraph, pos_etype="neg_statement", neg_etype="pos_statement")
        elif not self.no_contrastive:
            self.neg_statement_sampler = NegativeStatementSampler(anchor_etype=self.e_type)
            self.neg_statement_sampler.prepare_global(full_cvgraph, negatives=gda_negs if self.task == "gda" else None)
        self.neg_sampler = NegativeSampler(full_cvgraph, edge_type=("node", self.e_type, "node"))



    def _to_homogeneous_pyg(self, hetero):
        """
        Convert HeteroData -> homogeneous Data and return (hom_data, offsets).
        offsets[ntype] = starting global id for that node type in concatenated x.
        """
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
        hom_edge_index = torch.cat(hom_edges, dim=1) if hom_edges else torch.empty(
            2, 0, dtype=torch.long, device=self.device)
        hom_edge_index, _ = add_self_loops(hom_edge_index, num_nodes=hom_x.size(0))
        hom_data = Data(x=hom_x, edge_index=hom_edge_index).to(self.device)
        return hom_data, offsets

    def _prepare_pairs_and_labels(self, batch):
        """Build pair tensor to score (pos+neg) and labels, using your samplers unchanged."""
        pos_index = _get_pos_edge_index(self.model, self.e_type, batch)
        neg_src, neg_dst = self.neg_sampler.sample()
        neg_index = torch.stack([neg_src, neg_dst], dim=0)
        edge_index = torch.cat([pos_index, neg_index], dim=1)
        labels = torch.cat([torch.ones(pos_index.size(1), device=self.device),
            torch.zeros(neg_index.size(1), device=self.device)], dim=0)
        return edge_index, labels, pos_index

    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            edge_index_pairs, labels, pos_index = self._prepare_pairs_and_labels(batch)
            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch, pos_index)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif self.nstatement_sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif not self.no_contrastive: self.neg_statement_sampler.prepare_batch(batch)
            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                hom_data, offsets = self._to_homogeneous_pyg(batch)
                src = edge_index_pairs[0] + offsets[self.n_type]
                dst = edge_index_pairs[1] + offsets[self.n_type]
                mapped_pairs = torch.stack([src, dst], dim=0)
                z, out = self.model(hom_data, mapped_pairs)
            else: z, out = self.model(batch, edge_index_pairs)

            if not self.no_contrastive:
                args = (z, neg_stmt_idx) if 'neg_stmt_idx' in locals() else (z,)
                z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(*args)
                loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss = self.alpha * loss_contrast + self.loss_fn(out.squeeze(-1), labels)
            else: loss = self.loss_fn(out.squeeze(-1), labels)
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
                edge_index_pairs, labels, pos_index = self._prepare_pairs_and_labels(batch)
                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch, pos_index)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif self.nstatement_sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch)
                    neg_stmt_idx = self.neg_statement_sampler.sample()
                elif not self.no_contrastive: self.neg_statement_sampler.prepare_batch(batch)

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    hom_data, offsets = self._to_homogeneous_pyg(batch)
                    src = edge_index_pairs[0] + offsets[self.n_type]
                    dst = edge_index_pairs[1] + offsets[self.n_type]
                    mapped_pairs = torch.stack([src, dst], dim=0)
                    z, out = self.model(hom_data, mapped_pairs)
                else: z, out = self.model(batch, edge_index_pairs)

                if not self.no_contrastive:
                    args = (z, neg_stmt_idx) if 'neg_stmt_idx' in locals() else (z,)
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(*args)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    loss = self.alpha * loss_contrast + self.loss_fn(out.squeeze(-1), labels)
                else: loss = self.loss_fn(out.squeeze(-1), labels)
                total_loss += loss.item() * out.size(0)
                total_examples += out.size(0)
        avg_loss = total_loss / total_examples if total_examples else 0.0
        return avg_loss, (out, labels)

    def run(self):
        best_lr = None
        best_val_loss = float('inf')
        best_metrics  = None

        for lr_ in self.lrs:
            self.model.load_state_dict(self._init_state)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr_
                best_epoch = epoch
                best_metrics = self.metrics.update(out.detach().to("cpu"), lbls.to("cpu"))
                #torch.save(self.model.state_dict(), self.log.dir + 'model_' + self.model.__class__.__name__ + '.pth')

        print(f"\n*** Best LR = {best_lr}, Val Loss = {best_val_loss:.4f}  ***")
        for name, val in zip(self.metrics.get_names(), best_metrics):
            print(f"{name}: {val:.4f}")
        return best_lr, best_val_loss, best_metrics, best_epoch