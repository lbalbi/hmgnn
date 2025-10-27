import copy
import torch, dgl
from utils import Metrics, EarlyStopping
from samplers import (
    NegativeStatementSampler,
    PartialStatementSampler,
    NegativeSampler,
    RandomStatementSampler
)
from losses import DualContrastiveLoss_CE, DualContrastiveLoss_Margin

class Train:
    def __init__(
        self, model, optimizer, epochs,train_loader, val_loader,
        full_cvgraph, e_type,log, device, task, lrs=[0.001],
        gda_negs=None, pstatement_sampler=False,  nstatement_sampler=False,
        rstatement_sampler=False, contrastive_weight=0.1, state_list=None,
        no_contrastive=False
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
        self.e_type = e_type

        self._setup_samplers(
            full_cvgraph, state_list, gda_negs, pstatement_sampler, 
            nstatement_sampler, rstatement_sampler, no_contrastive)
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.metrics = Metrics()
        header = "Setting, Epoch, " + ", ".join(self.metrics.get_names())
        self.log.log(header)

    def _setup_samplers(
        self, full_cvgraph, state_list,  gda_negs, pstatement_sampler,
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
            self.neg_statement_sampler.prepare_global(
                full_cvgraph, pos_etype="neg_statement", neg_etype="pos_statement" )
        elif not no_contrastive:
            self.neg_statement_sampler = NegativeStatementSampler(anchor_etype=self.e_type)
            self.neg_statement_sampler.prepare_global(
                full_cvgraph, negatives = gda_negs if self.task == "gda" else None )

        self.neg_sampler = NegativeSampler(
            full_cvgraph, edge_type=("node", self.e_type, "node"))

    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            src, dst = batch.edges(etype=self.e_type)
            pos_index = torch.stack([src, dst], dim=0)

            # Prepare statement‐based negatives
            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch, pos_index)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif self.nstatement_sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_stmt_idx = self.neg_statement_sampler.sample()
            elif not self.no_contrastive: self.neg_statement_sampler.prepare_batch(batch)

            # Sample classification negatives
            neg_edge_g = self.neg_sampler.sample().to(self.device)
            neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
            neg_index = torch.stack([neg_src, neg_dst], dim=0)
            edge_index = torch.cat([pos_index, neg_index], dim=1)
            labels = torch.cat([torch.ones(pos_index.size(1),  device=self.device),
                torch.zeros(neg_index.size(1), device=self.device)
            ], dim=0)
            # Homogeneous conversion
            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()
                for ntype in batch.ntypes:
                    num = batch.num_nodes(ntype)
                    batch.nodes[ntype].data[dgl.NID] = torch.arange(num, device=batch.device)
                batch = dgl.to_homogeneous(batch, ndata=['feat', dgl.NID], store_type=True)
                batch = batch.to(self.device)
                batch = dgl.add_self_loop(batch)
                homo_nid = batch.ndata[dgl.NID]
                inv = torch.empty(batch.num_nodes(), dtype=torch.long, device=self.device)
                inv[homo_nid] = torch.arange(batch.num_nodes(), device=self.device)
                src = inv[orig_src]
                dst = inv[orig_dst]
                edge_index = torch.stack([src, dst], dim=0)
            z, out = self.model(batch, edge_index)

            # Contrastive loss if enabled
            if not self.no_contrastive:
                args = (z, neg_stmt_idx) if 'neg_stmt_idx' in locals() else (z,)
                z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(*args)
                loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss = self.alpha * loss_contrast + self.loss_fn(out.squeeze(-1), labels)
            else: loss = self.loss_fn(out.squeeze(-1), labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * out.size(0)
            total_examples+= out.size(0)

        avg_loss = total_loss / total_examples if total_examples else 0
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

                neg_edge_g = self.neg_sampler.sample().to(self.device)
                neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
                neg_index = torch.stack([neg_src, neg_dst], dim=0)

                edge_index = torch.cat([pos_index, neg_index], dim=1)
                labels = torch.cat([torch.ones(pos_index.size(1),  device=self.device),
                    torch.zeros(neg_index.size(1), device=self.device)], dim=0)

                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()
                    for ntype in batch.ntypes:
                        num = batch.num_nodes(ntype)
                        batch.nodes[ntype].data[dgl.NID] = torch.arange(num, device=batch.device)
                    batch = dgl.to_homogeneous(batch, ndata=['feat', dgl.NID], store_type=True)
                    batch = batch.to(self.device)
                    batch = dgl.add_self_loop(batch)
                    homo_nid = batch.ndata[dgl.NID]
                    inv = torch.empty(batch.num_nodes(), dtype=torch.long, device=self.device)
                    inv[homo_nid] = torch.arange(batch.num_nodes(), device=self.device)
                    src = inv[orig_src]
                    dst = inv[orig_dst]
                    edge_index = torch.stack([src, dst], dim=0)

                z, out = self.model(batch, edge_index)

                if not self.no_contrastive:
                    args = (z, neg_stmt_idx) if 'neg_stmt_idx' in locals() else (z,)
                    z_pos, z_pos_pos, z_pos_neg = self.neg_statement_sampler.get_contrastive_samples(*args)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    loss = self.alpha * loss_contrast + self.loss_fn(out.squeeze(-1), labels)
                else: loss = self.loss_fn(out.squeeze(-1), labels)

                total_loss += loss.item() * out.size(0)
                total_examples+= out.size(0)

        avg_loss = total_loss / total_examples if total_examples else 0
        return avg_loss, (out, labels)

    def run(self):
        best_lr = None
        best_val_loss = float('inf')
        best_metrics  = None

        for lr in self.lrs:
            self.model.load_state_dict(self._init_state)
            for pg in self.optimizer.param_groups: pg['lr'] = lr
            self.earlystopper = EarlyStopping()
            print(f"\n=== Starting sweep with LR = {lr} ===", flush=True)

            for epoch in range(1, self.epochs + 1):
                train_loss, _ = self.train_epoch()
                val_loss, (out, lbls) = self.validate_epoch()

                print(f"LR={lr} | Epoch {epoch}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
                    flush=True)
                if self.earlystopper.step(val_loss, self.model):
                    print(f" -- Early stopping at epoch {epoch}", flush=True)
                    break
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                best_metrics = self.metrics.update(out.detach().to("cpu"), lbls.to("cpu"))
                torch.save(self.model.state_dict(), self.log.dir + 'model_'+ self.model.__class__.__name__ +'.pth')

        print(f"\n*** Best LR = {best_lr}, Val Loss = {best_val_loss:.4f}  ***")
        for name, val in zip(self.metrics.get_names(), best_metrics):
            print(f"{name}: {val:.4f}")
        return best_lr, best_val_loss, best_metrics


