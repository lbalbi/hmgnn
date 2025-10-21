import torch, dgl
from utils import Metrics, EarlyStopping
from samplers import NegativeStatementSampler, PartialStatementSampler, NegativeSampler, RandomStatementSampler
from losses import DualContrastiveLoss_CE, DualContrastiveLoss_Margin

class Train_BestModel:
    def __init__(self, model, epochs, train_loader, val_loader, full_cvgraph, e_type, log, device, task, lr=0.001, mcfg=None, cfg=None,
        pstatement_sampler=False, nstatement_sampler=False, rstatement_sampler=False, contrastive_weight=0.1, gda_negs=None, state_list=None, 
        no_contrastive=False):
        self.model = model
        self.best_lr = lr
        self.train_loader = list(train_loader)
        self.val_loader = list(val_loader)
        self.e_type = e_type
        self.rstatement_sampler = rstatement_sampler
        self.pstatement_sampler = pstatement_sampler
        self.nstatement__sampler = nstatement_sampler
        self.no_contrastive = no_contrastive

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
            self.neg_statement_sampler.prepare_global(full_cvgraph, negatives = gda_negs if task == "gda" else None)

        self.neg_sampler = NegativeSampler(full_cvgraph, edge_type = ("node", e_type, "node"))
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss_CE()
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.log = log
        self.device = device
        self.task = task
        self.gda_negs = gda_negs
        self.epochs = epochs
        self.metrics = Metrics()
        self.log.log("Setting, Epoch, " + "".join(name + ", " for name in self.metrics.get_names())[:-2])
        self.train_neg_ptr = 0
        self.val_neg_ptr = 0
        self.test_neg_ptr = 0

    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            src, dst  = batch.edges(etype=self.e_type)
            edge_index = torch.stack([src, dst], dim=0)

            if self.rstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch,edge_index)
                neg_statement_index = self.neg_statement_sampler.sample()
            elif self.nstatement__sampler or self.pstatement_sampler:
                self.neg_statement_sampler.prepare_batch(batch)
                neg_statement_index = self.neg_statement_sampler.sample()
            elif self.no_contrastive: pass
            else: self.neg_statement_sampler.prepare_batch(batch)


            neg_edge_g = self.neg_sampler.sample()
            neg_edge_g = neg_edge_g.to(self.device)
            neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
            labels = torch.cat([torch.ones(edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
            edge_index = torch.cat([edge_index, neg_edge_index], dim=1)

            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()
                for ntype in batch.ntypes: num = batch.num_nodes(ntype); batch.nodes[ntype].data[dgl.NID] = torch.arange(num, device=batch.device, dtype=torch.long)
                batch = dgl.to_homogeneous(batch, ndata=['feat', dgl.NID], store_type=True).to(self.device)
                batch = dgl.add_self_loop(batch)
                homo_nid = batch.ndata[dgl.NID]
                inv = torch.empty(batch.num_nodes(), dtype=torch.long, device=self.device)
                inv[homo_nid] = torch.arange(batch.num_nodes(), device=self.device)
                src = inv[orig_src.to(self.device)]
                dst = inv[orig_dst.to(self.device)]
                edge_index = torch.stack([src, dst], dim=0)

            z, out = self.model(batch, edge_index)
            if self.no_contrastive: pass
            else:
                if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                    (z_pos, z_pos_pos, z_pos_neg) = self.neg_statement_sampler.get_contrastive_samples(z, neg_statement_index)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                else:
                    (z_pos, z_pos_pos, z_pos_neg) = self.neg_statement_sampler.get_contrastive_samples(z)
                    loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
            loss = self.loss_fn(out.squeeze(-1), labels)
            if not self.no_contrastive: loss_ = self.alpha * loss_contrast + loss
            else: loss_ = loss

            loss_.backward()
            self.optimizer.step()
            num_examples = out.size(0)
            total_loss += loss_.item() * num_examples
            total_examples += num_examples
        if total_examples == 0: return 0
        return (total_loss / total_examples), (out, labels)
    

    def validate_epoch(self):
        self.model.eval()
        total_loss = total_examples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                src, dst  = batch.edges(etype=self.e_type)
                edge_index = torch.stack([src, dst], dim=0)

                if self.rstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch,edge_index)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.nstatement__sampler or self.pstatement_sampler:
                    self.neg_statement_sampler.prepare_batch(batch)
                    neg_statement_index = self.neg_statement_sampler.sample()
                elif self.no_contrastive: pass
                else: self.neg_statement_sampler.prepare_batch(batch)

                neg_edge_g = self.neg_sampler.sample()
                neg_edge_g = neg_edge_g.to(self.device)
                neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
                labels = torch.cat([torch.ones(src.size(0), device=self.device),
                    torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
                
                if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                    orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()
                    for ntype in batch.ntypes: num = batch.num_nodes(ntype); batch.nodes[ntype].data[dgl.NID] = torch.arange(num, device=batch.device, dtype=torch.long)
                    batch = dgl.to_homogeneous(batch, ndata=['feat', dgl.NID], store_type=True).to(self.device)
                    batch = dgl.add_self_loop(batch)
                    homo_nid = batch.ndata[dgl.NID]
                    inv = torch.empty(batch.num_nodes(), dtype=torch.long, device=self.device)
                    inv[homo_nid] = torch.arange(batch.num_nodes(), device=self.device)
                    src = inv[orig_src.to(self.device)]
                    dst = inv[orig_dst.to(self.device)]
                    edge_index = torch.stack([src, dst], dim=0)
                z, out = self.model(batch, edge_index)

                if self.no_contrastive: pass
                else:
                    if self.rstatement_sampler or self.nstatement__sampler or self.pstatement_sampler:
                        (z_pos, z_pos_pos, z_pos_neg) = self.neg_statement_sampler.get_contrastive_samples(z, neg_statement_index)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                    else:
                        (z_pos, z_pos_pos, z_pos_neg) = self.neg_statement_sampler.get_contrastive_samples(z)
                        loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg)
                loss = self.loss_fn(out.squeeze(-1), labels)
                if not self.no_contrastive: loss_ = self.alpha * loss_contrast + loss
                else: loss_ = loss
                num_examples = out.size(0)
                total_loss += loss_.item() * num_examples
                total_examples += num_examples
            return (total_loss / total_examples), (out, labels)

    def run(self):
        self.model.load_state_dict(torch.load(self.log.dir + 'model_'+ self.model.__class__.__name__ +'.pth'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.best_lr)
        for epoch in range(self.epochs):
            train_loss, (out, labels) = self.train_epoch()
            if train_loss != 0: print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}', flush=True)
            if self.earlystopper.step(train_loss, self.model):
                print(f" --  Early stopping at epoch {epoch}", flush=True)
                break
        return train_loss, (out, labels)
        


class Test_BestModel:
    def __init__(self, model, test_loader, e_type, full_graph, log, device, task, gda_negs=None):
        self.model = model
        self.test_loader = list(test_loader)
        self.e_type = e_type
        self.log = log
        self.device = device
        self.task = task
        self.gda_negs = gda_negs
        self.neg_sampler = NegativeSampler(full_graph, edge_type = ("node", e_type, "node"))
        self.metrics = Metrics()
        self.test_neg_ptr = 0

    def test_epoch(self):
        self.model.eval()
        for batch in self.test_loader:
            batch = batch.to(self.device)
            src, dst  = batch.edges(etype=self.e_type)
            edge_index = torch.stack([src, dst], dim=0)

            neg_edge_g = self.neg_sampler.sample()
            neg_edge_g = neg_edge_g.to(self.device)
            neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
            edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
            labels = torch.cat([torch.ones(src.size(0), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
            
            if self.model.__class__.__name__ in {"GCN", "GAT", "GCN_GAE"}:
                orig_src, orig_dst = edge_index[0].clone(), edge_index[1].clone()
                for ntype in batch.ntypes: num = batch.num_nodes(ntype); batch.nodes[ntype].data[dgl.NID] = torch.arange(num, device=batch.device, dtype=torch.long)
                batch = dgl.to_homogeneous(batch, ndata=['feat', dgl.NID], store_type=True).to(self.device)
                batch = dgl.add_self_loop(batch)
                homo_nid = batch.ndata[dgl.NID]
                inv = torch.empty(batch.num_nodes(), dtype=torch.long, device=self.device)
                inv[homo_nid] = torch.arange(batch.num_nodes(), device=self.device)
                src = inv[orig_src.to(self.device)]
                dst = inv[orig_dst.to(self.device)]
                edge_index = torch.stack([src, dst], dim=0)

            with torch.no_grad():  z, out = self.model(batch, edge_index)
        return labels, out
    
    
    def run(self):

        labels, out = self.test_epoch()
        torch.save(out, self.log.dir + "predictions_" + self.model.__class__.__name__ + ".pth")
        torch.save(labels, self.log.dir +  "labels_" + self.model.__class__.__name__ + ".pth")

        acc, f1, precision_p, recall_p, precision_n, recall_n, roc_auc = self.metrics.update_all(out.detach().to("cpu"), labels.to("cpu"))
        print('Test Results:', flush=True)
        print(f'Accuracy: {acc:.4f}, F1 Score (W): {f1:.4f}, Precision (+): {precision_p:.4f}, Recall (+): {recall_p:.4f}, '
              f'Precision (-): {precision_n:.4f}, Recall (-): {recall_n:.4f}, Roc Auc: {roc_auc:.4f}', flush=True)
        self.log.log("Test, final," + str(acc) + "," + str(f1)+ "," + str(precision_p)+ "," + str(recall_p)+ 
        "," + str(precision_n)+ "," + str(recall_n)+ "," + str(roc_auc))
        self.log.close()