import torch
from utils import Metrics, EarlyStopping
from samplers import NegativeStatementSampler, PartialStatementSampler, NegativeSampler
from losses import DualContrastiveLoss

class Train:
    def __init__(self, model, optimizer, epochs, train_loader, val_loader, full_cvgraph, e_type, log, device, 
                 pstatement_sampler=False, nstatement_sampler=False, contrastive_weight=0.1, state_list=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = list(train_loader)
        self.val_loader = list(val_loader)
        self.e_type = e_type
        if pstatement_sampler: 
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        elif nstatement_sampler: 
            self.neg_statement_sampler = PartialStatementSampler(neg_edges=state_list)
            self.neg_statement_sampler.prepare_global(full_cvgraph, pos_etype="neg_statement", neg_etype="pos_statement")
        else: 
            self.neg_statement_sampler = NegativeStatementSampler()
            self.neg_statement_sampler.prepare_global(full_cvgraph)
        self.neg_sampler = NegativeSampler(full_cvgraph, edge_type = ("node", e_type, "node"))
        self.loss_fn = torch.nn.BCELoss()
        self.contrastive = DualContrastiveLoss(temperature=0.5)
        self.alpha = contrastive_weight
        self.earlystopper = EarlyStopping()
        self.log = log
        self.device = device
        self.epochs = epochs
        self.metrics = Metrics()
        self.log.log("Setting, Epoch, " + "".join(name + ", " for name in self.metrics.get_names())[:-2])


    def train_epoch(self):
        self.model.train()
        total_loss = total_examples = 0

        for batch in self.train_loader:

            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            src, dst  = batch.edges(etype=self.e_type)
            edge_index = torch.stack([src, dst], dim=0)
            self.neg_statement_sampler.prepare_batch(batch)
            neg_statement_index = self.neg_statement_sampler.sample()
            neg_edge_g = self.neg_sampler.sample()
            neg_edge_g = neg_edge_g.to(self.device)
            neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
            neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
            labels = torch.cat([torch.ones(edge_index.size(1), device=self.device),
                torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
            edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
            z, out = self.model(batch, edge_index)
            (z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg) = self.neg_statement_sampler.get_contrastive_samples(z, neg_statement_index)
            loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg)
            loss = self.loss_fn(out.squeeze(-1), labels)
            loss_ = self.alpha * loss_contrast + loss
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
                self.neg_statement_sampler.prepare_batch(batch)
                neg_statement_index = self.neg_statement_sampler.sample()
                neg_edge_g = self.neg_sampler.sample()
                neg_edge_g = neg_edge_g.to(self.device)
                neg_src, neg_dst = neg_edge_g.edges(etype=("node", self.e_type, "node"))
                neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
                edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
                labels = torch.cat([torch.ones(src.size(0), device=self.device),
                    torch.zeros(neg_edge_index.size(1), device=self.device)], dim=0)
                z, out = self.model(batch, edge_index)
                (z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg) = self.neg_statement_sampler.get_contrastive_samples(z, neg_statement_index)
                loss_contrast = self.contrastive(z_pos, z_pos_pos, z_pos_neg, z_neg, z_neg_pos, z_neg_neg)
                loss = self.loss_fn(out.squeeze(-1), labels)
                loss_ = self.alpha * loss_contrast + loss
                num_examples = out.size(0)
                total_loss += loss_.item() * num_examples
                total_examples += num_examples
            return (total_loss / total_examples), (out, labels)

    def run(self):
        for epoch in range(self.epochs):
            train_loss, (out, labels) = self.train_epoch()
            if train_loss != 0: print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}', flush=True)
            if self.earlystopper.step(train_loss, self.model):
                print(f" --  Early stopping at epoch {epoch}", flush=True)
                break
        return train_loss, (out, labels)
        


class Test:
    def __init__(self, model, test_loader, e_type, full_graph, log, device):
        self.model = model
        self.test_loader = list(test_loader)
        self.e_type = e_type
        self.log = log
        self.device = device
        self.neg_sampler = NegativeSampler(full_graph, edge_type = ("node", e_type, "node"))
        self.metrics = Metrics()

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
            with torch.no_grad():
                z, out = self.model(batch, edge_index)
            self.metrics.update(out.detach().to("cpu"), labels.to("cpu"))
        return labels, out
    
    
    def run(self):
        labels, out = self.test_epoch()
        acc, f1, precision, recall, roc_auc = self.metrics.update(out.detach().to("cpu"), labels.to("cpu"))
        print('Test Results:', flush=True)
        print(f'Accuracy: {acc:.4f}, F1 Score (W): {f1:.4f}, Precision (+): {precision:.4f}, Recall (+): {recall:.4f}, Roc Auc: {roc_auc:.4f}', flush=True)
        self.log.log("Test, final," + str(acc) + "," + str(f1)+ "," + str(precision)+ "," + str(recall)+ ","+ str(roc_auc))
        
        torch.save(self.model.state_dict(), 'model_'+ self.model.__class__.__name__ +'.pth')
        torch.save(out, "predictions_" + self.model.__class__.__name__ + ".pth")
        torch.save(labels, "labels_" + self.model.__class__.__name__ + ".pth")
        self.log.close()