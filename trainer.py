import torch

class Train:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, log, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log = log
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    

class Test:
    def __init__(self, model, test_loader, log, device):
        self.model = model
        self.test_loader = test_loader
        self.log = log
        self.device = device

    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        for batch in self.test_loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                out = self.model(batch)
                loss = self.loss_fn(out, batch.y)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)