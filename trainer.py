import torch
from utils import Metrics

class Train:
    def __init__(self, model, optimizer, epochs, train_loader, val_loader, log, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log = log
        self.device = device
        self.epochs = epochs
        self.metrics = Metrics()
        self.log.log("Setting, Epoch, " + "".join(name + ", " for name in self.metrics.get_names())[:-2])


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
    

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.loss_fn(out, batch.y)
                total_loss += loss.item()
        return (total_loss / len(self.val_loader)), out
    

    def run(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            if epoch % 5 == 0:
                val_loss, out = self.validate_epoch()
                self.log.log()
                acc, f1, precision, recall, roc_auc = self.metrics.update(val_loss, out)
                self.log.log("Validation, ", epoch, str(acc), str(f1), str(precision), str(recall), str(roc_auc))
                print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else: print(f'Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}')


class Test:
    def __init__(self, model, loss, test_loader, log, device):
        self.model = model
        self.test_loader = test_loader
        self.log = log
        self.device = device
        self.loss_fn = loss
        self.metrics = Metrics()

    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        for batch in self.test_loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                out = self.model(batch)
                loss = self.loss_fn(out, batch.y)
                total_loss += loss.item()
        self.metrics.update(loss, out)
        return (total_loss / len(self.test_loader)), out
    
    
    def run(self):
        test_loss, out = self.test_epoch()
        acc, f1, precision, recall, roc_auc = self.metrics.update(test_loss, out)
        self.log.log("Test, final", str(acc), str(f1), str(precision), str(recall), str(roc_auc))

        print('Test Results:')
        print(f'Loss: {test_loss:.4f}')
        print(f'Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Roc Auc: {roc_auc:.4f}')

        torch.save(self.model.state_dict(), 'model_'+ self.model.__class__.__name__ +'.pth')
        torch.save(out, "predictions_" + self.model.__class__.__name__ + ".pth")
        self.log.close()