class Logger:

    def __init__(self, name):
        self.name = name
        self.log_file = f"{name}.log"
        self.file = open(self.log_file, 'w')
        self.file.write(f"Results for {name}\n")
        self.file.write("=" * 50 + "\n")

    def log(self, message):
        print(message)
        self.file.write(message + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
    def reset(self):
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def is_early_stop(self):
        return self.early_stop
    
    def get_patience(self):
        return self.patience
    
    def get_best_score(self):
        return self.best_score
    
    def get_counter(self):
        return self.counter
    

class metrics:
    @staticmethod
    def accuracy(preds, labels):
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        return correct / total if total > 0 else 0

    @staticmethod
    def f1_score(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    @staticmethod
    def precision(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    @staticmethod
    def recall(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    @staticmethod
    def confusion_matrix(preds, labels):
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def roc_auc_score(preds, labels):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())