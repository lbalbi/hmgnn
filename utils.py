def ensure_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

class Logger:

    def __init__(self, name, dir= ""):
        self.name = name
        self.dir = "output/" + dir
        ensure_dir(self.dir)
        self.log_file = f"{name}.log"
        self.file = open(self.dir + self.log_file, 'a')
        self.file.write(f"Results for {name}\n")
        self.file.write("=" * 50 + "\n")
        self.file.close()

    def log(self, message):
        print(message)
        self.file = open(self.dir + self.log_file, 'a')
        self.file.write(message + "\n")
        self.file.flush()
        self.file.close()

    def close(self):
        self.file.close()

import torch, copy
class EarlyStopping:
    """
    Args:
        patience (int): how many epochs to wait after last time validation metric improved.
        mode (str): 'min' to stop when metric stops decreasing, 'max' for increasing.
        delta (float): minimum change in the monitored metric to qualify as an improvement.
    """
    def __init__(self, patience: int = 5, mode: str = 'min', delta: float = 0.0):
        self.patience   = patience
        self.mode = mode
        self.delta = delta
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.num_bad = 0
        self.best_state = None
        self._is_better = (lambda a, b: a < b - delta) if mode == 'min' else (lambda a, b: a > b + delta)

    def step(self, metric: float, model: torch.nn.Module) -> bool:
        if self._is_better(metric, self.best_score):
            self.best_score = metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.num_bad = 0
        else: self.num_bad += 1
        return self.num_bad > self.patience

    def best_state_dict(self):
        return self.best_state
    

    
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
class Metrics:

    @staticmethod
    def accuracy(preds, labels):
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        return correct / total if total > 0 else 0

    @staticmethod
    def f1score(preds, labels):
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
    def rocauc_score(preds, labels):
        return roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
    
    def update(self, preds, labels):
        preds = torch.round(preds)
        self.accuracy_value = accuracy_score(labels, preds)
        self.f1_value = f1_score(labels, preds, average='weighted', zero_division=0)
        self.precision_value = precision_score(labels, preds, zero_division=0)
        self.recall_value = recall_score(labels, preds, zero_division=0)
        self.roc_auc_value = roc_auc_score(labels, preds, average='weighted')
        return self.accuracy_value, self.f1_value, self.precision_value, self.recall_value, self.roc_auc_value
    
    def get_names(self):
        return ["accuracy, f1 score, precision, recall, roc auc"]

    
import json
def load_config(task, path="config.json"):
    with open(task + "_"+ path, "r") as f:
        return json.load(f)