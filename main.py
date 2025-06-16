import argparse, torch
from models import gnn
from trainer import Train, Test
from utils import Logger


def main():
    parser = argparse.ArgumentParser(description="Run HMGNN pipeline for composite positive and negative ~" \
    "representation learning over heterogeneous graphs with negation -based contradictions.")

    parser.add_argument('--model', type=str, required=True, choices= ["gcn","gat", "hgcn", "hgat", "bigcn", "bigat"])
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model = eval(args.model)
    model = args.model().to(device)
    optim = torch.optim.Adam(model.parameters())

    log = Logger(args.model.__name__)

    loss = train(model, optim, args.epochs, train_load, val_load, device)
    test_loss = test(model, test_load, device)
    