import argparse, torch
from models import gnn
from trainer import Train, Test
from utils import Logger
from data_loader import DataLoader, Dglloader


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

    # load data
    data_loader = DataLoader({"GO_links.csv", "PPIs.csv"})
    dgl_loader = Dglloader(data_loader, batch_size=args.batch_size, device=device)
    train_load, val_load, test_load = dgl_loader.get_split_graphs()

    log = Logger(args.model.__name__)
    loss = Train(model, optim, args.epochs, train_load, val_load, log, device)
    test_loss = Test(model, test_load, log, device)