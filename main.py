import argparse, torch
from models import *
from trainer import Train, Test
from utils import Logger
from data_loader import DataLoader, Dglloader
from utils import load_config
def main():
    parser = argparse.ArgumentParser(description="Run HMGNN pipeline for composite positive and negative ~" \
    "representation learning over heterogeneous graphs with negation -based contradictions.")

    parser.add_argument('--model', type=str, choices= ["gcn","gat", "hgcn", "hgat", "bigcn", "bigat"], default="hgcn")
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model = eval(args.model.upper())
    config = load_config()
    print(f"Using model: {args.model.__name__}")
    model = args.model().to(device)
    optim = torch.optim.Adam(model.parameters())
    log = Logger(args.model.__name__)
    print(f"Using optimizer: {optim.__class__.__name__}")

    data_loader = DataLoader({"GO_links.csv", "PPIs.csv"})
    print("here")
    dgl_loader = Dglloader(data_loader, batch_size=args.batch_size, device=device)
    train_load, val_load, test_load = dgl_loader.get_split_graphs()

    Train(model, optim, args.epochs, train_load, val_load, log, device)
    Test(model, test_load, log, device)


if __name__ == "__main__":
    main()
