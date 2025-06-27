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
    parser.add_argument('--path', type=str, default="data", help='Folder with data files, defaults to data/ directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model = eval(args.model.upper())
    config = load_config()
    model = args.model(config["models"][args.model.__name__].get("in_feats"),
        config["models"][args.model.__name__].get("hidden_dim"), config["models"][args.model.__name__].get("out_dim")).to(device)
    optim = torch.optim.Adam(model.parameters())
    log = Logger(args.model.__name__)

    print(f"Using model: {args.model.__name__}")
    print(f"Using optimizer: {optim.__class__.__name__}")

    data_loader = DataLoader(args.path + "/")
    data = data_loader.make_data_graph(data_loader.get_data())
    dgl_loader = Dglloader(data, batch_size=args.batch_size if args.batch_size else config.get("batch_size"), device=device)
    train_load, val_load, test_load = dgl_loader.get_split_graphs()

    train = Train(model, optim, args.epochs if args.epochs else config.get("epochs"), train_loader = train_load, 
          val_loader = val_load, log = log, device = device)
    train.run()
    test = Test(model = model, test_loader = test_load, log = log, device = device, loss = torch.nn.CrossEntropyLoss())
    test.run()



if __name__ == "__main__":
    main()
