import argparse, torch
from models import *
from trainer import Train, Test
from utils import Logger
from data_loader import DataLoader, Dglloader
from utils import load_config
from sklearn.model_selection import KFold

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices= ["gcn","gat", "hgcn", "hpgcn", "hgat", "bigcn", "bigat"], default="hgcn")
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--CV_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256*512, help='Batch size for training')
    parser.add_argument('--path', type=str, default="data", help='Folder with data files, defaults to data/ directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = eval(args.model.upper())
    cfg = load_config()
    mcfg = cfg["models"][ModelCls.__name__]
    dl = DataLoader(args.path + "/")
    full_graph = dl.make_data_graph(dl.get_data())
    ppi_etype  = mcfg["ppi_etype"]
    src_all, dst_all = full_graph.edges(etype=ppi_etype)
    pairs = list(zip(src_all.tolist(), dst_all.tolist()))
    n_splits = min(10, len(pairs))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_f1 = -1.0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(pairs), 1):
        print(f"\n=== Fold {fold}/{n_splits} ===", flush=True)

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]
        all_etypes = full_graph.canonical_etypes
        full_eids = {}
        for et in all_etypes:
            if et != ppi_etype:
                full_eids[et] = full_graph.edges(form='eid', etype=et)
        train_src, train_dst = zip(*train_pairs)
        val_src, val_dst = zip(*val_pairs)
        train_src = torch.tensor(train_src, dtype=torch.long)
        train_dst = torch.tensor(train_dst, dtype=torch.long)
        val_src = torch.tensor(val_src, dtype=torch.long)
        val_dst = torch.tensor(val_dst, dtype=torch.long)
        train_ppi_eids = full_graph.edge_ids(train_src, train_dst, etype=ppi_etype)
        val_ppi_eids = full_graph.edge_ids(val_src, val_dst, etype=ppi_etype)
        train_eid_map = {}
        val_eid_map   = {}
        for et in all_etypes:
            if et == ppi_etype:
                train_eid_map[et] = train_ppi_eids
                val_eid_map[et]   = val_ppi_eids
            else:
                train_eid_map[et] = full_eids[et]
                val_eid_map[et]   = full_eids[et]
        train_graph = full_graph.edge_subgraph(train_eid_map, relabel_nodes=True, store_ids=True)
        val_graph   = full_graph.edge_subgraph(val_eid_map, relabel_nodes=True, store_ids=True)
        train_loader = Dglloader(train_graph, batch_size=args.batch_size, device=device).train_batches()
        val_loader   = Dglloader(val_graph, batch_size=args.batch_size, device=device).train_batches()

        model = ModelCls(in_feats = mcfg["in_feats"], hidden_dim = mcfg["hidden_dim"],
        out_dim = mcfg["out_dim"], e_etypes = [tuple(e) for e in mcfg["edge_types"]],
        ppi_etype  = ppi_etype).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        log   = Logger(f"{ModelCls.__name__}_fold{fold}")
        trainer = Train(model, optim, args.CV_epochs, train_loader, val_loader,
                        e_type=ppi_etype, log=log, device=device, full_cvgraph=train_graph,
                        contrastive_weight=cfg["contrastive_weight"])
        t_loss, (out, labels) = trainer.run()
        val_loss, (logits, labels) = trainer.validate_epoch()
        acc, f1, prec, rec, rocauc = trainer.metrics.update(logits.detach().to("cpu"), labels.to("cpu"))
        log.log("Valid, Final epoch,"+ str(acc) + ","+ str(f1) + ","+str(prec) + ","+str(rec) + ","+str(rocauc))
        print(f"Fold {fold} AUC: {rocauc:.4f}", flush=True)
        if f1 > best_f1:
            best_f1   = f1
            best_state = model.state_dict()
    print(f"\nBest fold Weighted F1 = {best_f1:.4f} – saving model\n", flush=True)

    final_model = ModelCls(in_feats = mcfg["in_feats"], hidden_dim = mcfg["hidden_dim"], out_dim = mcfg["out_dim"],
        e_etypes = [tuple(e) for e in mcfg["edge_types"]], ppi_etype  = ppi_etype).to(device)
    final_model.load_state_dict(best_state)
    final_optim = torch.optim.Adam(final_model.parameters(), lr=cfg["lr"])
    full_train_loader = Dglloader(full_graph, batch_size = args.batch_size, device = device).train_batches()
    final_trainer = Train(final_model, final_optim, args.epochs, full_train_loader, [], full_cvgraph=full_graph,
                          e_type=ppi_etype, log=log, device=device, contrastive_weight=cfg["contrastive_weight"])
    loss, (out, labels) = final_trainer.run()
    print(f"Final training loss: {loss:.4f}", flush=True)

    final_log = Logger("final_test")
    test_loader = Dglloader(full_graph, batch_size = args.batch_size, device = device).test_batches()
    tester = Test(final_model, loss=torch.nn.BCEWithLogitsLoss(), test_loader=test_loader, log=final_log,
                  full_graph=full_graph, device=device)
    tester.run()

if __name__ == "__main__":
    main()
