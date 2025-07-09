import argparse, torch, pandas
from models import *
from trainer import Train, Test
from utils import Logger
from data_loader import DataLoader, Dglloader
from utils import load_config
from sklearn.model_selection import KFold, train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["hgcn", "hpgcn", "hgat"], default="hgcn")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--CV_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256*512)
    parser.add_argument('--use_pstatement_sampler', action='store_true')
    parser.add_argument('--use_nstatement_sampler', action='store_true')
    parser.add_argument('--path', type=str, default="data")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = eval(args.model.upper())
    cfg = load_config()
    mcfg = cfg["models"][ModelCls.__name__]

    dl = DataLoader(args.path + "/", use_pstatement_sampler=args.use_pstatement_sampler,
                    use_nstatement_sampler=args.use_nstatement_sampler)
    state_list = None
    if args.use_pstatement_sampler or args.use_nstatement_sampler: state_list = dl.get_state_list()
    if args.use_pstatement_sampler: mcfg["edge_types"] = mcfg["edge_types"].remove(("pos_statement"))
    elif args.use_nstatement_sampler: mcfg["edge_types"] = mcfg["edge_types"].remove(("neg_statement"))

    full_graph = dl.make_data_graph(dl.get_data())
    ppi_etype = mcfg["ppi_etype"]
    src_all, dst_all = full_graph.edges(etype=ppi_etype)
    edge_pairs = list(zip(src_all.tolist(), dst_all.tolist()))
    trainval_pairs, test_pairs = train_test_split(edge_pairs, test_size=0.1, random_state=42, shuffle=True) # train and test PPIs

    all_etypes = full_graph.canonical_etypes
    test_src, test_dst = zip(*test_pairs)
    test_src,test_dst = torch.tensor(test_src), torch.tensor(test_dst)
    test_eids = full_graph.edge_ids(test_src, test_dst, etype=ppi_etype)
    eid_dict = {et: full_graph.edges(form='eid', etype=et) for et in all_etypes}
    eid_dict[ppi_etype] = test_eids
    test_graph = full_graph.edge_subgraph(eid_dict, relabel_nodes=False, store_ids=True)

    trainval_src, trainval_dst = zip(*trainval_pairs)
    trainval_src = torch.tensor(trainval_src)
    trainval_dst = torch.tensor(trainval_dst)
    trainval_eids = full_graph.edge_ids(trainval_src, trainval_dst, etype=ppi_etype)
    eid_dict = {et: full_graph.edges(form='eid', etype=et) for et in all_etypes}
    eid_dict[ppi_etype] = trainval_eids
    trainval_graph = full_graph.edge_subgraph(eid_dict, relabel_nodes=False, store_ids=True)

    trainval_loader = Dglloader(trainval_graph, ppi_rel=ppi_etype[1], batch_size=args.batch_size,
                       val_split=0.1, device=device, seed=42)
    test_loader = Dglloader(test_graph, ppi_rel=ppi_etype[1], batch_size=args.batch_size,
                       val_split=0, device=device, seed=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_f1 = -1.0
    best_state = None
    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_pairs), 1):
        print(f"\n=== Fold {fold}/5 ===", flush=True)

        fold_train_pairs = [trainval_pairs[i] for i in train_idx]
        fold_val_pairs = [trainval_pairs[i] for i in val_idx]
        train_src, train_dst = zip(*fold_train_pairs)
        val_src, val_dst = zip(*fold_val_pairs)
        train_src = torch.tensor(train_src)
        train_dst = torch.tensor(train_dst)
        val_src = torch.tensor(val_src)
        val_dst = torch.tensor(val_dst)

        train_eids = full_graph.edge_ids(train_src, train_dst, etype=ppi_etype)
        val_eids = full_graph.edge_ids(val_src, val_dst, etype=ppi_etype)

        eid_dict_train = {et: full_graph.edges(form='eid', etype=et) for et in all_etypes}
        eid_dict_val = {et: full_graph.edges(form='eid', etype=et) for et in all_etypes}
        eid_dict_train[ppi_etype] = train_eids
        eid_dict_val[ppi_etype] = val_eids

        train_graph = full_graph.edge_subgraph(eid_dict_train, relabel_nodes=False, store_ids=True)
        val_graph = full_graph.edge_subgraph(eid_dict_val, relabel_nodes=False, store_ids=True)

        train_loader = Dglloader(train_graph, ppi_rel=ppi_etype[1], batch_size=args.batch_size, device=device)._batch_graphs(train_eids)
        val_loader = Dglloader(val_graph, ppi_rel=ppi_etype[1], batch_size=args.batch_size, device=device)._batch_graphs(val_eids)

        model = ModelCls(in_feats=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
                         e_etypes=[tuple(e) for e in mcfg["edge_types"]], ppi_etype=ppi_etype).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        log = Logger(f"{ModelCls.__name__}_fold{fold}")

        trainer = Train(model, optim, args.CV_epochs, train_loader, val_loader,
                        e_type=ppi_etype, log=log, device=device, full_cvgraph=train_graph,
                        contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
                        pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler)

        _, _ = trainer.run()
        _, (logits, labels) = trainer.validate_epoch()
        acc, f1, prec, rec, rocauc = trainer.metrics.update(logits.detach().to("cpu"), labels.to("cpu"))
        log.log("Valid, Final epoch," + str(acc) + "," + str(f1) + "," + str(prec) + "," + str(rec) + "," + str(rocauc))
        print(f"Fold {fold} AUC: {rocauc:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()

    print(f"\nBest fold Weighted F1 = {best_f1:.4f} – saving model\n", flush=True)
    final_model = ModelCls(in_feats=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
                           e_etypes=[tuple(e) for e in mcfg["edge_types"]], ppi_etype=ppi_etype).to(device)
    final_model.load_state_dict(best_state)
    final_optim = torch.optim.Adam(final_model.parameters(), lr=cfg["lr"])

    final_train_loader = Dglloader(trainval_graph, ppi_rel=ppi_etype[1], batch_size=args.batch_size, val_split=0, device=device).train_batches()
    final_trainer = Train(final_model, final_optim, args.epochs, final_train_loader, [], full_cvgraph=trainval_graph,
                          e_type=ppi_etype, log=Logger("final_train"), device=device, contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
                          pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler)
    loss, _ = final_trainer.run()
    print(f"Final training loss: {loss:.4f}")

    final_log = Logger("final_test")
    tester = Test(final_model, test_loader=test_loader, e_type=ppi_etype, log=final_log, full_graph=test_graph, device=device)
    tester.run()

if __name__ == "__main__":
    main()




# def main():

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, choices= ["hgcn", "hpgcn", "hgat", "bigcn", "bigat"], default="hgcn")
#     parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
#     parser.add_argument('--CV_epochs', type=int, default=200, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=256*512, help='Batch size for training')
#     parser.add_argument('--use_pstatement_sampler', action='store_true', help='Use PartialStatementSampler for positive statements')
#     parser.add_argument('--use_nstatement_sampler', action='store_true', help='Use PartialStatementSampler for negative statements')
#     parser.add_argument('--path', type=str, default="data", help='Folder with data files, defaults to data/ directory')
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ModelCls = eval(args.model.upper())
#     cfg = load_config()
#     mcfg = cfg["models"][ModelCls.__name__]

#     dl = DataLoader(args.path + "/",
#                     use_pstatement_sampler=args.use_pstatement_sampler,
#                     use_nstatement_sampler=args.use_nstatement_sampler)
#     state_list = None
#     if args.use_pstatement_sampler or args.use_nstatement_sampler: state_list = dl.get_state_list()
#     if args.use_pstatement_sampler: mcfg["edge_types"] = mcfg["edge_types"].remove(("pos_statement"))
#     elif args.use_nstatement_sampler: mcfg["edge_types"] = mcfg["edge_types"].remove(("neg_statement"))

#     full_graph = dl.make_data_graph(dl.get_data())
#     ppi_etype  = mcfg["ppi_etype"]
#     src_all, dst_all = full_graph.edges(etype=ppi_etype)
#     pairs = list(zip(src_all.tolist(), dst_all.tolist()))
#     n_splits = min(10, len(pairs))
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     best_f1 = -1.0
#     best_state = None

#     for fold, (train_idx, val_idx) in enumerate(kf.split(pairs), 1):
#         print(f"\n=== Fold {fold}/{n_splits} ===", flush=True)

#         train_pairs = [pairs[i] for i in train_idx]
#         val_pairs = [pairs[i] for i in val_idx]
#         all_etypes = full_graph.canonical_etypes
#         full_eids = {}
#         for et in all_etypes:
#             if et != ppi_etype:
#                 full_eids[et] = full_graph.edges(form='eid', etype=et)
#         train_src, train_dst = zip(*train_pairs)
#         val_src, val_dst = zip(*val_pairs)
#         train_src = torch.tensor(train_src, dtype=torch.long)
#         train_dst = torch.tensor(train_dst, dtype=torch.long)
#         val_src = torch.tensor(val_src, dtype=torch.long)
#         val_dst = torch.tensor(val_dst, dtype=torch.long)
#         train_ppi_eids = full_graph.edge_ids(train_src, train_dst, etype=ppi_etype)
#         val_ppi_eids = full_graph.edge_ids(val_src, val_dst, etype=ppi_etype)
#         train_eid_map = {}
#         val_eid_map   = {}
#         for et in all_etypes:
#             if et == ppi_etype:
#                 train_eid_map[et] = train_ppi_eids
#                 val_eid_map[et]   = val_ppi_eids
#             else:
#                 train_eid_map[et] = full_eids[et]
#                 val_eid_map[et]   = full_eids[et]
#         train_graph = full_graph.edge_subgraph(train_eid_map, relabel_nodes=True, store_ids=True)
#         val_graph   = full_graph.edge_subgraph(val_eid_map, relabel_nodes=True, store_ids=True)
#         train_loader = Dglloader(train_graph, batch_size=args.batch_size, device=device).train_batches()
#         val_loader   = Dglloader(val_graph, batch_size=args.batch_size, device=device).validation_batches()

#         model = ModelCls(in_feats = mcfg["in_feats"], hidden_dim = mcfg["hidden_dim"],
#         out_dim = mcfg["out_dim"], e_etypes = [tuple(e) for e in mcfg["edge_types"]],
#         ppi_etype  = ppi_etype).to(device)
#         optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
#         log   = Logger(f"{ModelCls.__name__}_fold{fold}")
#         trainer = Train(model, optim, args.CV_epochs, train_loader, val_loader,
#                         e_type=ppi_etype, log=log, device=device, full_cvgraph=train_graph,
#                         contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
#                         pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler)
#         t_loss, (out, labels) = trainer.run()
#         val_loss, (logits, labels) = trainer.validate_epoch()
#         acc, f1, prec, rec, rocauc = trainer.metrics.update(logits.detach().to("cpu"), labels.to("cpu"))
#         log.log("Valid, Final epoch,"+ str(acc) + ","+ str(f1) + ","+str(prec) + ","+str(rec) + ","+str(rocauc))
#         print(f"Fold {fold} AUC: {rocauc:.4f}", flush=True)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_state = model.state_dict()
#     print(f"\nBest fold Weighted F1 = {best_f1:.4f} – saving model\n", flush=True)

#     final_model = ModelCls(in_feats = mcfg["in_feats"], hidden_dim = mcfg["hidden_dim"], out_dim = mcfg["out_dim"],
#         e_etypes = [tuple(e) for e in mcfg["edge_types"]], ppi_etype  = ppi_etype).to(device)
#     final_model.load_state_dict(best_state)
#     final_optim = torch.optim.Adam(final_model.parameters(), lr=cfg["lr"])
#     full_train_loader = Dglloader(full_graph, batch_size = args.batch_size, device = device).train_batches()
#     final_trainer = Train(final_model, final_optim, args.epochs, full_train_loader, [], full_cvgraph=full_graph,
#                           e_type=ppi_etype, log=log, device=device, contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
#                         pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler)
#     loss, (out, labels) = final_trainer.run()
#     print(f"Final training loss: {loss:.4f}", flush=True)

#     final_log = Logger("final_test")
#     test_loader = Dglloader(full_graph, batch_size = args.batch_size, device = device).test_batches()
#     tester = Test(final_model, test_loader=test_loader, e_type=ppi_etype, log=final_log, full_graph=full_graph, device=device)
#     tester.run()

# if __name__ == "__main__":
#     main()
