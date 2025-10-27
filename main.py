import argparse, torch
from models import *
from trainer import Train
from trainer_bestmodel import Train_BestModel, Test_BestModel
from utils import Logger
from data_loader import DataLoader, Pygloader
from utils import load_config
from sklearn.model_selection import KFold, train_test_split
from statistics import mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="human", help="Task to run: human, cerevisae, melanogaster")
    parser.add_argument('--model', type=str, choices=["hgcn", "hpgcn", "hgat","gcn", "gae"], default="hgcn", help="Model to run")
    parser.add_argument('--epochs', type=int, default=250, help="Number of epochs for final training")
    parser.add_argument('--CV_epochs', type=int, default=250, help="Number of epochs for cross-validation")
    parser.add_argument('--batch_size', type=int, default=256*512, help="Batch size for training") 
    parser.add_argument('--use_pstatement_sampler', action='store_true', help="Use positive statement sampler")
    parser.add_argument('--use_nstatement_sampler', action='store_true', help="Use negative statement sampler")
    parser.add_argument('--use_rstatement_sampler', action='store_true', help="Use random statement sampler")
    parser.add_argument('--no_contrastive', action='store_true', help="Disable contrastive learning")
    parser.add_argument('--path', type=str, default="human_data", help="Path to the dataset directory")
    parser.add_argument('--output_dir', type=str, default="", help="Directory to save output logs and models")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    cfg = load_config(task=args.task)
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    dl = DataLoader(args.path + "/", use_pstatement_sampler=args.use_pstatement_sampler,
                    use_nstatement_sampler=args.use_nstatement_sampler, use_rstatement_sampler=args.use_rstatement_sampler)
    state_list = None
    if args.use_pstatement_sampler or args.use_nstatement_sampler: state_list = dl.get_state_list()
    if args.use_pstatement_sampler: mcfg["edge_types"].remove(["node","pos_statement","node"])
    elif args.use_nstatement_sampler: mcfg["edge_types"].remove(["node","neg_statement","node"])

    full_graph = dl.make_data_graph(dl.get_data())
    ppi_etype = mcfg["ppi_etype"]
    src_all, dst_all = full_graph.edges(etype=ppi_etype)
    edge_pairs = list(zip(src_all.tolist(), dst_all.tolist()))
    trainval_pairs, test_pairs = train_test_split(edge_pairs, test_size=0.1, random_state=42, shuffle=True) 

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

    trainval_loader = Pygloader(trainval_graph, ppi_rel=ppi_etype, batch_size=args.batch_size,
                       val_split=0.1, device=device, seed=42)
    test_loader = Pygloader(test_graph, ppi_rel=ppi_etype, batch_size=args.batch_size,
                       val_split=0, device=device, seed=42)

    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    best_f1 = -1.0
    best_lrs = []
    best_state = None
    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_pairs), 1):
        print(f"\n=== Fold {fold}/{cfg['k_folds']} ===", flush=True)

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

        train_loader = Pygloader(train_graph, ppi_rel=ppi_etype, val_split=0, batch_size=args.batch_size, device=device).train_batches()
        val_loader = Pygloader(val_graph, ppi_rel=ppi_etype, val_split=0, batch_size=args.batch_size, device=device).train_batches()

        model = ModelCls(in_feats=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
                         e_etypes=[tuple(e) for e in mcfg["edge_types"]], ppi_etype=ppi_etype).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        log = Logger(f"{ModelCls.__name__ if args.model != 'gae' else 'GAE'}_fold{fold}", dir=args.output_dir)

        trainer = Train(model, optim, args.CV_epochs, train_loader, val_loader,
                        e_type=ppi_etype, log=log, device=device, full_cvgraph=train_graph,
                        contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
                        pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
                        rstatement_sampler=args.use_rstatement_sampler, task=args.task,
                        gda_negs=dl.get_negative_edges() if args.path == "gda_data" or args.path == "dp_data" else None, no_contrastive=args.no_contrastive)

        lr, loss, _ = trainer.run()
        best_lrs.append(lr)
    best_lr = mode(best_lrs)
    
    final_model = ModelCls(in_feats=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
                           e_etypes=[tuple(e) for e in mcfg["edge_types"]], ppi_etype=ppi_etype).to(device)    
    
    final_trainer = Train_BestModel(final_model, args.epochs, trainval_loader, [], full_cvgraph=trainval_graph,
                          e_type=ppi_etype, log=Logger("final_train", dir=args.output_dir), device=device, 
                          contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
                          pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
                          rstatement_sampler=args.use_rstatement_sampler, task=args.task, lr=best_lr,
                          gda_negs=dl.get_negative_edges() if args.path == "gda_data"  or args.path == "dp_data" else None, no_contrastive=args.no_contrastive)
    loss, (pred, _) = final_trainer.run()
    print(f"Final training loss: {loss:.4f}")
    
    final_log = Logger("final_test", dir= args.output_dir)
    tester = Test_BestModel(final_model, test_loader=test_loader, e_type=ppi_etype, log=final_log, full_graph=test_graph, device=device, 
                 task=args.task, gda_negs=dl.get_negative_edges() if args.path == "gda_data"  or args.path == "dp_data" else None)
    tester.run()

if __name__ == "__main__":
    main()
