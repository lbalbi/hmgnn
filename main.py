import argparse, torch
from models import *
from trainer import Train
from trainer_bestmodel import Train_BestModel, Test_BestModel
from utils import Logger, load_config
from data_loader import DataLoader, Pygloader
from sklearn.model_selection import KFold, train_test_split
from statistics import mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="human", help="Task to run: human, cerevisae, melanogaster")
    parser.add_argument('--model', type=str, choices=["hgcn", "hpgcn", "hgat","gcn", "gae"], default="hgcn", help="Model to run")
    parser.add_argument('--epochs', type=int, default=300, help="Number of epochs for final training")
    parser.add_argument('--CV_epochs', type=int, default=250, help="Number of epochs for cross-validation")
    parser.add_argument('--batch_size', type=int, default=256*512, help="Batch size for training") 
    parser.add_argument('--use_pstatement_sampler', action='store_true', help="Use positive statement sampler to remove \
        positives from graph but keep in sampling")
    parser.add_argument('--use_nstatement_sampler', action='store_true', help="Use negative statement sampler to remove \
        negatives from graph but keep in sampling")
    parser.add_argument('--use_rstatement_sampler', action='store_true', help="Use random statement sampler to not use \
    statements in sampling")
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
    if args.use_pstatement_sampler and ["node","pos_statement","node"] in mcfg["edge_types"]:
        mcfg["edge_types"].remove(["node","pos_statement","node"])
    elif args.use_nstatement_sampler and ["node","neg_statement","node"] in mcfg["edge_types"]:
        mcfg["edge_types"].remove(["node","neg_statement","node"])

    full_graph = dl.make_data_graph(dl.get_data())
    ppi_rel = mcfg["ppi_etype"][1] if isinstance(mcfg["ppi_etype"], (list, tuple)) else mcfg["ppi_etype"]
    ppi_key = next((et for et in full_graph.edge_types if et[1] == ppi_rel), None)
    ppi_ei = full_graph[ppi_key].edge_index
    all_eids = torch.arange(ppi_ei.size(1))
    trainval_eids_np, test_eids_np = train_test_split(all_eids.cpu().numpy(), test_size=0.1, random_state=42, shuffle=True)
    trainval_eids = torch.tensor(trainval_eids_np, dtype=torch.long)
    test_eids = torch.tensor(test_eids_np, dtype=torch.long)
    #split_helper = Pygloader(full_graph, ppi_rel=ppi_rel, batch_size=args.batch_size, val_split=0, device=device)
    #trainval_graph = split_helper._create_split_graph(trainval_eids)
    #test_graph = split_helper._create_split_graph(test_eids)
    split_helper_train = Pygloader(full_graph, ppi_rel=ppi_rel, batch_size=args.batch_size, val_split=0,device=device)
    trainval_graph = split_helper_train._create_split_graph(trainval_eids, train=True)
    test_ppis  = full_graph[ppi_key].edge_index[:, test_eids]
    test_graph = trainval_graph
    #split_helper_test = Pygloader(full_graph, ppi_rel=ppi_rel,batch_size=args.batch_size, val_split=0,device=device)
    #test_graph, test_ppis = split_helper_test._create_split_graph(test_eids, train=False)

    trainval_loader = Pygloader(trainval_graph, ppi_rel=ppi_rel, batch_size=args.batch_size,
                                val_split=0.1, device=device, seed=42)
    test_loader = Pygloader(test_graph, ppi_rel=ppi_rel, batch_size=args.batch_size,
                            val_split=0.0, device=device, seed=42)

    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    best_lrs, best_f1, best_epochs = [], -1.0, []

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_eids), 1):
        print(f"\n=== Fold {fold}/{cfg['k_folds']} ===", flush=True)
        fold_train_eids = trainval_eids[torch.tensor(train_idx, dtype=torch.long)]
        fold_val_eids = trainval_eids[torch.tensor(val_idx, dtype=torch.long)]
        fold_train_graph = split_helper_train._create_split_graph(fold_train_eids)
        fold_val_graph, ppi_vei = split_helper_train._create_split_graph(fold_val_eids,train=False)
        train_loader = Pygloader(fold_train_graph, ppi_rel=ppi_rel, val_split=0,
                                 batch_size=args.batch_size, device=device)
        val_loader = Pygloader(fold_val_graph, ppi_rel=ppi_rel, val_split=0,
                               batch_size=args.batch_size, device=device)

        model = ModelCls(in_dim=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
            e_etypes=[tuple(e) for e in mcfg["edge_types"]], ppi_etype=ppi_rel).to(device)
        log = Logger(f"{ModelCls.__name__ if args.model != 'gae' else 'GAE'}_fold{fold}", dir=args.output_dir)
        gda_negs = dl.get_negative_edges() if hasattr(dl, "get_negative_edges") and \
                   (args.path == "gda_data" or args.path == "dp_data") else None
        trainer = Train(model, args.CV_epochs, train_loader, val_loader, e_type=ppi_rel, val_edges=ppi_vei, val_edge_batch_size=args.batch_size, 
        log=log, lrs=cfg["lr"], device=device, full_graph=full_graph, full_cvgraph=fold_train_graph, contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
            pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
            rstatement_sampler=args.use_rstatement_sampler, task=args.task,gda_negs=gda_negs, no_contrastive=args.no_contrastive)
        lr, loss, _, epoch_ = trainer.run()
        best_epochs.append(epoch_)
        best_lrs.append(lr)
    best_lr = mode(best_lrs)
    best_epoch = mode(best_epochs)

    final_model = ModelCls(in_dim=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
                e_etypes=[tuple(e) for e in mcfg["edge_types"]],ppi_etype=ppi_rel).to(device)
    
    final_log = Logger("final_train", dir=args.output_dir, non_verbose=True)
    gda_negs = dl.get_negative_edges() if hasattr(dl, "get_negative_edges") and \
        (args.path == "gda_data" or args.path == "dp_data") else None
    #final_trainer = Train_BestModel(final_model, best_epoch, trainval_loader, [], 
        # full_cvgraph=trainval_graph, e_type=ppi_rel, log=final_log, device=device, task=args.task, lr=best_lr,
        # contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
        # pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
        # rstatement_sampler=args.use_rstatement_sampler, gda_negs=gda_negs, no_contrastive=args.no_contrastive)
    final_trainer = Train_BestModel(final_model, best_epoch, trainval_loader, [],full_cvgraph=trainval_graph,full_graph=full_graph,
        e_type=ppi_rel, log=final_log, device=device, task=args.task, lr=best_lr,
        contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
        pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
        rstatement_sampler=args.use_rstatement_sampler, gda_negs=gda_negs, no_contrastive=args.no_contrastive)
    loss, (pred, _) = final_trainer.run()
    print(f"Final training loss: {loss:.4f}", flush=True)

    final_log_test = Logger("final_test", dir=args.output_dir)
    tester = Test_BestModel(final_model, test_loader=test_loader, e_type=ppi_rel, test_edges=test_ppis, test_edge_batch_size=args.batch_size,
    log=final_log_test, test_graph=test_graph, full_graph=full_graph, device=device, task=args.task, gda_negs=gda_negs)
    tester.run()

if __name__ == "__main__":
    main()
