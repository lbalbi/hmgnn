import argparse, torch
from models import *
from trainer import Train
from trainer_bestmodel import Train_BestModel, Test_BestModel
from utils import Logger, load_config
from data_loader import DataLoader, Dglloader
from sklearn.model_selection import KFold, train_test_split
from statistics import mode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="human")
    parser.add_argument('--model', type=str, choices=["hgcn", "hpgcn", "hgat","gcn", "gae"], default="hgcn")
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--CV_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=256*512)
    parser.add_argument('--use_pstatement_sampler', action='store_true')
    parser.add_argument('--use_nstatement_sampler', action='store_true')
    parser.add_argument('--use_rstatement_sampler', action='store_true')
    parser.add_argument('--no_contrastive', action='store_true')
    parser.add_argument('--path', type=str, default="human_data")
    parser.add_argument('--output_dir', type=str, default="")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    cfg = load_config(task=args.task)
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    dl = DataLoader(
        args.path + "/",
        use_pstatement_sampler=args.use_pstatement_sampler,
        use_nstatement_sampler=args.use_nstatement_sampler,
        use_rstatement_sampler=args.use_rstatement_sampler,
    )
    state_list = None
    if args.use_pstatement_sampler or args.use_nstatement_sampler:
        state_list = dl.get_state_list()
    if args.use_pstatement_sampler and ["node","pos_statement","node"] in mcfg["edge_types"]:
        mcfg["edge_types"].remove(["node","pos_statement","node"])
    elif args.use_nstatement_sampler and ["node","neg_statement","node"] in mcfg["edge_types"]:
        mcfg["edge_types"].remove(["node","neg_statement","node"])

    full_graph = dl.make_data_graph(dl.get_data())
    ppi_etype = mcfg["ppi_etype"]  # relation name, e.g. "PPI"

    # Split PPI edges into train+val vs test (by edges)
    src_all, dst_all = full_graph.edges(etype=ppi_etype)
    edge_pairs = list(zip(src_all.tolist(), dst_all.tolist()))
    trainval_pairs, test_pairs = train_test_split(
        edge_pairs, test_size=0.1, random_state=42, shuffle=True
    )

    all_etypes = full_graph.canonical_etypes

    # Train+val graph (message passing uses these PPI edges)
    trainval_src, trainval_dst = zip(*trainval_pairs)
    trainval_src = torch.tensor(trainval_src)
    trainval_dst = torch.tensor(trainval_dst)
    trainval_eids = full_graph.edge_ids(trainval_src, trainval_dst, etype=ppi_etype)
    eid_dict_trainval = {et: full_graph.edges(form='eid', etype=et) for et in all_etypes}
    eid_dict_trainval[next(et for et in all_etypes if et[1] == ppi_etype)] = trainval_eids
    trainval_graph = full_graph.edge_subgraph(
        eid_dict_trainval, relabel_nodes=False, store_ids=True
    )

    # Test edges: explicit [2, N_test] tensor; *not* in trainval_graph adjacency
    test_src, test_dst = zip(*test_pairs)
    test_src = torch.tensor(test_src)
    test_dst = torch.tensor(test_dst)
    test_edges = torch.stack([test_src, test_dst], dim=0)
    # test_graph for message passing is the *train+val* graph
    test_graph = trainval_graph

    trainval_loader = Dglloader(
        trainval_graph,
        ppi_rel=ppi_etype,
        batch_size=args.batch_size,
        val_split=0.1,
        device=device,
        seed=42,
    )
    test_loader = []  # not used by Test_BestModel, but keep for signature

    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    best_lrs = []

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
        eid_dict_train[next(et for et in all_etypes if et[1] == ppi_etype)] = train_eids
        eid_dict_val[next(et for et in all_etypes if et[1] == ppi_etype)] = val_eids

        train_graph = full_graph.edge_subgraph(
            eid_dict_train, relabel_nodes=False, store_ids=True
        )
        val_graph = full_graph.edge_subgraph(
            eid_dict_val, relabel_nodes=False, store_ids=True
        )

        train_loader = Dglloader(
            train_graph,
            ppi_rel=ppi_etype,
            val_split=0,
            batch_size=args.batch_size,
            device=device,
        ).train_batches()
        val_loader = Dglloader(
            val_graph,
            ppi_rel=ppi_etype,
            val_split=0,
            batch_size=args.batch_size,
            device=device,
        ).train_batches()

        model = ModelCls(
            in_feats=mcfg["in_feats"],
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg["out_dim"],
            e_etypes=[tuple(e) for e in mcfg["edge_types"]],
            ppi_etype=ppi_etype,
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        log = Logger(
            f"{ModelCls.__name__ if args.model != 'gae' else 'GAE'}_fold{fold}",
            dir=args.output_dir,
        )

        trainer = Train(
            model,
            optim,
            args.CV_epochs,
            train_loader,
            val_loader,
            full_graph=full_graph,
            full_cvgraph=train_graph,
            e_type=ppi_etype,
            log=log,
            device=device,
            task=args.task,
            lrs=[cfg["lr"]],
            contrastive_weight=cfg["contrastive_weight"],
            state_list=state_list,
            pstatement_sampler=args.use_pstatement_sampler,
            nstatement_sampler=args.use_nstatement_sampler,
            rstatement_sampler=args.use_rstatement_sampler,
            gda_negs=dl.get_negative_edges()
            if args.path in ("gda_data", "dp_data")
            else None,
            no_contrastive=args.no_contrastive,
        )

        lr, loss, _ = trainer.run()
        best_lrs.append(lr)

    best_lr = mode(best_lrs)

    final_model = ModelCls(
        in_feats=mcfg["in_feats"],
        hidden_dim=mcfg["hidden_dim"],
        out_dim=mcfg["out_dim"],
        e_etypes=[tuple(e) for e in mcfg["edge_types"]],
        ppi_etype=ppi_etype,
    ).to(device)

    final_trainer = Train_BestModel(
        final_model,
        args.epochs,
        trainval_loader.train_batches(),
        [],
        full_cvgraph=trainval_graph,
        full_graph=full_graph,
        e_type=ppi_etype,
        log=Logger("final_train", dir=args.output_dir),
        device=device,
        task=args.task,
        lr=best_lr,
        contrastive_weight=cfg["contrastive_weight"],
        state_list=state_list,
        pstatement_sampler=args.use_pstatement_sampler,
        nstatement_sampler=args.use_nstatement_sampler,
        rstatement_sampler=args.use_rstatement_sampler,
        gda_negs=dl.get_negative_edges()
        if args.path in ("gda_data", "dp_data")
        else None,
        no_contrastive=args.no_contrastive,
    )
    loss, (pred, _) = final_trainer.run()
    print(f"Final training loss: {loss:.4f}")

    final_log = Logger("final_test", dir=args.output_dir)
    tester = Test_BestModel(
        final_model,
        test_loader=test_loader,
        e_type=ppi_etype,
        test_graph=test_graph,
        full_graph=full_graph,
        log=final_log,
        device=device,
        task=args.task,
        gda_negs=dl.get_negative_edges()
        if args.path in ("gda_data", "dp_data")
        else None,
        test_edges=test_edges,
        test_edge_batch_size=args.batch_size,
    )
    tester.run()

if __name__ == "__main__":
    main()
