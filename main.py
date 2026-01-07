import argparse, torch, os
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
    parser.add_argument('--patience', type=int, default=20, help="Patience for early stopping")
    parser.add_argument('--no_contrastive', action='store_true', help="Disable contrastive learning")
    parser.add_argument('--path', type=str, default="human_data", help="Path to the dataset directory")
    parser.add_argument('--output_dir', type=str, default="output/", help="Directory to save output logs and models")
    parser.add_argument('--protein_splits', action='store_true', help='Prot-disjoint PPI split: train PPIs do not touch val/test proteins')
    parser.add_argument('--protein_degree_splits', action='store_true',
        help='Protein-disjoint PPI split (degree-aware): hold out proteins sampled from PPI sources with bias toward LOW-degree; '
            'HIGH-degree proteins tend to stay in training.')
    parser.add_argument('--degree_gamma', type=float,default=10.0,
         help='Strength of degree bias for --protein_degree_splits. Larger => more high-degree proteins in training.')
    args = parser.parse_args()
    
    if args.protein_degree_splits and args.protein_splits:
        print("[WARN] Both --protein_splits and --protein_degree_splits set; using --protein_degree_splits.", flush=True)
    use_protein_split = args.protein_splits or args.protein_degree_splits

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    cfg = load_config(task=args.task)
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    dl = DataLoader(args.path + "/", use_pstatement_sampler=args.use_pstatement_sampler,
        use_nstatement_sampler=args.use_nstatement_sampler, use_rstatement_sampler=args.use_rstatement_sampler)

    data_dict = dl.get_data()
    neg_ppi_edge_index = None
    if str(args.task).lower() == "huri":
        # For HURI we want to *use* neg_PPI for classification, but NOT include it in the message-passing graph.
        neg = data_dict.pop("neg_PPI", None)
        if neg is None:
            print("[WARN] --task huri set but no edge_type=='neg_PPI' found in the data files. Falling back to random negative sampling.", flush=True)
        else:
            neg_ppi_edge_index = torch.stack([neg[0], neg[1]], dim=0)
        # Safety: ensure neg_PPI is not used as a message-passing relation (even if present in config)
        if ["node", "neg_PPI", "node"] in mcfg.get("edge_types", []):
            mcfg["edge_types"].remove(["node", "neg_PPI", "node"])


    state_list = None
    if args.use_pstatement_sampler or args.use_nstatement_sampler: state_list = dl.get_state_list()
    if args.use_pstatement_sampler and ["node","pos_statement","node"] in mcfg["edge_types"]:
        mcfg["edge_types"].remove(["node","pos_statement","node"])
    elif args.use_nstatement_sampler and ["node","neg_statement","node"] in mcfg["edge_types"]:
        mcfg["edge_types"].remove(["node","neg_statement","node"])

    full_graph = dl.make_data_graph(data_dict)
    ppi_rel = mcfg["ppi_etype"][1] if isinstance(mcfg["ppi_etype"], (list, tuple)) else mcfg["ppi_etype"]
    ppi_key = next((et for et in full_graph.edge_types if et[1] == ppi_rel), None)
    ppi_ei = full_graph[ppi_key].edge_index
    all_eids = torch.arange(ppi_ei.size(1))

    split_helper_train = Pygloader(full_graph, ppi_rel=ppi_rel, 
            batch_size=args.batch_size, val_split=0, device=device)

    split_id = args.path.split("/")[0]
    split_dir = os.path.join(split_id, "splits_{}".format(args.task))
    os.makedirs(split_dir, exist_ok=True)


    def describe_ppi_split(ppi_ei: torch.Tensor, eids: torch.Tensor, name: str, ref_nodes: torch.Tensor | None = None):
        """
        Print basic stats for a set of PPI edges (selected by eids).
        If ref_nodes is provided, also reports how many of those nodes appear in these edges.
        """
        eids = eids.cpu()
        sub = ppi_ei[:, eids].cpu()
        src = sub[0]
        dst = sub[1]
        uniq_src = torch.unique(src)
        uniq_nodes = torch.unique(torch.cat([src, dst], dim=0))
        msg = (f"[{name}] #PPIs={sub.size(1)} | "
            f"#unique_src={len(uniq_src)} | "
            f"#unique_nodes(src∪dst)={len(uniq_nodes)}")
        if ref_nodes is not None:
            ref_nodes = ref_nodes.cpu()
            touched = torch.isin(uniq_nodes, ref_nodes).sum().item()
            msg += f" | #nodes_in_ref={touched}/{len(ref_nodes)}"
        print(msg, flush=True)


    def sample_test_proteins_degree_aware(src_all: torch.Tensor, dst_all: torch.Tensor,
            frac: float = 0.2, gamma: float = 1.0, seed: int = 42) -> torch.Tensor:
        """
        Pick held-out proteins from UNIQUE PPI SOURCE nodes, biased toward LOW PPI degree.
        Degree is computed undirected over PPI edges (counts incident edges on either endpoint).
        """
        src_all = src_all.cpu()
        dst_all = dst_all.cpu()
        unique_src = torch.unique(src_all)
        max_id = int(torch.max(torch.cat([src_all, dst_all])).item())
        deg = torch.bincount(src_all, minlength=max_id + 1) + torch.bincount(dst_all, minlength=max_id + 1)
        deg_unique = deg[unique_src].to(torch.float)
        weights = 1.0 / torch.pow(deg_unique + 1.0, float(gamma))
        weights = weights / weights.sum()
        n_test = max(1, int(len(unique_src) * frac))
        n_test = min(n_test, len(unique_src))
        g = torch.Generator().manual_seed(seed)
        idx = torch.multinomial(weights, n_test, replacement=False, generator=g)
        test_proteins = unique_src[idx]
        return torch.sort(test_proteins).values

    if args.protein_degree_splits:
        trainval_eids_path = os.path.join(split_dir, "trainval_eids_node_disjoint_degree.pt")
        test_eids_path = os.path.join(split_dir, "test_eids_node_disjoint_degree.pt")
        test_nodes_path = os.path.join(split_dir, "test_proteins_degree.pt")
    elif args.protein_splits:
        trainval_eids_path = os.path.join(split_dir, "trainval_eids_node_disjoint.pt")
        test_eids_path = os.path.join(split_dir, "test_eids_node_disjoint.pt")
        test_nodes_path = os.path.join(split_dir, "test_proteins.pt")
    else:
        trainval_eids_path = os.path.join(split_dir, "trainval_eids.pt")
        test_eids_path = os.path.join(split_dir, "test_eids.pt")
        test_nodes_path = None

    if os.path.exists(trainval_eids_path) and os.path.exists(test_eids_path):
        trainval_eids = torch.load(trainval_eids_path, weights_only=False)
        test_eids = torch.load(test_eids_path, weights_only=False)
        if use_protein_split and test_nodes_path is not None and os.path.exists(test_nodes_path):
            test_proteins = torch.load(test_nodes_path, weights_only=False)
        else: test_proteins = None
    else:
        if args.protein_degree_splits:
            src_all = ppi_ei[0].cpu()
            dst_all = ppi_ei[1].cpu()
            test_proteins = sample_test_proteins_degree_aware(
                src_all, dst_all, frac=0.2, gamma=args.degree_gamma, seed=42)
            test_edge_mask = torch.isin(src_all, test_proteins) | torch.isin(dst_all, test_proteins)
            test_eids = all_eids[test_edge_mask].to(torch.long)
            trainval_eids = all_eids[~test_edge_mask].to(torch.long)
            torch.save(trainval_eids, trainval_eids_path)
            torch.save(test_eids, test_eids_path)
            torch.save(test_proteins, test_nodes_path)
        elif args.protein_splits:
            src_all = ppi_ei[0].cpu()
            dst_all = ppi_ei[1].cpu()
            unique_src = torch.unique(src_all)
            g = torch.Generator().manual_seed(42)
            unique_src = unique_src[torch.randperm(len(unique_src), generator=g)]
            n_test = max(1, int(len(unique_src) * 0.15))
            test_proteins = torch.sort(unique_src[:n_test]).values
            test_edge_mask = torch.isin(src_all, test_proteins) | torch.isin(dst_all, test_proteins)
            test_eids = all_eids[test_edge_mask].to(torch.long)
            trainval_eids = all_eids[~test_edge_mask].to(torch.long)
            torch.save(trainval_eids, trainval_eids_path)
            torch.save(test_eids, test_eids_path)
            torch.save(test_proteins, test_nodes_path)
        else:
            trainval_eids_np, test_eids_np = train_test_split(
                all_eids.cpu().numpy(), test_size=0.15, random_state=42, shuffle=True)
            trainval_eids = torch.tensor(trainval_eids_np, dtype=torch.long)
            test_eids = torch.tensor(test_eids_np, dtype=torch.long)
            torch.save(trainval_eids, trainval_eids_path)
            torch.save(test_eids, test_eids_path)

    # ===== Split diagnostics =====
    describe_ppi_split(ppi_ei, trainval_eids, "TRAINVAL(+)")
    describe_ppi_split(ppi_ei, test_eids, "TEST(+)", ref_nodes=test_proteins if 'test_proteins' in locals() else None)

    # Extra checks for protein-disjoint mode
    if args.protein_splits and test_proteins is not None:
        src_tv = torch.unique(ppi_ei[:, trainval_eids].reshape(-1).cpu())
        src_te = torch.unique(ppi_ei[:, test_eids].reshape(-1).cpu())
        overlap = torch.isin(src_tv, src_te).sum().item()
        print(f"[CHECK] trainval_nodes ∩ test_nodes = {overlap}", flush=True)


    trainval_graph = split_helper_train._create_split_graph(trainval_eids, train=True)
    test_ppis = full_graph[ppi_key].edge_index[:, test_eids]
    test_graph = trainval_graph

    trainval_loader = Pygloader(trainval_graph, ppi_rel=ppi_rel, batch_size=args.batch_size,
                                val_split=0.1, device=device, seed=42)
    test_loader = Pygloader(test_graph, ppi_rel=ppi_rel, batch_size=args.batch_size,
                            val_split=0.0, device=device, seed=42)


    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    best_lrs, best_epochs, best_alphas = [], [], []

    def run_one_fold(fold_train_graph, fold_val_graph, ppi_vei, fold_num: int):
        """Train one CV fold and store best hyperparams."""
        train_loader = Pygloader(fold_train_graph, ppi_rel=ppi_rel, val_split=0,
            batch_size=args.batch_size, device=device)
        val_loader = Pygloader(fold_val_graph, ppi_rel=ppi_rel, val_split=0,
            batch_size=args.batch_size, device=device)

        model = ModelCls(in_dim=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg["out_dim"], e_etypes=[tuple(e) for e in mcfg["edge_types"]],
            ppi_etype=ppi_rel).to(device)
        log = Logger(f"{ModelCls.__name__ if args.model != 'gae' else 'GAE'}_fold{fold_num}",
            dir=args.output_dir)

        gda_negs = dl.get_negative_edges() if hasattr(dl, "get_negative_edges") and \
                   (args.path == "gda_data" or args.path == "dp_data") else None

        trainer = Train(model, args.CV_epochs, train_loader, val_loader,
            e_type=ppi_rel, val_edges=ppi_vei, val_edge_batch_size=args.batch_size,
            log=log, lrs=cfg["lr"], device=device, full_graph=full_graph, full_cvgraph=fold_train_graph,
            contrastive_weight=cfg["contrastive_weight"], state_list=state_list,
            pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
            rstatement_sampler=args.use_rstatement_sampler,
            task=args.task, gda_negs=gda_negs, no_contrastive=args.no_contrastive, patience=args.patience,
            neg_ppi_edge_index=neg_ppi_edge_index)

        lr, loss, _, epoch_, alpha_ = trainer.run()
        best_epochs.append(epoch_)
        best_lrs.append(lr)
        best_alphas.append(alpha_)

    if args.protein_splits or args.protein_degree_splits:
        src_all = ppi_ei[0].cpu()
        dst_all = ppi_ei[1].cpu()
        trainval_edge_mask = torch.zeros(ppi_ei.size(1), dtype=torch.bool)
        trainval_edge_mask[trainval_eids.cpu()] = True
        tv_nodes = torch.unique(src_all[trainval_edge_mask])

        for fold, (train_idx, val_idx) in enumerate(kf.split(tv_nodes.numpy()), 1):
            print(f"\n=== Fold {fold}/{cfg['k_folds']} ===", flush=True)

            fold_val_nodes = tv_nodes[val_idx]
            fold_val_mask = trainval_edge_mask & (
                torch.isin(src_all, fold_val_nodes) | torch.isin(dst_all, fold_val_nodes))
            fold_val_eids = all_eids[fold_val_mask]
            fold_train_eids = all_eids[trainval_edge_mask & (~fold_val_mask)]

            if fold_train_eids.numel() == 0 or fold_val_eids.numel() == 0:
                print(f"[WARN] Skipping fold {fold}: "
                    f"fold_train_eids={fold_train_eids.numel()}, fold_val_eids={fold_val_eids.numel()}",
                    flush=True)
                continue

            fold_train_graph = split_helper_train._create_split_graph(fold_train_eids, train=True)
            fold_val_graph, ppi_vei = split_helper_train._create_split_graph(fold_val_eids, train=False)
            run_one_fold(fold_train_graph, fold_val_graph, ppi_vei, fold)

    else:
        for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_eids), 1):
            print(f"\n=== Fold {fold}/{cfg['k_folds']} ===", flush=True)
            fold_train_eids = trainval_eids[torch.tensor(train_idx, dtype=torch.long)]
            fold_val_eids = trainval_eids[torch.tensor(val_idx, dtype=torch.long)]
            if fold_train_eids.numel() == 0 or fold_val_eids.numel() == 0:
                print(f"[WARN] Skipping fold {fold}: "
                    f"fold_train_eids={fold_train_eids.numel()}, fold_val_eids={fold_val_eids.numel()}",
                    flush=True)
                continue

            fold_train_graph = split_helper_train._create_split_graph(fold_train_eids, train=True)
            fold_val_graph, ppi_vei = split_helper_train._create_split_graph(fold_val_eids, train=False)
            run_one_fold(fold_train_graph, fold_val_graph, ppi_vei, fold)
    if len(best_lrs) == 0:
        raise RuntimeError("Cross-validation produced no trained folds (best_lrs is empty). "
            "This usually happens if every fold was skipped because the fold's "
            "train/val PPI edge set became empty under the current splitting rule. "
            "Try lowering k_folds.") # degree-aware protein split

    best_lr = mode(best_lrs)
    best_epoch = mode(best_epochs)
    best_alpha = mode(best_alphas)
    final_model = ModelCls(in_dim=mcfg["in_feats"], hidden_dim=mcfg["hidden_dim"], out_dim=mcfg["out_dim"],
                e_etypes=[tuple(e) for e in mcfg["edge_types"]],ppi_etype=ppi_rel).to(device)
    
    final_log = Logger("final_train", dir=args.output_dir, non_verbose=True)
    gda_negs = dl.get_negative_edges() if hasattr(dl, "get_negative_edges") and \
        (args.path == "gda_data" or args.path == "dp_data") else None
    final_trainer = Train_BestModel(final_model, best_epoch, trainval_loader, [],full_cvgraph=trainval_graph,full_graph=full_graph,
        e_type=ppi_rel, log=final_log, device=device, task=args.task, lr=best_lr,
        contrastive_weight=best_alpha, state_list=state_list,
        pstatement_sampler=args.use_pstatement_sampler, nstatement_sampler=args.use_nstatement_sampler,
        rstatement_sampler=args.use_rstatement_sampler, gda_negs=gda_negs, no_contrastive=args.no_contrastive, 
        patience=args.patience, neg_ppi_edge_index=neg_ppi_edge_index)
    loss, (pred, _) = final_trainer.run()
    print(f"Final training loss: {loss:.4f}", flush=True)
    
    final_log_test = Logger("final_test", dir=args.output_dir)
    tester = Test_BestModel(final_model, test_loader=test_loader, e_type=ppi_rel, test_edges=test_ppis, test_edge_batch_size=args.batch_size,
    log=final_log_test, test_graph=test_graph, full_graph=full_graph, device=device, task=args.task, gda_negs=gda_negs, neg_ppi_edge_index=neg_ppi_edge_index)
    tester.run()

if __name__ == "__main__":
    main()
