import argparse
import os
from statistics import mode

import torch
from sklearn.model_selection import KFold, train_test_split

from models import *  # noqa: F401,F403
from trainer import Train
from trainer_bestmodel import Train_BestModel, Test_BestModel
from utils import Logger, load_config
from dataloader import DataLoader
from pygloader import Pygloader


def run_for_relation(
    ppi_rel,
    dl: DataLoader,
    full_graph,
    cfg,
    mcfg,
    args,
    device,
):
    """Run the whole CV + final training + test pipeline for a single relation `ppi_rel`."""

    print(f"\n==============================")
    print(f"  Relation: {ppi_rel}")
    print(f"==============================", flush=True)

    # Use all edge types from the graph for message passing
    e_etypes = list(full_graph.edge_types)

    # Edge key for this relation
    ppi_key = next((et for et in full_graph.edge_types if et[1] == ppi_rel), None)
    if ppi_key is None:
        print(f"[WARN] No edges with relation {ppi_rel} in training data. Skipping.")
        return

    ppi_ei = full_graph[ppi_key].edge_index
    all_eids = torch.arange(ppi_ei.size(1))

    # Try to get explicit test edges for this relation
    explicit_test_ppis = dl.get_test_pairs(ppi_rel)  # [2, N_test] or empty

    split_helper = Pygloader(
        full_graph,
        ppi_rel=ppi_rel,
        batch_size=args.batch_size,
        val_split=0.0,
        device=device,
    )

    # Make split directory *per relation* to avoid clashes
    rel_str = str(ppi_rel).replace("/", "_")
    split_id = args.path.split("/")[0]
    split_dir = os.path.join(split_id, f"splits_{args.task}_{rel_str}")
    os.makedirs(split_dir, exist_ok=True)
    trainval_eids_path = os.path.join(split_dir, "trainval_eids.pt")
    test_eids_path = os.path.join(split_dir, "test_eids.pt")

    # ------------------- build trainval_graph / test_graph -------------------
    if explicit_test_ppis.numel() > 0:
        # Use explicit test positives; all training positives available for CV
        if os.path.exists(trainval_eids_path):
            trainval_eids = torch.load(trainval_eids_path, weights_only=False)
        else:
            trainval_eids = all_eids.clone()
            torch.save(trainval_eids, trainval_eids_path)

        trainval_graph = split_helper._create_split_graph(trainval_eids, train=True)
        test_ppis = explicit_test_ppis
        test_graph = trainval_graph

        # For negative sampling, avoid both train and test positives
        from copy import deepcopy

        sampler_full_graph = deepcopy(full_graph)
        sampler_full_graph[ppi_key].edge_index = torch.cat(
            [
                sampler_full_graph[ppi_key].edge_index,
                explicit_test_ppis.to(sampler_full_graph[ppi_key].edge_index.device),
            ],
            dim=1,
        )
    else:
        # Fallback: random split from training positives
        if os.path.exists(trainval_eids_path) and os.path.exists(test_eids_path):
            trainval_eids = torch.load(trainval_eids_path, weights_only=False)
            test_eids = torch.load(test_eids_path, weights_only=False)
        else:
            trainval_eids_np, test_eids_np = train_test_split(
                all_eids.cpu().numpy(),
                test_size=0.15,
                random_state=42,
                shuffle=True,
            )
            trainval_eids = torch.tensor(trainval_eids_np, dtype=torch.long)
            test_eids = torch.tensor(test_eids_np, dtype=torch.long)
            torch.save(trainval_eids, trainval_eids_path)
            torch.save(test_eids, test_eids_path)

        trainval_graph = split_helper._create_split_graph(trainval_eids, train=True)
        test_ppis = full_graph[ppi_key].edge_index[:, test_eids]
        test_graph = trainval_graph
        sampler_full_graph = full_graph

    # Loaders for CV and test
    trainval_loader = Pygloader(
        trainval_graph,
        ppi_rel=ppi_rel,
        batch_size=args.batch_size,
        val_split=0.1,
        device=device,
        seed=42,
    )
    test_loader = Pygloader(
        test_graph,
        ppi_rel=ppi_rel,
        batch_size=args.batch_size,
        val_split=0.0,
        device=device,
        seed=42,
    )

    # ------------------------------ Sampler state ----------------------------
    state_list = None
    if args.use_pstatement_sampler or args.use_nstatement_sampler:
        state_list = dl.get_state_list()

    # ------------------------------ Cross-validation -------------------------
    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)
    best_lrs, best_epochs, best_alphas = [], [], []

    # Model class
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_eids), 1):
        print(f"\n=== Rel {ppi_rel} | Fold {fold}/{cfg['k_folds']} ===", flush=True)

        fold_train_eids = trainval_eids[torch.tensor(train_idx, dtype=torch.long)]
        fold_val_eids = trainval_eids[torch.tensor(val_idx, dtype=torch.long)]

        fold_train_graph = split_helper._create_split_graph(fold_train_eids, train=True)
        fold_val_graph, ppi_vei = split_helper._create_split_graph(fold_val_eids, train=False)

        train_loader = Pygloader(
            fold_train_graph,
            ppi_rel=ppi_rel,
            val_split=0.0,
            batch_size=args.batch_size,
            device=device,
        )
        val_loader = Pygloader(
            fold_val_graph,
            ppi_rel=ppi_rel,
            val_split=0.0,
            batch_size=args.batch_size,
            device=device,
        )

        # Instantiate model for this relation
        model = ModelCls(
            in_dim=mcfg["in_feats"],
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg["out_dim"],
            e_etypes=e_etypes,
            ppi_etype=("node", ppi_rel, "node"),
        ).to(device)

        log_name = f"{ModelCls.__name__}_rel{ppi_rel}_fold{fold}"
        log = Logger(log_name, dir=args.output_dir)

        # Optional GDA negatives
        gda_negs = (
            dl.get_negative_edges()
            if hasattr(dl, "get_negative_edges")
            and (args.path == "gda_data" or args.path == "dp_data")
            else None
        )

        trainer = Train(
            model,
            args.CV_epochs,
            train_loader,
            val_loader,
            sampler_full_graph,   # graph used for negative sampling
            fold_train_graph,     # full_cvgraph for statement samplers
            e_type=ppi_rel,
            log=log,
            device=device,
            task=args.task,
            lrs=cfg["lr"],
            gda_negs=gda_negs,
            pstatement_sampler=args.use_pstatement_sampler,
            nstatement_sampler=args.use_nstatement_sampler,
            rstatement_sampler=args.use_rstatement_sampler,
            contrastive_weight=cfg.get("contrastive_weight", 0.1),
            state_list=state_list,
            no_contrastive=args.no_contrastive,
            val_edges=ppi_vei,
            val_edge_batch_size=args.batch_size,
        )

        lr, loss, _, epoch_, alpha_ = trainer.run()
        best_epochs.append(epoch_)
        best_lrs.append(lr)
        best_alphas.append(alpha_)

    best_lr = mode(best_lrs)
    best_epoch = mode(best_epochs)
    best_alpha = mode(best_alphas)

    # --------------------------- Final training ------------------------------
    final_model = ModelCls(
        in_dim=mcfg["in_feats"],
        hidden_dim=mcfg["hidden_dim"],
        out_dim=mcfg["out_dim"],
        e_etypes=e_etypes,
        ppi_etype=("node", ppi_rel, "node"),
    ).to(device)

    final_log = Logger(f"final_train_rel{ppi_rel}", dir=args.output_dir, non_verbose=True)
    gda_negs = (
        dl.get_negative_edges()
        if hasattr(dl, "get_negative_edges")
        and (args.path == "gda_data" or args.path == "dp_data")
        else None
    )

    final_trainer = Train_BestModel(
        final_model,
        best_epoch,
        trainval_loader,  # reuse loader over trainval_graph
        [],               # no separate val_loader at this stage
        full_cvgraph=trainval_graph,
        full_graph=sampler_full_graph,
        e_type=ppi_rel,
        log=final_log,
        device=device,
        task=args.task,
        lr=best_lr,
        contrastive_weight=best_alpha,
        state_list=state_list,
        pstatement_sampler=args.use_pstatement_sampler,
        nstatement_sampler=args.use_nstatement_sampler,
        rstatement_sampler=args.use_rstatement_sampler,
        gda_negs=gda_negs,
        no_contrastive=args.no_contrastive,
    )

    loss, _ = final_trainer.run()
    print(f"[Rel {ppi_rel}] Final training loss: {loss:.4f}", flush=True)

    # ---------------------------- Final test --------------------------------
    final_log_test = Logger(f"final_test_rel{ppi_rel}", dir=args.output_dir)
    tester = Test_BestModel(
        final_model,
        test_loader=test_loader,
        e_type=ppi_rel,
        test_graph=test_graph,
        full_graph=sampler_full_graph,
        log=final_log_test,
        device=device,
        task=args.task,
        gda_negs=gda_negs,
        test_edges=test_ppis,
        test_edge_batch_size=args.batch_size,
    )
    tester.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="human",
                        help="Task to run: human, cerevisae, melanogaster, wikidata, ...")
    parser.add_argument("--model", type=str,
                        choices=["hgcn", "hhgcn", "hgat", "gcn", "gae"],
                        default="hgcn", help="Model to run")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of epochs for final training")
    parser.add_argument("--CV_epochs", type=int, default=250,
                        help="Number of epochs for cross-validation")
    parser.add_argument("--batch_size", type=int, default=256 * 512,
                        help="Batch size for training")
    parser.add_argument("--use_pstatement_sampler", action="store_true",
                        help="Use positive statement sampler")
    parser.add_argument("--use_nstatement_sampler", action="store_true",
                        help="Use negative statement sampler")
    parser.add_argument("--use_rstatement_sampler", action="store_true",
                        help="Use random statement sampler")
    parser.add_argument("--no_contrastive", action="store_true",
                        help="Disable contrastive learning")
    parser.add_argument("--path", type=str, default="human_data",
                        help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="output/",
                        help="Directory to save output logs and models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model config
    cfg = load_config(task=args.task)
    # Use the right sub-config for your model
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    # ----------------------- Load graph from training files ------------------
    dl = DataLoader(
        args.path + "/",
        use_pstatement_sampler=args.use_pstatement_sampler,
        use_nstatement_sampler=args.use_nstatement_sampler,
        use_rstatement_sampler=args.use_rstatement_sampler,
    )

    # Full graph from train files, with all relation types
    full_graph = dl.make_data_graph(dl.get_data(), in_dim=mcfg.get("in_feats", 128))

    # Target relations = all edge types that appear in test data
    target_rels = dl.get_test_edge_types()
    if not target_rels:
        # If no explicit test file, fall back to "all relations in training data"
        target_rels = dl.get_edge_types()
        print("[WARN] No explicit test edges found; will create random test splits per relation.")

    print(f"Target relations (from test data): {target_rels}", flush=True)

    # Run full pipeline once per relation
    for ppi_rel in target_rels:
        run_for_relation(ppi_rel, dl, full_graph, cfg, mcfg, args, device)


if __name__ == "__main__":
    main()
