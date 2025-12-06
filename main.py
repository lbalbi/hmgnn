# main.py
import argparse
import os
from typing import Dict, List, Tuple
import statistics

import torch
import pandas as pd
from sklearn.model_selection import KFold

from models import *
from trainer import Train
from trainer_bestmodel import Train_BestModel, Test_BestModel
from utils import Logger, load_config
from dataloader import DataLoader
from negative_sampler import NegativeSampler
from negativestatement_sampler import NegativeStatementSampler


def build_triples_from_wikidata(
    data_dir: str,
) -> Dict[str, torch.Tensor]:
    """
    Read Wikidata-style KG files and build train / test triple tensors.

    Expected files inside `data_dir`:
      - train2id_pos.txt  with columns: source_node,target_node,edge_type
      - train2id_neg.txt  with columns: source_node,target_node,edge_type (e.g. NOT_3)
      - test2id_pos.txt   with columns: source_node,target_node,edge_type

    For training:
      - Positives use their edge_type directly  (e.g. '3') with label 1.
      - Negatives with edge_type 'NOT_k' are interpreted as relation k with label 0
        for the classifier, but keep the raw type 'NOT_k' for graph edges.
    """
    train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
    train_neg_path = os.path.join(data_dir, "train2id_neg.txt")
    test_pos_path = os.path.join(data_dir, "test2id_pos.txt")

    def _read(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        required = {"source_node", "target_node", "edge_type"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} must have columns {required}, got {list(df.columns)}"
            )
        df["edge_type"] = df["edge_type"].astype(str)
        return df

    df_train_pos = _read(train_pos_path)
    df_train_neg = _read(train_neg_path)
    df_test_pos = _read(test_pos_path)

    # ------------------ Relation names & mapping ------------------
    pos_rel_names: List[str] = df_train_pos["edge_type"].astype(str).tolist()
    neg_rel_names_raw: List[str] = df_train_neg["edge_type"].astype(str).tolist()

    # Base relation names for classifier (strip NOT_ for negatives)
    neg_rel_names_base: List[str] = []
    for r in neg_rel_names_raw:
        if r.startswith("NOT_"):
            neg_rel_names_base.append(r[4:])
        else:
            neg_rel_names_base.append(r)

    test_rel_names: List[str] = df_test_pos["edge_type"].astype(str).tolist()

    # Relations we learn embeddings for are the base positive relations
    all_rel_names = sorted(set(pos_rel_names) | set(neg_rel_names_base) | set(test_rel_names))
    rel2id: Dict[str, int] = {r: i for i, r in enumerate(all_rel_names)}
    id2rel: Dict[int, str] = {i: r for r, i in rel2id.items()}

    # ------------------ Train triples (pos + neg) -----------------
    pos_heads = torch.tensor(df_train_pos["source_node"].to_numpy(), dtype=torch.long)
    pos_tails = torch.tensor(df_train_pos["target_node"].to_numpy(), dtype=torch.long)
    pos_rels = torch.tensor([rel2id[r] for r in pos_rel_names], dtype=torch.long)
    pos_labels = torch.ones_like(pos_heads, dtype=torch.float)

    neg_heads = torch.tensor(df_train_neg["source_node"].to_numpy(), dtype=torch.long)
    neg_tails = torch.tensor(df_train_neg["target_node"].to_numpy(), dtype=torch.long)
    neg_rels = torch.tensor([rel2id[r] for r in neg_rel_names_base], dtype=torch.long)
    neg_labels = torch.zeros_like(neg_heads, dtype=torch.float)

    train_heads = torch.cat([pos_heads, neg_heads], dim=0)
    train_tails = torch.cat([pos_tails, neg_tails], dim=0)
    train_rels = torch.cat([pos_rels, neg_rels], dim=0)
    train_labels = torch.cat([pos_labels, neg_labels], dim=0)

    # Raw edge_type names as they appear in the files (for graph edges)
    train_raw_rels: List[str] = pos_rel_names + neg_rel_names_raw
    if len(train_raw_rels) != train_heads.size(0):
        raise RuntimeError("Length mismatch: train_raw_rels vs train_heads.")

    # For convenience: set of all statement edge types used in training (raw names)
    statement_edge_types_raw = sorted(set(train_raw_rels))

    # ------------------ Test triples (positives only) ------------------
    test_heads = torch.tensor(df_test_pos["source_node"].to_numpy(), dtype=torch.long)
    test_tails = torch.tensor(df_test_pos["target_node"].to_numpy(), dtype=torch.long)
    test_rels = torch.tensor([rel2id[r] for r in test_rel_names], dtype=torch.long)

    return {
        "train_heads": train_heads,
        "train_tails": train_tails,
        "train_rels": train_rels,
        "train_labels": train_labels,
        "train_raw_rels": train_raw_rels,
        "statement_edge_types_raw": statement_edge_types_raw,
        "test_heads": test_heads,
        "test_tails": test_tails,
        "test_rels": test_rels,
        "rel2id": rel2id,
        "id2rel": id2rel,
    }


def build_fold_graph(
    num_nodes: int,
    base_x: torch.Tensor,
    struct_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    train_heads: torch.Tensor,
    train_tails: torch.Tensor,
    train_raw_rels: List[str],
    train_idx: torch.Tensor,
) -> "HeteroData":
    """
    Build a HeteroData graph for a given fold, containing:
      - all structural edges (e.g., subclass_of) from struct_data,
      - only the statement edges corresponding to the training triples
        indexed by `train_idx` (no validation triples as edges).
    """
    from torch_geometric.data import HeteroData  # local import to avoid circulars
    from collections import defaultdict

    g = HeteroData()
    g["node"].num_nodes = int(num_nodes)
    g["node"].x = base_x.clone()

    # Structural edges: all edge types not used as classification statements
    for etype, (src, tgt) in struct_data.items():
        if src.numel() == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            edge_index = torch.stack([src, tgt], dim=0)
        g[("node", etype, "node")].edge_index = edge_index

    # Statement edges from training triples
    per_type_pairs: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    idx_list = train_idx.tolist()
    for i in idx_list:
        et = train_raw_rels[i]
        h = int(train_heads[i].item())
        t = int(train_tails[i].item())
        per_type_pairs[et].append((h, t))

    for etype, pairs in per_type_pairs.items():
        if not pairs:
            continue
        src_nodes = torch.tensor([h for (h, _) in pairs], dtype=torch.long)
        tgt_nodes = torch.tensor([t for (_, t) in pairs], dtype=torch.long)
        edge_index = torch.stack([src_nodes, tgt_nodes], dim=0)
        g[("node", etype, "node")].edge_index = edge_index

    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="wikidata",
        help="Task / dataset name (used to load config JSON).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["hgcn", "hhgcn", "hgat", "gcn", "gae"],
        default="hgcn",
        help="Model to run (RA_HGCN is relation-aware).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Max epochs per fold / final training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Triple batch size for training and testing.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="wikidata_data",
        help="Path to the dataset directory (containing train2id_*.txt etc.).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--num_neg_test",
        type=int,
        default=1,
        help="Number of negative test triples to sample per positive triple.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------- Config -----------------------------
    cfg = load_config(task=args.task)
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    lr_cfg = cfg.get("lr", 1e-3)
    if isinstance(lr_cfg, (list, tuple)):
        base_lr = float(lr_cfg[0])
    else:
        base_lr = float(lr_cfg)

    k_folds = int(cfg.get("k_folds", 10))
    contrastive_weight = float(cfg.get("contrastive_weight", 0.1))
    subclass_rel = cfg.get("subclass_rel", "subclass_of")
    instance_rel = cfg.get("instance_rel", None)
    contrastive_k = int(cfg.get("contrastive_k", 1))

    # -------------------------- Graph (PyG) ---------------------------
    # Training graph built from all training files EXCEPT test2id_pos.txt
    dl = DataLoader(args.path + "/")
    data_dict = dl.get_data()
    full_graph = dl.make_data_graph(data_dict, orthogonal=False)

    e_etypes = list(full_graph.edge_types)
    num_nodes = full_graph["node"].num_nodes
    base_x = full_graph["node"].x

    # ------------------------ Triples (KG) ----------------------------
    triples = build_triples_from_wikidata(args.path)
    train_heads = triples["train_heads"]
    train_tails = triples["train_tails"]
    train_rels = triples["train_rels"]
    train_labels = triples["train_labels"]
    train_raw_rels = triples["train_raw_rels"]
    statement_edge_types_raw = set(triples["statement_edge_types_raw"])

    test_heads = triples["test_heads"]
    test_tails = triples["test_tails"]
    test_rels = triples["test_rels"]
    rel2id = triples["rel2id"]
    id2rel = triples["id2rel"]

    print(f"#Train triples: {train_heads.size(0)} (pos+neg)")
    print(f"#Test  triples: {test_heads.size(0)}")
    print(f"#Relations (base positive): {len(rel2id)}")

    # Separate structural edges (e.g., subclass_of) from statement edges
    struct_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for etype, (src, tgt) in data_dict.items():
        # Any edge_type that appears in train_raw_rels is considered a statement type
        if etype in statement_edge_types_raw:
            continue
        struct_data[etype] = (src, tgt)

    # ---------------------- 10-fold triple-level CV -------------------
    num_triples = train_heads.size(0)
    indices = torch.arange(num_triples, dtype=torch.long)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_epochs: List[int] = []
    cv_metrics: List[torch.Tensor] = []

    for fold, (train_idx_np, val_idx_np) in enumerate(kf.split(range(num_triples)), start=1):
        train_idx = torch.tensor(train_idx_np, dtype=torch.long)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long)

        print(f"\n=== Fold {fold}/{k_folds} ===")
        print(f"  Train triples: {train_idx.numel()} | Val triples: {val_idx.numel()}")

        # Build fold-specific training graph:
        # - contains structural edges (subclass_of, etc.),
        # - contains statement edges ONLY for training triples of this fold.
        fold_graph = build_fold_graph(
            num_nodes=num_nodes,
            base_x=base_x,
            struct_data=struct_data,
            train_heads=train_heads,
            train_tails=train_tails,
            train_raw_rels=train_raw_rels,
            train_idx=train_idx,
        )

        # Model for this fold
        model_fold = ModelCls(
            in_dim=mcfg["in_feats"],
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
            e_etypes=list(fold_graph.edge_types),
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
            rel2id=rel2id,
        ).to(device)

        # NegativeStatementSampler for contrastive learning on this fold's training graph
        neg_stmt_sampler = NegativeStatementSampler(
            k=contrastive_k,
            subclass_rel=subclass_rel,
            neg_prefix="NOT_",
            instance_rel=instance_rel,
        )
        neg_stmt_sampler.prepare_global(fold_graph)

        log_fold = Logger(f"train_cv_fold{fold}", dir=args.output_dir)
        trainer_fold = Train(
            model=model_fold,
            graph=fold_graph,
            heads=train_heads,
            rel_ids=train_rels,
            tails=train_tails,
            labels=train_labels,
            lr=base_lr,
            epochs=args.epochs,
            device=device,
            log=log_fold,
            batch_size=args.batch_size,
            val_ratio=0.0,  # we provide explicit train/val indices
            early_stopping_patience=cfg.get("patience", 20),
            train_idx=train_idx,
            val_idx=val_idx,
            contrastive_sampler=neg_stmt_sampler,
            contrastive_weight=contrastive_weight,
        )

        best_val_loss, best_epoch, best_metrics = trainer_fold.run()
        best_epochs.append(int(best_epoch if best_epoch is not None else args.epochs))
        if best_metrics is not None:
            cv_metrics.append(best_metrics)

    if best_epochs:
        final_epochs = int(statistics.median(best_epochs))
    else:
        final_epochs = args.epochs

    print("\n=== Cross-validation summary ===")
    print(f"Per-fold best epochs: {best_epochs}")
    print(f"Chosen number of epochs for final training: {final_epochs}")

    # ------------------------ Final training -------------------------
    # Use the full training graph (all train triples as edges) for final training.
    # Here we reuse `full_graph` which already contains all training statements.
    final_model = ModelCls(
        in_dim=mcfg["in_feats"],
        hidden_dim=mcfg["hidden_dim"],
        out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
        e_etypes=e_etypes,
        n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        rel2id=rel2id,
    ).to(device)

    final_log = Logger("final_train_global", dir=args.output_dir, non_verbose=True)
    final_trainer = Train_BestModel(
        model=final_model,
        graph=full_graph,
        heads=train_heads,
        rel_ids=train_rels,
        tails=train_tails,
        labels=train_labels,
        lr=base_lr,
        epochs=final_epochs,
        device=device,
        log=final_log,
        batch_size=args.batch_size,
    )
    final_loss = final_trainer.run()
    print(f"[Final Train] Loss after {final_epochs} epochs: {final_loss:.4f}")

    # --------------------- Negative samplers (test) ------------------
    neg_samplers: Dict[str, NegativeSampler] = {}
    for rel_name in rel2id.keys():
        key = next((et for et in e_etypes if et[1] == rel_name), None)
        if key is None:
            continue
        all_pos_edge_index = full_graph[key].edge_index
        neg_samplers[rel_name] = NegativeSampler(
            full_graph,
            edge_type=key,
            all_pos_edge_index=all_pos_edge_index,
        )

    # ---------------------------- Testing ----------------------------
    test_log = Logger("test_global", dir=args.output_dir)
    tester = Test_BestModel(
        model=final_model,
        graph=full_graph,
        test_heads=test_heads,
        test_rels=test_rels,
        test_tails=test_tails,
        id2rel=id2rel,
        neg_samplers=neg_samplers,
        num_neg_per_pos=args.num_neg_test,
        device=device,
        log=test_log,
        batch_size=args.batch_size,
    )
    tester.run()


if __name__ == "__main__":
    main()
