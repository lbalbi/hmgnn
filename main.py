import argparse, os, statistics, torch
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.loader import NeighborLoader

from models import *
from trainer import Train
from trainer_bestmodel import (Train_BestModel, Test_BestModel)
from utils import Logger, load_config
from data_loader import DataLoader
from samplers import (PartialStatementSampler, NegativeStatementSampler, RandomStatementSampler,
    NegativeSampler)


def build_triples_from_wikidata(data_dir: str) -> Dict[str, torch.Tensor]:
    """ Read Wikidata KG files and build train / test triple tensors.
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
        if not os.path.exists(path): raise FileNotFoundError(f"Expected file not found: {path}")
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        required = {"source_node", "target_node", "edge_type"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} must have columns {required}, got {list(df.columns)}")
        df["edge_type"] = df["edge_type"].astype(str)
        return df

    df_train_pos = _read(train_pos_path)
    df_train_neg = _read(train_neg_path)
    df_test_pos = _read(test_pos_path)

    pos_rel_names: List[str] = df_train_pos["edge_type"].astype(str).tolist()
    neg_rel_names_raw: List[str] = df_train_neg["edge_type"].astype(str).tolist()
    neg_rel_names_base: List[str] = []
    for r in neg_rel_names_raw:
        if r.startswith("NOT_"): neg_rel_names_base.append(r[4:])
        else: neg_rel_names_base.append(r)

    test_rel_names: List[str] = df_test_pos["edge_type"].astype(str).tolist()
    all_rel_names = sorted(set(pos_rel_names) | set(neg_rel_names_base) | set(test_rel_names))
    rel2id: Dict[str, int] = {r: i for i, r in enumerate(all_rel_names)}
    id2rel: Dict[int, str] = {i: r for r, i in rel2id.items()}

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

    train_raw_rels = pos_rel_names + neg_rel_names_raw
    if len(train_raw_rels) != train_heads.size(0):
        raise RuntimeError("Length mismatch: train_raw_rels vs train_heads.")
    statement_edge_types_raw = sorted(set(train_raw_rels))

    test_heads = torch.tensor(df_test_pos["source_node"].to_numpy(), dtype=torch.long)
    test_tails = torch.tensor(df_test_pos["target_node"].to_numpy(), dtype=torch.long)
    test_rels = torch.tensor([rel2id[r] for r in test_rel_names], dtype=torch.long)

    return {"train_heads": train_heads, "train_tails": train_tails, "train_rels": train_rels,
        "train_labels": train_labels, "train_raw_rels": train_raw_rels,
        "statement_edge_types_raw": statement_edge_types_raw, "test_heads": test_heads,
        "test_tails": test_tails, "test_rels": test_rels, "rel2id": rel2id,"id2rel": id2rel}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="wikidata",
        help="Task / dataset name (used to load config JSON).")
    parser.add_argument("--model", type=str, choices=["hgcn", "ra_hgcn", "gcn", "gae"],
        default="hgcn", help="Model to run (default is relation-aware HGCN)")
    parser.add_argument("--epochs", type=int, default=250,
        help="Max epochs per fold / final training.")
    parser.add_argument("--batch_size", type=int, default=1024,
        help="Triple batch size for training and testing.")
    parser.add_argument("--path", type=str, default="wikidata_data",
        help="Path to the dataset directory (containing train2id_*.txt etc.).")
    parser.add_argument("--output_dir", type=str, default="output/",
        help="Directory to save logs.")
    parser.add_argument("--num_neg_test", type=int, default=1,
        help="Number of negative test triples to sample per positive triple.")
    parser.add_argument("--use_nstatementsampler", action="store_true",
        help="Use PartialStatementSampler on negative statements removed from the graph.")
    parser.add_argument("--use_pstatementsampler", action="store_true",
        help="Use PartialStatementSampler on positive statements removed from the graph.")
    parser.add_argument("--use_rstatement_sampler", action="store_true",
        help="Use RandomStatementSampler for negative statements.")
    parser.add_argument("--use_contrastive", action="store_true",
        help="Use contrastive learning with statement samplers.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cfg = load_config(task=args.task)
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    if args.model == "ra_hgcn":
        in_feats_cfg = mcfg["in_feats"]
        if isinstance(in_feats_cfg, dict):
            n_type_cfg = mcfg.get("n_type", "node")
            in_dim = int(in_feats_cfg.get(n_type_cfg, list(in_feats_cfg.values())[0]))
        else: in_dim = int(in_feats_cfg)
    else: in_dim = mcfg["in_feats"]

    lr_cfg = cfg.get("lr", 1e-3)
    if isinstance(lr_cfg, (list, tuple)): lr_candidates = [float(lr) for lr in lr_cfg]
    else: lr_candidates = [float(lr_cfg)]
    print(f"Learning rate candidates (per-fold sweep): {lr_candidates}")

    k_folds = int(cfg.get("k_folds", 10))
    contrastive_weight = float(cfg.get("contrastive_weight", 0.1))
    subclass_rel = cfg.get("subclass_rel", "subclass_of")
    instance_rel = cfg.get("instance_rel", "2")  # assuming "2" is instance_of / P31 relation idx
    contrastive_k = int(cfg.get("contrastive_k", 1))

    dl = DataLoader(args.path + "/", use_pstatement_sampler=args.use_pstatementsampler,
        use_nstatement_sampler=args.use_nstatementsampler, use_rstatement_sampler=args.use_rstatement_sampler)
    data_dict = dl.get_data()

    full_graph_all = dl.make_data_graph(data_dict, orthogonal=False)
    num_nodes = full_graph_all["node"].num_nodes
    base_x = full_graph_all["node"].x
    all_e_etypes = list(full_graph_all.edge_types)
    nflag, pflag, rflag = args.use_nstatementsampler, args.use_pstatementsampler, args.use_rstatement_sampler
    use_partial_sampler = nflag or pflag
    use_random_sampler = rflag
    external_edges = dl.get_state_list()

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

    print(f"#Train triples (pos+neg, all types): {train_heads.size(0)}")
    print(f"#Test triples: {test_heads.size(0)}")
    print(f"#Relations (base positive): {len(rel2id)}")

    if instance_rel not in data_dict:
        raise RuntimeError(f"Expected instance-of relation '{instance_rel}' in data_dict keys, "
            f"but got keys: {list(data_dict.keys())}")

    inst_src, _ = data_dict[instance_rel]
    inst_set = set(inst_src.tolist())
    is_instance = torch.zeros(num_nodes, dtype=torch.bool)
    if inst_set:
        inst_idx_tensor = torch.tensor(sorted(inst_set), dtype=torch.long)
        is_instance[inst_idx_tensor] = True

    head_is_inst = is_instance[train_heads]
    tail_is_inst = is_instance[train_tails]
    inst_inst_mask = head_is_inst & tail_is_inst
    inst_class_mask = head_is_inst ^ tail_is_inst
    cls_idx = torch.nonzero(inst_inst_mask, as_tuple=False).view(-1)
    cls_heads = train_heads[cls_idx]
    cls_tails = train_tails[cls_idx]
    cls_rels = train_rels[cls_idx]
    cls_labels = train_labels[cls_idx]
    cls_raw_rels = [train_raw_rels[i] for i in cls_idx.tolist()]

    relation_has_inst_class = set()
    inst_class_mask_list = inst_class_mask.tolist()
    for i, flag in enumerate(inst_class_mask_list):
        if flag: relation_has_inst_class.add(train_raw_rels[i])

    struct_edge_types = set()
    if instance_rel in data_dict: struct_edge_types.add(instance_rel)
    not2_name = f"NOT_{instance_rel}"
    if not2_name in data_dict: struct_edge_types.add(not2_name)
    struct_edge_types.update(relation_has_inst_class)

    if subclass_rel in data_dict: struct_edge_types.add(subclass_rel)

    from torch_geometric.data import HeteroData
    struct_graph = HeteroData()
    struct_graph["node"].num_nodes = int(num_nodes)
    struct_graph["node"].x = base_x.clone()

    for etype in struct_edge_types:
        if etype not in data_dict: continue
        src, tgt = data_dict[etype]
        if src.numel() == 0: edge_index = torch.empty(2, 0, dtype=torch.long)
        else: edge_index = torch.stack([src, tgt], dim=0)
        struct_graph[("node", etype, "node")].edge_index = edge_index
    e_etypes_struct = list(struct_graph.edge_types)

    print("\n=== Classification vs structural split ===")
    print(f"Total training triples:  {train_heads.size(0)}", flush=True)
    print(f"Classification triples (inst-inst): {cls_heads.size(0)}", flush=True)
    print(f"Structural edge types in encoder graph: {sorted(struct_edge_types)}", flush=True)
    print(f"#Encoder graph edge types: {len(e_etypes_struct)}", flush=True)

    neighbor_sizes = [20, 10]
    num_cls_triples = cls_heads.size(0)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    best_epochs: List[int] = []
    best_lrs: List[float] = []
    cv_metrics: List[torch.Tensor] = []

    for fold, (train_idx_np, val_idx_np) in enumerate(
        kf.split(range(num_cls_triples)), start=1):
        train_idx = torch.tensor(train_idx_np, dtype=torch.long)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long)

        train_nodes = torch.unique(torch.cat([cls_heads[train_idx], cls_tails[train_idx]], dim=0))
        val_nodes = torch.unique(torch.cat([cls_heads[val_idx], cls_tails[val_idx]], dim=0))
        num_neighbors = [20, 10]
        train_loader = NeighborLoader(struct_graph, input_nodes=("node", train_nodes),
            num_neighbors=num_neighbors, batch_size=args.batch_size, shuffle=True)
        val_loader = NeighborLoader(struct_graph, input_nodes=("node", val_nodes),
            num_neighbors=num_neighbors, batch_size=args.batch_size, shuffle=False)

        print(f"\n=== Fold {fold}/{k_folds} ===", flush=True)
        print(f"  Train classification triples: {train_idx.numel()} | "
            f"Val classification triples: {val_idx.numel()}", flush=True)

        fold_graph = struct_graph
        base_model_kwargs = dict(in_dim=in_dim, hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]), e_etypes=list(fold_graph.edge_types),
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"))
        if args.model == "ra_hgcn": model_fold = ModelCls(**base_model_kwargs, rel2id=rel2id).to(device)
        else: model_fold = ModelCls(**base_model_kwargs).to(device)

        if args.use_contrastive:
            if use_random_sampler:
                random_sampler = RandomStatementSampler(k=contrastive_k, external_negs=external_edges)
                random_sampler.prepare_global(fold_graph)
                neg_stmt_sampler = random_sampler
            elif use_partial_sampler:
                edges_are_negative = nflag
                neg_stmt_sampler = PartialStatementSampler(k=contrastive_k, neg_edges=external_edges,
                    edges_are_negative=edges_are_negative)
                neg_stmt_sampler.prepare_global(fold_graph)
            else:
                neg_stmt_sampler = NegativeStatementSampler(k=contrastive_k, subclass_rel=subclass_rel, 
                    neg_prefix="NOT_", instance_rel=instance_rel)
                neg_stmt_sampler.prepare_global(fold_graph)

        log_fold = Logger(f"train_cv_fold{fold}", dir=args.output_dir)
        trainer_fold = Train(model=model_fold, graph=fold_graph, heads=cls_heads,
            rel_ids=cls_rels, tails=cls_tails, labels=cls_labels, lr_candidates=lr_candidates,
            epochs=args.epochs, device=device, log=log_fold, batch_size=args.batch_size,
            val_ratio=0.0, early_stopping_patience=cfg.get("patience", 20), train_idx=train_idx,
            val_idx=val_idx, contrastive_sampler=neg_stmt_sampler, contrastive_weight=contrastive_weight,
            train_loader=train_loader, val_loader=val_loader, use_contrastive = args.use_contrastive)

        best_val_loss, best_epoch, best_metrics, best_lr = trainer_fold.run()
        best_epochs.append(int(best_epoch if best_epoch is not None else args.epochs))
        best_lrs.append(float(best_lr) if best_lr is not None else lr_candidates[0])
        if best_metrics is not None: cv_metrics.append(best_metrics)

    if best_epochs: final_epochs = int(statistics.median(best_epochs))
    else: final_epochs = args.epochs
    if best_lrs: final_lr = float(statistics.median(best_lrs))
    else: final_lr = lr_candidates[0]

    print("\n=== Cross-validation summary ===", flush=True)
    print(f"Per-fold best epochs: {best_epochs}", flush=True)
    print(f"Per-fold best learning rates: {best_lrs}", flush=True)
    print(f"Chosen number of epochs for final training (median): {final_epochs}", flush=True)
    print(f"Chosen learning rate for final training (median):  {final_lr}", flush=True)

    final_nodes = torch.unique(torch.cat([cls_heads, cls_tails], dim=0))
    final_loader = NeighborLoader(struct_graph, input_nodes=("node", final_nodes),
        num_neighbors=neighbor_sizes, batch_size=args.batch_size, shuffle=True)

    final_base_kwargs = dict(in_dim=in_dim, hidden_dim=mcfg["hidden_dim"],
        out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]), e_etypes=e_etypes_struct,
        n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"))

    if args.model == "ra_hgcn": final_model = ModelCls(**final_base_kwargs, rel2id=rel2id).to(device)
    else: final_model = ModelCls(**final_base_kwargs).to(device)
    final_log = Logger("final_train_global", dir=args.output_dir, non_verbose=True)

    if args.use_contrastive:
        if use_random_sampler:
            final_contrastive_sampler = RandomStatementSampler(
                k=contrastive_k, external_negs=external_edges)
            final_contrastive_sampler.prepare_global(struct_graph)
        elif use_partial_sampler:
            edges_are_negative = nflag
            final_contrastive_sampler = PartialStatementSampler(k=contrastive_k,
                neg_edges=external_edges, edges_are_negative=edges_are_negative)
            final_contrastive_sampler.prepare_global(struct_graph)
        else:
            final_contrastive_sampler = NegativeStatementSampler(
                k=contrastive_k,subclass_rel=subclass_rel, neg_prefix="NOT_",instance_rel=instance_rel)
            final_contrastive_sampler.prepare_global(struct_graph)

    final_trainer = Train_BestModel(final_model, graph=struct_graph,
        heads=cls_heads, rel_ids=cls_rels, tails=cls_tails, labels=cls_labels, lr=final_lr,
        epochs=final_epochs, device=device, log=final_log, batch_size=args.batch_size,
        contrastive_sampler=final_contrastive_sampler, contrastive_weight=contrastive_weight, loader=final_loader,
        use_contrastive = args.use_contrastive)
    final_loss = final_trainer.run()
    print(f"[Final Train] Loss after {final_epochs} epochs (lr={final_lr:.3g}): {final_loss:.4f}", flush=True)


    neg_samplers: Dict[str, NegativeSampler] = {}
    for rel_name in rel2id.keys():
        key = next((et for et in all_e_etypes if et[1] == rel_name), None)
        if key is None: continue
        all_pos_edge_index = full_graph_all[key].edge_index
        neg_samplers[rel_name] = NegativeSampler(full_graph_all,
            edge_type=key, all_pos_edge_index=all_pos_edge_index)

    test_log = Logger("test_global", dir=args.output_dir)
    tester = Test_BestModel(model=final_model, graph=struct_graph,
        test_heads=test_heads, test_rels=test_rels, test_tails=test_tails,
        id2rel=id2rel, neg_samplers=neg_samplers, num_neg_per_pos=args.num_neg_test,
        device=device, log=test_log, batch_size=args.batch_size)
    tester.run()


if __name__ == "__main__":
    main()
