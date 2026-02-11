# main.py
import os
import hashlib
import re
import statistics
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

from data_loader import DataLoader
from models import *
from trainer_NEW import Train
from trainer_bestmodel_NEW import Train_BestModel, Test_BestModel
from utils import Logger, load_config
from samplers import (
    NegativeInstanceSampler_NEW, RandomInstanceSampler, PartialInstanceSampler
)

NEG_PREFIX = "NOT_"
CLS_EDGE_TYPE = ("node", "cls_link", "node")

def _with_cls_edges(
    base_graph: HeteroData,
    heads: torch.Tensor,
    tails: torch.Tensor,
) -> HeteroData:
    g = base_graph.clone()
    device = heads.device
    if heads.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.stack([heads, tails], dim=0)
    g[CLS_EDGE_TYPE].edge_index = edge_index
    return g

def print_encoder_and_cls_totals(
    *,
    encoder_graph: HeteroData,
    subclass_rel: str,
    neg_prefix: str,
    train_heads: torch.Tensor,
    train_rels: torch.Tensor,
    train_tails: torch.Tensor,
    train_labels: torch.Tensor,
    test_heads: torch.Tensor,
    test_rels: torch.Tensor,
    test_tails: torch.Tensor,
    test_labels: torch.Tensor,
) -> None:
    # 1) positives in encoder graph: instance_of + any non-subclass, non-NOT_* (and pos_statement if present)
    # 2) negatives in encoder graph: all NOT_* (and neg_statement if present)
    # 3) subclass_of (subsumption) in encoder graph
    pos_cnt = 0
    neg_cnt = 0
    sub_cnt = 0

    for et in encoder_graph.edge_types:
        if et == CLS_EDGE_TYPE:
            continue

        s, rel, d = et
        if s != "node" or d != "node":
            continue

        ei = encoder_graph[et].edge_index
        n = int(ei.size(1)) if (ei is not None and ei.numel() > 0) else 0
        rel_s = str(rel)

        if rel_s == str(subclass_rel):
            sub_cnt += n
        elif rel_s.startswith(str(neg_prefix)) or rel_s == "neg_statement":
            neg_cnt += n
        elif rel_s == "pos_statement":
            pos_cnt += n
        else:
            pos_cnt += n

    # 4) unique classification examples used in total (train + test)
    def _unique_examples(h, r, t, y) -> torch.Tensor:
        if h is None or h.numel() == 0:
            return torch.empty((0, 4), dtype=torch.long)
        yb = (y.detach().cpu() > 0.5).long()
        return torch.stack(
            [h.detach().cpu().long(), r.detach().cpu().long(), t.detach().cpu().long(), yb],
            dim=1,
        )

    all_ex = torch.cat(
        [
            _unique_examples(train_heads, train_rels, train_tails, train_labels),
            _unique_examples(test_heads, test_rels, test_tails, test_labels),
        ],
        dim=0,
    )
    uniq_cls_total = int(torch.unique(all_ex, dim=0).size(0)) if all_ex.numel() else 0

    print(f"1) how many positive statements/triples are in the encoder graph: {pos_cnt}")
    print(f"2) how many negative statements/triples are in the encoder graph: {neg_cnt}")
    print(f"3) how many ontology subsumption relations (subclass_of) are in the encoder graph: {sub_cnt}")
    print(f"4) how many unique classification examples are used in total (training + testing): {uniq_cls_total}")

def print_final_train_encoder_statement_counts(
    *,
    encoder_graph: HeteroData,
    subclass_rel: str,
    neg_prefix: str,
) -> None:
    """Print counts for the full/final train encoder graph (excluding subclass_of)."""
    pos_cnt = 0
    neg_cnt = 0

    for et in encoder_graph.edge_types:
        if et == CLS_EDGE_TYPE:
            continue

        s, rel, d = et
        if s != "node" or d != "node":
            continue

        ei = encoder_graph[et].edge_index
        n = int(ei.size(1)) if (ei is not None and ei.numel() > 0) else 0
        rel_s = str(rel)

        if rel_s == str(subclass_rel):
            continue
        if rel_s.startswith(str(neg_prefix)) or rel_s == "neg_statement":
            neg_cnt += n
        else:
            pos_cnt += n

    print("\n=== Final train encoder graph statement counts ===")
    print(f"Positive statements (excluding subclass_of): {pos_cnt}")
    print(f"Negative statements:                         {neg_cnt}")


def report_unseen_test_nodes(
    *,
    test_heads: torch.Tensor,
    test_tails: torch.Tensor,
    train_heads: torch.Tensor,
    train_tails: torch.Tensor,
    encoder_graph: HeteroData,
    num_nodes: int,
    topk: int = 25,
) -> None:
    train_nodes = torch.unique(torch.cat([train_heads, train_tails], dim=0)).detach().cpu().long()
    test_nodes = torch.unique(torch.cat([test_heads, test_tails], dim=0)).detach().cpu().long()

    seen = torch.zeros(int(num_nodes), dtype=torch.bool)
    if train_nodes.numel() > 0:
        seen[train_nodes] = True

    enc_nodes_chunks = []
    for et in encoder_graph.edge_types:
        ei = encoder_graph[et].edge_index
        if ei is None or ei.numel() == 0:
            continue
        enc_nodes_chunks.append(ei.reshape(-1).detach().cpu().long())
    if enc_nodes_chunks:
        enc_nodes = torch.unique(torch.cat(enc_nodes_chunks, dim=0))
        seen[enc_nodes] = True
    else:
        enc_nodes = torch.empty(0, dtype=torch.long)

    unseen_test_nodes = test_nodes[~seen[test_nodes]]

    test_heads_u = torch.unique(test_heads.detach().cpu().long())
    test_tails_u = torch.unique(test_tails.detach().cpu().long())
    unseen_heads = test_heads_u[~seen[test_heads_u]]
    unseen_tails = test_tails_u[~seen[test_tails_u]]

    print("\n================ TEST NODE COVERAGE CHECK ================")
    print(f"Unique TRAIN nodes (from train cls examples):           {int(train_nodes.numel())}")
    print(f"Unique ENCODER-edge nodes (any encoder edge endpoint): {int(enc_nodes.numel())}")
    print(f"Unique TEST nodes (from test heads/tails):             {int(test_nodes.numel())}")
    print(f"Unique TEST nodes unseen in BOTH:                      {int(unseen_test_nodes.numel())}")

    if test_nodes.numel() > 0:
        frac = float(unseen_test_nodes.numel()) / float(test_nodes.numel())
        print(f"Unseen fraction of unique test nodes: {frac:.4%}")

    print(f"Unique TEST head nodes unseen: {int(unseen_heads.numel())}")
    print(f"Unique TEST tail nodes unseen: {int(unseen_tails.numel())}")

    if unseen_test_nodes.numel() > 0:
        ex = unseen_test_nodes[:topk].tolist()
        print(f"Example unseen node ids (first {min(topk, len(ex))}): {ex}")

def report_cls_nodes_not_in_encoder(
    *,
    train_heads: torch.Tensor,
    train_tails: torch.Tensor,
    test_heads: torch.Tensor,
    test_tails: torch.Tensor,
    encoder_graph: HeteroData,
    num_nodes: int,
    topk: int = 25,
) -> None:
    # Unique nodes in classification triples
    train_nodes = torch.unique(torch.cat([train_heads, train_tails], dim=0)).detach().cpu().long()
    test_nodes  = torch.unique(torch.cat([test_heads,  test_tails],  dim=0)).detach().cpu().long()
    all_cls_nodes = torch.unique(torch.cat([train_nodes, test_nodes], dim=0)).detach().cpu().long()

    # Unique nodes that appear as endpoints of ANY encoder edge
    enc_nodes_chunks = []
    for et in encoder_graph.edge_types:
        ei = encoder_graph[et].edge_index
        if ei is None or ei.numel() == 0:
            continue
        enc_nodes_chunks.append(ei.reshape(-1).detach().cpu().long())

    if enc_nodes_chunks:
        enc_nodes = torch.unique(torch.cat(enc_nodes_chunks, dim=0))
    else:
        enc_nodes = torch.empty(0, dtype=torch.long)

    seen_enc = torch.zeros(int(num_nodes), dtype=torch.bool)
    if enc_nodes.numel() > 0:
        seen_enc[enc_nodes] = True

    # Missing nodes = in classification triples, but not in ANY encoder edge endpoint
    missing_all  = all_cls_nodes[~seen_enc[all_cls_nodes]]
    missing_tr   = train_nodes[~seen_enc[train_nodes]]
    missing_te   = test_nodes[~seen_enc[test_nodes]]

    def _frac(missing: torch.Tensor, total: torch.Tensor) -> float:
        return float(missing.numel()) / float(total.numel()) if total.numel() else 0.0

    print("\n============= CLASSIFICATION NODE COVERAGE IN ENCODER =============")
    print(f"Unique encoder-edge endpoint nodes:            {int(enc_nodes.numel())}")
    print(f"Unique classification nodes (train+test):      {int(all_cls_nodes.numel())}")
    print(f"Classification nodes NOT in encoder edges:     {int(missing_all.numel())} ({_frac(missing_all, all_cls_nodes):.4%})")
    print(f"  - Train cls nodes NOT in encoder edges:      {int(missing_tr.numel())} ({_frac(missing_tr, train_nodes):.4%})")
    print(f"  - Test  cls nodes NOT in encoder edges:      {int(missing_te.numel())} ({_frac(missing_te, test_nodes):.4%})")

    if missing_all.numel() > 0:
        ex = missing_all[:topk].tolist()
        print(f"Example missing node ids (first {min(topk, len(ex))}): {ex}")


# ============================================================
# Deterministic helpers + split cache
# ============================================================

def _stable_hash_int(s: str, mod: int = 1_000_000) -> int:
    h = hashlib.md5(str(s).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % int(mod)

def _sanitize_component(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

def _file_fingerprint(path: str) -> Dict[str, object]:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}

def _make_split_cache_path(
    data_dir: str,
    task: str,
    test_ratio: float,
    seed: int,
    subclass_rel: str,
    instance_rel: str,
    min_pos_per_rel: int,
) -> str:
    cache_dir = os.path.join(data_dir, "split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fname = (
        f"splits_{_sanitize_component(task)}"
        f"_testr{float(test_ratio):.3f}"
        f"_seed{int(seed)}"
        f"_sub{_sanitize_component(subclass_rel)}"
        f"_inst{_sanitize_component(instance_rel)}"
        f"_minpos{int(min_pos_per_rel)}"
        ".pt"
    )
    return os.path.join(cache_dir, fname)

def _build_split_cache_meta(
    data_dir: str,
    task: str,
    test_ratio: float,
    seed: int,
    subclass_rel: str,
    instance_rel: str,
    min_pos_per_rel: int,
) -> Dict[str, object]:
    train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
    train_neg_path = os.path.join(data_dir, "train2id_neg.txt")
    return {
        "version": 3,  # bumped because encoder-augmentation semantics changed
        "task": str(task),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "subclass_rel": str(subclass_rel),
        "instance_rel": str(instance_rel),
        "min_pos_per_rel": int(min_pos_per_rel),
        "files": {
            "train2id_pos": _file_fingerprint(train_pos_path),
            "train2id_neg": _file_fingerprint(train_neg_path),
        },
    }

def _meta_matches(cached: Dict[str, object], current: Dict[str, object]) -> bool:
    keys = ["version", "task", "test_ratio", "seed", "subclass_rel", "instance_rel", "min_pos_per_rel"]
    for k in keys:
        if cached.get(k) != current.get(k):
            return False
    cfiles = cached.get("files", {})
    nfiles = current.get("files", {})
    for name in ["train2id_pos", "train2id_neg"]:
        if cfiles.get(name) != nfiles.get(name):
            return False
    return True

def _try_load_split_cache(cache_path: str, expected_meta: Dict[str, object]) -> Optional[Dict[str, object]]:
    if not os.path.exists(cache_path):
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict) or "meta" not in payload or "split" not in payload:
            print(f"[SplitCache] Found cache but format is unexpected: {cache_path}")
            return None
        if not _meta_matches(payload["meta"], expected_meta):
            print("[SplitCache] Cache exists but is NOT valid for current data/params. Recomputing.")
            return None
        print(f"[SplitCache] Loaded cached split: {cache_path}")
        return payload["split"]
    except Exception as e:
        print(f"[SplitCache] Failed to load cache ({cache_path}): {e}")
        return None

def _save_split_cache(cache_path: str, meta: Dict[str, object], split_out: Dict[str, object]) -> None:
    tmp_path = cache_path + ".tmp"
    torch.save({"meta": meta, "split": split_out}, tmp_path)
    os.replace(tmp_path, cache_path)
    print(f"[SplitCache] Saved split cache: {cache_path}")


# ============================================================
# Data IO + encoder graph helper
# ============================================================

def _read_edge_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = {"source_node", "target_node", "edge_type"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must have columns {required}, got {list(df.columns)}")
    df["edge_type"] = df["edge_type"].astype(str)
    return df

def _strip_not(rel: str, neg_prefix: str = NEG_PREFIX) -> str:
    rel = str(rel)
    return rel[len(neg_prefix):] if rel.startswith(neg_prefix) else rel

def _ensure_not_prefixed(rel: str, neg_prefix: str = NEG_PREFIX) -> str:
    rel = str(rel)
    return rel if rel.startswith(neg_prefix) else (neg_prefix + rel)

def _append_edges(struct_graph: HeteroData, rel: str, src: torch.Tensor, tgt: torch.Tensor) -> None:
    rel = str(rel)
    src = src.long()
    tgt = tgt.long()
    edge_index = torch.stack([src, tgt], dim=0) if src.numel() else torch.empty(2, 0, dtype=torch.long)

    key = ("node", rel, "node")
    if key in struct_graph.edge_types:
        old = struct_graph[key].edge_index
        if old is None or old.numel() == 0:
            struct_graph[key].edge_index = edge_index
        elif edge_index.numel() == 0:
            pass
        else:
            struct_graph[key].edge_index = torch.cat([old, edge_index], dim=1)
    else:
        struct_graph[key].edge_index = edge_index

def build_hgcn_encoder_graph(
    struct_graph: HeteroData,
    subclass_rel: str = "subclass_of",
    neg_prefix: str = "NOT_",
) -> HeteroData:
    num_nodes = int(struct_graph["node"].num_nodes)
    x = struct_graph["node"].x

    enc_graph = HeteroData()
    enc_graph["node"].num_nodes = num_nodes
    enc_graph["node"].x = x.clone()

    subclass_ei = None
    pos_edges = []
    neg_edges = []

    for (s, rel, d) in struct_graph.edge_types:
        if s != "node" or d != "node":
            continue
        edge_index = struct_graph[(s, rel, d)].edge_index
        if rel == subclass_rel:
            subclass_ei = edge_index if subclass_ei is None else torch.cat([subclass_ei, edge_index], dim=1)
        elif str(rel).startswith(neg_prefix):
            if edge_index.numel() > 0:
                neg_edges.append(edge_index)
        else:
            if edge_index.numel() > 0:
                pos_edges.append(edge_index)

    if subclass_ei is None:
        subclass_ei = torch.empty(2, 0, dtype=torch.long)

    enc_graph[("node", subclass_rel, "node")].edge_index = subclass_ei
    pos_ei = torch.cat(pos_edges, dim=1) if pos_edges else torch.empty(2, 0, dtype=torch.long)
    neg_ei = torch.cat(neg_edges, dim=1) if neg_edges else torch.empty(2, 0, dtype=torch.long)

    enc_graph[("node", "pos_statement", "node")].edge_index = pos_ei
    enc_graph[("node", "neg_statement", "node")].edge_index = neg_ei
    return enc_graph


# ============================================================
# Classification dataset construction + splitting
# ============================================================

def _balanced_downsample_per_relation(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    *,
    seed: int,
    min_pos_per_rel: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int, int]]]:
    """
    Build a per-relation 1:1 balanced dataset by downsampling larger class to smaller class.
    Returns:
      df_balanced
      stats[r] = (npos, nneg, m_kept_each)
    """
    stats: Dict[str, Tuple[int, int, int]] = {}
    chunks: List[pd.DataFrame] = []

    rels_pos = set(df_pos["rel_base"].unique().tolist())
    rels_neg = set(df_neg["rel_base"].unique().tolist())
    rels = sorted(list(rels_pos & rels_neg))

    dropped_minpos = []
    dropped_empty = []

    for r in rels:
        pos_rows = df_pos[df_pos["rel_base"] == r]
        neg_rows = df_neg[df_neg["rel_base"] == r]
        npos = int(len(pos_rows))
        nneg = int(len(neg_rows))

        if min_pos_per_rel and npos < int(min_pos_per_rel):
            dropped_minpos.append(r)
            continue

        m = min(npos, nneg)
        if m <= 0:
            dropped_empty.append(r)
            continue

        rng = np.random.RandomState(int(seed) + _stable_hash_int(r, 1_000_000))
        pos_idx = pos_rows.index.to_numpy(dtype=np.int64).copy()
        neg_idx = neg_rows.index.to_numpy(dtype=np.int64).copy()
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        pos_sel = pos_idx[:m]
        neg_sel = neg_idx[:m]

        pos_chunk = df_pos.loc[pos_sel].copy()
        neg_chunk = df_neg.loc[neg_sel].copy()

        chunks.append(pos_chunk)
        chunks.append(neg_chunk)
        stats[r] = (npos, nneg, int(m))

    if dropped_minpos:
        print(f"\n[WARN] Dropped {len(dropped_minpos)} relations due to min_pos_per_rel={min_pos_per_rel}. (first 50)")
        print(dropped_minpos[:50])
    if dropped_empty:
        print(f"\n[WARN] Dropped {len(dropped_empty)} relations with empty pos or neg after filtering. (first 50)")
        print(dropped_empty[:50])

    if not chunks:
        return pd.DataFrame(columns=["source_node", "target_node", "edge_type", "rel_base", "label"]), stats

    df_bal = pd.concat(chunks, ignore_index=True)
    return df_bal, stats

def _per_relation_label_split_indices(
    df: pd.DataFrame,
    *,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[int, Tuple[int, int]]]]:
    train_idx: List[int] = []
    test_idx: List[int] = []
    counts: Dict[str, Dict[int, Tuple[int, int]]] = {}

    if len(df) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), counts

    rels = sorted(df["rel_base"].unique().tolist())
    for r in rels:
        pos_ix = df.index[(df["rel_base"] == r) & (df["label"] > 0.5)].to_numpy(dtype=np.int64)
        neg_ix = df.index[(df["rel_base"] == r) & (df["label"] <= 0.5)].to_numpy(dtype=np.int64)

        if min(len(pos_ix), len(neg_ix)) < 2:
            train_idx.extend(pos_ix.tolist())
            train_idx.extend(neg_ix.tolist())
            counts[r] = {1: (int(len(pos_ix)), 0), 0: (int(len(neg_ix)), 0)}
            continue

        counts[r] = {}
        for lab, ix0 in [(1, pos_ix), (0, neg_ix)]:
            ix = ix0.copy()
            rng = np.random.RandomState(int(seed) + _stable_hash_int(f"{r}|{lab}", 1_000_000))
            rng.shuffle(ix)

            n = int(len(ix))
            n_test = int(round(float(test_ratio) * n))
            n_test = max(1, min(n - 1, n_test))

            te = ix[:n_test]
            tr = ix[n_test:]

            test_idx.extend(te.tolist())
            train_idx.extend(tr.tolist())
            counts[r][lab] = (int(len(tr)), int(len(te)))

    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64), counts

def _df_to_tensors(df: pd.DataFrame, rel2id: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = torch.tensor(df["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    tails = torch.tensor(df["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    rels = torch.tensor(df["rel_base"].map(rel2id).to_numpy(dtype=np.int64), dtype=torch.long)
    labels = torch.tensor(df["label"].to_numpy(dtype=np.float32), dtype=torch.float)
    return heads, rels, tails, labels

def _counts_per_relation(rels: torch.Tensor, labels: torch.Tensor, id2rel: Dict[int, str]) -> pd.DataFrame:
    rels_cpu = rels.detach().cpu().long()
    lab_cpu = labels.detach().cpu().float()
    num_rel = int(rels_cpu.max().item()) + 1 if rels_cpu.numel() else 0
    if num_rel == 0:
        return pd.DataFrame(columns=["rel_id", "rel_name", "pos", "neg", "total"])

    pos = torch.bincount(rels_cpu[lab_cpu > 0.5], minlength=num_rel)
    neg = torch.bincount(rels_cpu[lab_cpu <= 0.5], minlength=num_rel)
    total = pos + neg

    rows = []
    for rid in range(num_rel):
        t = int(total[rid].item())
        if t == 0:
            continue
        rows.append({
            "rel_id": rid,
            "rel_name": id2rel.get(rid, str(rid)),
            "pos": int(pos[rid].item()),
            "neg": int(neg[rid].item()),
            "total": t,
        })
    return pd.DataFrame(rows).sort_values(["total", "rel_name"], ascending=[False, True]).reset_index(drop=True)

def print_pos_neg_summary_train_test(
    train_rels: torch.Tensor,
    train_labels: torch.Tensor,
    test_rels: torch.Tensor,
    test_labels: torch.Tensor,
    id2rel: Dict[int, str],
    *,
    topk: int = 200,
) -> None:
    print("\n================ POS/NEG COUNTS PER RELATION ================")
    df_train = _counts_per_relation(train_rels, train_labels, id2rel)
    df_test = _counts_per_relation(test_rels, test_labels, id2rel) if test_rels.numel() else pd.DataFrame()

    print("\n--- TRAIN (pos/neg/total) ---")
    print("(empty)" if len(df_train) == 0 else df_train.head(topk).to_string(index=False))

    print("\n--- TEST (pos/neg/total) ---")
    if test_rels.numel() == 0 or len(df_test) == 0:
        print("(empty)")
    else:
        print(df_test.head(topk).to_string(index=False))

    if train_rels.numel():
        print("\n--- TRAIN totals ---")
        print("Train positives:", int((train_labels > 0.5).sum().item()))
        print("Train negatives:", int((train_labels <= 0.5).sum().item()))
        print("Train relations:", int(train_rels.unique().numel()))
    if test_rels.numel():
        print("\n--- TEST totals ---")
        print("Test positives:", int((test_labels > 0.5).sum().item()))
        print("Test negatives:", int((test_labels <= 0.5).sum().item()))
        print("Test relations:", int(test_rels.unique().numel()))

def build_classification_splits_from_train_files(
    data_dir: str,
    *,
    subclass_rel: str,
    instance_rel: str,
    test_ratio: float,
    seed: int,
    min_pos_per_rel: int = 0,
) -> Dict[str, object]:
    """
    - Uses ONLY train2id_pos + train2id_neg.
    - Excludes subclass_of / instance_of from classification.
    - Balances per relation (1:1) by downsampling.
    - Splits 80/20 per relation AND per label -> keeps balance per relation.
    - Returns balance_stats so we can later add LEFTOVERS into encoder graph.
    """
    train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
    train_neg_path = os.path.join(data_dir, "train2id_neg.txt")

    df_pos_all = _read_edge_file(train_pos_path)
    df_neg_all = _read_edge_file(train_neg_path)

    df_pos_all["rel_base"] = df_pos_all["edge_type"].astype(str)
    df_pos_all["label"] = 1.0

    df_neg_all["rel_base"] = df_neg_all["edge_type"].astype(str).map(_strip_not)
    df_neg_all["label"] = 0.0

    excluded = {str(subclass_rel), str(instance_rel), "subclass_of", "instance_of"}

    df_pos_cls = df_pos_all[~df_pos_all["rel_base"].isin(excluded)].copy()
    df_neg_cls = df_neg_all[~df_neg_all["rel_base"].isin(excluded)].copy()

    if min_pos_per_rel and int(min_pos_per_rel) > 0:
        pos_counts = df_pos_cls["rel_base"].value_counts().to_dict()
        keep_rels = {r for r, c in pos_counts.items() if int(c) >= int(min_pos_per_rel)}
        before = len(df_pos_cls)
        df_pos_cls = df_pos_cls[df_pos_cls["rel_base"].isin(keep_rels)].copy()
        df_neg_cls = df_neg_cls[df_neg_cls["rel_base"].isin(keep_rels)].copy()
        print(f"\n[INFO] min_pos_per_rel={min_pos_per_rel}: kept {len(keep_rels)} relations. "
              f"pos rows: {before} -> {len(df_pos_cls)}")

    df_balanced, balance_stats = _balanced_downsample_per_relation(
        df_pos_cls, df_neg_cls, seed=int(seed), min_pos_per_rel=int(min_pos_per_rel) if min_pos_per_rel else 0
    )
    if len(df_balanced) == 0:
        raise RuntimeError("After filtering/balancing, classification dataset is empty.")

    rels = sorted(df_balanced["rel_base"].unique().tolist())
    rel2id = {r: i for i, r in enumerate(rels)}
    id2rel = {i: r for r, i in rel2id.items()}

    train_idx, test_idx, split_counts = _per_relation_label_split_indices(
        df_balanced, test_ratio=float(test_ratio), seed=int(seed)
    )

    df_train = df_balanced.loc[train_idx].copy().reset_index(drop=True)
    df_test = df_balanced.loc[test_idx].copy().reset_index(drop=True)

    tr_h, tr_r, tr_t, tr_y = _df_to_tensors(df_train, rel2id)
    te_h, te_r, te_t, te_y = _df_to_tensors(df_test, rel2id)

    print("\n=== Classification dataset (from TRAIN files only) ===")
    print(f"Excluded from classification (encoder-only): {sorted(list(excluded))}")
    print(f"Relations in classification: {len(rel2id)}")
    print(f"Balanced dataset total examples: {len(df_balanced)} "
          f"(pos={int((df_balanced['label']>0.5).sum())}, neg={int((df_balanced['label']<=0.5).sum())})")
    print(f"Train examples: {len(df_train)} | Test examples: {len(df_test)}")

    show = 20
    print(f"\n[Split sanity] Showing first {show} relations with (train_pos, test_pos, train_neg, test_neg):")
    for r in rels[:show]:
        c = split_counts.get(r, {})
        tp = c.get(1, (0, 0))
        tn = c.get(0, (0, 0))
        print(f"  rel={r:>20}  pos(tr={tp[0]:>5}, te={tp[1]:>5})  neg(tr={tn[0]:>5}, te={tn[1]:>5})")

    return {
        "train_heads": tr_h,
        "train_rels": tr_r,
        "train_tails": tr_t,
        "train_labels": tr_y,
        "test_heads": te_h,
        "test_rels": te_r,
        "test_tails": te_t,
        "test_labels": te_y,
        "rel2id": rel2id,
        "id2rel": id2rel,
        "rel_list": rels,
        "balance_stats": balance_stats,  # <-- used to compute leftovers for encoder
    }


# ============================================================
# Compute leftover (unused) classification examples for encoder
# ============================================================
# def compute_leftover_examples_for_encoder(
#     data_dir: str,
#     *,
#     rel_list: List[str],
#     balance_stats: Dict[str, Tuple[int, int, int]],
#     seed: int,
#     subclass_rel: str,
#     instance_rel: str,
#     min_pos_per_rel: int = 0,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Returns:
#       leftover_pos_df: rows for rel_list that were NOT used in balanced dataset (positives)
#       leftover_neg_df: rows for rel_list that were NOT used in balanced dataset (negatives)
#       enc_only_neg_df: ONLY negatives for subclass_of/instance_of (if exist) from train2id_neg
#     """
#     train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
#     train_neg_path = os.path.join(data_dir, "train2id_neg.txt")

#     df_pos_all = _read_edge_file(train_pos_path)
#     df_neg_all = _read_edge_file(train_neg_path)

#     df_pos_all["rel_base"] = df_pos_all["edge_type"].astype(str)
#     df_neg_all["rel_base"] = df_neg_all["edge_type"].astype(str).map(_strip_not)

#     encoder_only_bases = {str(subclass_rel), str(instance_rel), "subclass_of", "instance_of"}

#     # Only these negatives go to encoder (from file), per your requirement:
#     enc_only_neg_df = df_neg_all[df_neg_all["rel_base"].isin(encoder_only_bases)].copy()

#     # Leftovers: only for relations that are classification relations (rel_list)
#     df_pos_cls = df_pos_all[~df_pos_all["rel_base"].isin(encoder_only_bases)].copy()
#     df_neg_cls = df_neg_all[~df_neg_all["rel_base"].isin(encoder_only_bases)].copy()

#     if min_pos_per_rel and int(min_pos_per_rel) > 0:
#         pos_counts = df_pos_cls["rel_base"].value_counts().to_dict()
#         keep_rels = {r for r, c in pos_counts.items() if int(c) >= int(min_pos_per_rel)}
#         df_pos_cls = df_pos_cls[df_pos_cls["rel_base"].isin(keep_rels)].copy()
#         df_neg_cls = df_neg_cls[df_neg_cls["rel_base"].isin(keep_rels)].copy()

#     leftover_pos_chunks = []
#     leftover_neg_chunks = []

#     rel_set = set(rel_list)
#     for r in rel_list:
#         if r not in balance_stats:
#             continue
#         pos_rows = df_pos_cls[df_pos_cls["rel_base"] == r]
#         neg_rows = df_neg_cls[df_neg_cls["rel_base"] == r]
#         npos = int(len(pos_rows))
#         nneg = int(len(neg_rows))
#         m = int(balance_stats[r][2])

#         # Safety: if stats say keep m, but data changed, recompute m.
#         m = max(0, min(m, npos, nneg))

#         rng = np.random.RandomState(int(seed) + _stable_hash_int(r, 1_000_000))
#         pos_idx = pos_rows.index.to_numpy(dtype=np.int64).copy()
#         neg_idx = neg_rows.index.to_numpy(dtype=np.int64).copy()
#         rng.shuffle(pos_idx)
#         rng.shuffle(neg_idx)

#         pos_left = pos_idx[m:]
#         neg_left = neg_idx[m:]

#         if pos_left.size:
#             leftover_pos_chunks.append(df_pos_cls.loc[pos_left].copy())
#         if neg_left.size:
#             leftover_neg_chunks.append(df_neg_cls.loc[neg_left].copy())

#     leftover_pos_df = pd.concat(leftover_pos_chunks, ignore_index=True) if leftover_pos_chunks else pd.DataFrame(columns=df_pos_all.columns.tolist())
#     leftover_neg_df = pd.concat(leftover_neg_chunks, ignore_index=True) if leftover_neg_chunks else pd.DataFrame(columns=df_neg_all.columns.tolist())

#     return leftover_pos_df, leftover_neg_df, enc_only_neg_df
def compute_leftover_examples_for_encoder(data_dir: str, *,
    rel_list: List[str], balance_stats: Dict[str, Tuple[int, int, int]],
    seed: int, subclass_rel: str, instance_rel: str,
    min_pos_per_rel: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
    train_neg_path = os.path.join(data_dir, "train2id_neg.txt")
    df_pos_all = _read_edge_file(train_pos_path)
    df_neg_all = _read_edge_file(train_neg_path)
    df_pos_all["rel_base"] = df_pos_all["edge_type"].astype(str)
    df_neg_all["rel_base"] = df_neg_all["edge_type"].astype(str).map(_strip_not)

    encoder_only_bases = {str(subclass_rel), str(instance_rel), "subclass_of", "instance_of"}
    enc_only_neg_df = df_neg_all[df_neg_all["rel_base"].isin(encoder_only_bases)].copy()
    df_pos_cls = df_pos_all[~df_pos_all["rel_base"].isin(encoder_only_bases)].copy()
    df_neg_cls = df_neg_all[~df_neg_all["rel_base"].isin(encoder_only_bases)].copy()

    used_rels = set(balance_stats.keys())
    all_rels = sorted(set(df_pos_cls["rel_base"].unique().tolist()) |
        set(df_neg_cls["rel_base"].unique().tolist()))

    leftover_pos_chunks: List[pd.DataFrame] = []
    leftover_neg_chunks: List[pd.DataFrame] = []
    dropped_rels: List[str] = []

    for r in all_rels:
        pos_rows = df_pos_cls[df_pos_cls["rel_base"] == r]
        neg_rows = df_neg_cls[df_neg_cls["rel_base"] == r]
        if r in used_rels:
            npos = int(len(pos_rows))
            nneg = int(len(neg_rows))
            m = int(balance_stats[r][2])
            m = max(0, min(m, npos, nneg))

            rng = np.random.RandomState(int(seed) + _stable_hash_int(r, 1_000_000))
            pos_idx = pos_rows.index.to_numpy(dtype=np.int64).copy()
            neg_idx = neg_rows.index.to_numpy(dtype=np.int64).copy()
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)
            pos_left = pos_idx[m:]
            neg_left = neg_idx[m:]

            if pos_left.size: leftover_pos_chunks.append(df_pos_cls.loc[pos_left].copy())
            if neg_left.size: leftover_neg_chunks.append(df_neg_cls.loc[neg_left].copy())
        else:
            dropped_rels.append(r)
            if len(pos_rows) > 0:
                leftover_pos_chunks.append(pos_rows.copy())
            if len(neg_rows) > 0:
                leftover_neg_chunks.append(neg_rows.copy())
    leftover_pos_df = (pd.concat(leftover_pos_chunks, ignore_index=True)
        if leftover_pos_chunks else pd.DataFrame(columns=df_pos_all.columns.tolist()))
    leftover_neg_df = (pd.concat(leftover_neg_chunks, ignore_index=True)
        if leftover_neg_chunks else pd.DataFrame(columns=df_neg_all.columns.tolist()))
    if dropped_rels:
        print(f"\n[INFO] Added dropped relations fully to encoder leftovers: {len(dropped_rels)} (first 50)")
        print(dropped_rels[:50])
    return leftover_pos_df, leftover_neg_df, enc_only_neg_df


class FileNegativeSampler:
    """
    File-backed negatives, compatible with trainer_bestmodel.Test_BestModel,
    which calls sampler.sample_for_heads(...).
    """

    def __init__(
        self,
        neg_heads: torch.Tensor,
        neg_tails: torch.Tensor,
        *,
        num_nodes: int,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.device = device if device is not None else torch.device("cpu")
        self.rng = np.random.RandomState(int(seed))

        nh = neg_heads.detach().cpu().long().numpy() if neg_heads is not None else np.zeros((0,), dtype=np.int64)
        nt = neg_tails.detach().cpu().long().numpy() if neg_tails is not None else np.zeros((0,), dtype=np.int64)

        self.head2tails: Dict[int, List[int]] = defaultdict(list)
        for h, t in zip(nh.tolist(), nt.tolist()):
            self.head2tails[int(h)].append(int(t))

        self.global_tails: List[int] = nt.tolist()

        # deterministic shuffles
        for h in list(self.head2tails.keys()):
            self.rng.shuffle(self.head2tails[h])
        self.rng.shuffle(self.global_tails)

    def _pick_tail(
        self,
        h: int,
        *,
        invalid_ids: Set[int],
    ) -> int:
        """
        Pick a single tail for head h from file negatives:
        - prefer head-specific pool
        - fallback to global pool
        - fallback to random tail if pools empty
        Resamples a few times to avoid invalid_ids.
        """
        pool = self.head2tails.get(h, None)
        if pool is None or len(pool) == 0:
            pool = self.global_tails

        # Try a few times to avoid invalid ids
        if pool and len(pool) > 0:
            for _ in range(20):
                t = pool[self.rng.randint(0, len(pool))]
                if (h * self.num_nodes + t) not in invalid_ids:
                    return int(t)

            # If everything seems invalid, just return something (best effort)
            return int(pool[self.rng.randint(0, len(pool))])

        # Absolute fallback
        return int(self.rng.randint(0, self.num_nodes))

    def sample_for_heads(
        self,
        heads: torch.Tensor,
        num_negs_per_head: torch.Tensor,
        *,
        extra_invalid_ids: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match NegativeSampler API used by Test_BestModel:
          heads: [H]
          num_negs_per_head: [H] (how many negatives for each head)
        Returns:
          neg_src: [sum_k]
          neg_dst: [sum_k]
        """
        if extra_invalid_ids is None:
            extra_invalid_ids = set()

        h_list = heads.detach().cpu().long().tolist()
        k_list = num_negs_per_head.detach().cpu().long().tolist()

        neg_src: List[int] = []
        neg_dst: List[int] = []

        for h, k in zip(h_list, k_list):
            h = int(h)
            k = int(k)
            if k <= 0:
                continue
            for _ in range(k):
                t = self._pick_tail(h, invalid_ids=extra_invalid_ids)
                neg_src.append(h)
                neg_dst.append(int(t))

        if len(neg_src) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )

        return (
            torch.tensor(neg_src, dtype=torch.long, device=self.device),
            torch.tensor(neg_dst, dtype=torch.long, device=self.device),
        )


# ============================================================
# CV stratification labels (relation + pos/neg) (unchanged)
# ============================================================

def make_rel_label_strat_y(rels: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    rels_np = rels.detach().cpu().numpy().astype(np.int64)
    lab_np = (labels.detach().cpu().numpy() > 0.5).astype(np.int64)
    return (rels_np * 2 + lab_np).astype(np.int64)


# ============================================================
# MAIN
# ============================================================

def main():
    import resource
    print("RLIMIT_NOFILE:", resource.getrlimit(resource.RLIMIT_NOFILE))
    print("Open FDs now:", len(os.listdir("/proc/self/fd")))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="wikidata")
    parser.add_argument(
        "--model",
        type=str,
        choices=["hgcn", "ra_hgcn", "ra_rgcn", "ra_hgat", "sra_hgcn", "gcn", "gae", "gat", "sgnn", "sgat"],
        default="hgcn",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=3024 * 2)
    parser.add_argument("--path", type=str, default="wikidata_data")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--num_neg_test", type=int, default=1)
    parser.add_argument("--use_nstatementsampler", action="store_true")
    parser.add_argument("--use_pstatementsampler", action="store_true")
    parser.add_argument("--use_rstatement_sampler", action="store_true")
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--finaltrain_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--final_lr", type=float, default=None)
    parser.add_argument("--final_epochs", type=int, default=None)

    parser.add_argument("--balanced_test_ratio", type=float, default=0.20)
    parser.add_argument("--balanced_seed", type=int, default=42)
    parser.add_argument("--min_pos_per_rel", type=int, default=0)

    parser.add_argument("--cv_val_ratio", type=float, default=0.15)
    parser.add_argument("--split_cache_path", type=str, default=None)
    parser.add_argument("--force_resplit", action="store_true")

    args = parser.parse_args()
    print("output_dir:", args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(task=args.task)
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    if args.model in ("ra_hgcn", "sra_hgcn", "ra_rgcn"):
        in_feats_cfg = mcfg["in_feats"]
        if isinstance(in_feats_cfg, dict):
            n_type_cfg = mcfg.get("n_type", "node")
            in_dim = int(in_feats_cfg.get(n_type_cfg, list(in_feats_cfg.values())[0]))
        else:
            in_dim = int(in_feats_cfg)
    else:
        in_dim = mcfg["in_feats"]

    lr_cfg = cfg.get("lr", 1e-3)
    lr_candidates = [float(lr) for lr in lr_cfg] if isinstance(lr_cfg, (list, tuple)) else [float(lr_cfg)]
    print(f"Learning rate candidates (per-fold sweep): {lr_candidates}")

    k_folds = int(cfg.get("k_folds", 10))
    contrastive_weight = float(cfg.get("contrastive_weight", 0.1))
    subclass_rel = str(cfg.get("subclass_rel", "subclass_of"))
    instance_rel = str(cfg.get("instance_rel", "instance_of"))
    contrastive_k = int(cfg.get("contrastive_k", 1))

    dl = DataLoader(
        args.path + "/",
        use_pstatement_sampler=args.use_pstatementsampler,
        use_nstatement_sampler=args.use_nstatementsampler,
        use_rstatement_sampler=args.use_rstatement_sampler,
    )
    data_dict = dl.get_data()
    full_graph_all = dl.make_data_graph(data_dict, orthogonal=False)

    num_nodes = int(full_graph_all["node"].num_nodes)
    base_x = full_graph_all["node"].x

    nflag, pflag, rflag = args.use_nstatementsampler, args.use_pstatementsampler, args.use_rstatement_sampler
    use_partial_sampler = nflag or pflag
    use_random_sampler = rflag

    external_edges = dl.get_state_list()
    if nflag:
        neg_edges_all = []
        for etype, (src, tgt) in data_dict.items():
            if isinstance(etype, str) and etype.startswith(NEG_PREFIX):
                neg_edges_all.extend(zip(src.tolist(), etype, tgt.tolist()))
        if neg_edges_all:
            external_edges = neg_edges_all
        print(f"##### Total external negative edges: {len(external_edges)}")

    if instance_rel not in data_dict:
        raise RuntimeError(
            f"Expected instance-of relation '{instance_rel}' in data_dict keys, but got keys: {list(data_dict.keys())}"
        )

    inst_src, _ = data_dict[instance_rel]
    inst_set = set(inst_src.tolist())
    is_instance = torch.zeros(num_nodes, dtype=torch.bool)
    if inst_set:
        is_instance[torch.tensor(sorted(inst_set), dtype=torch.long)] = True
    print(f"Total instance nodes in graph: {int(is_instance.sum().item())} / {num_nodes}")

    cache_path = args.split_cache_path or _make_split_cache_path(
        data_dir=args.path,
        task=args.task,
        test_ratio=float(args.balanced_test_ratio),
        seed=int(args.balanced_seed),
        subclass_rel=subclass_rel,
        instance_rel=instance_rel,
        min_pos_per_rel=int(args.min_pos_per_rel),
    )
    expected_meta = _build_split_cache_meta(
        data_dir=args.path,
        task=args.task,
        test_ratio=float(args.balanced_test_ratio),
        seed=int(args.balanced_seed),
        subclass_rel=subclass_rel,
        instance_rel=instance_rel,
        min_pos_per_rel=int(args.min_pos_per_rel),
    )

    split_out = None
    if not args.force_resplit:
        split_out = _try_load_split_cache(cache_path, expected_meta)

    if split_out is None:
        split_out = build_classification_splits_from_train_files(
            args.path,
            subclass_rel=subclass_rel,
            instance_rel=instance_rel,
            test_ratio=float(args.balanced_test_ratio),
            seed=int(args.balanced_seed),
            min_pos_per_rel=int(args.min_pos_per_rel),
        )
        _save_split_cache(cache_path, expected_meta, split_out)

    cls_heads = split_out["train_heads"]
    cls_rels = split_out["train_rels"]
    cls_tails = split_out["train_tails"]
    cls_labels = split_out["train_labels"]

    test_heads_all = split_out["test_heads"]
    test_rels_all = split_out["test_rels"]
    test_tails_all = split_out["test_tails"]
    test_labels_all = split_out["test_labels"]

    rel2id = split_out["rel2id"]
    id2rel = split_out["id2rel"]
    rel_list = split_out["rel_list"]
    balance_stats = split_out["balance_stats"]

    print_pos_neg_summary_train_test(
        train_rels=cls_rels,
        train_labels=cls_labels,
        test_rels=test_rels_all,
        test_labels=test_labels_all,
        id2rel=id2rel,
        topk=200,
    )

    print("\n=== Final classification dataset sizes ===")
    print(f"#TRAIN classification examples: {cls_heads.numel()}")
    print(f"#TEST  classification examples: {test_heads_all.numel()}")
    print(f"#Classification relations:      {len(rel2id)}")

    # -----------------------------------------------------------------
    # Build encoder graph:
    #   - include ALL graph relations from data_dict
    #   - add ONLY:
    #       (a) file negatives for subclass_of / instance_of (NOT_ prefixed)
    #       (b) leftovers from balancing for classification relations (pos->rel, neg->NOT_rel)
    # -----------------------------------------------------------------
    struct_graph = HeteroData()
    struct_graph["node"].num_nodes = int(num_nodes)
    struct_graph["node"].x = base_x.clone()

    # # Add all relations from data_dict (graph positives etc.)
    # for etype in sorted(list(data_dict.keys()), key=lambda x: str(x)):
    #     src, tgt = data_dict[etype]
    #     _append_edges(struct_graph, str(etype), src, tgt)

    # Add ONLY structural relations needed for the encoder: subclass_of + instance_of
    for etype in (subclass_rel, instance_rel):
        if etype not in data_dict:
            print(f"[WARN] Structural relation missing from data_dict: {etype}")
            continue
        src, tgt = data_dict[etype]
        _append_edges(struct_graph, str(etype), src, tgt)

    # Compute leftovers + encoder-only subclass/instance negatives from file
    leftover_pos_df, leftover_neg_df, enc_only_neg_df = compute_leftover_examples_for_encoder(
        args.path, rel_list=rel_list, balance_stats=balance_stats, seed=int(args.balanced_seed),
        subclass_rel=subclass_rel, instance_rel=instance_rel, min_pos_per_rel=int(args.min_pos_per_rel))

    # (a) Add ONLY subclass/instance negatives from file
    if (not nflag) and len(enc_only_neg_df) > 0:
        enc_only_neg_df = enc_only_neg_df.copy()
        enc_only_neg_df["edge_type"] = enc_only_neg_df["rel_base"].astype(str).map(_ensure_not_prefixed)
        for et, g in enc_only_neg_df.groupby("edge_type"):
            src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(struct_graph, str(et), src, tgt)
        print(f"\n[INFO] Encoder got ONLY subclass/instance negatives from train2id_neg: {len(enc_only_neg_df)}")
    elif nflag: print("\n[INFO] --use_nstatementsampler: skipping subclass/instance NOT_* edges for encoder")
    # (b) Add leftover classification examples (unused after balancing)
    n_left_pos = int(len(leftover_pos_df))
    n_left_neg = int(len(leftover_neg_df))
    if n_left_pos:
        for r, g in leftover_pos_df.groupby("rel_base"):
            src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(struct_graph, str(r), src, tgt)
    if (not nflag) and n_left_neg:
        for r, g in leftover_neg_df.groupby("rel_base"):
            et = _ensure_not_prefixed(str(r))  # -> NOT_*
            src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(struct_graph, et, src, tgt)
    elif nflag and n_left_neg:
        print(f"\n[INFO] --use_nstatementsampler: skipping leftover NOT_* edges for encoder (count={n_left_neg})")
    print(f"\n[INFO] Encoder augmented with leftover (unused) classification examples:")
    print(f"       leftover positives added: {n_left_pos}")
    print(f"       leftover negatives added: {0 if nflag else n_left_neg}")


    # # (a) Add ONLY subclass/instance negatives from file
    # if len(enc_only_neg_df) > 0:
    #     enc_only_neg_df = enc_only_neg_df.copy()
    #     enc_only_neg_df["edge_type"] = enc_only_neg_df["rel_base"].astype(str).map(_ensure_not_prefixed)
    #     for et, g in enc_only_neg_df.groupby("edge_type"):
    #         src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    #         tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    #         _append_edges(struct_graph, str(et), src, tgt)
    #     print(f"\n[INFO] Encoder got ONLY subclass/instance negatives from train2id_neg: {len(enc_only_neg_df)}")

    # (b) Add leftover classification examples (unused after balancing)
    # n_left_pos = int(len(leftover_pos_df))
    # n_left_neg = int(len(leftover_neg_df))
    # if n_left_pos or n_left_neg:
    #     if n_left_pos:
    #         for r, g in leftover_pos_df.groupby("rel_base"):
    #             src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    #             tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    #             _append_edges(struct_graph, str(r), src, tgt)

    #     if n_left_neg:
    #         for r, g in leftover_neg_df.groupby("rel_base"):
    #             et = _ensure_not_prefixed(str(r))
    #             src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    #             tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
    #             _append_edges(struct_graph, et, src, tgt)

    #     print(f"\n[INFO] Encoder augmented with leftover (unused) classification examples:")
    #     print(f"       leftover positives added: {n_left_pos}")
    #     print(f"       leftover negatives added: {n_left_neg}")
    # else:
    #     print("\n[INFO] No leftover classification examples to add to encoder.")

    # e_etypes_struct = list(struct_graph.edge_types)
    # if args.model == "hgcn":
    #     encoder_graph = build_hgcn_encoder_graph(struct_graph, subclass_rel=subclass_rel, neg_prefix=NEG_PREFIX)
    #     encoder_e_etypes = list(encoder_graph.edge_types)
    # else:
    #     encoder_graph = struct_graph
    #     encoder_e_etypes = e_etypes_struct


    e_etypes_struct = list(struct_graph.edge_types)
    if args.model == "hgcn":
        encoder_graph = build_hgcn_encoder_graph(struct_graph, subclass_rel=subclass_rel, neg_prefix=NEG_PREFIX)
    else: encoder_graph = struct_graph
    if CLS_EDGE_TYPE not in encoder_graph.edge_types:
        encoder_graph[CLS_EDGE_TYPE].edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        ei = encoder_graph[CLS_EDGE_TYPE].edge_index
        if ei is None: encoder_graph[CLS_EDGE_TYPE].edge_index = torch.empty((2, 0), dtype=torch.long)
    MP_EDGE_TYPES = [et for et in encoder_graph.edge_types if et != CLS_EDGE_TYPE]
    encoder_e_etypes = MP_EDGE_TYPES

    print_encoder_and_cls_totals(
        encoder_graph=encoder_graph,
        subclass_rel=subclass_rel,
        neg_prefix=NEG_PREFIX,
        train_heads=cls_heads,
        train_rels=cls_rels,
        train_tails=cls_tails,
        train_labels=cls_labels,
        test_heads=test_heads_all,
        test_rels=test_rels_all,
        test_tails=test_tails_all,
        test_labels=test_labels_all)

    print("\n=== Encoder graph summary ===")
    print(f"#Structural edge types: {len(e_etypes_struct)}")
    print(f"#Encoder edge types:    {len(encoder_e_etypes)}")

    report_unseen_test_nodes(
        test_heads=test_heads_all,
        test_tails=test_tails_all,
        train_heads=cls_heads,
        train_tails=cls_tails,
        encoder_graph=encoder_graph,
        num_nodes=num_nodes,
        topk=25,
    )

    report_cls_nodes_not_in_encoder(
        train_heads=cls_heads,
        train_tails=cls_tails,
        test_heads=test_heads_all,
        test_tails=test_tails_all,
        encoder_graph=encoder_graph,
        num_nodes=num_nodes,
        topk=25)


    neighbor_sizes = [15, 10]
    # best_epochs: List[int] = []
    # best_lrs: List[float] = []
    # cv_metrics: List[torch.Tensor] = []

    lr_to_fold_losses = defaultdict(list)
    lr_to_fold_epochs = defaultdict(list)
    lr_to_fold_metrics = defaultdict(list)
    cv_metrics: List[torch.Tensor] = []


    if args.finaltrain_only:
        if args.final_lr is None or args.final_epochs is None:
            raise ValueError("When using --finaltrain_only you must provide BOTH --final_lr and --final_epochs")
        final_lr = float(args.final_lr)
        final_epochs = int(args.final_epochs)
        print("\n=== Final-only mode (CV skipped) ===")
        print(f"final_epochs: {final_epochs}")
        print(f"final_lr:     {final_lr}")

    elif args.test_only:
        model_path = os.path.join("output/" + args.output_dir, f"final_model_{args.model}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for testing not found: {model_path}")

        final_base_kwargs = dict(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
            e_etypes=encoder_e_etypes,
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        )
        if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat"):
            final_model = ModelCls(**final_base_kwargs, rel2id=rel2id).to(device)
        else:
            final_model = ModelCls(**final_base_kwargs).to(device)

        state = torch.load(model_path, map_location=device)
        missing, unexpected = final_model.load_state_dict(state, strict=False)
        print("Loaded model.")
        print("  missing keys:", missing)
        print("  unexpected keys:", unexpected)

    else:
        val_ratio = float(args.cv_val_ratio)
        val_ratio = max(0.01, min(0.49, val_ratio))

        y = make_rel_label_strat_y(cls_rels, cls_labels)

        class_counts = np.bincount(y) if y.size else np.array([], dtype=np.int64)
        rare_classes = set(np.where(class_counts < 2)[0].tolist()) if class_counts.size else set()
        if rare_classes:
            rare_mask = np.isin(y, np.array(sorted(list(rare_classes)), dtype=np.int64))
            rare_idx = np.where(rare_mask)[0]
            main_idx = np.where(~rare_mask)[0]
            print(f"\n[CV] Found {len(rare_classes)} rare (rel,label) classes with <2 samples; "
                  f"{len(rare_idx)} examples forced into TRAIN for every fold.")
        else:
            rare_idx = np.array([], dtype=np.int64)
            main_idx = np.arange(len(y), dtype=np.int64)

        splitter = StratifiedShuffleSplit(n_splits=int(k_folds), test_size=val_ratio, random_state=42)
        split_iter = splitter.split(np.zeros(len(main_idx), dtype=np.int64), y[main_idx]) if len(main_idx) else []

        for fold, (tr_sub, va_sub) in enumerate(split_iter, start=1):
            train_idx_np = main_idx[tr_sub]
            val_idx_np = main_idx[va_sub]
            if rare_idx.size:
                train_idx_np = np.concatenate([train_idx_np, rare_idx], axis=0)

            train_idx = torch.tensor(train_idx_np, dtype=torch.long)
            val_idx = torch.tensor(val_idx_np, dtype=torch.long)

            # train_nodes = torch.unique(torch.cat([cls_heads[train_idx], cls_tails[train_idx]], dim=0))
            # val_nodes = torch.unique(torch.cat([cls_heads[val_idx], cls_tails[val_idx]], dim=0))
            # num_neighbors = [20, 10]
            # train_loader = NeighborLoader(
            #     encoder_graph,
            #     input_nodes=("node", train_nodes),
            #     num_neighbors=num_neighbors,
            #     batch_size=args.batch_size,
            #     shuffle=True,
            #     num_workers=2,
            #     persistent_workers=True,
            #     pin_memory=(device.type == "cuda"),
            # )
            # val_loader = NeighborLoader(
            #     encoder_graph,
            #     input_nodes=("node", val_nodes),
            #     num_neighbors=num_neighbors,
            #     batch_size=args.batch_size,
            #     shuffle=False,
            #     num_workers=2,
            #     persistent_workers=True,
            #     pin_memory=(device.type == "cuda"),
            # )

            train_edge_label_index = torch.stack([cls_heads[train_idx], cls_tails[train_idx]], dim=0)
            train_edge_label = cls_labels[train_idx].to(torch.float)
            val_edge_label_index = torch.stack([cls_heads[val_idx], cls_tails[val_idx]], dim=0)
            val_edge_label = cls_labels[val_idx].to(torch.float)

            fold_encoder_graph = _with_cls_edges(
                encoder_graph,
                cls_heads[train_idx],
                cls_tails[train_idx],
            )

            # num_neighbors = {et: [20, 10] for et in encoder_graph.edge_types}
            num_neighbors = {et: [20, 10] for et in MP_EDGE_TYPES}
            num_neighbors[CLS_EDGE_TYPE] = [0, 0]   # <-- IMPORTANT: don't sample along cls_link

            train_loader = LinkNeighborLoader(fold_encoder_graph, num_neighbors=num_neighbors,
                edge_label_index=(CLS_EDGE_TYPE, train_edge_label_index), edge_label=train_edge_label,
                batch_size=args.batch_size, shuffle=True, num_workers=2, persistent_workers=True,
                pin_memory=(device.type == "cuda"), neg_sampling_ratio=0.0)

            val_loader = LinkNeighborLoader(fold_encoder_graph, num_neighbors=num_neighbors,
                edge_label_index=(CLS_EDGE_TYPE, val_edge_label_index), edge_label=val_edge_label,
                batch_size=args.batch_size, shuffle=False, num_workers=2, persistent_workers=True,
                pin_memory=(device.type == "cuda"), neg_sampling_ratio=0.0,)


            print(f"\n=== CV Split {fold}/{k_folds} ===")
            print(f"  Train cls examples: {train_idx.numel()} | Val cls examples: {val_idx.numel()}")

            base_model_kwargs = dict(
                in_dim=in_dim,
                hidden_dim=mcfg["hidden_dim"],
                out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
                # e_etypes=list(encoder_graph.edge_types),
                e_etypes=encoder_e_etypes,
                n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
            )
            if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat"):
                model_fold = ModelCls(**base_model_kwargs, rel2id=rel2id).to(device)
            else:
                model_fold = ModelCls(**base_model_kwargs).to(device)

            sampler_graph = fold_encoder_graph
            if not args.no_contrastive:
                if use_random_sampler:
                    neg_stmt_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                    neg_stmt_sampler.prepare_global(sampler_graph)
                elif use_partial_sampler:
                    edges_are_negative = nflag
                    neg_stmt_sampler = PartialInstanceSampler(
                        k=contrastive_k, neg_edges=external_edges, edges_are_negative=edges_are_negative
                    )
                    neg_stmt_sampler.prepare_global(sampler_graph)
                else:
                    neg_stmt_sampler = NegativeInstanceSampler_NEW(
                        k=contrastive_k,
                        subclass_rel=subclass_rel,
                        neg_prefix=NEG_PREFIX,
                        instance_rel=instance_rel,
                        cache_dir="data/cache",
                        cache_key=f"{args.path}|{args.output_dir}|fold{fold}",
                    )
                    neg_stmt_sampler.prepare_global(sampler_graph)
            else:
                neg_stmt_sampler = None

            log_fold = Logger(f"train_cv_split{fold}", dir=args.output_dir)
            trainer_fold = Train(
                model=model_fold,
                graph=fold_encoder_graph,
                heads=cls_heads,
                rel_ids=cls_rels,
                tails=cls_tails,
                labels=cls_labels,
                lr_candidates=lr_candidates,
                epochs=args.epochs,
                device=device,
                log=log_fold,
                batch_size=args.batch_size,
                val_ratio=0.0,
                early_stopping_patience=cfg.get("patience", 15),
                train_idx=train_idx,
                val_idx=val_idx,
                contrastive_sampler=neg_stmt_sampler,
                contrastive_weight=contrastive_weight,
                train_loader=train_loader,
                val_loader=val_loader,
                no_contrastive=args.no_contrastive,
            )

            # best_val_loss, best_epoch, best_metrics, best_lr = trainer_fold.run()
            # best_epochs.append(int(best_epoch if best_epoch is not None else args.epochs))
            # best_lrs.append(float(best_lr if best_lr is not None else lr_candidates[0]))
            # if best_metrics is not None:
            #     cv_metrics.append(best_metrics)
            best_val_loss, best_epoch, best_metrics, best_lr, per_lr = trainer_fold.run()
            for lr in lr_candidates:
                lr = float(lr)
                rec = per_lr.get(lr, None)
                if rec is None: continue
                loss = float(rec["best_val_loss"])
                ep = int(rec["best_epoch"])

                if loss != float("inf"): lr_to_fold_losses[lr].append(loss)
                if ep > 0: lr_to_fold_epochs[lr].append(ep)
                if rec.get("best_metrics", None) is not None: lr_to_fold_metrics[lr].append(rec["best_metrics"])
            if best_metrics is not None:
                cv_metrics.append(best_metrics)


        # final_epochs = int(statistics.median(best_epochs)) if best_epochs else int(args.epochs)
        # final_lr = float(statistics.mode(best_lrs)) if best_lrs else float(lr_candidates[0])

        # print("\n=== Cross-validation summary (TRAIN split only) ===")
        # print(f"Per-split best epochs: {best_epochs}")
        # print(f"Per-split best learning rates: {best_lrs}")
        # print(f"Chosen final_epochs (median): {final_epochs}")
        # print(f"Chosen final_lr (mode):       {final_lr}")

        lr_summary = []
        for lr in lr_candidates:
            lr = float(lr)
            losses = lr_to_fold_losses.get(lr, [])
            if not losses: continue
            mean_loss = statistics.mean(losses)
            std_loss = statistics.pstdev(losses) if len(losses) > 1 else 0.0
            n = len(losses)
            lr_summary.append((mean_loss, std_loss, n, lr))
        if not lr_summary:
            final_lr = float(lr_candidates[0])
            final_epochs = int(args.epochs)
        else:
            lr_summary.sort(key=lambda x: (x[0], x[1], x[3]))
            best_mean, best_std, best_n, final_lr = lr_summary[0]
            epochs_for_lr = lr_to_fold_epochs.get(final_lr, [])
            final_epochs = int(statistics.median(epochs_for_lr)) if epochs_for_lr else int(args.epochs)
        print("\n=== Cross-validation summary (TRAIN split only) ===")
        print("LR aggregates (mean_val_loss ± std over folds):")
        for mean_loss, std_loss, n, lr in sorted(lr_summary, key=lambda x: (x[0], x[1], x[3])):
            print(f"  lr={lr:.3g} | mean={mean_loss:.6f} | std={std_loss:.6f} | folds={n}")
        print(f"\nChosen final_lr (best mean over folds): {final_lr:.6g}")
        print(f"Epochs for chosen LR across folds:      {lr_to_fold_epochs.get(final_lr, [])}")
        print(f"Chosen final_epochs (median for LR):    {final_epochs}")


    # -----------------------------------------------------------------
    # Final training on TRAIN split
    # -----------------------------------------------------------------
    if not args.test_only:
        train_encoder_graph = _with_cls_edges(encoder_graph, cls_heads, cls_tails)
        print_final_train_encoder_statement_counts(
            encoder_graph=train_encoder_graph,
            subclass_rel=subclass_rel,
            neg_prefix=NEG_PREFIX,
        )

        final_edge_label_index = torch.stack([cls_heads, cls_tails], dim=0)
        final_edge_label = cls_labels.to(torch.float)
        # num_neighbors = {et: neighbor_sizes for et in encoder_graph.edge_types}
        num_neighbors = {et: neighbor_sizes for et in MP_EDGE_TYPES}
        num_neighbors[CLS_EDGE_TYPE] = [0, 0]
        
        final_loader = LinkNeighborLoader(train_encoder_graph, num_neighbors=num_neighbors,
            edge_label_index=(CLS_EDGE_TYPE, final_edge_label_index),neg_sampling_ratio=0.0,
            edge_label=final_edge_label, batch_size=args.batch_size, shuffle=True,
            num_workers=2, persistent_workers=True, pin_memory=(device.type == "cuda"))

        # final_nodes = torch.unique(torch.cat([cls_heads, cls_tails], dim=0))
        # final_loader = NeighborLoader(
        #     encoder_graph,
        #     input_nodes=("node", final_nodes),
        #     num_neighbors=neighbor_sizes,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=2,
        #     persistent_workers=True,
        #     pin_memory=(device.type == "cuda"),
        # )

        final_base_kwargs = dict(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
            e_etypes=encoder_e_etypes,
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        )
        if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gat", "gae"):
            final_model = ModelCls(**final_base_kwargs, rel2id=rel2id).to(device)
        else:
            final_model = ModelCls(**final_base_kwargs).to(device)

        final_log = Logger("final_train_global", dir=args.output_dir, non_verbose=True)

        if not args.no_contrastive:
            if use_random_sampler:
                final_contrastive_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                final_contrastive_sampler.prepare_global(train_encoder_graph)
            elif use_partial_sampler:
                edges_are_negative = nflag
                final_contrastive_sampler = PartialInstanceSampler(
                    k=contrastive_k, neg_edges=external_edges, edges_are_negative=edges_are_negative
                )
                final_contrastive_sampler.prepare_global(train_encoder_graph)
            else:
                final_contrastive_sampler = NegativeInstanceSampler_NEW(
                    k=contrastive_k, subclass_rel=subclass_rel, neg_prefix=NEG_PREFIX, instance_rel=instance_rel,
                    cache_dir="data/cache", cache_key=f"{args.path}|{args.output_dir}|final"
                )
                final_contrastive_sampler.prepare_global(train_encoder_graph)
        else:
            final_contrastive_sampler = None

        if args.finaltrain_only:
            final_lr = float(args.final_lr)
            final_epochs = int(args.final_epochs)

        final_trainer = Train_BestModel(
            final_model,
            graph=train_encoder_graph,
            heads=cls_heads,
            rel_ids=cls_rels,
            tails=cls_tails,
            labels=cls_labels,
            lr=float(final_lr),
            epochs=int(final_epochs),
            device=device,
            log=final_log,
            batch_size=args.batch_size,
            contrastive_sampler=final_contrastive_sampler,
            contrastive_weight=contrastive_weight,
            loader=final_loader,
            no_contrastive=args.no_contrastive,
        )
        final_loss = final_trainer.run()
        print(f"[Final Train] Loss after {final_epochs} epochs (lr={final_lr:.3g}): {final_loss:.4f}")
    test_log = Logger("test_global", dir=args.output_dir)

    train_encoder_graph = _with_cls_edges(encoder_graph, cls_heads, cls_tails)

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_path = os.path.join("output/" + args.output_dir, f"final_model_{args.model}.pt")
    if not args.test_only:
        torch.save(final_model.state_dict(), model_path)

    test_pos_mask = (test_labels_all > 0.5)
    test_heads = test_heads_all[test_pos_mask]
    test_rels = test_rels_all[test_pos_mask]
    test_tails = test_tails_all[test_pos_mask]

    test_neg_mask = (test_labels_all <= 0.5)
    neg_heads_all = test_heads_all[test_neg_mask]
    neg_rels_all = test_rels_all[test_neg_mask]
    neg_tails_all = test_tails_all[test_neg_mask]

    neg_samplers: Dict[str, FileNegativeSampler] = {}
    if neg_rels_all.numel() > 0:
        for rid in torch.unique(neg_rels_all).detach().cpu().long().tolist():
            rid = int(rid)
            m = (neg_rels_all == rid)
            rel_name = id2rel.get(rid, str(rid))
            neg_samplers[rel_name] = FileNegativeSampler(
                neg_heads=neg_heads_all[m],
                neg_tails=neg_tails_all[m],
                num_nodes=num_nodes,
                seed=int(args.balanced_seed) + 17 + rid,
                device=torch.device("cpu"))

    else:
        print("[WARN] No TEST negatives found; Test_BestModel may not be able to evaluate properly.")

    cache_path = os.path.join("data/neg_cache", f"test_negs_{args.task}_numneg{args.num_neg_test}.pt")
    print("neg cache abs path:", os.path.abspath(cache_path))

    tester = Test_BestModel(
        model=final_model,
        graph=train_encoder_graph,
        test_heads=test_heads,
        test_rels=test_rels,
        test_tails=test_tails,
        id2rel=id2rel,
        neg_samplers=neg_samplers,
        num_neg_per_pos=args.num_neg_test,
        device=device,
        log=test_log,
        batch_size=args.batch_size,
        neg_cache_path=cache_path,
    )
    tester.run()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
