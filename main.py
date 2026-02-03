# main.py
import os, hashlib, re, statistics, inspect, builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader

from data_loader import DataLoader
from models import *
from trainer import Train
from trainer_bestmodel import Train_BestModel, Test_BestModel
from utils import Logger, load_config
from samplers import (
    NegativeEntitySampler,
    NegativeEntitySampler2,
    NegativeInstanceSampler,
    RandomInstanceSampler,
    PartialInstanceSampler,
    FileNegativeSampler,
)

NEG_PREFIX = "NOT_"
CLS_EDGE_TYPE = ("node", "cls_link", "node")

# Minimal output: suppress all prints in this module unless explicitly using _p.
MINIMAL_OUTPUT = True

def _p(*args, **kwargs):
    builtins.print(*args, **kwargs)

def print(*args, **kwargs):
    if not MINIMAL_OUTPUT:
        builtins.print(*args, **kwargs)

def filter_train_by_neg_coverage(
    *,
    train_heads: torch.Tensor,
    train_rels: torch.Tensor,
    train_tails: torch.Tensor,
    train_labels: torch.Tensor,
    neg_node_mask: torch.Tensor,
    id2rel: Dict[int, str],
    min_keep_per_group: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop train triples where neither endpoint has a neg-edge in the encoder graph.
    Tries to keep at least min_keep_per_group per (relation,label) group to preserve stratification."""
    if train_heads.numel() == 0:
        return train_heads, train_rels, train_tails, train_labels, torch.empty(0, dtype=torch.long)
    mask_has_neg = neg_node_mask[train_heads.long()] | neg_node_mask[train_tails.long()]
    if mask_has_neg.all():
        return train_heads, train_rels, train_tails, train_labels, torch.empty(0, dtype=torch.long)

    kept_idx: List[int] = []
    dropped = 0
    rels_cpu = train_rels.detach().cpu().long()
    labels_cpu = train_labels.detach().cpu().float()
    idx_all = torch.arange(train_heads.numel(), dtype=torch.long)
    for rel in rels_cpu.unique().tolist():
        for lab in (0.0, 1.0):
            rel_mask = (rels_cpu == int(rel))
            lab_mask = (labels_cpu > 0.5) if lab > 0.5 else (labels_cpu <= 0.5)
            group_idx = idx_all[rel_mask & lab_mask]
            if group_idx.numel() == 0:
                continue
            group_keep = group_idx[mask_has_neg[group_idx]]
            if group_keep.numel() == 0 and min_keep_per_group > 0:
                # keep a small number to avoid removing the group entirely
                group_keep = group_idx[:min_keep_per_group]
            kept_idx.extend(group_keep.tolist())
            dropped += int(group_idx.numel() - group_keep.numel())

    kept_idx = sorted(set(kept_idx))
    if not kept_idx:
        print("[WARN] filter_train_by_neg_coverage removed all training triples; skipping filter.")
        return train_heads, train_rels, train_tails, train_labels, torch.empty(0, dtype=torch.long)

    kept_idx_t = torch.tensor(kept_idx, dtype=torch.long)
    all_idx = torch.arange(train_heads.numel(), dtype=torch.long)
    keep_mask = torch.zeros(train_heads.numel(), dtype=torch.bool)
    keep_mask[kept_idx_t] = True
    removed_idx_t = all_idx[~keep_mask]
    before = int(train_heads.numel())
    after = int(kept_idx_t.numel())
    print("\n=== Filter train triples by NEG-edge coverage ===")
    print(f"Train kept: {after}/{before} (dropped={before - after})")
    print(f"Fallback kept per (rel,label) group: min_keep_per_group={min_keep_per_group}")

    # Optional per-relation summary (top 10)
    if id2rel:
        kept_rels = train_rels[kept_idx_t]
        kept_labels = train_labels[kept_idx_t]
        df_before = _counts_per_relation(train_rels, train_labels, id2rel).head(10)
        df_after = _counts_per_relation(kept_rels, kept_labels, id2rel).head(10)
        print("\n--- Top relations before filter ---")
        print(df_before.to_string(index=False))
        print("\n--- Top relations after filter ---")
        print(df_after.to_string(index=False))

    return (
        train_heads[kept_idx_t],
        train_rels[kept_idx_t],
        train_tails[kept_idx_t],
        train_labels[kept_idx_t],
        removed_idx_t,
    )

def _prune_splits_by_neg_source_coverage(
    *,
    train_heads: torch.Tensor,
    train_rels: torch.Tensor,
    train_tails: torch.Tensor,
    train_labels: torch.Tensor,
    test_heads: torch.Tensor,
    test_rels: torch.Tensor,
    test_tails: torch.Tensor,
    test_labels: torch.Tensor,
    neg_node_mask: torch.Tensor,
    min_per_label: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove a minimal number of examples (prefer those without neg coverage)
    while keeping >= min_per_label per (relation,label) in both splits.
    Relations failing this are removed entirely from both splits."""
    def _group_indices(rels, labels, rel_id, is_pos: bool):
        lab_mask = (labels > 0.5) if is_pos else (labels <= 0.5)
        return torch.where((rels == rel_id) & lab_mask)[0]

    rel_ids = torch.unique(torch.cat([train_rels, test_rels], dim=0)).tolist()
    remove_rel_ids = set()

    train_keep = torch.zeros(train_heads.numel(), dtype=torch.bool)
    test_keep = torch.zeros(test_heads.numel(), dtype=torch.bool)

    for rid in rel_ids:
        rid = int(rid)
        # compute group indices
        tr_pos = _group_indices(train_rels, train_labels, rid, True)
        tr_neg = _group_indices(train_rels, train_labels, rid, False)
        te_pos = _group_indices(test_rels, test_labels, rid, True)
        te_neg = _group_indices(test_rels, test_labels, rid, False)

        # if any group total < min_per_label, drop relation entirely
        if (tr_pos.numel() < min_per_label or tr_neg.numel() < min_per_label or
            te_pos.numel() < min_per_label or te_neg.numel() < min_per_label):
            remove_rel_ids.add(rid)
            continue

        def _select_keep(idx: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
            if idx.numel() == 0:
                return idx
            has_neg = neg_node_mask[heads[idx].long()]
            keep_idx = idx[has_neg]
            if keep_idx.numel() >= min_per_label:
                return keep_idx
            # keep all with coverage, then pad with uncovered to reach min_per_label
            need = int(min_per_label - keep_idx.numel())
            fallback = idx[~has_neg][:need]
            return torch.cat([keep_idx, fallback], dim=0)

        tr_pos_keep = _select_keep(tr_pos, train_heads)
        tr_neg_keep = _select_keep(tr_neg, train_heads)
        te_pos_keep = _select_keep(te_pos, test_heads)
        te_neg_keep = _select_keep(te_neg, test_heads)

        train_keep[tr_pos_keep] = True
        train_keep[tr_neg_keep] = True
        test_keep[te_pos_keep] = True
        test_keep[te_neg_keep] = True

    if remove_rel_ids:
        rel_mask_train = ~torch.isin(train_rels, torch.tensor(list(remove_rel_ids), dtype=train_rels.dtype))
        rel_mask_test = ~torch.isin(test_rels, torch.tensor(list(remove_rel_ids), dtype=test_rels.dtype))
        train_keep &= rel_mask_train
        test_keep &= rel_mask_test

    removed_train = ~train_keep
    removed_test = ~test_keep

    return (
        train_heads[train_keep], train_rels[train_keep], train_tails[train_keep], train_labels[train_keep],
        test_heads[test_keep], test_rels[test_keep], test_tails[test_keep], test_labels[test_keep],
        train_heads[removed_train], train_rels[removed_train], train_tails[removed_train], train_labels[removed_train],
        test_heads[removed_test], test_rels[removed_test], test_tails[removed_test], test_labels[removed_test],
    )

def _append_cls_triples_to_encoder(
    encoder_graph: HeteroData,
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    labels: torch.Tensor,
    id2rel: Dict[int, str],
    *,
    nflag: bool,
) -> int:
    if heads.numel() == 0:
        return 0
    added = 0
    # positives
    pos_mask = labels > 0.5
    if pos_mask.any():
        df_pos = pd.DataFrame({
            "h": heads[pos_mask].detach().cpu().numpy(),
            "r": rels[pos_mask].detach().cpu().numpy(),
            "t": tails[pos_mask].detach().cpu().numpy(),
        })
        for rid, g in df_pos.groupby("r"):
            rel_name = id2rel.get(int(rid), str(int(rid)))
            src = torch.tensor(g["h"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["t"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(encoder_graph, str(rel_name), src, tgt)
            added += int(len(g))
    # negatives
    neg_mask = labels <= 0.5
    if neg_mask.any():
        if nflag:
            print("[INFO] --use_nstatementsampler: skipping NOT_* edges for encoder")
        else:
            df_neg = pd.DataFrame({
                "h": heads[neg_mask].detach().cpu().numpy(),
                "r": rels[neg_mask].detach().cpu().numpy(),
                "t": tails[neg_mask].detach().cpu().numpy(),
            })
            for rid, g in df_neg.groupby("r"):
                rel_name = _ensure_not_prefixed(str(id2rel.get(int(rid), str(int(rid)))))
                src = torch.tensor(g["h"].to_numpy(dtype=np.int64), dtype=torch.long)
                tgt = torch.tensor(g["t"].to_numpy(dtype=np.int64), dtype=torch.long)
                _append_edges(encoder_graph, rel_name, src, tgt)
                added += int(len(g))
    return added


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
    if src.numel() == 0 or tgt.numel() == 0:
        return
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
    min_keep_per_rel: int = 20,
    coverage_priority: bool = True,
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
    dropped_small = []

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
        if int(m) < int(min_keep_per_rel):
            dropped_small.append(r)
            continue

        rng = np.random.RandomState(int(seed) + _stable_hash_int(r, 1_000_000))

        def _select_idx(rows: pd.DataFrame) -> np.ndarray:
            if coverage_priority and "has_neg" in rows.columns:
                covered = rows[rows["has_neg"]]
                uncovered = rows[~rows["has_neg"]]
                cov_idx = covered.index.to_numpy(dtype=np.int64).copy()
                unc_idx = uncovered.index.to_numpy(dtype=np.int64).copy()
                rng.shuffle(cov_idx)
                rng.shuffle(unc_idx)
                need = int(m)
                take_cov = cov_idx[:need]
                remaining = need - len(take_cov)
                if remaining > 0:
                    take_unc = unc_idx[:remaining]
                    return np.concatenate([take_cov, take_unc], axis=0)
                return take_cov
            idx = rows.index.to_numpy(dtype=np.int64).copy()
            rng.shuffle(idx)
            return idx[:m]

        pos_sel = _select_idx(pos_rows)
        neg_sel = _select_idx(neg_rows)

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
    if dropped_small:
        print(f"\n[WARN] Dropped {len(dropped_small)} relations with m < {min_keep_per_rel} after balancing. (first 50)")
        print(dropped_small[:50])

    if not chunks:
        return pd.DataFrame(columns=["source_node", "target_node", "edge_type", "rel_base", "label"]), stats

    df_bal = pd.concat(chunks, ignore_index=True)
    return df_bal, stats

def _per_relation_label_split_indices(df: pd.DataFrame, *, test_ratio: float,
    seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[int, Tuple[int, int]]]]:
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
    # _p("\n================ POS/NEG COUNTS PER RELATION ================")
    df_train = _counts_per_relation(train_rels, train_labels, id2rel)
    df_test = _counts_per_relation(test_rels, test_labels, id2rel) if test_rels.numel() else pd.DataFrame()

    # Per-relation table prints suppressed.
    # _p("\n--- TRAIN (pos/neg/total) ---")
    # _p("(empty)" if len(df_train) == 0 else df_train.head(topk).to_string(index=False))
    # _p("\n--- TEST (pos/neg/total) ---")
    # if test_rels.numel() == 0 or len(df_test) == 0:
    #     _p("(empty)")
    # else:
    #     _p(df_test.head(topk).to_string(index=False))

    if train_rels.numel():
        _p("\n--- TRAIN totals ---")
        _p("Train positives:", int((train_labels > 0.5).sum().item()))
        _p("Train negatives:", int((train_labels <= 0.5).sum().item()))
        _p("Train relations:", int(train_rels.unique().numel()))
    if test_rels.numel():
        _p("\n--- TEST totals ---")
        _p("Test positives:", int((test_labels > 0.5).sum().item()))
        _p("Test negatives:", int((test_labels <= 0.5).sum().item()))
        _p("Test relations:", int(test_rels.unique().numel()))

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

    # Coverage-aware balancing: mark rows whose endpoints have at least one NOT_* edge.
    neg_nodes = set(df_neg_all["source_node"].tolist()) | set(df_neg_all["target_node"].tolist())
    if neg_nodes:
        df_pos_cls["has_neg"] = df_pos_cls["source_node"].isin(neg_nodes) | df_pos_cls["target_node"].isin(neg_nodes)
        df_neg_cls["has_neg"] = df_neg_cls["source_node"].isin(neg_nodes) | df_neg_cls["target_node"].isin(neg_nodes)

    if min_pos_per_rel and int(min_pos_per_rel) > 0:
        pos_counts = df_pos_cls["rel_base"].value_counts().to_dict()
        keep_rels = {r for r, c in pos_counts.items() if int(c) >= int(min_pos_per_rel)}
        before = len(df_pos_cls)
        df_pos_cls = df_pos_cls[df_pos_cls["rel_base"].isin(keep_rels)].copy()
        df_neg_cls = df_neg_cls[df_neg_cls["rel_base"].isin(keep_rels)].copy()
        print(f"\n[INFO] min_pos_per_rel={min_pos_per_rel}: kept {len(keep_rels)} relations. "
              f"pos rows: {before} -> {len(df_pos_cls)}")

    df_balanced, balance_stats = _balanced_downsample_per_relation(
        df_pos_cls, df_neg_cls, seed=int(seed),
        min_pos_per_rel=int(min_pos_per_rel) if min_pos_per_rel else 0,
        min_keep_per_rel=20,
        coverage_priority=True,
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

def _remove_cls_from_leftovers(
    *,
    leftover_pos_df: pd.DataFrame,
    leftover_neg_df: pd.DataFrame,
    train_heads: torch.Tensor,
    train_rels: torch.Tensor,
    train_tails: torch.Tensor,
    train_labels: torch.Tensor,
    test_heads: torch.Tensor,
    test_rels: torch.Tensor,
    test_tails: torch.Tensor,
    test_labels: torch.Tensor,
    id2rel: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if (leftover_pos_df is None) or (leftover_neg_df is None):
        return leftover_pos_df, leftover_neg_df

    def _cls_df(h, r, t, y, *, label_pos: bool) -> pd.DataFrame:
        if h.numel() == 0:
            return pd.DataFrame(columns=["source_node", "target_node", "rel_base"])
        mask = (y > 0.5) if label_pos else (y <= 0.5)
        if not mask.any():
            return pd.DataFrame(columns=["source_node", "target_node", "rel_base"])
        rel_names = [id2rel.get(int(x), str(int(x))) for x in r[mask].tolist()]
        return pd.DataFrame({
            "source_node": h[mask].detach().cpu().numpy(),
            "target_node": t[mask].detach().cpu().numpy(),
            "rel_base": rel_names,
        })

    cls_pos = pd.concat([
        _cls_df(train_heads, train_rels, train_tails, train_labels, label_pos=True),
        _cls_df(test_heads, test_rels, test_tails, test_labels, label_pos=True),
    ], ignore_index=True)
    cls_neg = pd.concat([
        _cls_df(train_heads, train_rels, train_tails, train_labels, label_pos=False),
        _cls_df(test_heads, test_rels, test_tails, test_labels, label_pos=False),
    ], ignore_index=True)

    if len(leftover_pos_df):
        merged = leftover_pos_df.merge(
            cls_pos, on=["source_node", "target_node", "rel_base"], how="left", indicator=True
        )
        leftover_pos_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    if len(leftover_neg_df):
        merged = leftover_neg_df.merge(
            cls_neg, on=["source_node", "target_node", "rel_base"], how="left", indicator=True
        )
        leftover_neg_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return leftover_pos_df, leftover_neg_df

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
    # _p("RLIMIT_NOFILE:", resource.getrlimit(resource.RLIMIT_NOFILE))
    # _p("Open FDs now:", len(os.listdir("/proc/self/fd")))

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
    parser.add_argument("--use_negativeentity_sampler", action="store_true")
    parser.add_argument("--use_negativeentity_sampler2", action="store_true")
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--finaltrain_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--final_lr", type=float, default=None)
    parser.add_argument("--final_epochs", type=int, default=None)
    parser.add_argument("--final_contrastive_temperature", type=float, default=None)

    parser.add_argument("--balanced_test_ratio", type=float, default=0.20)
    parser.add_argument("--balanced_seed", type=int, default=42)
    parser.add_argument("--min_pos_per_rel", type=int, default=0)
    parser.add_argument("--cv_val_ratio", type=float, default=0.15)
    parser.add_argument("--split_cache_path", type=str, default=None)
    parser.add_argument("--force_resplit", action="store_true")
    parser.add_argument("--parallel_grid", action="store_true")
    parser.add_argument("--parallel_grid_workers", type=int, default=None)
    parser.add_argument("--parallel_grid_loader_workers", type=int, default=None)
    parser.add_argument("--parallel_grid_pin_memory", action="store_true")
    parser.add_argument("--contrastive_weight", type=float, default=None)
    args = parser.parse_args()

    # _p("output_dir:", args.output_dir, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _p(f"device: {device}", flush=True)

    cfg = load_config(task=args.task)
    ModelCls = eval(args.model.upper()) if args.model != "gae" else eval("GCN_" + args.model.upper())
    mcfg = cfg["models"][ModelCls.__name__ if args.model != "gae" else "GAE"]

    if args.model in ("ra_hgcn", "sra_hgcn", "ra_rgcn"):
        in_feats_cfg = mcfg["in_feats"]
        if isinstance(in_feats_cfg, dict):
            n_type_cfg = mcfg.get("n_type", "node")
            in_dim = int(in_feats_cfg.get(n_type_cfg, list(in_feats_cfg.values())[0]))
        else: in_dim = int(in_feats_cfg)
    else: in_dim = mcfg["in_feats"]

    lr_cfg = cfg.get("lr", 1e-3)
    lr_candidates = [float(lr) for lr in lr_cfg] if isinstance(lr_cfg, (list, tuple)) else [float(lr_cfg)]

    dropout_candidates = [0.2]
    contrastive_weight_candidates = [0.1]
    if args.contrastive_weight is not None:
        contrastive_weight_candidates = [float(args.contrastive_weight)]

    contrastive_temp_cfg = cfg.get("contrastive_temperature", 0.5)
    contrastive_temp_candidates = (
        [float(t) for t in contrastive_temp_cfg]
        if isinstance(contrastive_temp_cfg, (list, tuple))
        else [float(contrastive_temp_cfg)])

    clone_inputs = bool(cfg.get("clone_inputs", True))
    cv_prune_ratio = float(cfg.get("cv_prune_ratio", 0.0))
    cv_prune_warmup = int(cfg.get("cv_prune_warmup", 5))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    timing_profile = bool(cfg.get("timing_profile", False))
    parallel_grid = bool(cfg.get("parallel_grid", False) or args.parallel_grid)
    parallel_grid_workers = args.parallel_grid_workers
    if parallel_grid_workers is None:
        parallel_grid_workers = int(cfg.get("parallel_grid_workers", 2))
    parallel_grid_loader_workers = args.parallel_grid_loader_workers
    if parallel_grid_loader_workers is None:
        parallel_grid_loader_workers = int(cfg.get("parallel_grid_loader_workers", 0))
    parallel_grid_pin_memory = bool(cfg.get("parallel_grid_pin_memory", False) or args.parallel_grid_pin_memory)
    print("timing_profile:", timing_profile)

    print(f"Learning rate candidates (per-fold sweep): {lr_candidates}")

    k_folds = int(cfg.get("k_folds", 10))
    contrastive_weight = float(contrastive_weight_candidates[0])
    subclass_rel = str(cfg.get("subclass_rel", "subclass_of"))
    instance_rel = str(cfg.get("instance_rel", "instance_of"))
    contrastive_k = int(cfg.get("contrastive_k", 1))
    contrastive_pool_cap = int(cfg.get("contrastive_pool_cap", 512))
    if contrastive_pool_cap <= 0:
        contrastive_pool_cap = None

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
    use_negativeentity_sampler = bool(args.use_negativeentity_sampler)
    use_negativeentity_sampler2 = bool(args.use_negativeentity_sampler2)

    external_edges = dl.get_state_list()
    if nflag:
        neg_edges_all = []
        for etype, (src, tgt) in data_dict.items():
            if isinstance(etype, str) and etype.startswith(NEG_PREFIX):
                neg_edges_all.extend(zip(src.tolist(), etype, tgt.tolist()))
        if neg_edges_all:
            external_edges = neg_edges_all
        print(f"##### Total external negative edges: {len(external_edges)}", flush=True)

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

    # Anchor coverage vs classification nodes (computed later after cls_* are set)

    min_pos_fixed = 80
    cache_path = args.split_cache_path or _make_split_cache_path(
        data_dir=args.path,
        task=args.task,
        test_ratio=float(args.balanced_test_ratio),
        seed=int(args.balanced_seed),
        subclass_rel=subclass_rel,
        instance_rel=instance_rel,
        min_pos_per_rel=min_pos_fixed,
    )
    expected_meta = _build_split_cache_meta(
        data_dir=args.path,
        task=args.task,
        test_ratio=float(args.balanced_test_ratio),
        seed=int(args.balanced_seed),
        subclass_rel=subclass_rel,
        instance_rel=instance_rel,
        min_pos_per_rel=min_pos_fixed,
    )

    split_out = _try_load_split_cache(cache_path, expected_meta)

    if split_out is None:
        split_out = build_classification_splits_from_train_files(
            args.path,
            subclass_rel=subclass_rel,
            instance_rel=instance_rel,
            test_ratio=float(args.balanced_test_ratio),
            seed=int(args.balanced_seed),
            min_pos_per_rel=min_pos_fixed,
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

    # print_pos_neg_summary_train_test(
    #     train_rels=cls_rels,
    #     train_labels=cls_labels,
    #     test_rels=test_rels_all,
    #     test_labels=test_labels_all,
    #     id2rel=id2rel,
    #     topk=max(200, len(id2rel)),
    # )

    print("\n=== Final classification dataset sizes ===")
    print(f"#TRAIN classification examples: {cls_heads.numel()}")
    print(f"#TEST  classification examples: {test_heads_all.numel()}")
    print(f"#Classification relations:      {len(rel2id)}")

    # -----------------------------------------------------------------
    # Build encoder graph:
    # -----------------------------------------------------------------
    struct_graph = HeteroData()
    struct_graph["node"].num_nodes = int(num_nodes)
    struct_graph["node"].x = base_x.clone()

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
    # Ensure leftovers do not overlap final train/test classification examples
    leftover_pos_df, leftover_neg_df = _remove_cls_from_leftovers(
        leftover_pos_df=leftover_pos_df,
        leftover_neg_df=leftover_neg_df,
        train_heads=cls_heads,
        train_rels=cls_rels,
        train_tails=cls_tails,
        train_labels=cls_labels,
        test_heads=test_heads_all,
        test_rels=test_rels_all,
        test_tails=test_tails_all,
        test_labels=test_labels_all,
        id2rel=id2rel,
    )

    # Add ALL negatives from train2id_neg that are NOT in final train/test classification negatives.
    # This preserves the split while maximizing neg coverage without introducing new negatives.
    train_neg_path = os.path.join(args.path, "train2id_neg.txt")
    df_neg_all = _read_edge_file(train_neg_path)
    df_neg_all["rel_base"] = df_neg_all["edge_type"].astype(str).map(_strip_not)
    encoder_only_bases = {str(subclass_rel), str(instance_rel), "subclass_of", "instance_of"}
    df_neg_all_cls = df_neg_all[~df_neg_all["rel_base"].isin(encoder_only_bases)].copy()

    cls_neg_df = pd.concat([
        pd.DataFrame({
            "source_node": cls_heads[cls_labels <= 0.5].detach().cpu().numpy(),
            "target_node": cls_tails[cls_labels <= 0.5].detach().cpu().numpy(),
            "rel_base": [id2rel.get(int(x), str(int(x))) for x in cls_rels[cls_labels <= 0.5].tolist()],
        }),
        pd.DataFrame({
            "source_node": test_heads_all[test_labels_all <= 0.5].detach().cpu().numpy(),
            "target_node": test_tails_all[test_labels_all <= 0.5].detach().cpu().numpy(),
            "rel_base": [id2rel.get(int(x), str(int(x))) for x in test_rels_all[test_labels_all <= 0.5].tolist()],
        }),
    ], ignore_index=True)

    if len(df_neg_all_cls) > 0 and len(cls_neg_df) > 0:
        merged = df_neg_all_cls.merge(
            cls_neg_df, on=["source_node", "target_node", "rel_base"], how="left", indicator=True
        )
        safe_neg_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    else:
        safe_neg_df = df_neg_all_cls

    # Replace leftover_neg_df with the safe full-neg set (non-leaking)
    leftover_neg_df = safe_neg_df.reset_index(drop=True)

    # (b) Add leftover classification examples (unused after balancing)
    n_left_pos = int(len(leftover_pos_df))
    n_left_neg = int(len(leftover_neg_df))
    if n_left_pos:
        for r, g in leftover_pos_df.groupby("rel_base"):
            src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(struct_graph, str(r), src, tgt)
    if n_left_neg:
        for r, g in leftover_neg_df.groupby("rel_base"):
            et = _ensure_not_prefixed(str(r))  # -> NOT_*
            src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(struct_graph, et, src, tgt)
    print(f"\n[INFO] Encoder augmented with leftover (unused) classification examples:")
    print(f"       leftover positives added: {n_left_pos}")
    print(f"       leftover negatives added: {0 if nflag else n_left_neg}")

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

    # (suppressed) detailed encoder/classification totals


    # -----------------------------------------------------------------
    # Classification examples with at least one NEG relation in encoder graph
    # -----------------------------------------------------------------
    neg_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for et in encoder_graph.edge_types:
        if et == CLS_EDGE_TYPE:
            continue
        rel_s = str(et[1])
        if rel_s.startswith(NEG_PREFIX) or rel_s == "neg_statement":
            ei = encoder_graph[et].edge_index
            if ei is None or ei.numel() == 0:
                continue
            neg_node_mask[ei[0].long()] = True
            neg_node_mask[ei[1].long()] = True

    def _count_with_neg(h: torch.Tensor, t: torch.Tensor) -> Tuple[int, int]:
        if h.numel() == 0:
            return 0, 0
        mask = neg_node_mask[h.long()] #| neg_node_mask[t.long()]
        return int(mask.sum().item()), int(mask.numel())

    n_train_with_neg, n_train_total = _count_with_neg(cls_heads, cls_tails)
    n_test_with_neg, n_test_total = _count_with_neg(test_heads_all, test_tails_all)
    n_all_with_neg = n_train_with_neg + n_test_with_neg
    n_all_total = n_train_total + n_test_total

    def _pct(num: int, den: int) -> float:
        return (100.0 * num / den) if den > 0 else 0.0

    # (moved) test src NOT_* coverage print after all filtering/pruning

    # Optionally remove train triples where neither endpoint has a neg-edge in encoder graph
    min_keep = int(cfg.get("train_neg_coverage_min_keep", 5))
    removed_train_idx = torch.empty(0, dtype=torch.long)
    if int(cfg.get("filter_train_no_neg_edges", 1)) == 1:
        orig_cls_heads = cls_heads
        orig_cls_rels = cls_rels
        orig_cls_tails = cls_tails
        orig_cls_labels = cls_labels
        cls_heads, cls_rels, cls_tails, cls_labels, removed_train_idx = filter_train_by_neg_coverage(
            train_heads=cls_heads,
            train_rels=cls_rels,
            train_tails=cls_tails,
            train_labels=cls_labels,
            neg_node_mask=neg_node_mask,
            id2rel=id2rel,
            min_keep_per_group=min_keep,
        )

        # print("\n=== Train set sizes after NEG-edge coverage filter ===")
        # print(f"#TRAIN classification examples: {cls_heads.numel()}")
        # print(f"#TEST  classification examples: {test_heads_all.numel()}")
        # print_pos_neg_summary_train_test(
        #     train_rels=cls_rels,
        #     train_labels=cls_labels,
        #     test_rels=test_rels_all,
        #     test_labels=test_labels_all,
        #     id2rel=id2rel,
        #     topk=max(200, len(id2rel)),
        # )

        # Remove relations with too few test examples (pos or neg), drop from both train/test.
        min_test_per_label = 3
        if test_rels_all.numel() > 0:
            rels_test = test_rels_all.detach().cpu().long()
            labs_test = test_labels_all.detach().cpu().float()
            rel_ids = rels_test.unique().tolist()
            bad_rels: List[int] = []
            for rid in rel_ids:
                pos_n = int(((rels_test == rid) & (labs_test > 0.5)).sum().item())
                neg_n = int(((rels_test == rid) & (labs_test <= 0.5)).sum().item())
                if pos_n < min_test_per_label or neg_n < min_test_per_label:
                    bad_rels.append(int(rid))

            if bad_rels:
                bad_set = set(bad_rels)
                train_mask = ~torch.isin(cls_rels, torch.tensor(bad_rels, dtype=cls_rels.dtype))
                test_mask = ~torch.isin(test_rels_all, torch.tensor(bad_rels, dtype=test_rels_all.dtype))

                removed_train_h = cls_heads[~train_mask]
                removed_train_r = cls_rels[~train_mask]
                removed_train_t = cls_tails[~train_mask]
                removed_train_y = cls_labels[~train_mask]

                removed_test_h = test_heads_all[~test_mask]
                removed_test_r = test_rels_all[~test_mask]
                removed_test_t = test_tails_all[~test_mask]
                removed_test_y = test_labels_all[~test_mask]

                cls_heads = cls_heads[train_mask]
                cls_rels = cls_rels[train_mask]
                cls_tails = cls_tails[train_mask]
                cls_labels = cls_labels[train_mask]

                test_heads_all = test_heads_all[test_mask]
                test_rels_all = test_rels_all[test_mask]
                test_tails_all = test_tails_all[test_mask]
                test_labels_all = test_labels_all[test_mask]

                removed_count = int(removed_train_h.numel() + removed_test_h.numel())
                print(f"\n[INFO] Dropping relations with test pos/neg < {min_test_per_label}: {sorted(bad_set)}")
                print(f"[INFO] Removed train+test triples: {removed_count}")
                _append_cls_triples_to_encoder(
                    encoder_graph,
                    torch.cat([removed_train_h, removed_test_h], dim=0),
                    torch.cat([removed_train_r, removed_test_r], dim=0),
                    torch.cat([removed_train_t, removed_test_t], dim=0),
                    torch.cat([removed_train_y, removed_test_y], dim=0),
                    id2rel,
                    nflag=nflag,
                )

                print("\n=== Train/Test sizes after test-min filter ===")
                print(f"#TRAIN classification examples: {cls_heads.numel()}")
                print(f"#TEST  classification examples: {test_heads_all.numel()}")

        # Add removed train triples to encoder graph (as edges)
        if removed_train_idx.numel() > 0:
            rem_h = orig_cls_heads[removed_train_idx]
            rem_r = orig_cls_rels[removed_train_idx]
            rem_t = orig_cls_tails[removed_train_idx]
            rem_y = orig_cls_labels[removed_train_idx]
            removed_count = int(removed_train_idx.numel())
            print(f"\n[INFO] Adding removed train triples to encoder graph: {removed_count}")
            _append_cls_triples_to_encoder(
                encoder_graph,
                rem_h, rem_r, rem_t, rem_y,
                id2rel,
                nflag=nflag,
            )
    # After all filtering, prune minimally to improve neg-coverage while keeping >=3 per (rel,label)
    (cls_heads, cls_rels, cls_tails, cls_labels,
     test_heads_all, test_rels_all, test_tails_all, test_labels_all,
     rem_tr_h, rem_tr_r, rem_tr_t, rem_tr_y,
     rem_te_h, rem_te_r, rem_te_t, rem_te_y) = _prune_splits_by_neg_source_coverage(
        train_heads=cls_heads,
        train_rels=cls_rels,
        train_tails=cls_tails,
        train_labels=cls_labels,
        test_heads=test_heads_all,
        test_rels=test_rels_all,
        test_tails=test_tails_all,
        test_labels=test_labels_all,
        neg_node_mask=neg_node_mask,
        min_per_label=3,
    )

    if (rem_tr_h.numel() + rem_te_h.numel()) > 0:
        _append_cls_triples_to_encoder(
            encoder_graph,
            torch.cat([rem_tr_h, rem_te_h], dim=0),
            torch.cat([rem_tr_r, rem_te_r], dim=0),
            torch.cat([rem_tr_t, rem_te_t], dim=0),
            torch.cat([rem_tr_y, rem_te_y], dim=0),
            id2rel,
            nflag=nflag,
        )

    # Final per-relation counts for train/test
    print_pos_neg_summary_train_test(
        train_rels=cls_rels,
        train_labels=cls_labels,
        test_rels=test_rels_all,
        test_labels=test_labels_all,
        id2rel=id2rel,
        topk=max(200, len(id2rel)),
    )

    # Updated number of test triples whose SOURCE node has no NOT_* edge in encoder
    n_test_total = int(test_heads_all.numel())
    test_src_no_neg = int((~neg_node_mask[test_heads_all.long()]).sum().item()) if n_test_total else 0
    _p("\n=== Test triples with source node lacking NOT_* coverage ===")
    _p(f"Test src without NOT_* coverage: {test_src_no_neg}/{n_test_total} ({_pct(test_src_no_neg, n_test_total):.2f}%)")
    # (suppressed) initial per-relation counts; will print after all filtering

    # If --use_nstatementsampler, remove NOT_* edges only after pruning/splitting.
    if nflag:
        for et in list(encoder_graph.edge_types):
            if et == CLS_EDGE_TYPE:
                continue
            rel_s = str(et[1])
            if rel_s.startswith(NEG_PREFIX) or rel_s == "neg_statement":
                encoder_graph[et].edge_index = torch.empty((2, 0), dtype=torch.long)

    # Recompute edge-type lists in case encoder_graph changed (e.g., after filtering).
    MP_EDGE_TYPES = [et for et in encoder_graph.edge_types if et != CLS_EDGE_TYPE]
    encoder_e_etypes = MP_EDGE_TYPES

    # Encoder graph summary: subclass_of, instance_of, others
    sub_edges = 0
    inst_edges = 0
    other_edges = 0
    other_rels = set()
    for et in encoder_graph.edge_types:
        if et == CLS_EDGE_TYPE:
            continue
        rel_s = str(et[1])
        n = int(encoder_graph[et].edge_index.size(1)) if encoder_graph[et].edge_index is not None else 0
        if rel_s == str(subclass_rel):
            sub_edges += n
        elif rel_s == str(instance_rel):
            inst_edges += n
        else:
            other_edges += n
            other_rels.add(rel_s)
    _p("\n=== Encoder graph summary ===")
    _p(f"subclass_of edges: {sub_edges}")
    _p(f"instance_of edges: {inst_edges}")
    _p(f"other relations:   {len(other_rels)}")
    _p(f"other edges:       {other_edges}")

    # (suppressed) node coverage reports


    model_sig = inspect.signature(ModelCls.__init__)
    model_params = set(model_sig.parameters.keys())
    model_params.discard("self")
    supports_dropout = ("dropout" in model_params) or ("attn_dropout" in model_params)
    if not supports_dropout:
        if any(d is not None for d in dropout_candidates):
            print(f"[WARN] Model {ModelCls.__name__} does not accept dropout/attn_dropout; "
                  "ignoring dropout grid.")
        dropout_candidates = [None]
    print(f"Dropout candidates: {dropout_candidates}")

    use_contrastive = (not args.no_contrastive)
    if not use_contrastive:
        contrastive_weight_candidates = [0.0]
        contrastive_temp_candidates = [contrastive_temp_candidates[0]]
    print(f"Contrastive weight candidates: {contrastive_weight_candidates}")
    print(f"Contrastive temperature inits:  {contrastive_temp_candidates}")
    print(f"Train clone_inputs: {clone_inputs}")
    if cv_prune_ratio > 0.0:
        print(f"CV prune: ratio={cv_prune_ratio:.3g}, warmup_epochs={cv_prune_warmup}")
    if timing_profile:
        print("Timing profile: enabled")
        if torch.cuda.is_available():
            print(f"[GPU] device={torch.cuda.get_device_name(0)}")
            print("[GPU] utilization logging not available (nvidia-smi not found); "
                  "use nvidia-smi externally if needed.")
    if parallel_grid:
        print(f"Parallel grid: enabled (workers={parallel_grid_workers})")
        print(f"Parallel grid loaders: num_workers={parallel_grid_loader_workers}, "
              f"pin_memory={parallel_grid_pin_memory}")

    def _filter_model_kwargs(kwargs: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in kwargs.items() if k in model_params}

    def _build_model_kwargs(base_kwargs: Dict[str, object], dropout_value: Optional[float]) -> Dict[str, object]:
        kwargs = dict(base_kwargs)
        if dropout_value is not None:
            if "dropout" in model_params:
                kwargs["dropout"] = float(dropout_value)
            elif "attn_dropout" in model_params:
                kwargs["attn_dropout"] = float(dropout_value)
        return _filter_model_kwargs(kwargs)

    neighbor_sizes = [15, 10]
    hp_to_fold_losses = defaultdict(list)
    hp_to_fold_epochs = defaultdict(list)
    hp_to_fold_metrics = defaultdict(list)
    hp_to_fold_temperatures = defaultdict(list)
    cv_metrics: List[torch.Tensor] = []


    if args.finaltrain_only:
        if args.final_lr is None or args.final_epochs is None:
            raise ValueError("When using --finaltrain_only you must provide BOTH --final_lr and --final_epochs")
        final_lr = float(args.final_lr)
        final_epochs = int(args.final_epochs)
        final_dropout = dropout_candidates[0]
        final_contrastive_weight = float(contrastive_weight_candidates[0])
        if args.final_contrastive_temperature is not None:
            final_contrastive_temp_init = float(args.final_contrastive_temperature)
            final_contrastive_temperature = float(args.final_contrastive_temperature)
        else:
            final_contrastive_temp_init = float(contrastive_temp_candidates[0])
            final_contrastive_temperature = float(contrastive_temp_candidates[0])
        print("\n=== Final-only mode (CV skipped) ===")
        print(f"final_epochs: {final_epochs}")
        print(f"final_lr:     {final_lr}")
        print(f"final_contrastive_temperature: {final_contrastive_temperature}")

    elif args.test_only:
        model_path = os.path.join("output/" + args.output_dir, f"final_model_{args.model}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for testing not found: {model_path}")
        final_dropout = dropout_candidates[0]
        final_contrastive_weight = float(contrastive_weight_candidates[0])
        final_contrastive_temp_init = float(contrastive_temp_candidates[0])
        final_contrastive_temperature = float(contrastive_temp_candidates[0])

        final_base_kwargs = dict(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
            e_etypes=encoder_e_etypes,
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        )
        final_model_kwargs = _build_model_kwargs(final_base_kwargs, final_dropout)
        if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat"):
            final_model = ModelCls(**final_model_kwargs, rel2id=rel2id).to(device)
        else:
            final_model = ModelCls(**final_model_kwargs).to(device)

        state = torch.load(model_path, map_location=device)
        missing, unexpected = final_model.load_state_dict(state, strict=False)
        print("Loaded model.")
        print("  missing keys:", missing)
        print("  unexpected keys:", unexpected)

    else:
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

        splitter = StratifiedKFold(n_splits=int(k_folds), shuffle=True, random_state=42)
        print(f"[CV] StratifiedKFold with k={k_folds} (val size ≈ {1.0 / max(k_folds,1):.2f})")
        split_iter = splitter.split(np.zeros(len(main_idx), dtype=np.int64), y[main_idx]) if len(main_idx) else []

        hyperparam_grid = list(product(dropout_candidates, contrastive_weight_candidates, contrastive_temp_candidates))

        for fold, (tr_sub, va_sub) in enumerate(split_iter, start=1):
            train_idx_np = main_idx[tr_sub]
            val_idx_np = main_idx[va_sub]
            if rare_idx.size:
                train_idx_np = np.concatenate([train_idx_np, rare_idx], axis=0)

            train_idx = torch.tensor(train_idx_np, dtype=torch.long)
            val_idx = torch.tensor(val_idx_np, dtype=torch.long)

            train_edge_label_index = torch.stack([cls_heads[train_idx], cls_tails[train_idx]], dim=0)
            train_edge_label = cls_labels[train_idx].to(torch.float)
            val_edge_label_index = torch.stack([cls_heads[val_idx], cls_tails[val_idx]], dim=0)
            val_edge_label = cls_labels[val_idx].to(torch.float)

            print(f"\n=== CV Split {fold}/{k_folds} ===")
            print(f"  Train cls examples: {train_idx.numel()} | Val cls examples: {val_idx.numel()}")
            if parallel_grid:
                if cv_prune_ratio > 0.0:
                    print("[WARN] parallel_grid enabled; pruning disabled for parallel runs.")

                def _run_hparam(hparams):
                    dropout_val, contr_w, contr_temp_init = hparams
                    num_neighbors = {et: [20, 10] for et in MP_EDGE_TYPES}
                    num_neighbors[CLS_EDGE_TYPE] = [0, 0]
                    loader_workers = int(parallel_grid_loader_workers)
                    loader_kwargs = dict(
                        batch_size=args.batch_size,
                        num_workers=loader_workers,
                        persistent_workers=(loader_workers > 0),
                        pin_memory=parallel_grid_pin_memory,
                        neg_sampling_ratio=0.0,
                    )
                    if loader_workers > 0:
                        loader_kwargs["prefetch_factor"] = prefetch_factor

                    train_loader = LinkNeighborLoader(
                        encoder_graph,
                        num_neighbors=num_neighbors,
                        edge_label_index=(CLS_EDGE_TYPE, train_edge_label_index),
                        edge_label=train_edge_label,
                        shuffle=True,
                        **loader_kwargs,
                    )
                    val_loader = LinkNeighborLoader(
                        encoder_graph,
                        num_neighbors=num_neighbors,
                        edge_label_index=(CLS_EDGE_TYPE, val_edge_label_index),
                        edge_label=val_edge_label,
                        shuffle=False,
                        **loader_kwargs,
                    )

                    base_model_kwargs = dict(
                        in_dim=in_dim,
                        hidden_dim=mcfg["hidden_dim"],
                        out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
                        e_etypes=encoder_e_etypes,
                        n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
                    )
                    model_kwargs = _build_model_kwargs(base_model_kwargs, dropout_val)
                    if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat"):
                        model_fold = ModelCls(**model_kwargs, rel2id=rel2id).to(device)
                    else:
                        model_fold = ModelCls(**model_kwargs).to(device)

                    sampler_graph = struct_graph
                    if not args.no_contrastive:
                        if use_random_sampler:
                            neg_stmt_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                            neg_stmt_sampler.prepare_global(sampler_graph)
                        elif use_partial_sampler:
                            edges_are_negative = nflag
                            neg_stmt_sampler = PartialInstanceSampler(
                                k=contrastive_k,
                                neg_edges=external_edges,
                                edges_are_negative=edges_are_negative,
                            )
                            neg_stmt_sampler.prepare_global(sampler_graph)
                        elif use_negativeentity_sampler2:
                            neg_stmt_sampler = NegativeEntitySampler2(
                                k=contrastive_k,
                                max_pool_size=contrastive_pool_cap,
                                subclass_rel=subclass_rel,
                                neg_prefix=NEG_PREFIX,
                                instance_rel=instance_rel,
                            )
                            neg_stmt_sampler.prepare_global(sampler_graph)
                        elif use_negativeentity_sampler:
                            neg_stmt_sampler = NegativeEntitySampler(
                                k=contrastive_k,
                                max_pool_size=contrastive_pool_cap,
                                subclass_rel=subclass_rel,
                                neg_prefix=NEG_PREFIX,
                                instance_rel=instance_rel,
                            )
                            neg_stmt_sampler.prepare_global(sampler_graph)
                        else:
                            neg_stmt_sampler = NegativeInstanceSampler(
                                k=contrastive_k,
                                subclass_rel=subclass_rel,
                                neg_prefix=NEG_PREFIX,
                                instance_rel=instance_rel,
                            )
                            neg_stmt_sampler.prepare_global(sampler_graph)
                    else:
                        neg_stmt_sampler = None

                    dtag = "default" if dropout_val is None else f"{dropout_val:.3g}"
                    log_fold = Logger(f"train_cv_split{fold}_d{dtag}_cw{contr_w:.3g}_t{contr_temp_init:.3g}",
                                      dir=args.output_dir)
                    log_prefix = f"[fold={fold} d={dtag} cw={contr_w:.3g} t0={contr_temp_init:.3g}]"
                    trainer_fold = Train(
                        model=model_fold,
                        graph=encoder_graph,
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
                        contrastive_weight=contr_w,
                        contrastive_temperature=contr_temp_init,
                        learnable_contrastive_temperature=True,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        no_contrastive=args.no_contrastive,
                        clone_inputs=clone_inputs,
                        prune_ratio=0.0,
                        prune_warmup_epochs=cv_prune_warmup,
                        prune_target=None,
                        timing_profile=timing_profile,
                        log_prefix=log_prefix,
                    )
                    return (dropout_val, contr_w, contr_temp_init, *trainer_fold.run())

                with ThreadPoolExecutor(max_workers=int(parallel_grid_workers)) as executor:
                    futures = [executor.submit(_run_hparam, h) for h in hyperparam_grid]
                    for fut in as_completed(futures):
                        dropout_val, contr_w, contr_temp_init, best_val_loss, best_epoch, best_metrics, best_lr, per_lr = fut.result()
                        for lr in lr_candidates:
                            lr = float(lr)
                            rec = per_lr.get(lr, None)
                            if rec is None: continue
                            loss = float(rec["best_val_loss"])
                            ep = int(rec["best_epoch"])
                            temp = rec.get("best_temperature", None)
                            key = (lr, dropout_val, float(contr_w), float(contr_temp_init))
                            if loss != float("inf"): hp_to_fold_losses[key].append(loss)
                            if ep > 0: hp_to_fold_epochs[key].append(ep)
                            if temp is not None: hp_to_fold_temperatures[key].append(float(temp))
                            if rec.get("best_metrics", None) is not None: hp_to_fold_metrics[key].append(rec["best_metrics"])
                        if best_metrics is not None:
                            cv_metrics.append(best_metrics)
            else:
                num_neighbors = {et: [20, 10] for et in MP_EDGE_TYPES}
                num_neighbors[CLS_EDGE_TYPE] = [0, 0]   # <-- IMPORTANT: don't sample along cls_link

                train_loader = LinkNeighborLoader(encoder_graph, num_neighbors=num_neighbors,
                    edge_label_index=(CLS_EDGE_TYPE, train_edge_label_index), edge_label=train_edge_label,
                    batch_size=args.batch_size, shuffle=True, num_workers=2, persistent_workers=True,
                    pin_memory=(device.type == "cuda"), neg_sampling_ratio=0.0,
                    prefetch_factor=prefetch_factor)

                val_loader = LinkNeighborLoader(encoder_graph, num_neighbors=num_neighbors,
                    edge_label_index=(CLS_EDGE_TYPE, val_edge_label_index), edge_label=val_edge_label,
                    batch_size=args.batch_size, shuffle=False, num_workers=2, persistent_workers=True,
                    pin_memory=(device.type == "cuda"), neg_sampling_ratio=0.0,
                    prefetch_factor=prefetch_factor)

                sampler_graph = struct_graph
                if not args.no_contrastive:
                    if use_random_sampler:
                        neg_stmt_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    elif use_partial_sampler:
                        edges_are_negative = nflag
                        neg_stmt_sampler = PartialInstanceSampler(
                            k=contrastive_k,
                            neg_edges=external_edges,
                            edges_are_negative=edges_are_negative,
                        )
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    elif use_negativeentity_sampler2:
                        neg_stmt_sampler = NegativeEntitySampler2(
                            k=contrastive_k,
                            max_pool_size=contrastive_pool_cap,
                            subclass_rel=subclass_rel,
                            neg_prefix=NEG_PREFIX,
                            instance_rel=instance_rel,
                        )
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    elif use_negativeentity_sampler:
                        neg_stmt_sampler = NegativeEntitySampler(
                            k=contrastive_k,
                            max_pool_size=contrastive_pool_cap,
                            subclass_rel=subclass_rel,
                            neg_prefix=NEG_PREFIX,
                            instance_rel=instance_rel,
                        )
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    else:
                        neg_stmt_sampler = NegativeInstanceSampler(
                            k=contrastive_k,
                            subclass_rel=subclass_rel,
                            neg_prefix=NEG_PREFIX,
                            instance_rel=instance_rel,
                        )
                        neg_stmt_sampler.prepare_global(sampler_graph)
                else:
                    neg_stmt_sampler = None

                fold_best_val = float("inf")
                for hp_i, (dropout_val, contr_w, contr_temp_init) in enumerate(hyperparam_grid, start=1):
                    base_model_kwargs = dict(
                        in_dim=in_dim,
                        hidden_dim=mcfg["hidden_dim"],
                        out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
                        e_etypes=encoder_e_etypes,
                        n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
                    )
                    model_kwargs = _build_model_kwargs(base_model_kwargs, dropout_val)
                    if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat"):
                        model_fold = ModelCls(**model_kwargs, rel2id=rel2id).to(device)
                    else:
                        model_fold = ModelCls(**model_kwargs).to(device)

                    dtag = "default" if dropout_val is None else f"{dropout_val:.3g}"
                    log_fold = Logger(f"train_cv_split{fold}_d{dtag}_cw{contr_w:.3g}_t{contr_temp_init:.3g}",
                                      dir=args.output_dir)
                    log_prefix = f"[fold={fold} d={dtag} cw={contr_w:.3g} t0={contr_temp_init:.3g}]"
                    trainer_fold = Train(
                        model=model_fold,
                        graph=encoder_graph,
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
                        contrastive_weight=contr_w,
                        contrastive_temperature=contr_temp_init,
                        learnable_contrastive_temperature=True,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        no_contrastive=args.no_contrastive,
                        clone_inputs=clone_inputs,
                        prune_ratio=cv_prune_ratio,
                        prune_warmup_epochs=cv_prune_warmup,
                        prune_target=(fold_best_val if fold_best_val < float("inf") else None),
                        timing_profile=timing_profile,
                        log_prefix=log_prefix,
                    )

                    best_val_loss, best_epoch, best_metrics, best_lr, per_lr = trainer_fold.run()
                    if best_val_loss is not None and best_val_loss < fold_best_val:
                        fold_best_val = float(best_val_loss)
                    for lr in lr_candidates:
                        lr = float(lr)
                        rec = per_lr.get(lr, None)
                        if rec is None: continue
                        loss = float(rec["best_val_loss"])
                        ep = int(rec["best_epoch"])
                        temp = rec.get("best_temperature", None)

                        key = (lr, dropout_val, float(contr_w), float(contr_temp_init))
                        if loss != float("inf"): hp_to_fold_losses[key].append(loss)
                        if ep > 0: hp_to_fold_epochs[key].append(ep)
                        if temp is not None: hp_to_fold_temperatures[key].append(float(temp))
                        if rec.get("best_metrics", None) is not None: hp_to_fold_metrics[key].append(rec["best_metrics"])
                    if best_metrics is not None:
                        cv_metrics.append(best_metrics)

        hp_summary = []
        for key, losses in hp_to_fold_losses.items():
            if not losses:
                continue
            mean_loss = statistics.mean(losses)
            std_loss = statistics.pstdev(losses) if len(losses) > 1 else 0.0
            n = len(losses)
            hp_summary.append((mean_loss, std_loss, n, key))
        if not hp_summary:
            final_lr = float(lr_candidates[0])
            final_dropout = dropout_candidates[0]
            final_contrastive_weight = float(contrastive_weight_candidates[0])
            final_contrastive_temp_init = float(contrastive_temp_candidates[0])
            final_contrastive_temperature = float(contrastive_temp_candidates[0])
            final_epochs = int(args.epochs)
        else:
            hp_summary.sort(key=lambda x: (
                x[0], x[1], x[3][0], x[3][1] if x[3][1] is not None else -1.0, x[3][2], x[3][3]
            ))
            best_mean, best_std, best_n, best_key = hp_summary[0]
            final_lr, final_dropout, final_contrastive_weight, final_contrastive_temp_init = best_key
            epochs_for_key = hp_to_fold_epochs.get(best_key, [])
            final_epochs = int(statistics.median(epochs_for_key)) if epochs_for_key else int(args.epochs)
            temps_for_key = hp_to_fold_temperatures.get(best_key, [])
            if temps_for_key:
                final_contrastive_temperature = float(statistics.median(temps_for_key))
            else:
                final_contrastive_temperature = float(final_contrastive_temp_init)
        _p("\n=== Cross-validation summary (TRAIN split only) ===")
        _p("Grid aggregates (mean_val_loss ± std over folds):")
        for mean_loss, std_loss, n, key in sorted(hp_summary, key=lambda x: (x[0], x[1], x[3][0])):
            lr, d, cw, ct = key
            dtag = "default" if d is None else f"{d:.3g}"
            _p(f"  lr={lr:.3g} | dropout={dtag} | c_w={cw:.3g} | c_t0={ct:.3g} | "
                  f"mean={mean_loss:.6f} | std={std_loss:.6f} | folds={n}")
        dtag = "default" if final_dropout is None else f"{final_dropout:.3g}"
        _p(f"\nChosen final_lr (best mean over folds): {final_lr:.6g}")
        _p(f"Chosen final_dropout:                    {dtag}")
        _p(f"Chosen final_contrastive_weight:         {final_contrastive_weight:.6g}")
        _p(f"Chosen final_contrastive_temperature:    {final_contrastive_temperature:.6g}")
        final_key = (final_lr, final_dropout, float(final_contrastive_weight), float(final_contrastive_temp_init))
        _p(f"Epochs for chosen grid key across folds: {hp_to_fold_epochs.get(final_key, [])}")
        _p(f"Chosen final_epochs (median for key):    {final_epochs}")


    # -----------------------------------------------------------------
    # Final training on TRAIN split
    # -----------------------------------------------------------------
    if not args.test_only:
        final_edge_label_index = torch.stack([cls_heads, cls_tails], dim=0)
        final_edge_label = cls_labels.to(torch.float)
        num_neighbors = {et: neighbor_sizes for et in MP_EDGE_TYPES}
        num_neighbors[CLS_EDGE_TYPE] = [0, 0]
        
        final_loader = LinkNeighborLoader(encoder_graph, num_neighbors=num_neighbors,
            edge_label_index=(CLS_EDGE_TYPE, final_edge_label_index),neg_sampling_ratio=0.0,
            edge_label=final_edge_label, batch_size=args.batch_size, shuffle=True,
            num_workers=2, persistent_workers=True, pin_memory=(device.type == "cuda"),
            prefetch_factor=prefetch_factor)

        final_base_kwargs = dict(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
            e_etypes=encoder_e_etypes,
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        )
        final_model_kwargs = _build_model_kwargs(final_base_kwargs, final_dropout)
        if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gat", "gae"):
            final_model = ModelCls(**final_model_kwargs, rel2id=rel2id).to(device)
        else:
            final_model = ModelCls(**final_model_kwargs).to(device)

        final_log = Logger("final_train_global", dir=args.output_dir, non_verbose=True)

        if not args.no_contrastive:
            if use_random_sampler:
                final_contrastive_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                final_contrastive_sampler.prepare_global(struct_graph)
            elif use_partial_sampler:
                edges_are_negative = nflag
                final_contrastive_sampler = PartialInstanceSampler(
                    k=contrastive_k,
                    neg_edges=external_edges,
                    edges_are_negative=edges_are_negative,
                )
                final_contrastive_sampler.prepare_global(struct_graph)
            elif use_negativeentity_sampler2:
                final_contrastive_sampler = NegativeEntitySampler2(
                    k=contrastive_k,
                    max_pool_size=contrastive_pool_cap,
                    subclass_rel=subclass_rel,
                    neg_prefix=NEG_PREFIX,
                    instance_rel=instance_rel,
                )
                final_contrastive_sampler.prepare_global(struct_graph)
            elif use_negativeentity_sampler:
                final_contrastive_sampler = NegativeEntitySampler(
                    k=contrastive_k,
                    max_pool_size=contrastive_pool_cap,
                    subclass_rel=subclass_rel,
                    neg_prefix=NEG_PREFIX,
                    instance_rel=instance_rel,
                )
                final_contrastive_sampler.prepare_global(struct_graph)
            else:
                final_contrastive_sampler = NegativeInstanceSampler(
                    k=contrastive_k,
                    subclass_rel=subclass_rel,
                    neg_prefix=NEG_PREFIX,
                    instance_rel=instance_rel,
                )
                final_contrastive_sampler.prepare_global(struct_graph)
        else:
            final_contrastive_sampler = None

        if args.finaltrain_only:
            final_lr = float(args.final_lr)
            final_epochs = int(args.final_epochs)

        final_trainer = Train_BestModel(
            final_model,
            graph=encoder_graph,
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
            contrastive_weight=final_contrastive_weight,
            loader=final_loader,
            no_contrastive=args.no_contrastive,
            contrastive_temperature=final_contrastive_temperature,
            learnable_contrastive_temperature=True,
            timing_profile=timing_profile,
        )
        final_loss = final_trainer.run()
        _p(f"[Final Train] Loss after {final_epochs} epochs (lr={final_lr:.3g}): {final_loss:.4f}")
    test_log = Logger("test_global", dir=args.output_dir)

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
        _p("[WARN] No TEST negatives found; Test_BestModel may not be able to evaluate properly.")

    cache_path = os.path.join("data/neg_cache", f"test_negs_{args.task}_numneg{args.num_neg_test}.pt")
    _p("neg cache abs path:", os.path.abspath(cache_path))

    tester = Test_BestModel(
        model=final_model,
        graph=encoder_graph,
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
