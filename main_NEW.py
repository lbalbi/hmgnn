# main.py
import os
import hashlib
import re
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader

from data_loader import DataLoader
from models import *
from trainer_NEW import Train
from trainer_bestmodel_NEW import Train_BestModel, LinkPredictionEvaluator
from utils import Logger, load_config, ensure_dir
from samplers import (
    NegativeInstanceSampler_NEWER, NegativeInstanceSampler_NEW, RandomInstanceSampler, TypedInstanceSampler, PartialInstanceSampler
)

NEG_PREFIX = "NOT_"
CLS_EDGE_TYPE = ("node", "cls_link", "node")
CLS_EDGE_SUFFIX = "__cls"

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

def _add_cls_edges_for_sampler(
    g: HeteroData,
    *,
    heads: torch.Tensor,
    tails: torch.Tensor,
    rels: torch.Tensor,
    labels: torch.Tensor,
    id2rel: Dict[int, str],
) -> Tuple[int, int]:
    """Append train classification triples as typed edges using base relation names."""
    n_cls_pos = 0
    n_cls_neg = 0
    if heads is None or heads.numel() == 0:
        return 0, 0
    h = heads.detach().cpu().numpy()
    t = tails.detach().cpu().numpy()
    r = rels.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    for rid in np.unique(r):
        rel = str(id2rel[int(rid)])
        mask_r = (r == rid)
        mask_pos = mask_r & (y > 0.5)
        mask_neg = mask_r & (y <= 0.5)
        if mask_pos.any():
            src = torch.tensor(h[mask_pos], dtype=torch.long)
            tgt = torch.tensor(t[mask_pos], dtype=torch.long)
            et = rel
            _append_edges(g, et, src, tgt)
            n_cls_pos += int(mask_pos.sum())
        if mask_neg.any():
            src = torch.tensor(h[mask_neg], dtype=torch.long)
            tgt = torch.tensor(t[mask_neg], dtype=torch.long)
            et_base = _ensure_not_prefixed(rel)
            et = et_base
            _append_edges(g, et, src, tgt)
            n_cls_neg += int(mask_neg.sum())
    return n_cls_pos, n_cls_neg


def _message_passing_edge_types(
    graph: HeteroData,
    *,
    subclass_rel: str,
    drop_subclass: bool,
    cls_edge_suffix: str = CLS_EDGE_SUFFIX,
) -> List[Tuple[str, str, str]]:
    etypes = [
        et for et in graph.edge_types
        if et != CLS_EDGE_TYPE and not str(et[1]).endswith(cls_edge_suffix)
    ]
    if drop_subclass:
        etypes = [et for et in etypes if str(et[1]) != str(subclass_rel)]
    return etypes


def _build_hr_paired_train_indices(
    *,
    heads: torch.Tensor,
    rels: torch.Tensor,
    labels: torch.Tensor,
    candidate_idx: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Build a supervision index where each positive triple is paired with one
    explicit negative triple that has the same (head, relation).

    Returns:
      paired_idx_global: shape [2 * n_pairs], interleaved as [pos, neg, pos, neg, ...]
      stats: summary counters for logging
    """
    cand = candidate_idx.detach().cpu().long().numpy()
    if cand.size == 0:
        return torch.empty((0,), dtype=torch.long), {
            "raw_total": 0,
            "raw_pos": 0,
            "raw_neg": 0,
            "paired_pos": 0,
            "paired_neg": 0,
            "dropped_pos_no_hr_neg": 0,
        }

    h = heads.detach().cpu().long().numpy()[cand]
    r = rels.detach().cpu().long().numpy()[cand]
    y = labels.detach().cpu().numpy()[cand]

    pos_local = np.where(y > 0.5)[0]
    neg_local = np.where(y <= 0.5)[0]

    hr_to_neg_locals: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for li in neg_local:
        hr_to_neg_locals[(int(h[li]), int(r[li]))].append(int(li))

    rng = np.random.default_rng(int(seed))
    paired_pos_local: List[int] = []
    paired_neg_local: List[int] = []
    dropped_pos = 0
    for li in pos_local:
        key = (int(h[li]), int(r[li]))
        candidates = hr_to_neg_locals.get(key, None)
        if not candidates:
            dropped_pos += 1
            continue
        neg_li = int(candidates[int(rng.integers(low=0, high=len(candidates)))])
        paired_pos_local.append(int(li))
        paired_neg_local.append(neg_li)

    n_pairs = len(paired_pos_local)
    if n_pairs == 0:
        return torch.empty((0,), dtype=torch.long), {
            "raw_total": int(cand.size),
            "raw_pos": int(pos_local.size),
            "raw_neg": int(neg_local.size),
            "paired_pos": 0,
            "paired_neg": 0,
            "dropped_pos_no_hr_neg": int(dropped_pos),
        }

    paired_local = np.empty((2 * n_pairs,), dtype=np.int64)
    paired_local[0::2] = np.asarray(paired_pos_local, dtype=np.int64)
    paired_local[1::2] = np.asarray(paired_neg_local, dtype=np.int64)
    paired_global = cand[paired_local]
    paired_idx_global = torch.tensor(paired_global, dtype=torch.long)

    return paired_idx_global, {
        "raw_total": int(cand.size),
        "raw_pos": int(pos_local.size),
        "raw_neg": int(neg_local.size),
        "paired_pos": int(n_pairs),
        "paired_neg": int(n_pairs),
        "dropped_pos_no_hr_neg": int(dropped_pos),
    }


def _drop_val_examples_seen_in_train(
    *,
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Leak guard for CV: remove validation examples that exactly match any
    (h, r, t, y) example present in that fold's train split.
    """
    tr = train_idx.detach().cpu().long().numpy()
    va = val_idx.detach().cpu().long().numpy()
    if va.size == 0 or tr.size == 0:
        return val_idx, {
            "val_before": int(va.size),
            "val_removed_seen_in_train": 0,
            "val_after": int(va.size),
        }

    h = heads.detach().cpu().long().numpy()
    r = rels.detach().cpu().long().numpy()
    t = tails.detach().cpu().long().numpy()
    y = (labels.detach().cpu().numpy() > 0.5).astype(np.int8)

    train_keys = set(
        zip(h[tr].tolist(), r[tr].tolist(), t[tr].tolist(), y[tr].tolist())
    )
    keep_mask = np.array(
        [
            (int(h[i]), int(r[i]), int(t[i]), int(y[i])) not in train_keys
            for i in va.tolist()
        ],
        dtype=bool,
    )
    va_new = va[keep_mask]
    removed = int(va.size - va_new.size)
    return torch.tensor(va_new, dtype=torch.long), {
        "val_before": int(va.size),
        "val_removed_seen_in_train": removed,
        "val_after": int(va_new.size),
    }

def _assert_shared_id_alignment(
    *,
    base_graph: HeteroData,
    contrastive_graph: HeteroData,
) -> None:
    """
    Ensure nodes/edges that already existed in base_graph keep the same ids in
    contrastive_graph. Contrastive graph may append nodes/edges, but existing
    ids must be stable.
    """
    if "node" not in base_graph.node_types or "node" not in contrastive_graph.node_types:
        raise RuntimeError("Alignment check requires node type 'node' in both graphs.")

    n_base = int(base_graph["node"].num_nodes)
    n_con = int(contrastive_graph["node"].num_nodes)
    if n_con < n_base:
        raise RuntimeError(
            f"Contrastive graph shrank node space ({n_con}) below base graph ({n_base})."
        )

    bx = base_graph["node"].x
    cx = contrastive_graph["node"].x
    if bx is not None and cx is not None and bx.numel() > 0 and cx.numel() > 0:
        if cx.size(0) < bx.size(0) or cx.size(1) != bx.size(1):
            raise RuntimeError("Node feature shape mismatch between base and contrastive graphs.")
        if not torch.equal(cx[:n_base], bx):
            raise RuntimeError(
                "Shared node features differ between base and contrastive graphs; id alignment broken."
            )

    # Existing edges must be preserved (same ids/order as prefix) for every base edge type.
    for et in base_graph.edge_types:
        be = base_graph[et].edge_index
        if be is None:
            continue
        if et not in contrastive_graph.edge_types:
            raise RuntimeError(f"Contrastive graph missing base edge type: {et}")
        ce = contrastive_graph[et].edge_index
        if ce is None:
            raise RuntimeError(f"Contrastive graph edge_index is None for edge type: {et}")
        if ce.size(1) < be.size(1):
            raise RuntimeError(f"Contrastive graph has fewer edges for edge type {et}.")
        if be.numel() > 0 and not torch.equal(ce[:, : be.size(1)], be):
            raise RuntimeError(
                f"Base edge ids/order not preserved for edge type {et}; alignment broken."
            )

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

def compute_neg_neighbor_similarity_chart(
    *,
    model: torch.nn.Module,
    encoder_graph: HeteroData,
    cls_heads: torch.Tensor,
    cls_tails: torch.Tensor,
    subclass_rel: str,
    neg_prefix: str,
    instance_rel: str,
    output_dir: str,
    max_pool_size: int = 100,
    chart_bins: int = 40,
) -> None:
    cls_nodes = torch.unique(torch.cat([cls_heads, cls_tails], dim=0)).detach().cpu().long()
    if cls_nodes.numel() == 0:
        print("[NegNeighborSim] No classification nodes found; skipping.")
        return

    sampler = NegativeInstanceSampler_NEW(
        k=1,
        subclass_rel=subclass_rel,
        neg_prefix=neg_prefix,
        instance_rel=instance_rel,
        max_pool_size=max_pool_size,
        cache_dir="data/cache",
        cache_key=f"neg_neighbor_sim|{output_dir}",
    )
    graph_cpu = encoder_graph.cpu()
    sampler.prepare_global(graph_cpu)

    n_type = getattr(model, "n_type", "node")
    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")

    model_cpu = model.cpu()
    model_cpu.eval()
    with torch.inference_mode():
        h_dict = model_cpu.encode(graph_cpu)
        z = h_dict[n_type].detach().cpu()
    if orig_device.type != "cpu":
        model.to(orig_device)

    z_norm = F.normalize(z, dim=1)

    def _collect_neighbors(u: int) -> Tuple[torch.Tensor, int, int]:
        empty = torch.empty(0, dtype=torch.long)
        pos_pool = empty
        neg_pool = empty
        if getattr(sampler, "pool_pos_to_u_neg_by_pred", None) is not None:
            pos_maps = sampler.pool_pos_to_u_neg_by_pred[u]
            neg_maps = sampler.pool_neg_to_u_pos_by_pred[u]
            pos_parts = [t for t in pos_maps.values() if t is not None and t.numel() > 0]
            neg_parts = [t for t in neg_maps.values() if t is not None and t.numel() > 0]
            if pos_parts:
                pos_pool = torch.unique(torch.cat(pos_parts, dim=0))
            if neg_parts:
                neg_pool = torch.unique(torch.cat(neg_parts, dim=0))
        else:
            if sampler.pool_pos_to_u_neg and u < len(sampler.pool_pos_to_u_neg):
                pos_pool = sampler.pool_pos_to_u_neg[u]
            if sampler.pool_neg_to_u_pos and u < len(sampler.pool_neg_to_u_pos):
                neg_pool = sampler.pool_neg_to_u_pos[u]

        if pos_pool.numel() == 0 and neg_pool.numel() == 0:
            return empty, 0, 0
        if pos_pool.numel() == 0:
            return neg_pool, 0, int(neg_pool.numel())
        if neg_pool.numel() == 0:
            return pos_pool, int(pos_pool.numel()), 0
        neighbors = torch.unique(torch.cat([pos_pool, neg_pool], dim=0))
        return neighbors, int(pos_pool.numel()), int(neg_pool.numel())

    def _collect_same_sign_neighbors(u: int) -> Tuple[torch.Tensor, torch.Tensor]:
        empty = torch.empty(0, dtype=torch.long)
        pos_pool = empty
        if getattr(sampler, "pool_shared_pos_by_pred", None) is not None:
            pos_maps = sampler.pool_shared_pos_by_pred[u]
            pos_parts = [t for t in pos_maps.values() if t is not None and t.numel() > 0]
            if pos_parts:
                pos_pool = torch.unique(torch.cat(pos_parts, dim=0))
        else:
            if sampler.pool_shared_pos and u < len(sampler.pool_shared_pos):
                pos_pool = sampler.pool_shared_pos[u]
        return pos_pool, empty

    # Negative neighbors defined by ANY negative edge connected to the node (1-hop)
    cls_set = set(int(x) for x in cls_nodes.tolist())
    neg_edge_neighbors: Dict[int, set[int]] = {u: set() for u in cls_set}
    for (s, rel, d), eidx in encoder_graph.edge_index_dict.items():
        if s != "node" or d != "node":
            continue
        rel_s = str(rel)
        if rel_s == str(subclass_rel):
            continue
        if not (rel_s.startswith(str(neg_prefix)) or rel_s == "neg_statement"):
            continue
        if eidx is None or eidx.numel() == 0:
            continue
        src = eidx[0].detach().cpu().long().tolist()
        dst = eidx[1].detach().cpu().long().tolist()
        for a, b in zip(src, dst):
            if a in cls_set:
                neg_edge_neighbors[a].add(b)
            if b in cls_set:
                neg_edge_neighbors[b].add(a)

    rows = []
    skipped_no_neighbors = 0
    skipped_oob = 0
    for u in cls_nodes.tolist():
        if u < 0 or u >= sampler.num_nodes:
            skipped_oob += 1
            continue
        neighbors, n_pos2neg, n_neg2pos = _collect_neighbors(u)
        if neighbors.numel() == 0:
            skipped_no_neighbors += 1
            continue
        sim = (z_norm[u].unsqueeze(0) * z_norm[neighbors]).sum(dim=1)
        rows.append(
            {
                "node_id": int(u),
                "avg_similarity": float(sim.mean().item()),
                "num_neighbors": int(neighbors.numel()),
                "num_pos2neg": int(n_pos2neg),
                "num_neg2pos": int(n_neg2pos),
            }
        )

    if not rows:
        print("[NegNeighborSim] No nodes with neg-neighbor pools; skipping chart.")
        return

    df = pd.DataFrame(rows)
    out_dir = os.path.join("output", output_dir)
    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, "neg_neighbor_similarity.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)

    import matplotlib.pyplot as plt
    vals = df["avg_similarity"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(vals, bins=int(chart_bins), color="#2a6f9a", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Avg cosine similarity to neg neighbors")
    ax.set_ylabel("Classification node count")
    ax.set_title("Neg-neighbor similarity (pos->u_neg, neg->u_pos)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    chart_path = os.path.join(out_dir, "neg_neighbor_similarity_hist.png")
    fig.savefig(chart_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(
        "[NegNeighborSim] "
        f"nodes_with_neighbors={len(rows)} "
        f"skipped_no_neighbors={skipped_no_neighbors} "
        f"skipped_oob={skipped_oob} "
        f"mean={float(vals.mean()):.4f} median={float(np.median(vals)):.4f}"
    )
    print(f"[NegNeighborSim] Wrote: {tsv_path}")
    print(f"[NegNeighborSim] Chart: {chart_path}")

    # Same-sign neighbors: positive and negative separately
    pos_rows = []
    neg_rows = []
    skipped_no_pos = 0
    skipped_no_neg = 0
    skipped_oob = 0
    for u in cls_nodes.tolist():
        if u < 0 or u >= sampler.num_nodes:
            skipped_oob += 1
            continue
        pos_pool, _ = _collect_same_sign_neighbors(u)
        if pos_pool.numel() > 0:
            sim_pos = (z_norm[u].unsqueeze(0) * z_norm[pos_pool]).sum(dim=1)
            pos_rows.append(
                {
                    "node_id": int(u),
                    "avg_similarity": float(sim_pos.mean().item()),
                    "num_neighbors": int(pos_pool.numel()),
                }
            )
        else:
            skipped_no_pos += 1

        neg_list = neg_edge_neighbors.get(int(u), set())
        if neg_list:
            neg_pool = torch.tensor(sorted(neg_list), dtype=torch.long)
            sim_neg = (z_norm[u].unsqueeze(0) * z_norm[neg_pool]).sum(dim=1)
            neg_rows.append(
                {
                    "node_id": int(u),
                    "avg_similarity": float(sim_neg.mean().item()),
                    "num_neighbors": int(neg_pool.numel()),
                }
            )
        else:
            skipped_no_neg += 1

    out_dir = os.path.join("output", output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if pos_rows:
        df_pos = pd.DataFrame(pos_rows)
        tsv_pos = os.path.join(out_dir, "pos_neighbor_similarity.tsv")
        df_pos.to_csv(tsv_pos, sep="\t", index=False)
        vals = df_pos["avg_similarity"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        ax.hist(vals, bins=int(chart_bins), color="#2a9d8f", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Avg cosine similarity to pos neighbors")
        ax.set_ylabel("Classification node count")
        ax.set_title("Pos-neighbor similarity (shared positive statements)")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        chart_pos = os.path.join(out_dir, "pos_neighbor_similarity_hist.png")
        fig.savefig(chart_pos, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(
            "[PosNeighborSim] "
            f"nodes_with_neighbors={len(pos_rows)} "
            f"skipped_no_neighbors={skipped_no_pos} "
            f"skipped_oob={skipped_oob} "
            f"mean={float(vals.mean()):.4f} median={float(np.median(vals)):.4f}"
        )
        print(f"[PosNeighborSim] Wrote: {tsv_pos}")
        print(f"[PosNeighborSim] Chart: {chart_pos}")
    else:
        print("[PosNeighborSim] No nodes with positive neighbors; skipping chart.")

    if neg_rows:
        df_neg = pd.DataFrame(neg_rows)
        tsv_neg = os.path.join(out_dir, "neg_neighbor_similarity_same.tsv")
        df_neg.to_csv(tsv_neg, sep="\t", index=False)
        vals = df_neg["avg_similarity"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        ax.hist(vals, bins=int(chart_bins), color="#e76f51", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Avg cosine similarity to neg neighbors")
        ax.set_ylabel("Classification node count")
        ax.set_title("Neg-neighbor similarity (1-hop negative edges)")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        chart_neg = os.path.join(out_dir, "neg_neighbor_similarity_same_hist.png")
        fig.savefig(chart_neg, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(
            "[NegNeighborSameSim] "
            f"nodes_with_neighbors={len(neg_rows)} "
            f"skipped_no_neighbors={skipped_no_neg} "
            f"skipped_oob={skipped_oob} "
            f"mean={float(vals.mean()):.4f} median={float(np.median(vals)):.4f}"
        )
        print(f"[NegNeighborSameSim] Wrote: {tsv_neg}")
        print(f"[NegNeighborSameSim] Chart: {chart_neg}")
    else:
        print("[NegNeighborSameSim] No nodes with negative neighbors; skipping chart.")


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
) -> Dict[str, torch.Tensor]:
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

    return {
        "missing_all": missing_all,
        "missing_train": missing_tr,
        "missing_test": missing_te,
        "train_nodes": train_nodes,
        "test_nodes": test_nodes,
        "all_cls_nodes": all_cls_nodes,
        "enc_nodes": enc_nodes,
    }


def _load_retrieved_node_idx_set(
    *,
    retrieved_dir: str,
    map_out_name: str = "qid2idx_retrieved.tsv",
) -> Set[int]:
    map_path = os.path.join(retrieved_dir, map_out_name)
    if not os.path.exists(map_path):
        return set()
    try:
        df = pd.read_csv(map_path, sep="\t")
        if "node_idx" not in df.columns:
            return set()
        vals = pd.to_numeric(df["node_idx"], errors="coerce").dropna().astype(np.int64).tolist()
        return set(int(v) for v in vals)
    except Exception as e:
        print(f"[WARN] Could not parse retrieved node map at {map_path}: {e}")
        return set()


def write_missing_cls_nodes_with_sources(
    *,
    missing_nodes: torch.Tensor,
    train_nodes: torch.Tensor,
    test_nodes: torch.Tensor,
    retrieved_node_ids: Optional[Set[int]],
    out_path: str,
) -> None:
    missing_set = set(int(x) for x in missing_nodes.detach().cpu().long().tolist())
    train_set = set(int(x) for x in train_nodes.detach().cpu().long().tolist())
    test_set = set(int(x) for x in test_nodes.detach().cpu().long().tolist())
    retrieved_set = set(retrieved_node_ids or set())

    rows: List[Dict[str, object]] = []
    for nid in sorted(missing_set):
        in_train = int(nid in train_set)
        in_test = int(nid in test_set)
        in_retrieved = int(nid in retrieved_set)
        tags: List[str] = []
        if in_train:
            tags.append("train")
        if in_test:
            tags.append("test")
        if in_retrieved:
            tags.append("retrieved")
        rows.append(
            {
                "node_id": int(nid),
                "in_train_files": in_train,
                "in_test_files": in_test,
                "in_retrieved_files": in_retrieved,
                "sources": "|".join(tags),
            }
        )

    out_df = pd.DataFrame(
        rows,
        columns=["node_id", "in_train_files", "in_test_files", "in_retrieved_files", "sources"],
    )
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"[INFO] Wrote missing classification-node provenance file: {out_path}")


def _load_original_subclass_pairs(
    *,
    data_dir: str,
    subclass_rel: str,
    neg_prefix: str = NEG_PREFIX,
) -> Set[Tuple[int, int]]:
    """
    Keep-list for subclass edges considered "original", restricted to:
      - train2id_pos.txt
      - train2id_neg.txt
      - test2id_pos.txt
    """
    keep: Set[Tuple[int, int]] = set()
    files = ["train2id_pos.txt", "train2id_neg.txt", "test2id_pos.txt"]
    subclass_aliases = {str(subclass_rel), "subclass_of"}
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            df = _read_edge_file(path)
        except Exception as e:
            print(f"[WARN] Could not read {path} for subclass keep-list: {e}")
            continue
        rel_base = df["edge_type"].astype(str).map(lambda r: _strip_not(r, neg_prefix=neg_prefix))
        m = rel_base.isin(subclass_aliases)
        if not bool(m.any()):
            continue
        src = pd.to_numeric(df.loc[m, "source_node"], errors="coerce").dropna().astype(np.int64).tolist()
        tgt = pd.to_numeric(df.loc[m, "target_node"], errors="coerce").dropna().astype(np.int64).tolist()
        for s, t in zip(src, tgt):
            keep.add((int(s), int(t)))
    return keep


def _filter_rel_edges_by_allowlist(
    *,
    graph: HeteroData,
    rel: str,
    allowed_pairs: Set[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    In-place filter of edge type ('node', rel, 'node') so only allowed (src, tgt)
    pairs remain. Returns (before_count, after_count).
    """
    key = ("node", str(rel), "node")
    if key not in graph.edge_types:
        return 0, 0
    ei = graph[key].edge_index
    if ei is None or ei.numel() == 0:
        return 0, 0

    before = int(ei.size(1))
    if not allowed_pairs:
        graph[key].edge_index = torch.empty((2, 0), dtype=torch.long)
        return before, 0

    src = ei[0].detach().cpu().tolist()
    tgt = ei[1].detach().cpu().tolist()
    keep_mask = torch.tensor(
        [(int(s), int(t)) in allowed_pairs for s, t in zip(src, tgt)],
        dtype=torch.bool,
    )
    graph[key].edge_index = ei[:, keep_mask]
    after = int(graph[key].edge_index.size(1))
    return before, after


def _append_contrastive_extra_negatives_from_file(
    *,
    graph: HeteroData,
    data_dir: str,
    file_name: str = "train2id_neg_extras_for_contrastive.txt",
) -> Dict[str, int]:
    """
    Append extra negative triples only to the contrastive graph.
    File must contain: source_node,target_node,edge_type
    """
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path):
        return {"rows": 0, "added": 0}

    df = _read_edge_file(path)
    if len(df) == 0:
        return {"rows": 0, "added": 0}

    rows = int(len(df))
    total_added = 0
    for rel, g in df.groupby("edge_type"):
        rel_s = str(rel)
        src_vals = pd.to_numeric(g["source_node"], errors="coerce").dropna().astype(np.int64).tolist()
        tgt_vals = pd.to_numeric(g["target_node"], errors="coerce").dropna().astype(np.int64).tolist()
        if not src_vals or not tgt_vals:
            continue
        pairs = list(zip(src_vals, tgt_vals))
        existing = _edge_set_from_graph(graph, rel_s)
        new_pairs = [(int(s), int(t)) for s, t in pairs if (int(s), int(t)) not in existing]
        if not new_pairs:
            continue
        src = torch.tensor([s for s, _ in new_pairs], dtype=torch.long)
        tgt = torch.tensor([t for _, t in new_pairs], dtype=torch.long)
        _append_edges(graph, rel_s, src, tgt)
        total_added += int(len(new_pairs))

    return {"rows": rows, "added": int(total_added)}


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
    min_neg_per_rel: int,
    cls_keep_frac: float,
) -> str:
    cache_dir = os.path.join(data_dir, "split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    fname = (
        f"splits_{_sanitize_component(task)}"
        "_origfiles"
        f"_sub{_sanitize_component(subclass_rel)}"
        f"_inst{_sanitize_component(instance_rel)}"
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
    min_neg_per_rel: int,
    cls_keep_frac: float,
) -> Dict[str, object]:
    train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
    train_neg_path = os.path.join(data_dir, "train2id_neg.txt")
    test_pos_path = os.path.join(data_dir, "test2id_pos.txt")
    test_neg_path = os.path.join(data_dir, "test2id_neg.txt")

    def _maybe_fingerprint(path: str) -> Optional[Dict[str, object]]:
        return _file_fingerprint(path) if os.path.exists(path) else None

    return {
        "version": 6,  # bumped: preserve original split files for classification
        "task": str(task),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "subclass_rel": str(subclass_rel),
        "instance_rel": str(instance_rel),
        "min_pos_per_rel": int(min_pos_per_rel),
        "min_neg_per_rel": int(min_neg_per_rel),
        "cls_keep_frac": float(cls_keep_frac),
        "files": {
            "train2id_pos": _maybe_fingerprint(train_pos_path),
            "train2id_neg": _maybe_fingerprint(train_neg_path),
            "test2id_pos": _maybe_fingerprint(test_pos_path),
            "test2id_neg": _maybe_fingerprint(test_neg_path),
        },
    }

def _meta_matches(cached: Dict[str, object], current: Dict[str, object]) -> bool:
    keys = ["version", "task", "test_ratio", "seed", "subclass_rel", "instance_rel", "min_pos_per_rel", "min_neg_per_rel", "cls_keep_frac"]
    for k in keys:
        if cached.get(k) != current.get(k):
            return False
    cfiles = cached.get("files", {})
    nfiles = current.get("files", {})
    for name in ["train2id_pos", "train2id_neg", "test2id_pos", "test2id_neg"]:
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


def _drop_relation_edges(g: HeteroData, rel: str) -> int:
    """Remove ('node', rel, 'node') from graph, returning removed edge count."""
    key = ("node", str(rel), "node")
    if key not in g.edge_types:
        return 0
    ei = g[key].edge_index
    removed = int(ei.size(1)) if (ei is not None and ei.numel() > 0) else 0
    del g[key]
    return removed

def _extract_qid(val: str) -> Optional[str]:
    m = re.search(r"(Q[1-9][0-9]*)", str(val))
    return m.group(1) if m else None

def _load_qid_maps(data_dir: str, entity_file: str, class_file: str) -> Tuple[Dict[str, int], Set[str]]:
    ent_path = os.path.join(data_dir, entity_file)
    cls_path = os.path.join(data_dir, class_file)

    ent_df = pd.read_csv(ent_path)
    cls_df = pd.read_csv(cls_path)

    ent_cols = list(ent_df.columns)
    cls_cols = list(cls_df.columns)
    ent_id_col = "entity_id" if "entity_id" in ent_cols else ("protein_id" if "protein_id" in ent_cols else ent_cols[0])
    ent_idx_col = "node_idx" if "node_idx" in ent_cols else ent_cols[1]
    cls_id_col = "entity_id" if "entity_id" in cls_cols else cls_cols[0]
    cls_idx_col = "node_idx" if "node_idx" in cls_cols else cls_cols[1]

    qid_to_idx: Dict[str, int] = {}
    class_qids: Set[str] = set()

    for _, row in ent_df.iterrows():
        qid = _extract_qid(row.get(ent_id_col, ""))
        if qid is None:
            continue
        try:
            idx = int(row.get(ent_idx_col, -1))
        except Exception:
            continue
        if qid not in qid_to_idx:
            qid_to_idx[qid] = idx

    for _, row in cls_df.iterrows():
        qid = _extract_qid(row.get(cls_id_col, ""))
        if qid is None:
            continue
        class_qids.add(qid)
        try:
            idx = int(row.get(cls_idx_col, -1))
        except Exception:
            continue
        if qid not in qid_to_idx:
            qid_to_idx[qid] = idx

    return qid_to_idx, class_qids

def _load_rel_map(data_dir: str, rel_file: str = "relation2id.txt") -> Dict[str, str]:
    rel_path = os.path.join(data_dir, rel_file)
    df = pd.read_csv(rel_path)
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        rel_id = str(row.get("relation_id", ""))
        edge_idx = str(row.get("edge_idx", "")).strip()
        prop = rel_id.rsplit("/", 1)[-1]
        if prop.startswith("P") and edge_idx != "":
            out[prop] = edge_idx
    return out

def _edge_set_from_graph(g: HeteroData, rel: str) -> Set[Tuple[int, int]]:
    key = ("node", str(rel), "node")
    if key not in g.edge_types:
        return set()
    ei = g[key].edge_index
    if ei is None or ei.numel() == 0:
        return set()
    src = ei[0].detach().cpu().tolist()
    tgt = ei[1].detach().cpu().tolist()
    return set(zip(src, tgt))

def _graph_stats(g: HeteroData, *, subclass_rel: str, instance_rel: str, neg_prefix: str) -> Dict[str, int]:
    num_nodes = int(g["node"].num_nodes) if "node" in g.node_types else 0
    total_edges = 0
    inst_edges = 0
    sub_edges = 0
    neg_edges = 0

    for (s, rel, d) in g.edge_types:
        if s != "node" or d != "node":
            continue
        ei = g[(s, rel, d)].edge_index
        n = int(ei.size(1)) if (ei is not None and ei.numel() > 0) else 0
        total_edges += n
        if str(rel) == str(instance_rel):
            inst_edges += n
        if str(rel) == str(subclass_rel):
            sub_edges += n
        if str(rel).startswith(str(neg_prefix)):
            neg_edges += n

    return {
        "num_nodes": num_nodes,
        "total_edges": total_edges,
        "instance_edges": inst_edges,
        "subclass_edges": sub_edges,
        "neg_edges": neg_edges,
    }

def _augment_with_retrieved(
    *,
    struct_graph: HeteroData,
    base_x: torch.Tensor,
    num_nodes: int,
    data_dir: str,
    retrieved_dir: str,
    subclass_rel: str,
    instance_rel: str,
    neg_prefix: str,
    entity_file: str = "entity2id.txt",
    class_file: str = "class2id.txt",
    map_out_name: str = "qid2idx_retrieved.tsv",
) -> Tuple[HeteroData, torch.Tensor, int, Dict[str, int]]:
    rdir = Path(retrieved_dir)
    if not rdir.exists():
        print(f"[WARN] Retrieved dir not found: {retrieved_dir} (skipping)")
        return struct_graph, base_x, num_nodes, {}

    qid_to_idx, class_qids = _load_qid_maps(data_dir, entity_file, class_file)
    rel_map = _load_rel_map(data_dir)
    p31_edge = rel_map.get("P31", str(instance_rel))

    inst_path = rdir / "instance_of_new.tsv"
    neg_path = rdir / "negative_instance_of.tsv"
    sub_path = rdir / "negative_class_subclasses.tsv"
    label_path = rdir / "unique_names.tsv"

    label_map: Dict[str, str] = {}
    if label_path.exists():
        df_lab = pd.read_csv(label_path, sep="\t", usecols=["qid", "label"], dtype=str)
        for _, row in df_lab.iterrows():
            qid = _extract_qid(row.get("qid", ""))
            if qid and qid not in label_map:
                label = str(row.get("label", "") or "").strip()
                if label:
                    label_map[qid] = label

    inst_edges_qid: List[Tuple[str, str]] = []
    if inst_path.exists():
        df_inst = pd.read_csv(inst_path, sep="\t", usecols=["item_qid", "item_label", "class_qid", "class_label"], dtype=str)
        for _, row in df_inst.iterrows():
            s = _extract_qid(row.get("item_qid", ""))
            t = _extract_qid(row.get("class_qid", ""))
            if s and t:
                inst_edges_qid.append((s, t))
            if s and s not in label_map:
                lab = str(row.get("item_label", "") or "").strip()
                if lab:
                    label_map[s] = lab
            if t and t not in label_map:
                lab = str(row.get("class_label", "") or "").strip()
                if lab:
                    label_map[t] = lab
    else:
        print(f"[WARN] Missing {inst_path} (skipping instance_of_new)")

    neg_edges_qid: List[Tuple[str, str, str]] = []
    neg_pred_counts: Dict[str, int] = defaultdict(int)
    neg_pred_skipped_non_p31 = 0
    if neg_path.exists():
        df_neg = pd.read_csv(
            neg_path,
            sep="\t",
            usecols=[
                "subject_qid",
                "subject_label",
                "predicate_id",
                "predicate_label",
                "object_qid",
                "object_label",
            ],
            dtype=str,
        )
        for _, row in df_neg.iterrows():
            s = _extract_qid(row.get("subject_qid", ""))
            t = _extract_qid(row.get("object_qid", ""))
            pred = str(row.get("predicate_id", "") or "").strip()
            if not s or not t or not pred:
                continue
            # Retrieved negatives are restricted to NOT_P31 / NOT_<instance_rel> only.
            base_rel = rel_map.get(pred, pred)
            if not (str(pred) == "P31" or str(base_rel) == str(instance_rel)):
                neg_pred_skipped_non_p31 += 1
                continue
            neg_edges_qid.append((s, pred, t))
            neg_pred_counts[pred] += 1

            if s and s not in label_map:
                lab = str(row.get("subject_label", "") or "").strip()
                if lab:
                    label_map[s] = lab
            if t and t not in label_map:
                lab = str(row.get("object_label", "") or "").strip()
                if lab:
                    label_map[t] = lab
    else:
        print(f"[WARN] Missing {neg_path} (skipping negative statements)")

    sub_edges_qid: List[Tuple[str, str]] = []
    if sub_path.exists():
        df_sub = pd.read_csv(
            sub_path,
            sep="\t",
            usecols=["child_qid", "child_label", "parent_qid", "parent_label"],
            dtype=str,
        )
        for _, row in df_sub.iterrows():
            c = _extract_qid(row.get("child_qid", ""))
            p = _extract_qid(row.get("parent_qid", ""))
            if c and p:
                sub_edges_qid.append((c, p))
            if c and c not in label_map:
                lab = str(row.get("child_label", "") or "").strip()
                if lab:
                    label_map[c] = lab
            if p and p not in label_map:
                lab = str(row.get("parent_label", "") or "").strip()
                if lab:
                    label_map[p] = lab
    else:
        print(f"[WARN] Missing {sub_path} (skipping subclass edges)")

    all_qids: Set[str] = set()
    for s, t in inst_edges_qid:
        all_qids.add(s)
        all_qids.add(t)
    for s, pred, t in neg_edges_qid:
        all_qids.add(s)
        all_qids.add(t)
    for c, p in sub_edges_qid:
        all_qids.add(c)
        all_qids.add(p)

    if not all_qids:
        print("[INFO] No retrieved QIDs found to add.")
        return struct_graph, base_x, num_nodes, {}

    max_idx = max(qid_to_idx.values()) if qid_to_idx else -1
    new_qids = sorted([q for q in all_qids if q not in qid_to_idx])
    new_qid_to_idx: Dict[str, int] = {}
    next_idx = max_idx + 1
    for qid in new_qids:
        qid_to_idx[qid] = next_idx
        new_qid_to_idx[qid] = next_idx
        next_idx += 1

    new_num_nodes = max(num_nodes, next_idx)
    if new_num_nodes > num_nodes:
        feat_dim = int(base_x.size(1)) if base_x is not None and base_x.numel() > 0 else 128
        extra = torch.randn((new_num_nodes - num_nodes, feat_dim), dtype=base_x.dtype if base_x is not None else torch.float)
        base_x = torch.cat([base_x, extra], dim=0) if base_x is not None and base_x.numel() > 0 else extra
        struct_graph["node"].num_nodes = int(new_num_nodes)
        struct_graph["node"].x = base_x.clone()
        num_nodes = int(new_num_nodes)

    new_class_qids: Set[str] = set()
    for _, t in inst_edges_qid:
        new_class_qids.add(t)
    for c, p in sub_edges_qid:
        new_class_qids.add(c)
        new_class_qids.add(p)
    for s, pred, t in neg_edges_qid:
        if pred == "P31":
            new_class_qids.add(t)

    map_rows = []
    for qid in sorted(all_qids):
        idx = qid_to_idx.get(qid)
        if idx is None:
            continue
        is_new = 1 if qid in new_qid_to_idx else 0
        is_class = 1 if (qid in class_qids or qid in new_class_qids) else 0
        map_rows.append(
            {
                "qid": qid,
                "node_idx": int(idx),
                "is_new": is_new,
                "is_class": is_class,
                "label": label_map.get(qid, ""),
            }
        )
    map_out = rdir / map_out_name
    pd.DataFrame(map_rows).to_csv(map_out, sep="\t", index=False)
    print(f"[INFO] Wrote retrieved QID mapping: {map_out}")

    existing_by_rel: Dict[str, Set[Tuple[int, int]]] = {}
    def _filter_new_edges(rel: str, pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not pairs:
            return []
        if rel not in existing_by_rel:
            existing_by_rel[rel] = _edge_set_from_graph(struct_graph, rel)
        existing = existing_by_rel[rel]
        seen: Set[Tuple[int, int]] = set()
        out: List[Tuple[int, int]] = []
        for u, v in pairs:
            key = (int(u), int(v))
            if key in existing or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    inst_pairs = [(qid_to_idx[s], qid_to_idx[t]) for s, t in inst_edges_qid if s in qid_to_idx and t in qid_to_idx]
    inst_pairs = _filter_new_edges(p31_edge, inst_pairs)

    sub_pairs = [(qid_to_idx[c], qid_to_idx[p]) for c, p in sub_edges_qid if c in qid_to_idx and p in qid_to_idx]
    sub_pairs = _filter_new_edges(subclass_rel, sub_pairs)

    neg_pairs_by_rel: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    missing_pred = 0
    for s, pred, t in neg_edges_qid:
        if s not in qid_to_idx or t not in qid_to_idx:
            continue
        base_rel = rel_map.get(pred)
        if base_rel is None:
            base_rel = pred
            missing_pred += 1
        rel = f"{neg_prefix}{base_rel}"
        neg_pairs_by_rel[rel].append((qid_to_idx[s], qid_to_idx[t]))

    neg_pairs_added: Dict[str, int] = {}
    for rel, pairs in neg_pairs_by_rel.items():
        filtered = _filter_new_edges(rel, pairs)
        neg_pairs_by_rel[rel] = filtered
        neg_pairs_added[rel] = len(filtered)

    if inst_pairs:
        src = torch.tensor([u for u, _ in inst_pairs], dtype=torch.long)
        tgt = torch.tensor([v for _, v in inst_pairs], dtype=torch.long)
        _append_edges(struct_graph, str(p31_edge), src, tgt)

    if sub_pairs:
        src = torch.tensor([u for u, _ in sub_pairs], dtype=torch.long)
        tgt = torch.tensor([v for _, v in sub_pairs], dtype=torch.long)
        _append_edges(struct_graph, str(subclass_rel), src, tgt)

    for rel, pairs in neg_pairs_by_rel.items():
        if not pairs:
            continue
        src = torch.tensor([u for u, _ in pairs], dtype=torch.long)
        tgt = torch.tensor([v for _, v in pairs], dtype=torch.long)
        _append_edges(struct_graph, str(rel), src, tgt)

    stats = {
        "new_nodes": len(new_qid_to_idx),
        "new_class_nodes": sum(1 for q in new_qid_to_idx if q in class_qids or q in new_class_qids),
        "inst_edges_added": len(inst_pairs),
        "sub_edges_added": len(sub_pairs),
        "neg_edges_added": sum(neg_pairs_added.values()),
        "neg_pred_missing": missing_pred,
        "neg_skipped_non_p31": int(neg_pred_skipped_non_p31),
    }

    top_preds = sorted(neg_pred_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_preds:
        print("[INFO] Retrieved negative predicates (top 10):")
        for p, c in top_preds:
            print(f"  {p}: {c}")
    if neg_pred_skipped_non_p31 > 0:
        print(f"[INFO] Retrieved negatives skipped (not P31/instance_rel): {neg_pred_skipped_non_p31}")

    return struct_graph, base_x, num_nodes, stats

def _save_embeddings_and_neighbors(
    *,
    model: torch.nn.Module,
    encoder_graph: HeteroData,
    out_dir: str,
    sampler: Optional[object],
    subclass_rel: Optional[str] = None,
    cls_edge_suffix: str = CLS_EDGE_SUFFIX,
) -> None:
    ensure_dir(out_dir)
    emb_path = os.path.join(out_dir, "embeddings.pt")
    nb_path = os.path.join(out_dir, "neighbors.pt")

    try:
        orig_device = next(model.parameters()).device
    except StopIteration:
        orig_device = torch.device("cpu")
    model_cpu = model.to("cpu")
    graph_cpu = encoder_graph.cpu()
    # Remove __cls edges so embeddings match message-passing edges used at test time
    for et in list(graph_cpu.edge_types):
        rel = str(et[1])
        if rel.endswith(str(cls_edge_suffix)):
            graph_cpu[et].edge_index = torch.empty((2, 0), dtype=torch.long)
    model_cpu.eval()
    with torch.no_grad():
        h_dict = model_cpu.encode(graph_cpu)
    node_emb = h_dict.get("node")
    if node_emb is None:
        raise RuntimeError("Model encode did not return embeddings for node type 'node'.")
    torch.save({"embeddings": node_emb.cpu()}, emb_path)
    if orig_device.type != "cpu":
        model.to(orig_device)
    print(f"[INFO] Saved node embeddings: {emb_path}")

    if sampler is None:
        print("[WARN] No contrastive sampler available; saving empty neighbors.pt")
        anchors = []
    else:
        anchors = getattr(sampler, "anchors", None)
        if anchors is None:
            print("[WARN] Sampler has no anchors; saving empty neighbors.pt")
            anchors = []

    empty = torch.empty(0, dtype=torch.long)

    def _pool_from_by_pred(by_pred_list, u: int) -> Optional[torch.Tensor]:
        if by_pred_list is None or u >= len(by_pred_list):
            return None
        dd = by_pred_list[u]
        parts = [t for t in dd.values() if t is not None and t.numel() > 0]
        if not parts:
            return empty
        return torch.unique(torch.cat(parts, dim=0))

    def _pool_from_list(pool_list, u: int) -> Optional[torch.Tensor]:
        if pool_list is None or u >= len(pool_list):
            return None
        t = pool_list[u]
        if hasattr(t, "detach"):
            return t.detach().cpu()
        return torch.tensor(t)

    def _get_pool(name: str, by_pred_name: Optional[str] = None):
        out = []
        by_pred = getattr(sampler, by_pred_name, None) if by_pred_name else None
        pool_list = getattr(sampler, name, None)
        for u in anchors:
            t = None
            if by_pred is not None:
                t = _pool_from_by_pred(by_pred, u)
            if t is None:
                t = _pool_from_list(pool_list, u)
            if t is None:
                t = empty
            out.append(t)
        return out

    payload = {
        "anchors": torch.tensor(anchors, dtype=torch.long),
        "pool_shared_pos": _get_pool("pool_shared_pos", "pool_shared_pos_by_pred"),
        "pool_shared_neg": _get_pool("pool_shared_neg", "pool_shared_neg_by_pred"),
        "pool_pos_to_u_neg": _get_pool("pool_pos_to_u_neg", "pool_pos_to_u_neg_by_pred"),
        "pool_neg_to_u_pos": _get_pool("pool_neg_to_u_pos", "pool_neg_to_u_pos_by_pred"),
    }
    torch.save(payload, nb_path)
    print(f"[INFO] Saved neighbor pools: {nb_path}")

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
    min_neg_per_rel: int = 0,
    cls_keep_frac: float = 1.0,
    prefer_col: str = "source_node",
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
    reduced_rels: List[str] = []
    reduced_keep: List[str] = []

    for r in rels:
        pos_rows = df_pos[df_pos["rel_base"] == r]
        neg_rows = df_neg[df_neg["rel_base"] == r]
        npos = int(len(pos_rows))
        nneg = int(len(neg_rows))

        if min_pos_per_rel and npos < int(min_pos_per_rel):
            dropped_minpos.append(r)
            continue
        if min_neg_per_rel and nneg < int(min_neg_per_rel):
            dropped_minpos.append(r)
            continue

        m = min(npos, nneg)
        if m <= 0:
            dropped_empty.append(r)
            continue

        rng = np.random.RandomState(int(seed) + _stable_hash_int(r, 1_000_000))
        pos_idx = pos_rows.index.to_numpy(dtype=np.int64).copy()
        neg_idx = neg_rows.index.to_numpy(dtype=np.int64).copy()

        # Prefer examples whose anchor node appears in BOTH pos and neg for this relation
        shared_heads = set(pos_rows[prefer_col]) & set(neg_rows[prefer_col])
        prefer_pos = pos_rows[prefer_col].isin(shared_heads).to_numpy()
        prefer_neg = neg_rows[prefer_col].isin(shared_heads).to_numpy()

        pos_pref = pos_idx[prefer_pos]
        pos_rest = pos_idx[~prefer_pos]
        neg_pref = neg_idx[prefer_neg]
        neg_rest = neg_idx[~prefer_neg]

        # If we can fill m *entirely* from shared heads while respecting min counts,
        # reduce m to maximize overlap (more nodes with both pos/neg statements).
        m_pref = min(len(pos_pref), len(neg_pref))
        m_min = max(int(min_pos_per_rel), int(min_neg_per_rel))
        if m_pref >= max(1, m_min) and m_pref < m:
            reduced_rels.append(r)
            m = m_pref

        # Optional global shrink while preserving minimum per-relation counts
        if cls_keep_frac is not None and float(cls_keep_frac) < 1.0:
            keep_frac = max(0.0, float(cls_keep_frac))
            m_keep = int(round(float(m) * keep_frac))
            m_keep = max(m_keep, m_min)
            if m_keep < m:
                reduced_keep.append(r)
                m = m_keep

        rng.shuffle(pos_pref)
        rng.shuffle(pos_rest)
        rng.shuffle(neg_pref)
        rng.shuffle(neg_rest)

        if len(pos_pref) >= m:
            pos_sel = pos_pref[:m]
        else:
            pos_sel = np.concatenate([pos_pref, pos_rest[: max(0, m - len(pos_pref))]])

        if len(neg_pref) >= m:
            neg_sel = neg_pref[:m]
        else:
            neg_sel = np.concatenate([neg_pref, neg_rest[: max(0, m - len(neg_pref))]])

        pos_chunk = df_pos.loc[pos_sel].copy()
        neg_chunk = df_neg.loc[neg_sel].copy()

        chunks.append(pos_chunk)
        chunks.append(neg_chunk)
        stats[r] = (npos, nneg, int(m))

    if dropped_minpos:
        print(
            f"\n[WARN] Dropped {len(dropped_minpos)} relations due to "
            f"min_pos_per_rel={min_pos_per_rel} or min_neg_per_rel={min_neg_per_rel}. (first 50)"
        )
        print(dropped_minpos[:50])
    if reduced_rels:
        print(f"\n[INFO] Reduced classification size to maximize shared heads: {len(reduced_rels)} relations (first 50)")
        print(reduced_rels[:50])
    if reduced_keep:
        print(f"\n[INFO] Reduced classification size by cls_keep_frac={cls_keep_frac:.3f}: {len(reduced_keep)} relations (first 50)")
        print(reduced_keep[:50])
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
    min_neg_per_rel: int = 0,
    cls_keep_frac: float = 1.0,
) -> Dict[str, object]:
    """
    Build classification tensors preserving the ORIGINAL file split:
      - TRAIN from train2id_pos + train2id_neg
      - TEST  from test2id_pos (+ optional test2id_neg if present)
    Ontology relations are excluded from classification:
      - subclass_of
      - instance_of
      - config instance_rel

    NOTE:
      min_pos_per_rel / min_neg_per_rel / cls_keep_frac are ignored here
      because preserving original split means no balancing/downsampling.
    """
    train_pos_path = os.path.join(data_dir, "train2id_pos.txt")
    train_neg_path = os.path.join(data_dir, "train2id_neg.txt")
    test_pos_path = os.path.join(data_dir, "test2id_pos.txt")
    test_neg_path = os.path.join(data_dir, "test2id_neg.txt")

    df_train_pos = _read_edge_file(train_pos_path)
    df_train_neg = _read_edge_file(train_neg_path)
    df_test_pos = _read_edge_file(test_pos_path)
    df_test_neg = (
        _read_edge_file(test_neg_path)
        if os.path.exists(test_neg_path)
        else pd.DataFrame(columns=["source_node", "target_node", "edge_type"])
    )

    df_train_pos["rel_base"] = df_train_pos["edge_type"].astype(str)
    df_train_pos["label"] = 1.0

    df_train_neg["rel_base"] = df_train_neg["edge_type"].astype(str).map(_strip_not)
    df_train_neg["label"] = 0.0

    df_test_pos["rel_base"] = df_test_pos["edge_type"].astype(str)
    df_test_pos["label"] = 1.0

    df_test_neg["rel_base"] = df_test_neg["edge_type"].astype(str).map(_strip_not)
    df_test_neg["label"] = 0.0

    excluded = {str(subclass_rel), str(instance_rel), "subclass_of", "instance_of"}

    for frame in [df_train_pos, df_train_neg, df_test_pos, df_test_neg]:
        if len(frame) > 0:
            frame.drop(frame[frame["rel_base"].isin(excluded)].index, inplace=True)

    if int(min_pos_per_rel) > 0 or int(min_neg_per_rel) > 0 or float(cls_keep_frac) < 1.0:
        print("[INFO] Preserving original split: min_pos_per_rel/min_neg_per_rel/cls_keep_frac are ignored.")

    df_train = pd.concat([df_train_pos, df_train_neg], ignore_index=True)
    df_test = pd.concat([df_test_pos, df_test_neg], ignore_index=True)
    if len(df_train) == 0:
        raise RuntimeError("Classification TRAIN set is empty after ontology filtering.")

    rels = sorted(set(df_train["rel_base"].unique().tolist()) | set(df_test["rel_base"].unique().tolist()))
    if not rels:
        raise RuntimeError("No classification relations available after ontology filtering.")

    df_train = df_train[df_train["rel_base"].isin(rels)].copy().reset_index(drop=True)
    df_test = df_test[df_test["rel_base"].isin(rels)].copy().reset_index(drop=True)
    rel2id = {r: i for i, r in enumerate(rels)}
    id2rel = {i: r for r, i in rel2id.items()}

    tr_h, tr_r, tr_t, tr_y = _df_to_tensors(df_train, rel2id)
    te_h, te_r, te_t, te_y = _df_to_tensors(df_test, rel2id)

    print("\n=== Classification dataset (preserved original file split) ===")
    print(f"Excluded from classification (encoder-only): {sorted(list(excluded))}")
    print(f"Relations in classification: {len(rel2id)}")
    print("Train source files: train2id_pos.txt + train2id_neg.txt")
    print(f"Test source files:  test2id_pos.txt{' + test2id_neg.txt' if os.path.exists(test_neg_path) else ''}")
    print(f"Train examples: {len(df_train)} | Test examples: {len(df_test)}")

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
        "balance_stats": {},
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
    min_pos_per_rel: int = 0,
    keep_all_used_relations: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

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

    used_rels = set(rel_list) if keep_all_used_relations else set(balance_stats.keys())
    all_rels = sorted(set(df_pos_cls["rel_base"].unique().tolist()) |
        set(df_neg_cls["rel_base"].unique().tolist()))

    leftover_pos_chunks: List[pd.DataFrame] = []
    leftover_neg_chunks: List[pd.DataFrame] = []
    dropped_rels: List[str] = []

    for r in all_rels:
        pos_rows = df_pos_cls[df_pos_cls["rel_base"] == r]
        neg_rows = df_neg_cls[df_neg_cls["rel_base"] == r]
        if r in used_rels:
            if keep_all_used_relations:
                continue
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
        choices=["hgcn", "ra_hgcn", "ra_rgcn", "ra_hgat", "sra_hgcn", "gcn", "gae", "gat", "sgnn", "sgat", "nbfnet"],
        default="hgcn",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=532)
    parser.add_argument("--path", type=str, default="wikidata_data")
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--num_neg_test", type=int, default=1)
    parser.add_argument("--use_nstatementsampler", action="store_true")
    parser.add_argument("--use_pstatementsampler", action="store_true")
    parser.add_argument("--use_rstatement_sampler", action="store_true")
    parser.add_argument("--use_tstatement_sampler", action="store_true",
                        help="Use type-constrained random instance sampler for corruption.")
    parser.add_argument("--alt_contrastive_sampler", action="store_true",
                        help="Use NegativeInstanceSampler_NEWER instead of NEW (default).")
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--finaltrain_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--finaltest_only", action="store_true")
    parser.add_argument("--final_lr", type=float, default=None)
    parser.add_argument("--final_epochs", type=int, default=None)
    parser.add_argument("--use_retrieved", action="store_true",
                        help="Include retrieved edges from output/retrieve_wiki into encoder graph.")
    parser.add_argument("--retrieved_dir", type=str, default="output/retrieve_wiki")
    parser.add_argument("--retrieved_map_out", type=str, default="qid2idx_retrieved.tsv")
    parser.add_argument("--print_sampler_stats", action="store_true",
                        help="Print pool-size stats from the negative sampler.")

    parser.add_argument("--balanced_test_ratio", type=float, default=0.20)
    parser.add_argument("--balanced_seed", type=int, default=42)
    parser.add_argument("--min_pos_per_rel", type=int, default=20)
    parser.add_argument("--min_neg_per_rel", type=int, default=20)
    parser.add_argument("--cls_keep_frac", type=float, default=1.0,
                        help="Keep fraction of balanced classification triples per relation (<=1.0).")

    parser.add_argument("--cv_val_ratio", type=float, default=0.0005)
    parser.add_argument("--val_lp_eval_every", type=int, default=1,
                        help="Compute expensive validation LP metrics every N epochs (>=1).")
    parser.add_argument(
        "--max_contrastive_anchors",
        type=int,
        default=0,
        help="Optional cap (>0) on number of contrastive anchors sampled per batch (0 disables cap).",
    )
    parser.add_argument(
        "--cv_contrastive_weight_grid",
        type=str,
        default="",
        help="Optional comma-separated contrastive weights to sweep in CV (only when contrastive is enabled).",
    )
    parser.add_argument(
        "--cv_contrastive_temp_grid",
        type=str,
        default="",
        help="Optional comma-separated contrastive temperatures to sweep in CV (only when contrastive is enabled).",
    )
    parser.add_argument("--split_cache_path", type=str, default=None)
    parser.add_argument("--force_resplit", action="store_true")

    args = parser.parse_args()
    if args.finaltest_only:
        args.test_only = True
    if args.finaltrain_only and args.test_only:
        raise ValueError("Cannot use --finaltrain_only together with --test_only/--finaltest_only.")
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
    contrastive_temperature = float(cfg.get("contrastive_temperature", 0.5))
    subclass_rel = str(cfg.get("subclass_rel", "subclass_of"))
    instance_rel = str(cfg.get("instance_rel", "instance_of"))
    contrastive_k = int(cfg.get("contrastive_k", 1))

    def _parse_float_grid_arg(spec: str, default_value: float, name: str) -> List[float]:
        txt = str(spec or "").strip()
        if not txt:
            return [float(default_value)]
        vals: List[float] = []
        for piece in txt.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                vals.append(float(piece))
            except Exception:
                raise ValueError(f"Invalid float '{piece}' in --{name}")
        if not vals:
            return [float(default_value)]
        # Stable unique, preserve input order.
        seen = set()
        out: List[float] = []
        for v in vals:
            key = float(v)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    if args.no_contrastive:
        cv_contrastive_weights = [float(contrastive_weight)]
        cv_contrastive_temps = [float(contrastive_temperature)]
    else:
        cv_contrastive_weights = _parse_float_grid_arg(
            args.cv_contrastive_weight_grid, contrastive_weight, "cv_contrastive_weight_grid"
        )
        cv_contrastive_temps = _parse_float_grid_arg(
            args.cv_contrastive_temp_grid, contrastive_temperature, "cv_contrastive_temp_grid"
        )
    cv_contrastive_combos: List[Tuple[float, float]] = [
        (float(w), float(t)) for w in cv_contrastive_weights for t in cv_contrastive_temps
    ]
    selected_contrastive_weight = float(contrastive_weight)
    selected_contrastive_temperature = float(contrastive_temperature)
    if not args.no_contrastive:
        print(f"Contrastive weight grid (CV): {cv_contrastive_weights}")
        print(f"Contrastive temperature grid (CV): {cv_contrastive_temps}")
        if int(args.max_contrastive_anchors) > 0:
            print(f"Contrastive anchors cap per batch: {int(args.max_contrastive_anchors)}")

    dl = DataLoader(
        args.path + "/",
        use_pstatement_sampler=args.use_pstatementsampler,
        use_nstatement_sampler=args.use_nstatementsampler,
        use_rstatement_sampler=(args.use_rstatement_sampler or args.use_tstatement_sampler),
    )
    data_dict = dl.get_data()
    full_graph_all = dl.make_data_graph(data_dict, orthogonal=False)

    num_nodes = int(full_graph_all["node"].num_nodes)
    base_x = full_graph_all["node"].x

    nflag, pflag, rflag, tflag = (
        args.use_nstatementsampler,
        args.use_pstatementsampler,
        args.use_rstatement_sampler,
        args.use_tstatement_sampler,
    )
    use_partial_sampler = nflag or pflag
    use_typed_sampler = tflag
    use_random_sampler = rflag and (not tflag)
    if rflag and tflag:
        print("[WARN] Both --use_rstatement_sampler and --use_tstatement_sampler were set; "
              "using TypedInstanceSampler.", flush=True)

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
        min_neg_per_rel=int(args.min_neg_per_rel),
        cls_keep_frac=float(args.cls_keep_frac),
    )
    expected_meta = _build_split_cache_meta(
        data_dir=args.path,
        task=args.task,
        test_ratio=float(args.balanced_test_ratio),
        seed=int(args.balanced_seed),
        subclass_rel=subclass_rel,
        instance_rel=instance_rel,
        min_pos_per_rel=int(args.min_pos_per_rel),
        min_neg_per_rel=int(args.min_neg_per_rel),
        cls_keep_frac=float(args.cls_keep_frac),
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
            min_neg_per_rel=int(args.min_neg_per_rel),
            cls_keep_frac=float(args.cls_keep_frac),
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
    # Global leakage audit: exact overlap between TRAIN and TEST classification examples.
    tr_key = torch.stack(
        [
            cls_heads.detach().cpu().long(),
            cls_rels.detach().cpu().long(),
            cls_tails.detach().cpu().long(),
            (cls_labels.detach().cpu() > 0.5).long(),
        ],
        dim=1,
    )
    te_key = torch.stack(
        [
            test_heads_all.detach().cpu().long(),
            test_rels_all.detach().cpu().long(),
            test_tails_all.detach().cpu().long(),
            (test_labels_all.detach().cpu() > 0.5).long(),
        ],
        dim=1,
    )
    tr_set = set(map(tuple, tr_key.tolist()))
    te_set = set(map(tuple, te_key.tolist()))
    tr_te_overlap = len(tr_set & te_set)
    print(
        f"[LeakAudit] Exact TRAIN/TEST overlap on (h,r,t,label): {tr_te_overlap} "
        f"(train_unique={len(tr_set)}, test_unique={len(te_set)})"
    )

    print("\n=== Final classification dataset sizes ===")
    print(f"#TRAIN classification examples: {cls_heads.numel()}")
    print(f"#TEST  classification examples: {test_heads_all.numel()}")
    print(f"#Classification relations:      {len(rel2id)}")

    run_out_dir = os.path.join("output", args.output_dir)
    os.makedirs(run_out_dir, exist_ok=True)
    split_payload = {
        "train_heads": cls_heads.detach().cpu().long(),
        "train_rels": cls_rels.detach().cpu().long(),
        "train_tails": cls_tails.detach().cpu().long(),
        "train_labels": cls_labels.detach().cpu().float(),
        "test_heads": test_heads_all.detach().cpu().long(),
        "test_rels": test_rels_all.detach().cpu().long(),
        "test_tails": test_tails_all.detach().cpu().long(),
        "test_labels": test_labels_all.detach().cpu().float(),
        "rel2id": rel2id,
        "id2rel": id2rel,
        "rel_list": rel_list,
        "balance_stats": balance_stats,
    }
    torch.save(split_payload, os.path.join(run_out_dir, "lp_labels_and_splits.pt"))
    split_rows = []
    for split_name, h_t, r_t, t_t, y_t in [
        ("train", cls_heads, cls_rels, cls_tails, cls_labels),
        ("test", test_heads_all, test_rels_all, test_tails_all, test_labels_all),
    ]:
        for h_i, r_i, t_i, y_i in zip(h_t.tolist(), r_t.tolist(), t_t.tolist(), y_t.tolist()):
            r_i = int(r_i)
            split_rows.append({
                "split": split_name,
                "head": int(h_i),
                "rel_id": r_i,
                "rel_name": id2rel.get(r_i, str(r_i)),
                "tail": int(t_i),
                "label": float(y_i),
            })
    pd.DataFrame(split_rows).to_csv(
        os.path.join(run_out_dir, "lp_labels_and_splits.tsv"), sep="\t", index=False)

    # -----------------------------------------------------------------
    # Build encoder graph:
    #   - include structural relations from data_dict
    #   - add ONLY:
    #       (a) file negatives for subclass_of / instance_of (NOT_ prefixed)
    #       (b) leftover non-classification relations from train files (if any)
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

    original_subclass_pairs = _load_original_subclass_pairs(
        data_dir=args.path,
        subclass_rel=subclass_rel,
        neg_prefix=NEG_PREFIX,
    )
    print(
        f"[INFO] Original-file subclass keep-list size "
        f"(train2id_pos/train2id_neg/test2id_pos): {len(original_subclass_pairs)}"
    )

    # Compute leftovers + encoder-only subclass/instance negatives from file
    leftover_pos_df, leftover_neg_df, enc_only_neg_df = compute_leftover_examples_for_encoder(
        args.path, rel_list=rel_list, balance_stats=balance_stats, seed=int(args.balanced_seed),
        subclass_rel=subclass_rel, instance_rel=instance_rel, min_pos_per_rel=int(args.min_pos_per_rel),
        keep_all_used_relations=True)

    # (a) Add ONLY subclass/instance negatives from file
    if len(enc_only_neg_df) > 0:
        enc_only_neg_df = enc_only_neg_df.copy()
        enc_only_neg_df["edge_type"] = enc_only_neg_df["rel_base"].astype(str).map(_ensure_not_prefixed)
        for et, g in enc_only_neg_df.groupby("edge_type"):
            src = torch.tensor(g["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            tgt = torch.tensor(g["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
            _append_edges(struct_graph, str(et), src, tgt)
        print(f"\n[INFO] Encoder got ONLY subclass/instance negatives from train2id_neg: {len(enc_only_neg_df)}")

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
    print(f"       leftover negatives added: {n_left_neg}")

    # (c) TRAIN classification triples will be added per-CV-fold (and for final training)
    #     so that each fold only sees its own train triples in the encoder graph.
    #
    # Contrastive graph starts as the FULL structural graph (includes non-original
    # subclass edges). Encoder graph is filtered later to keep only original-file
    # subclass edges for message passing.
    contrastive_struct_graph = struct_graph.clone()
    if args.use_retrieved and not args.no_contrastive:
        contrastive_base_x = contrastive_struct_graph["node"].x.clone()
        contrastive_num_nodes = int(contrastive_struct_graph["node"].num_nodes)
        before_stats = _graph_stats(
            contrastive_struct_graph,
            subclass_rel=subclass_rel,
            instance_rel=instance_rel,
            neg_prefix=NEG_PREFIX,
        )
        contrastive_struct_graph, contrastive_base_x, contrastive_num_nodes, aug_stats = _augment_with_retrieved(
            struct_graph=contrastive_struct_graph,
            base_x=contrastive_base_x,
            num_nodes=contrastive_num_nodes,
            data_dir=args.path,
            retrieved_dir=args.retrieved_dir,
            subclass_rel=subclass_rel,
            instance_rel=instance_rel,
            neg_prefix=NEG_PREFIX,
            map_out_name=args.retrieved_map_out,
        )
        _assert_shared_id_alignment(
            base_graph=struct_graph,
            contrastive_graph=contrastive_struct_graph,
        )
        after_stats = _graph_stats(
            contrastive_struct_graph,
            subclass_rel=subclass_rel,
            instance_rel=instance_rel,
            neg_prefix=NEG_PREFIX,
        )

        print("\n=== Retrieved augmentation summary (contrastive-only graph) ===")
        print(f"New nodes added:             {aug_stats.get('new_nodes', 0)}")
        print(f"New class nodes added:       {aug_stats.get('new_class_nodes', 0)}")
        print(f"Instance_of edges added:     {aug_stats.get('inst_edges_added', 0)}")
        print(f"Subclass_of edges added:     {aug_stats.get('sub_edges_added', 0)}")
        print(f"Negative edges added:        {aug_stats.get('neg_edges_added', 0)}")
        if aug_stats.get("neg_pred_missing", 0) > 0:
            print(f"Negative predicates missing in relation2id: {aug_stats.get('neg_pred_missing', 0)}")
        if aug_stats.get("neg_skipped_non_p31", 0) > 0:
            print(f"Negative predicates skipped (not P31/instance_rel): {aug_stats.get('neg_skipped_non_p31', 0)}")

        print("\n=== Contrastive graph size (before vs after retrieved) ===")
        print(f"Nodes:       {before_stats['num_nodes']} -> {after_stats['num_nodes']}")
        print(f"Edges total: {before_stats['total_edges']} -> {after_stats['total_edges']}")
        print(f"Instance_of: {before_stats['instance_edges']} -> {after_stats['instance_edges']}")
        print(f"Subclass_of: {before_stats['subclass_edges']} -> {after_stats['subclass_edges']}")
        print(f"Neg edges:   {before_stats['neg_edges']} -> {after_stats['neg_edges']}")
        print("[INFO] Retrieved contrastive graph alignment check passed (shared node ids preserved).")
    elif args.use_retrieved and args.no_contrastive:
        print("\n[INFO] --use_retrieved requested but --no_contrastive is active; retrieved data will not be used.")

    if not args.no_contrastive:
        extra_stats = _append_contrastive_extra_negatives_from_file(
            graph=contrastive_struct_graph,
            data_dir=args.path,
            file_name="train2id_neg_extras_for_contrastive.txt",
        )
        print(
            "[INFO] Contrastive-only extra negatives file "
            f"train2id_neg_extras_for_contrastive.txt: rows={extra_stats.get('rows', 0)} "
            f"added_edges={extra_stats.get('added', 0)}"
        )


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
    b, a = _filter_rel_edges_by_allowlist(
        graph=encoder_graph,
        rel=str(subclass_rel),
        allowed_pairs=original_subclass_pairs,
    )
    removed = int(b - a)
    print(
        f"[INFO] Encoder subclass filtering: kept {a}/{b} '{subclass_rel}' edges "
        f"from original files; removed {removed} non-original subclass edges "
        f"(e.g., subclass2id/retrieved provenance)."
    )

    drop_subclass_mp_when_no_contrastive = bool(args.no_contrastive and len(original_subclass_pairs) == 0)
    MP_EDGE_TYPES = _message_passing_edge_types(
        encoder_graph,
        subclass_rel=str(subclass_rel),
        drop_subclass=drop_subclass_mp_when_no_contrastive,
        cls_edge_suffix=CLS_EDGE_SUFFIX,
    )
    if drop_subclass_mp_when_no_contrastive:
        print(f"[INFO] --no_contrastive: removed '{subclass_rel}' from message passing (no original-file subclass edges).")
    encoder_e_etypes = MP_EDGE_TYPES

    # Contrastive-only graph (sampler input). Encoder graph above remains retrieval-free.
    if args.model == "hgcn":
        contrastive_encoder_graph = build_hgcn_encoder_graph(
            contrastive_struct_graph, subclass_rel=subclass_rel, neg_prefix=NEG_PREFIX
        )
    else:
        contrastive_encoder_graph = contrastive_struct_graph
    if CLS_EDGE_TYPE not in contrastive_encoder_graph.edge_types:
        contrastive_encoder_graph[CLS_EDGE_TYPE].edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        cei = contrastive_encoder_graph[CLS_EDGE_TYPE].edge_index
        if cei is None:
            contrastive_encoder_graph[CLS_EDGE_TYPE].edge_index = torch.empty((2, 0), dtype=torch.long)

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

    cls_cov = report_cls_nodes_not_in_encoder(
        train_heads=cls_heads,
        train_tails=cls_tails,
        test_heads=test_heads_all,
        test_tails=test_tails_all,
        encoder_graph=encoder_graph,
        num_nodes=num_nodes,
        topk=25)
    retrieved_nodes_for_report: Set[int] = set()
    if args.use_retrieved:
        retrieved_nodes_for_report = _load_retrieved_node_idx_set(
            retrieved_dir=args.retrieved_dir,
            map_out_name=args.retrieved_map_out,
        )
    write_missing_cls_nodes_with_sources(
        missing_nodes=cls_cov["missing_all"],
        train_nodes=cls_cov["train_nodes"],
        test_nodes=cls_cov["test_nodes"],
        retrieved_node_ids=retrieved_nodes_for_report,
        out_path=os.path.join(run_out_dir, "missing_cls_nodes_with_sources.tsv"),
    )


    neighbor_sizes = [10, 7]
    # best_epochs: List[int] = []
    # best_lrs: List[float] = []
    # cv_metrics: List[torch.Tensor] = []
    sampler_stats_printed = False

    cv_metrics: List[torch.Tensor] = []
    cv_fold_splits = []
    cv_summary_rows = []
    cv_combo_summary_rows = []
    cv_combo_lr_summary_rows = []

    final_model = None
    final_contrastive_sampler = None
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
        test_encoder_graph_for_mp = _with_cls_edges(encoder_graph, cls_heads, cls_tails)
        _add_cls_edges_for_sampler(
            test_encoder_graph_for_mp,
            heads=cls_heads,
            tails=cls_tails,
            rels=cls_rels,
            labels=cls_labels,
            id2rel=id2rel,
        )

        final_base_kwargs = dict(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
            e_etypes=_message_passing_edge_types(
                test_encoder_graph_for_mp,
                subclass_rel=str(subclass_rel),
                drop_subclass=drop_subclass_mp_when_no_contrastive,
                cls_edge_suffix=CLS_EDGE_SUFFIX,
            ),
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        )
        if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat", "sgat", "nbfnet"):
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
        val_ratio = max(0.0005, min(0.49, val_ratio))

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
        split_indices = list(splitter.split(np.zeros(len(main_idx), dtype=np.int64), y[main_idx])) if len(main_idx) else []
        combo_best_key = None
        selected_lr_summary_rows: List[Dict[str, object]] = []

        for combo_idx, (combo_weight, combo_temp) in enumerate(cv_contrastive_combos, start=1):
            combo_weight = float(combo_weight)
            combo_temp = float(combo_temp)
            combo_tag = _sanitize_component(f"cw{combo_weight:.6g}_ct{combo_temp:.6g}")
            print(
                f"\n=== CV Contrastive Combo {combo_idx}/{len(cv_contrastive_combos)} "
                f"(weight={combo_weight:.6g}, temp={combo_temp:.6g}) ==="
            )

            lr_to_fold_losses = defaultdict(list)
            lr_to_fold_epochs = defaultdict(list)
            lr_to_fold_metrics = defaultdict(list)

            for fold, (tr_sub, va_sub) in enumerate(split_indices, start=1):
                train_idx_np = main_idx[tr_sub]
                val_idx_np = main_idx[va_sub]
                if rare_idx.size:
                    train_idx_np = np.concatenate([train_idx_np, rare_idx], axis=0)

                train_idx = torch.tensor(train_idx_np, dtype=torch.long)
                val_idx = torch.tensor(val_idx_np, dtype=torch.long)
                val_idx, val_guard_stats = _drop_val_examples_seen_in_train(
                    heads=cls_heads,
                    rels=cls_rels,
                    tails=cls_tails,
                    labels=cls_labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                )
                if val_idx.numel() == 0:
                    raise RuntimeError(
                        f"Fold {fold}: validation set became empty after removing examples also present in train."
                    )
                paired_train_idx, pair_stats = _build_hr_paired_train_indices(
                    heads=cls_heads,
                    rels=cls_rels,
                    labels=cls_labels,
                    candidate_idx=train_idx,
                    seed=42 + int(combo_idx) * 1000 + int(fold),
                )
                if paired_train_idx.numel() == 0:
                    raise RuntimeError(
                        f"Fold {fold}: no (head,relation)-matched pos/neg pairs could be built "
                        "for training supervision."
                    )
                cv_fold_splits.append({
                    "fold": int(fold),
                    "combo_index": int(combo_idx),
                    "contrastive_weight": float(combo_weight),
                    "contrastive_temperature": float(combo_temp),
                    "train_idx": train_idx.detach().cpu().long(),
                    "train_idx_paired_hr": paired_train_idx.detach().cpu().long(),
                    "val_idx": val_idx.detach().cpu().long(),
                    "train_heads": cls_heads[train_idx].detach().cpu().long(),
                    "train_rels": cls_rels[train_idx].detach().cpu().long(),
                    "train_tails": cls_tails[train_idx].detach().cpu().long(),
                    "train_labels": cls_labels[train_idx].detach().cpu().float(),
                    "val_heads": cls_heads[val_idx].detach().cpu().long(),
                    "val_rels": cls_rels[val_idx].detach().cpu().long(),
                    "val_tails": cls_tails[val_idx].detach().cpu().long(),
                    "val_labels": cls_labels[val_idx].detach().cpu().float(),
                })

                train_edge_label_index = torch.stack([cls_heads[paired_train_idx], cls_tails[paired_train_idx]], dim=0)
                train_edge_label = cls_labels[paired_train_idx].to(torch.float)
                val_edge_label_index = torch.stack([cls_heads[val_idx], cls_tails[val_idx]], dim=0)
                val_edge_label = cls_labels[val_idx].to(torch.float)

                fold_encoder_graph = _with_cls_edges(
                    encoder_graph,
                    cls_heads[train_idx],
                    cls_tails[train_idx],
                )
                n_cls_pos_fold, n_cls_neg_fold = _add_cls_edges_for_sampler(
                    fold_encoder_graph,
                    heads=cls_heads[train_idx],
                    tails=cls_tails[train_idx],
                    rels=cls_rels[train_idx],
                    labels=cls_labels[train_idx],
                    id2rel=id2rel,
                )
                if fold == 1:
                    print(
                        f"\n[INFO] Fold {fold} encoder got TRAIN cls edges: "
                        f"pos={n_cls_pos_fold} neg={n_cls_neg_fold}"
                    )
                print(f"[INFO] Fold {fold} encoder summary after adding TRAIN cls edges:")
                print_final_train_encoder_statement_counts(
                    encoder_graph=fold_encoder_graph,
                    subclass_rel=subclass_rel,
                    neg_prefix=NEG_PREFIX,
                )

                fold_mp_edge_types = _message_passing_edge_types(
                    fold_encoder_graph,
                    subclass_rel=str(subclass_rel),
                    drop_subclass=drop_subclass_mp_when_no_contrastive,
                    cls_edge_suffix=CLS_EDGE_SUFFIX,
                )
                # Include train classification edges in message passing for this fold.
                num_neighbors = {et: [12, 8] for et in fold_encoder_graph.edge_types}
                num_neighbors[CLS_EDGE_TYPE] = [0, 0]
                for et in fold_encoder_graph.edge_types:
                    if str(et[1]).endswith(CLS_EDGE_SUFFIX):
                        num_neighbors[et] = [0, 0]
                if drop_subclass_mp_when_no_contrastive:
                    num_neighbors[("node", str(subclass_rel), "node")] = [0, 0]

                train_loader = LinkNeighborLoader(
                    fold_encoder_graph, num_neighbors=num_neighbors,
                    edge_label_index=(CLS_EDGE_TYPE, train_edge_label_index), edge_label=train_edge_label,
                    batch_size=args.batch_size, shuffle=True, num_workers=2, persistent_workers=True,
                    pin_memory=(device.type == "cuda"), neg_sampling_ratio=0.0
                )
                val_loader = LinkNeighborLoader(
                    fold_encoder_graph, num_neighbors=num_neighbors,
                    edge_label_index=(CLS_EDGE_TYPE, val_edge_label_index), edge_label=val_edge_label,
                    batch_size=args.batch_size, shuffle=False, num_workers=2, persistent_workers=True,
                    pin_memory=(device.type == "cuda"), neg_sampling_ratio=0.0,
                )

                print(f"\n=== CV Split {fold}/{k_folds} ===")
                print(f"  Train cls examples: {train_idx.numel()} | Val cls examples: {val_idx.numel()}")
                if val_guard_stats["val_removed_seen_in_train"] > 0:
                    print(
                        "  [LeakGuard] Removed val examples that exactly matched train examples: "
                        f"{val_guard_stats['val_removed_seen_in_train']} "
                        f"(before={val_guard_stats['val_before']}, after={val_guard_stats['val_after']})"
                    )
                print(
                    "  Paired train supervision: "
                    f"raw_pos={pair_stats['raw_pos']} raw_neg={pair_stats['raw_neg']} "
                    f"-> paired_pos={pair_stats['paired_pos']} paired_neg={pair_stats['paired_neg']} "
                    f"(dropped_pos_no_hr_neg={pair_stats['dropped_pos_no_hr_neg']})"
                )

                base_model_kwargs = dict(
                    in_dim=in_dim,
                    hidden_dim=mcfg["hidden_dim"],
                    out_dim=mcfg.get("out_dim", mcfg["hidden_dim"]),
                    e_etypes=fold_mp_edge_types,
                    n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
                )
                if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gae", "gat", "sgat", "nbfnet"):
                    model_fold = ModelCls(**base_model_kwargs, rel2id=rel2id).to(device)
                else:
                    model_fold = ModelCls(**base_model_kwargs).to(device)

                sampler_graph = fold_encoder_graph
                if not args.no_contrastive:
                    sampler_graph = _with_cls_edges(
                        contrastive_encoder_graph, cls_heads[train_idx], cls_tails[train_idx]
                    )
                    _add_cls_edges_for_sampler(
                        sampler_graph,
                        heads=cls_heads[train_idx],
                        tails=cls_tails[train_idx],
                        rels=cls_rels[train_idx],
                        labels=cls_labels[train_idx],
                        id2rel=id2rel,
                    )
                    if args.use_retrieved:
                        print(f"[INFO] Fold {fold} sampler uses contrastive graph (full subclass + retrieved edges).")
                    else:
                        print(f"[INFO] Fold {fold} sampler uses contrastive graph (full subclass edges).")

                if not args.no_contrastive:
                    if use_typed_sampler:
                        neg_stmt_sampler = TypedInstanceSampler(k=contrastive_k, external_negs=external_edges)
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    elif use_random_sampler:
                        neg_stmt_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    elif use_partial_sampler:
                        edges_are_negative = nflag
                        neg_stmt_sampler = PartialInstanceSampler(
                            k=contrastive_k, neg_edges=external_edges, edges_are_negative=edges_are_negative
                        )
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    else:
                        SamplerCls = NegativeInstanceSampler_NEWER if args.alt_contrastive_sampler else NegativeInstanceSampler_NEW
                        neg_stmt_sampler = SamplerCls(
                            k=contrastive_k,
                            subclass_rel=subclass_rel,
                            neg_prefix=NEG_PREFIX,
                            instance_rel=instance_rel,
                            max_contrastive_anchors=int(args.max_contrastive_anchors),
                            cache_dir="data/cache",
                            cache_key=(
                                f"{args.path}|{combo_tag}|fold{fold}|clsedge=1|"
                                f"retrieved_contrastive={int(args.use_retrieved and not args.no_contrastive)}|"
                                f"sampler={SamplerCls.__name__}"
                            ),
                        )
                        neg_stmt_sampler.prepare_global(sampler_graph)
                    if (args.print_sampler_stats or args.use_retrieved) and not sampler_stats_printed:
                        if hasattr(neg_stmt_sampler, "print_pool_stats"):
                            neg_stmt_sampler.print_pool_stats(prefix="[NegSamplerStats]")
                            sampler_stats_printed = True
                else:
                    neg_stmt_sampler = None

                log_name = f"train_cv_split{fold}" if len(cv_contrastive_combos) == 1 else f"train_cv_split{fold}_{combo_tag}"
                log_fold = Logger(log_name, dir=args.output_dir)
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
                    train_idx=paired_train_idx,
                    val_idx=val_idx,
                    contrastive_sampler=neg_stmt_sampler,
                    contrastive_weight=combo_weight,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    no_contrastive=args.no_contrastive,
                    val_lp_eval_every=int(args.val_lp_eval_every),
                    contrastive_temperature=combo_temp,
                )

                best_val_loss, best_epoch, best_metrics, best_lr, per_lr = trainer_fold.run()
                for lr in lr_candidates:
                    lr = float(lr)
                    rec = per_lr.get(lr, None)
                    if rec is None:
                        continue
                    loss = float(rec["best_val_loss"])
                    ep = int(rec["best_epoch"])
                    row = {
                        "fold": int(fold),
                        "combo_index": int(combo_idx),
                        "contrastive_weight": float(combo_weight),
                        "contrastive_temperature": float(combo_temp),
                        "lr": lr,
                        "best_epoch": ep,
                        "best_val_loss": loss,
                        "is_best_lr_for_fold": int(best_lr is not None and float(best_lr) == lr),
                    }
                    metrics_for_lr = rec.get("best_metrics", None)
                    if metrics_for_lr is not None:
                        for metric_name, metric_value in metrics_for_lr.items():
                            row[f"best_{metric_name.replace('@', 'at')}"] = float(metric_value)
                    cv_summary_rows.append(row)

                    if loss != float("inf"):
                        lr_to_fold_losses[lr].append(loss)
                    if ep > 0:
                        lr_to_fold_epochs[lr].append(ep)
                    if rec.get("best_metrics", None) is not None:
                        lr_to_fold_metrics[lr].append(rec["best_metrics"])
                if best_metrics is not None:
                    cv_metrics.append(best_metrics)

            lr_summary = []
            for lr in lr_candidates:
                lr = float(lr)
                losses = lr_to_fold_losses.get(lr, [])
                if not losses:
                    continue
                mean_loss = statistics.mean(losses)
                std_loss = statistics.pstdev(losses) if len(losses) > 1 else 0.0
                n = len(losses)
                lr_summary.append((mean_loss, std_loss, n, lr))

            if not lr_summary:
                combo_final_lr = float(lr_candidates[0])
                combo_final_epochs = int(args.epochs)
                combo_best_mean = float("inf")
                combo_best_std = float("inf")
            else:
                lr_summary.sort(key=lambda x: (x[0], x[1], x[3]))
                combo_best_mean, combo_best_std, combo_best_n, combo_final_lr = lr_summary[0]
                epochs_for_lr = lr_to_fold_epochs.get(combo_final_lr, [])
                combo_final_epochs = int(statistics.median(epochs_for_lr)) if epochs_for_lr else int(args.epochs)

            print(
                f"\n[CV Combo Summary] weight={combo_weight:.6g}, temp={combo_temp:.6g} | "
                f"chosen_lr={combo_final_lr:.6g}, chosen_epochs={combo_final_epochs}"
            )
            for mean_loss, std_loss, n, lr in sorted(lr_summary, key=lambda x: (x[0], x[1], x[3])):
                print(f"  lr={lr:.3g} | mean={mean_loss:.6f} | std={std_loss:.6f} | folds={n}")

            cv_combo_summary_rows.append(
                {
                    "combo_index": int(combo_idx),
                    "contrastive_weight": float(combo_weight),
                    "contrastive_temperature": float(combo_temp),
                    "chosen_lr": float(combo_final_lr),
                    "chosen_epochs": int(combo_final_epochs),
                    "best_mean_val_loss": float(combo_best_mean),
                    "best_std_val_loss": float(combo_best_std),
                }
            )
            for mean_loss, std_loss, n, lr in sorted(lr_summary, key=lambda x: (x[0], x[1], x[3])):
                cv_combo_lr_summary_rows.append(
                    {
                        "combo_index": int(combo_idx),
                        "contrastive_weight": float(combo_weight),
                        "contrastive_temperature": float(combo_temp),
                        "lr": float(lr),
                        "mean_val_loss": float(mean_loss),
                        "std_val_loss": float(std_loss),
                        "folds": int(n),
                        "chosen_for_combo": int(float(lr) == float(combo_final_lr)),
                    }
                )

            combo_key = (float(combo_best_mean), float(combo_best_std), float(combo_weight), float(combo_temp))
            if combo_best_key is None or combo_key < combo_best_key:
                combo_best_key = combo_key
                final_lr = float(combo_final_lr)
                final_epochs = int(combo_final_epochs)
                selected_contrastive_weight = float(combo_weight)
                selected_contrastive_temperature = float(combo_temp)
                selected_lr_summary_rows = [
                    {
                        "lr": float(lr),
                        "mean_val_loss": float(mean_loss),
                        "std_val_loss": float(std_loss),
                        "folds": int(n),
                        "chosen_final_lr": int(float(lr) == float(combo_final_lr)),
                    }
                    for mean_loss, std_loss, n, lr in sorted(lr_summary, key=lambda x: (x[0], x[1], x[3]))
                ]

        print("\n=== Cross-validation summary (TRAIN split only) ===")
        if not args.no_contrastive:
            print(
                f"Selected contrastive combo: weight={selected_contrastive_weight:.6g}, "
                f"temperature={selected_contrastive_temperature:.6g}"
            )
        print(f"Chosen final_lr: {final_lr:.6g}")
        print(f"Chosen final_epochs: {final_epochs}")

        torch.save(
            {
                "folds": cv_fold_splits,
                "rel2id": rel2id,
                "id2rel": id2rel,
                "cv_val_ratio": val_ratio,
                "chosen_final_lr": float(final_lr),
                "chosen_final_epochs": int(final_epochs),
                "chosen_contrastive_weight": float(selected_contrastive_weight),
                "chosen_contrastive_temperature": float(selected_contrastive_temperature),
            },
            os.path.join(run_out_dir, "cv_fold_splits.pt"),
        )
        if cv_summary_rows:
            pd.DataFrame(cv_summary_rows).to_csv(
                os.path.join(run_out_dir, "cv_fold_summary.tsv"), sep="\t", index=False
            )
        if selected_lr_summary_rows:
            pd.DataFrame(selected_lr_summary_rows).to_csv(
                os.path.join(run_out_dir, "cv_lr_summary.tsv"), sep="\t", index=False
            )
        if cv_combo_summary_rows:
            pd.DataFrame(cv_combo_summary_rows).to_csv(
                os.path.join(run_out_dir, "cv_contrastive_combo_summary.tsv"), sep="\t", index=False
            )
        if cv_combo_lr_summary_rows:
            pd.DataFrame(cv_combo_lr_summary_rows).to_csv(
                os.path.join(run_out_dir, "cv_contrastive_lr_summary.tsv"), sep="\t", index=False
            )


    # -----------------------------------------------------------------
    # Final training on TRAIN split
    # -----------------------------------------------------------------
    if not args.test_only:
        train_encoder_graph = _with_cls_edges(encoder_graph, cls_heads, cls_tails)
        n_cls_pos_all, n_cls_neg_all = _add_cls_edges_for_sampler(
            train_encoder_graph,
            heads=cls_heads,
            tails=cls_tails,
            rels=cls_rels,
            labels=cls_labels,
            id2rel=id2rel,
        )
        print(f"\n[INFO] Final encoder got ALL TRAIN cls edges: "
              f"pos={n_cls_pos_all} neg={n_cls_neg_all}")
        print_final_train_encoder_statement_counts(
            encoder_graph=train_encoder_graph,
            subclass_rel=subclass_rel,
            neg_prefix=NEG_PREFIX,
        )

        contrastive_train_graph = train_encoder_graph
        if not args.no_contrastive:
            contrastive_train_graph = _with_cls_edges(contrastive_encoder_graph, cls_heads, cls_tails)
            _add_cls_edges_for_sampler(
                contrastive_train_graph,
                heads=cls_heads,
                tails=cls_tails,
                rels=cls_rels,
                labels=cls_labels,
                id2rel=id2rel,
            )
            if args.use_retrieved:
                print("[INFO] Final contrastive sampler uses contrastive graph (full subclass + retrieved edges).")
            else:
                print("[INFO] Final contrastive sampler uses contrastive graph (full subclass edges).")

        final_candidate_idx = torch.arange(cls_labels.numel(), dtype=torch.long)
        final_paired_idx, final_pair_stats = _build_hr_paired_train_indices(
            heads=cls_heads,
            rels=cls_rels,
            labels=cls_labels,
            candidate_idx=final_candidate_idx,
            seed=777,
        )
        if final_paired_idx.numel() == 0:
            raise RuntimeError(
                "Final training: no (head,relation)-matched pos/neg pairs could be built for supervision."
            )
        print(
            "[Final Train] Paired supervision: "
            f"raw_pos={final_pair_stats['raw_pos']} raw_neg={final_pair_stats['raw_neg']} "
            f"-> paired_pos={final_pair_stats['paired_pos']} paired_neg={final_pair_stats['paired_neg']} "
            f"(dropped_pos_no_hr_neg={final_pair_stats['dropped_pos_no_hr_neg']})"
        )

        final_heads_sup = cls_heads[final_paired_idx]
        final_rels_sup = cls_rels[final_paired_idx]
        final_tails_sup = cls_tails[final_paired_idx]
        final_labels_sup = cls_labels[final_paired_idx]

        final_edge_label_index = torch.stack([final_heads_sup, final_tails_sup], dim=0)
        final_edge_label = final_labels_sup.to(torch.float)
        # num_neighbors = {et: neighbor_sizes for et in encoder_graph.edge_types}
        # Include train classification edges in message passing for final training.
        final_mp_edge_types = _message_passing_edge_types(
            train_encoder_graph,
            subclass_rel=str(subclass_rel),
            drop_subclass=drop_subclass_mp_when_no_contrastive,
            cls_edge_suffix=CLS_EDGE_SUFFIX,
        )
        num_neighbors = {et: neighbor_sizes for et in train_encoder_graph.edge_types}
        num_neighbors[CLS_EDGE_TYPE] = [0, 0]
        for et in train_encoder_graph.edge_types:
            if str(et[1]).endswith(CLS_EDGE_SUFFIX):
                num_neighbors[et] = [0, 0]
        if drop_subclass_mp_when_no_contrastive:
            num_neighbors[("node", str(subclass_rel), "node")] = [0, 0]
        
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
            e_etypes=final_mp_edge_types,
            n_type=(mcfg.get("n_type", "node") if isinstance(mcfg, dict) else "node"),
        )
        if args.model in ("ra_hgcn", "ra_rgcn", "sra_hgcn", "ra_hgat", "gcn", "gat", "gae", "sgat", "nbfnet"):
            final_model = ModelCls(**final_base_kwargs, rel2id=rel2id).to(device)
        else:
            final_model = ModelCls(**final_base_kwargs).to(device)

        final_log = Logger("final_train_global", dir=args.output_dir, non_verbose=True)

        if not args.no_contrastive:
            if use_typed_sampler:
                final_contrastive_sampler = TypedInstanceSampler(k=contrastive_k, external_negs=external_edges)
                final_contrastive_sampler.prepare_global(contrastive_train_graph)
            elif use_random_sampler:
                final_contrastive_sampler = RandomInstanceSampler(k=contrastive_k, external_negs=external_edges)
                final_contrastive_sampler.prepare_global(contrastive_train_graph)
            elif use_partial_sampler:
                edges_are_negative = nflag
                final_contrastive_sampler = PartialInstanceSampler(
                    k=contrastive_k, neg_edges=external_edges, edges_are_negative=edges_are_negative
                )
                final_contrastive_sampler.prepare_global(contrastive_train_graph)
            else:
                SamplerCls = NegativeInstanceSampler_NEWER if args.alt_contrastive_sampler else NegativeInstanceSampler_NEW
                final_contrastive_sampler = SamplerCls(
                    k=contrastive_k, subclass_rel=subclass_rel, neg_prefix=NEG_PREFIX, instance_rel=instance_rel,
                    max_contrastive_anchors=int(args.max_contrastive_anchors),
                    cache_dir="data/cache",
                    cache_key=(
                        f"{args.path}|final|clsedge=1|"
                        f"retrieved_contrastive={int(args.use_retrieved and not args.no_contrastive)}|"
                        f"sampler={SamplerCls.__name__}"
                    )
                )
                final_contrastive_sampler.prepare_global(contrastive_train_graph)
            if (args.print_sampler_stats or args.use_retrieved) and not sampler_stats_printed:
                if hasattr(final_contrastive_sampler, "print_pool_stats"):
                    final_contrastive_sampler.print_pool_stats(prefix="[NegSamplerStats]")
                    sampler_stats_printed = True
        else:
            final_contrastive_sampler = None

        if args.finaltrain_only:
            final_lr = float(args.final_lr)
            final_epochs = int(args.final_epochs)
            selected_contrastive_weight = float(contrastive_weight)
            selected_contrastive_temperature = float(contrastive_temperature)

        if not args.no_contrastive:
            print(
                f"[Final Train] Using contrastive weight={selected_contrastive_weight:.6g}, "
                f"temperature={selected_contrastive_temperature:.6g}"
            )

        final_trainer = Train_BestModel(
            final_model,
            graph=train_encoder_graph,
            heads=final_heads_sup,
            rel_ids=final_rels_sup,
            tails=final_tails_sup,
            labels=final_labels_sup,
            lr=float(final_lr),
            epochs=int(final_epochs),
            device=device,
            log=final_log,
            batch_size=args.batch_size,
            contrastive_sampler=final_contrastive_sampler,
            contrastive_weight=selected_contrastive_weight,
            loader=final_loader,
            no_contrastive=args.no_contrastive,
            contrastive_temperature=selected_contrastive_temperature,
        )
        final_loss = final_trainer.run()
        print(f"[Final Train] Loss after {final_epochs} epochs (lr={final_lr:.3g}): {final_loss:.4f}")
        pd.DataFrame([{
            "model": args.model,
            "final_lr": float(final_lr),
            "final_epochs": int(final_epochs),
            "final_train_bce_loss": float(final_loss),
        }]).to_csv(os.path.join(run_out_dir, "final_train_summary.tsv"), sep="\t", index=False)
    test_log = Logger("test_global", dir=args.output_dir)

    train_encoder_graph = _with_cls_edges(encoder_graph, cls_heads, cls_tails)
    _add_cls_edges_for_sampler(
        train_encoder_graph,
        heads=cls_heads,
        tails=cls_tails,
        rels=cls_rels,
        labels=cls_labels,
        id2rel=id2rel,
    )

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if final_model is None:
        raise RuntimeError("final_model was not initialized. Use --test_only to load a model or run training.")

    model_path = os.path.join("output/" + args.output_dir, f"final_model_{args.model}.pt")
    if not args.test_only:
        torch.save(final_model.state_dict(), model_path)

    out_dir = os.path.join("output", args.output_dir)
    _save_embeddings_and_neighbors(
        model=final_model,
        encoder_graph=encoder_graph,
        out_dir=out_dir,
        sampler=final_contrastive_sampler,
        subclass_rel=subclass_rel,
        cls_edge_suffix=CLS_EDGE_SUFFIX,
    )

    test_pos_mask = (test_labels_all > 0.5)
    test_heads = test_heads_all[test_pos_mask]
    test_rels = test_rels_all[test_pos_mask]
    test_tails = test_tails_all[test_pos_mask]

    # Build filtered-LP "known true" positives from the authoritative source
    # (train2id_pos) instead of reconstructed leftovers.
    known_h_parts = [test_heads]
    known_r_parts = [test_rels]
    known_t_parts = [test_tails]

    train_pos_path = os.path.join(args.path, "train2id_pos.txt")
    df_train_pos_all = _read_edge_file(train_pos_path)
    df_train_pos_all["rel_base"] = df_train_pos_all["edge_type"].astype(str)
    df_train_pos_cls = df_train_pos_all[df_train_pos_all["rel_base"].isin(set(rel2id.keys()))].copy()
    if len(df_train_pos_cls) > 0:
        tr_h = torch.tensor(df_train_pos_cls["source_node"].to_numpy(dtype=np.int64), dtype=torch.long)
        tr_t = torch.tensor(df_train_pos_cls["target_node"].to_numpy(dtype=np.int64), dtype=torch.long)
        tr_r = torch.tensor(df_train_pos_cls["rel_base"].map(rel2id).to_numpy(dtype=np.int64), dtype=torch.long)
        known_h_parts.append(tr_h)
        known_r_parts.append(tr_r)
        known_t_parts.append(tr_t)

    known_h = torch.cat(known_h_parts, dim=0).detach().cpu().long()
    known_r = torch.cat(known_r_parts, dim=0).detach().cpu().long()
    known_t = torch.cat(known_t_parts, dim=0).detach().cpu().long()

    known_true_tails: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    known_true_heads: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for h, r, t in zip(known_h.tolist(), known_r.tolist(), known_t.tolist()):
        h_i = int(h)
        r_i = int(r)
        t_i = int(t)
        known_true_tails[(h_i, r_i)].add(t_i)
        known_true_heads[(t_i, r_i)].add(h_i)

    tester = LinkPredictionEvaluator(
        model=final_model,
        graph=train_encoder_graph,
        test_heads=test_heads,
        test_rels=test_rels,
        test_tails=test_tails,
        known_true_tails=dict(known_true_tails),
        known_true_heads=dict(known_true_heads),
        id2rel=id2rel,
        device=device,
        log=test_log,
        batch_size=args.batch_size,
        save_embeddings_path=os.path.join("output", args.output_dir, "embeddings.pt"),
        rankings_save_path=os.path.join(out_dir, "test_lp_rankings.pt"),
        rankings_tsv_path=os.path.join(out_dir, "test_lp_rankings.tsv"),
        metrics_save_path=os.path.join(out_dir, "test_lp_metrics.pt"),
        metrics_tsv_path=os.path.join(out_dir, "test_lp_metrics.tsv"),
        predictions_save_path=os.path.join(out_dir, "test_predictions.pt"),
    )
    tester.run()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
