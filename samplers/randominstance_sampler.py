import random, torch
from typing import Dict, Iterable, List, Optional, Tuple, Union
from torch_geometric.data import HeteroData
from .negativeinstance_sampler import NegativeInstanceSampler

# ExternalStmt = Union[Tuple[int, int], Tuple[int, str, int], Tuple[int, int, int]]

# class RandomInstanceSampler(NegativeInstanceSampler):
#     """ RandomInstanceSampler
#       1) Avoid per-epoch Python dict-of-lists for (key -> instances) when `prepare_batch()`
#          is called on the *full graph* (full-graph training mode). We build compact CSR
#          maps and cache them.
#       2) Avoid allocating `torch.arange(N)` for anchors and (optionally) avoid a dense
#          `_is_instance` mask when N is large.
#       3) External negatives are stored in a compact CSR index (inst -> keys), and the
#          original Python list is dropped to free RAM.
#     Two modes for anchor POS keys:
#       - global_pos_keys=True  (default): build global CSR (inst -> pos keys).
#         This avoids false negatives but costs memory proportional to #POS statements.
#       - global_pos_keys=False: derive POS keys only from the current prepared batch (cheap).
#         This reduces RAM a lot but can introduce occasional false negatives.
#     """

#     def __init__(self, k: int = 1, instance_rel: str = "2",
#         subclass_rel: str = "subclass_of", neg_prefix: str = "NOT_",
#         external_negs: Optional[List[ExternalStmt]] = None, max_corrupt_per_anchor: Optional[int] = None,
#         seed: Optional[int] = None, global_pos_keys: bool = True,
#         dense_instance_mask_threshold: int = 20_000_000):
#         super().__init__(k=k, subclass_rel=subclass_rel, neg_prefix=neg_prefix,
#             instance_rel=instance_rel, neg_expansion_hops=0)
#         self.seed = seed
#         self.external_negs = external_negs or []
#         self.max_corrupt_per_anchor = max_corrupt_per_anchor
#         self.global_pos_keys = bool(global_pos_keys)
#         self.dense_instance_mask_threshold = int(dense_instance_mask_threshold)
#         self._stride: int = 0
#         self._base_rel2id: Dict[str, int] = {}
#         self._universe_by_rid: Dict[int, torch.Tensor] = {}
#         self._global_universe: torch.Tensor = torch.empty(0, dtype=torch.int32)
#         self._inst_unique: Optional[torch.Tensor] = None
#         self._is_instance: Optional[torch.Tensor] = None
#         self._pos_inst: Optional[torch.Tensor] = None
#         self._pos_ptr: Optional[torch.Tensor] = None
#         self._pos_keys_flat: Optional[torch.Tensor] = None
#         self._g_pos_key_u: Optional[torch.Tensor] = None
#         self._g_pos_ptr: Optional[torch.Tensor] = None
#         self._g_pos_inst_flat: Optional[torch.Tensor] = None
#         self._g_neg_key_u: Optional[torch.Tensor] = None
#         self._g_neg_ptr: Optional[torch.Tensor] = None
#         self._g_neg_inst_flat: Optional[torch.Tensor] = None

#         # external negs: inst -> keys (CSR)
#         self._ext_inst_u: Optional[torch.Tensor] = None
#         self._ext_ptr: Optional[torch.Tensor] = None
#         self._ext_keys_flat: Optional[torch.Tensor] = None

#         # batch caches
#         self._batch_local_is_global: bool = False
#         self._batch_num_rows: int = 0
#         self._batch_global_for_row: Optional[torch.Tensor] = None  # int64 (only if needed)

#         # CSR maps for the current prepared batch
#         self._b_pos_key_u: Optional[torch.Tensor] = None
#         self._b_pos_ptr: Optional[torch.Tensor] = None
#         self._b_pos_inst_flat: Optional[torch.Tensor] = None
#         self._b_neg_key_u: Optional[torch.Tensor] = None
#         self._b_neg_ptr: Optional[torch.Tensor] = None
#         self._b_neg_inst_flat: Optional[torch.Tensor] = None

#         # low-memory batch-derived pos keys per anchor-local (only used if global_pos_keys=False)
#         self._b_poskeys_by_inst: Optional[Dict[int, List[int]]] = None

#         # full-graph caching flag: if prepare_batch gets the full graph repeatedly, reuse globals
#         self._fullgraph_cache_ready: bool = False

#         print("Using RandomInstanceSampler (memory-optimized)")

#     # -----------------------
#     # typed-key helpers
#     # -----------------------
#     def _base_rel(self, rel: str) -> str:
#         r = str(rel)
#         if r.startswith(self.neg_prefix):
#             return r[len(self.neg_prefix):]
#         if self.neg_prefix in r:
#             return r.replace(self.neg_prefix, "", 1)
#         return r

#     def _rel_id(self, base_rel: str) -> int:
#         base_rel = str(base_rel)
#         rid = self._base_rel2id.get(base_rel)
#         if rid is None:
#             rid = len(self._base_rel2id)
#             self._base_rel2id[base_rel] = rid
#         return rid

#     def _mk_key(self, rid: int, cls: int) -> int:
#         # key = rid * stride + cls
#         return int(rid) * int(self._stride) + int(cls)

#     def _parse_external_item(self, item: ExternalStmt) -> Tuple[int, str, int]:
#         if len(item) == 2:
#             a, b = item  # type: ignore
#             return int(a), "__external__", int(b)
#         if len(item) == 3:
#             a, rel, b = item  # type: ignore
#             return int(a), str(rel), int(b)
#         raise ValueError(f"Invalid external statement entry: {item}")

#     # -----------------------
#     # instance membership / orientation
#     # -----------------------
#     def _is_instance_vec(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Return a bool tensor for whether each element of x is an instance.
#         Uses a dense mask if available, else searchsorted into _inst_unique.
#         """
#         if self._is_instance is not None:
#             return self._is_instance[x.to(torch.int64)]
#         if self._inst_unique is None or self._inst_unique.numel() == 0:
#             return torch.zeros_like(x, dtype=torch.bool)
#         inst_u = self._inst_unique
#         idx = torch.searchsorted(inst_u, x)
#         out = torch.zeros_like(x, dtype=torch.bool)
#         m = idx < inst_u.numel()
#         if m.any():
#             out[m] = inst_u[idx[m]] == x[m]
#         return out

#     def _as_inst_class(self, a: int, b: int) -> Tuple[Optional[int], Optional[int]]:
#         """
#         Given two node ids, return (inst, cls) using instance membership if known.
#         """
#         if self._inst_unique is None and self._is_instance is None:
#             return int(a), int(b)
#         # check membership (scalar)
#         aa = torch.tensor([int(a)], dtype=torch.int32)
#         bb = torch.tensor([int(b)], dtype=torch.int32)
#         a_is = bool(self._is_instance_vec(aa)[0].item())
#         b_is = bool(self._is_instance_vec(bb)[0].item())
#         if a_is and (not b_is):
#             return int(a), int(b)
#         if b_is and (not a_is):
#             return int(b), int(a)
#         return None, None

#     # -----------------------
#     # corruption
#     # -----------------------
#     def _corrupt_for_rel(self, rid: int, pos_set: set[int], n_draw: int) -> List[int]:
#         uni = self._universe_by_rid.get(int(rid))
#         if uni is None or uni.numel() == 0:
#             uni = self._global_universe
#         if uni is None or uni.numel() == 0:
#             return []

#         if self.max_corrupt_per_anchor is not None:
#             n_draw = min(n_draw, int(self.max_corrupt_per_anchor))
#         n_draw = max(1, int(n_draw))

#         M = int(uni.numel())
#         if not pos_set:
#             if n_draw <= M:
#                 idx = torch.randperm(M)[:n_draw]
#                 return uni[idx].to(torch.int64).tolist()
#             idx = torch.randint(0, M, (n_draw,))
#             return uni[idx].to(torch.int64).tolist()

#         out: set[int] = set()
#         attempts = 0
#         max_attempts = max(200, 50 + 20 * n_draw)
#         while len(out) < n_draw and attempts < max_attempts:
#             c = int(uni[random.randrange(M)].item())
#             attempts += 1
#             if c in pos_set:
#                 continue
#             out.add(c)
#         return list(out)

#     # -----------------------
#     # CSR helpers
#     # -----------------------
#     @staticmethod
#     def _build_csr_by_first(first: torch.Tensor, second: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Build CSR grouped by `first`:
#           - first_u: unique sorted values of first
#           - ptr: offsets into second_flat
#           - second_flat: second values aligned to first_u segments
#         """
#         if first.numel() == 0:
#             first_u = torch.empty(0, dtype=first.dtype)
#             ptr = torch.zeros(1, dtype=torch.int64)
#             second_flat = torch.empty(0, dtype=second.dtype)
#             return first_u, ptr, second_flat

#         perm = torch.argsort(first)
#         first_s = first[perm]
#         second_s = second[perm]

#         first_u, counts = torch.unique_consecutive(first_s, return_counts=True)
#         ptr = torch.zeros(int(first_u.numel()) + 1, dtype=torch.int64)
#         ptr[1:] = torch.cumsum(counts.to(torch.int64), dim=0)
#         return first_u, ptr, second_s

#     @staticmethod
#     def _gather_from_csr(keys: Iterable[int], key_u: Optional[torch.Tensor],
#                         ptr: Optional[torch.Tensor], flat: Optional[torch.Tensor]) -> torch.Tensor:
#         if key_u is None or ptr is None or flat is None:
#             return torch.empty(0, dtype=torch.int64)
#         if key_u.numel() == 0:
#             return torch.empty(0, dtype=torch.int64)

#         parts: List[torch.Tensor] = []
#         for k in keys:
#             k_int = int(k)
#             idx = int(torch.searchsorted(key_u, torch.tensor(k_int, dtype=key_u.dtype)).item())
#             if idx < int(key_u.numel()) and int(key_u[idx].item()) == k_int:
#                 start = int(ptr[idx].item())
#                 end = int(ptr[idx + 1].item())
#                 if end > start:
#                     parts.append(flat[start:end])
#         if not parts:
#             return torch.empty(0, dtype=torch.int64)
#         if len(parts) == 1:
#             return parts[0].to(torch.int64)
#         # NOTE: do NOT unique here; unique can be expensive for large candidate pools.
#         return torch.cat(parts, dim=0).to(torch.int64)

#     # -----------------------
#     # global prep
#     # -----------------------
#     def prepare_global(self, full_g: HeteroData) -> None:
#         if self.seed is not None:
#             random.seed(self.seed)
#             torch.manual_seed(self.seed)

#         if len(full_g.node_types) != 1:
#             raise ValueError("RandomInstanceSampler assumes a single node type.")
#         self.node_type = full_g.node_types[0]
#         N = int(self._infer_num_nodes(full_g, self.node_type))
#         self.num_nodes = N
#         self._stride = N
#         self._base_rel2id = {}

#         # Pre-register relation ids so keys computed in prepare_batch match global keys.
#         for (_, rel, _dst) in full_g.edge_types:
#             rel_s = str(rel)
#             if rel_s == self.subclass_rel:
#                 continue
#             self._rel_id(self._base_rel(rel_s))

#         # Identify instances (store sorted ids; build dense mask only if N is not too big)
#         self._inst_unique = None
#         self._is_instance = None
#         if self.instance_rel is not None:
#             inst_key = self._find_edge_key(full_g, self.instance_rel)
#             if inst_key is not None and "edge_index" in full_g[inst_key]:
#                 inst_src = full_g[inst_key].edge_index[0].detach()
#                 # keep on CPU for membership tests while building indices
#                 inst_src = inst_src.to("cpu", non_blocking=True).to(torch.int64)
#                 inst_src = inst_src[(inst_src >= 0) & (inst_src < N)]
#                 if inst_src.numel() > 0:
#                     inst_unique = torch.unique(inst_src).to(torch.int32)
#                     inst_unique, _ = torch.sort(inst_unique)
#                     self._inst_unique = inst_unique
#                     if N <= self.dense_instance_mask_threshold:
#                         is_instance = torch.zeros(N, dtype=torch.bool)
#                         is_instance[inst_unique.to(torch.int64)] = True
#                         self._is_instance = is_instance

#         # Universes + (optionally) global pos keys + global key->inst maps
#         uni_acc: Dict[int, List[torch.Tensor]] = {}

#         pos_inst_chunks: List[torch.Tensor] = []
#         pos_key_chunks: List[torch.Tensor] = []

#         # global key->inst chunks (for candidates)
#         g_pos_key_chunks: List[torch.Tensor] = []
#         g_pos_inst_chunks: List[torch.Tensor] = []
#         g_neg_key_chunks: List[torch.Tensor] = []
#         g_neg_inst_chunks: List[torch.Tensor] = []

#         for (s, rel, d), eidx in full_g.edge_index_dict.items():
#             if s != self.node_type or d != self.node_type:
#                 continue
#             if eidx.numel() == 0:
#                 continue

#             rel_s = str(rel)
#             if rel_s == self.subclass_rel:
#                 continue

#             is_neg = rel_s.startswith(self.neg_prefix)
#             is_pos = not is_neg
#             base_rel = self._base_rel(rel_s)
#             rid = self._rel_id(base_rel)

#             e = eidx.detach()
#             # Keep on CPU to avoid doubling GPU+CPU memory in full-graph mode.
#             e = e.to("cpu", non_blocking=True).to(torch.int64)
#             src = e[0]
#             dst = e[1]
#             valid = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
#             if not bool(valid.any()):
#                 continue
#             src = src[valid]
#             dst = dst[valid]

#             if self._inst_unique is not None or self._is_instance is not None:
#                 src_i = self._is_instance_vec(src.to(torch.int32))
#                 dst_i = self._is_instance_vec(dst.to(torch.int32))
#                 ok = src_i ^ dst_i
#                 if not bool(ok.any()):
#                     continue
#                 src_ok = src[ok]
#                 dst_ok = dst[ok]
#                 src_i_ok = src_i[ok]
#                 inst = torch.where(src_i_ok, src_ok, dst_ok)
#                 cls = torch.where(src_i_ok, dst_ok, src_ok)
#             else:
#                 inst = src
#                 cls = dst

#             if inst.numel() == 0:
#                 continue

#             inst_i32 = inst.to(torch.int32)
#             cls_i32 = cls.to(torch.int32)
#             key_i64 = cls_i32.to(torch.int64) + int(rid) * int(self._stride)

#             # build universes from POS statements only
#             if is_pos:
#                 uni_acc.setdefault(int(rid), []).append(torch.unique(cls_i32))

#             # global key->inst maps for candidates (both POS and NEG)
#             if is_pos:
#                 g_pos_key_chunks.append(key_i64)
#                 g_pos_inst_chunks.append(inst_i32)
#             else:
#                 g_neg_key_chunks.append(key_i64)
#                 g_neg_inst_chunks.append(inst_i32)

#             # global inst->poskeys cache (optional)
#             if self.global_pos_keys and is_pos:
#                 pos_inst_chunks.append(inst_i32)
#                 pos_key_chunks.append(key_i64)

#         # Finalize universes (CPU int32).
#         self._universe_by_rid = {}
#         uni_all: List[torch.Tensor] = []
#         for rid, parts in uni_acc.items():
#             u = torch.unique(torch.cat(parts, dim=0)).to(torch.int32)
#             self._universe_by_rid[int(rid)] = u
#             if u.numel() > 0:
#                 uni_all.append(u)
#         self._global_universe = torch.unique(torch.cat(uni_all, dim=0)).to(torch.int32) if uni_all else torch.empty(0, dtype=torch.int32)

#         # Build global inst->poskeys CSR (or disable)
#         if self.global_pos_keys and pos_inst_chunks:
#             pos_inst = torch.cat(pos_inst_chunks, dim=0).to(torch.int32)
#             pos_keys = torch.cat(pos_key_chunks, dim=0).to(torch.int64)
#             # sort by inst (not key) for CSR
#             self._pos_inst, self._pos_ptr, self._pos_keys_flat = self._build_csr_by_first(pos_inst, pos_keys)
#         else:
#             self._pos_inst = torch.empty(0, dtype=torch.int32)
#             self._pos_ptr = torch.zeros(1, dtype=torch.int64)
#             self._pos_keys_flat = torch.empty(0, dtype=torch.int64)

#         # Build global key->inst CSRs (POS and NEG)
#         if g_pos_key_chunks:
#             gk = torch.cat(g_pos_key_chunks, dim=0).to(torch.int64)
#             gi = torch.cat(g_pos_inst_chunks, dim=0).to(torch.int32)
#             self._g_pos_key_u, self._g_pos_ptr, self._g_pos_inst_flat = self._build_csr_by_first(gk, gi)
#         else:
#             self._g_pos_key_u = torch.empty(0, dtype=torch.int64)
#             self._g_pos_ptr = torch.zeros(1, dtype=torch.int64)
#             self._g_pos_inst_flat = torch.empty(0, dtype=torch.int32)

#         if g_neg_key_chunks:
#             gk = torch.cat(g_neg_key_chunks, dim=0).to(torch.int64)
#             gi = torch.cat(g_neg_inst_chunks, dim=0).to(torch.int32)
#             self._g_neg_key_u, self._g_neg_ptr, self._g_neg_inst_flat = self._build_csr_by_first(gk, gi)
#         else:
#             self._g_neg_key_u = torch.empty(0, dtype=torch.int64)
#             self._g_neg_ptr = torch.zeros(1, dtype=torch.int64)
#             self._g_neg_inst_flat = torch.empty(0, dtype=torch.int32)

#         # Parse external negs into compact CSR (inst -> keys) and drop the Python list to free RAM.
#         ext_inst_list: List[int] = []
#         ext_key_list: List[int] = []
#         warned_pairs = False

#         for item in self.external_negs:
#             a, rel, b = self._parse_external_item(item)
#             if rel == "__external__":
#                 warned_pairs = True
#             if not (0 <= a < N and 0 <= b < N):
#                 continue
#             inst, cls = self._as_inst_class(int(a), int(b))
#             if inst is None or cls is None:
#                 continue
#             rid = self._rel_id(self._base_rel(rel))
#             ext_inst_list.append(int(inst))
#             ext_key_list.append(self._mk_key(rid, int(cls)))

#         if ext_inst_list:
#             ext_inst = torch.tensor(ext_inst_list, dtype=torch.int32)
#             ext_keys = torch.tensor(ext_key_list, dtype=torch.int64)
#             self._ext_inst_u, self._ext_ptr, self._ext_keys_flat = self._build_csr_by_first(ext_inst, ext_keys)
#         else:
#             self._ext_inst_u = torch.empty(0, dtype=torch.int32)
#             self._ext_ptr = torch.zeros(1, dtype=torch.int64)
#             self._ext_keys_flat = torch.empty(0, dtype=torch.int64)

#         # drop reference to potentially huge Python list to actually free RAM
#         self.external_negs = []

#         if warned_pairs and (self._ext_inst_u is not None and self._ext_inst_u.numel() > 0):
#             print(
#                 "[RandomInstanceSampler] WARNING: external_negs provided as (src,dst) pairs. "
#                 "They are treated as rel='__external__'. Provide triples (src, rel, dst) for type-aware behavior."
#             )

#         # For full-graph training, we can directly reuse the global maps as the "batch maps".
#         self._fullgraph_cache_ready = True
#         self._batch_local_is_global = True
#         self._batch_num_rows = N
#         self._batch_global_for_row = None
#         self._b_pos_key_u, self._b_pos_ptr, self._b_pos_inst_flat = self._g_pos_key_u, self._g_pos_ptr, self._g_pos_inst_flat
#         self._b_neg_key_u, self._b_neg_ptr, self._b_neg_inst_flat = self._g_neg_key_u, self._g_neg_ptr, self._g_neg_inst_flat
#         self._b_poskeys_by_inst = None  # computed per-batch only when global_pos_keys=False

#     # -----------------------
#     # get pos keys for an anchor
#     # -----------------------
#     def _get_pos_keys_global(self, u_global: int) -> List[int]:
#         if self._pos_inst is None or self._pos_ptr is None or self._pos_keys_flat is None:
#             return []
#         if self._pos_inst.numel() == 0:
#             return []
#         u = int(u_global)
#         # search in int32 domain to avoid per-call casting of the whole tensor
#         idx = int(torch.searchsorted(self._pos_inst, torch.tensor(u, dtype=self._pos_inst.dtype)).item())
#         if idx >= int(self._pos_inst.numel()) or int(self._pos_inst[idx].item()) != u:
#             return []
#         start = int(self._pos_ptr[idx].item())
#         end = int(self._pos_ptr[idx + 1].item())
#         if end <= start:
#             return []
#         return self._pos_keys_flat[start:end].tolist()

#     def _synth_neg_keys_from_pos(self, pos_keys: List[int]) -> List[int]:
#         if not pos_keys:
#             return []
#         stride = int(self._stride)
#         rid2pos: Dict[int, set[int]] = {}
#         for k in pos_keys:
#             kk = int(k)
#             rid = int(kk // stride)
#             cls = int(kk - rid * stride)
#             rid2pos.setdefault(rid, set()).add(cls)

#         neg_keys: List[int] = []
#         for rid, pos_set in rid2pos.items():
#             n_draw = len(pos_set) if pos_set else 1
#             neg_classes = self._corrupt_for_rel(rid, pos_set, n_draw)
#             for c in neg_classes:
#                 neg_keys.append(int(rid * stride + int(c)))
#         return neg_keys

#     # -----------------------
#     # batch-local prep
#     # -----------------------
#     def prepare_batch(self, batch: HeteroData) -> None:
#         """
#         Builds compact CSR maps for this batch:
#           POS: key -> instances
#           NEG: key -> instances
#         Also optionally builds a tiny local mapping inst_local -> pos_keys when global_pos_keys=False.

#         Crucially, if this is the full graph (as in trainer full-graph mode),
#         we reuse the cached globals instead of rebuilding every epoch.
#         """
#         ntype = self.node_type or "node"
#         if ntype not in batch.node_types:
#             ntype = batch.node_types[0]

#         B = int(batch[ntype].num_nodes)
#         self._batch_num_rows = B

#         # Detect the common "full-graph training" case: no n_id and same node count.
#         if ("n_id" not in batch[ntype]) and (self._fullgraph_cache_ready and B == int(self.num_nodes)):
#             self._batch_local_is_global = True
#             self._batch_global_for_row = None
#             self._b_pos_key_u, self._b_pos_ptr, self._b_pos_inst_flat = self._g_pos_key_u, self._g_pos_ptr, self._g_pos_inst_flat
#             self._b_neg_key_u, self._b_neg_ptr, self._b_neg_inst_flat = self._g_neg_key_u, self._g_neg_ptr, self._g_neg_inst_flat
#             self._b_poskeys_by_inst = None
#             return

#         # Otherwise, this is a subgraph batch (NeighborLoader etc.)
#         if "n_id" in batch[ntype]:
#             global_for_row = batch[ntype].n_id.detach().to("cpu", non_blocking=True).to(torch.int64)
#             self._batch_global_for_row = global_for_row
#             self._batch_local_is_global = False
#         else:
#             # local == global, but smaller than full graph
#             self._batch_global_for_row = None
#             self._batch_local_is_global = True

#         # Collect key/inst pairs as tensors (avoid Python dict-of-lists).
#         pos_key_chunks: List[torch.Tensor] = []
#         pos_inst_chunks: List[torch.Tensor] = []
#         neg_key_chunks: List[torch.Tensor] = []
#         neg_inst_chunks: List[torch.Tensor] = []

#         # optionally build local inst -> pos_keys for low-memory mode
#         poskeys_by_inst: Optional[Dict[int, set[int]]] = None
#         if not self.global_pos_keys:
#             poskeys_by_inst = {}

#         for (s, rel, d), eidx in batch.edge_index_dict.items():
#             if s != ntype or d != ntype:
#                 continue
#             if eidx.numel() == 0:
#                 continue

#             rel_s = str(rel)
#             if rel_s == self.subclass_rel:
#                 continue

#             is_neg = rel_s.startswith(self.neg_prefix)
#             is_pos = not is_neg
#             base_rel = self._base_rel(rel_s)
#             rid = self._rel_id(base_rel)

#             e = eidx.detach()
#             e = e.to("cpu", non_blocking=True).to(torch.int64)
#             src_l = e[0]
#             dst_l = e[1]

#             if self._batch_global_for_row is None:
#                 src_g = src_l
#                 dst_g = dst_l
#             else:
#                 src_g = self._batch_global_for_row[src_l]
#                 dst_g = self._batch_global_for_row[dst_l]

#             # orient inst/class
#             if self._inst_unique is not None or self._is_instance is not None:
#                 src_i = self._is_instance_vec(src_g.to(torch.int32))
#                 dst_i = self._is_instance_vec(dst_g.to(torch.int32))
#                 ok = src_i ^ dst_i
#                 if not bool(ok.any()):
#                     continue
#                 src_l = src_l[ok]
#                 dst_l = dst_l[ok]
#                 src_g = src_g[ok]
#                 dst_g = dst_g[ok]
#                 src_i = src_i[ok]
#                 inst_l = torch.where(src_i, src_l, dst_l).to(torch.int32)
#                 cls_g = torch.where(src_i, dst_g, src_g).to(torch.int32)
#             else:
#                 inst_l = src_l.to(torch.int32)
#                 cls_g = dst_g.to(torch.int32)

#             if inst_l.numel() == 0:
#                 continue

#             key = cls_g.to(torch.int64) + int(rid) * int(self._stride)

#             if is_pos:
#                 pos_key_chunks.append(key)
#                 pos_inst_chunks.append(inst_l)
#                 if poskeys_by_inst is not None:
#                     # batch edges are small; Python sets are ok here
#                     for il, kk in zip(inst_l.tolist(), key.tolist()):
#                         poskeys_by_inst.setdefault(int(il), set()).add(int(kk))
#             else:
#                 neg_key_chunks.append(key)
#                 neg_inst_chunks.append(inst_l)

#         # Add external negs for instances in this batch (inst_global -> keys via CSR)
#         if self._ext_inst_u is not None and self._ext_inst_u.numel() > 0:
#             for il in range(B):
#                 # recover global id for local row
#                 if self._batch_global_for_row is None:
#                     g = il
#                 else:
#                     g = int(self._batch_global_for_row[il].item())
#                 # lookup in ext CSR
#                 idx = int(torch.searchsorted(self._ext_inst_u.to(torch.int64), torch.tensor(g, dtype=torch.int64)).item())
#                 if idx >= int(self._ext_inst_u.numel()) or int(self._ext_inst_u[idx].item()) != g:
#                     continue
#                 start = int(self._ext_ptr[idx].item())
#                 end = int(self._ext_ptr[idx + 1].item())
#                 if end <= start:
#                     continue
#                 keys = self._ext_keys_flat[start:end]
#                 neg_key_chunks.append(keys.to(torch.int64))
#                 neg_inst_chunks.append(torch.full((keys.numel(),), int(il), dtype=torch.int32))

#         # Finalize CSR maps for this batch
#         if pos_key_chunks:
#             pk = torch.cat(pos_key_chunks, dim=0).to(torch.int64)
#             pi = torch.cat(pos_inst_chunks, dim=0).to(torch.int32)
#             self._b_pos_key_u, self._b_pos_ptr, self._b_pos_inst_flat = self._build_csr_by_first(pk, pi)
#         else:
#             self._b_pos_key_u = torch.empty(0, dtype=torch.int64)
#             self._b_pos_ptr = torch.zeros(1, dtype=torch.int64)
#             self._b_pos_inst_flat = torch.empty(0, dtype=torch.int32)

#         if neg_key_chunks:
#             nk = torch.cat(neg_key_chunks, dim=0).to(torch.int64)
#             ni = torch.cat(neg_inst_chunks, dim=0).to(torch.int32)
#             self._b_neg_key_u, self._b_neg_ptr, self._b_neg_inst_flat = self._build_csr_by_first(nk, ni)
#         else:
#             self._b_neg_key_u = torch.empty(0, dtype=torch.int64)
#             self._b_neg_ptr = torch.zeros(1, dtype=torch.int64)
#             self._b_neg_inst_flat = torch.empty(0, dtype=torch.int32)

#         # finalize per-inst pos keys (low-memory mode only)
#         if poskeys_by_inst is not None:
#             self._b_poskeys_by_inst = {k: list(v) for k, v in poskeys_by_inst.items()}
#         else:
#             self._b_poskeys_by_inst = None

#     # -----------------------
#     # sampling
#     # -----------------------
#     @staticmethod
#     def _sample_k(cands: torch.Tensor, anchor_local: int, k: int, device: torch.device) -> torch.Tensor:
#         """Sample up to k indices from `cands` without allocating a huge randperm/unique."""
#         if cands.numel() == 0:
#             return torch.full((k,), anchor_local, dtype=torch.long, device=device)

#         M = int(cands.numel())
#         # If the pool is small-ish, randperm is fine.
#         if M <= 200_000 and M >= k:
#             idx = torch.randperm(M)[:k]
#             out = cands[idx]
#             return out.to(torch.long).to(device)

#         # For large pools, sample with replacement in a small oversample and then de-dup.
#         # This avoids `torch.unique(cands)` or `torch.randperm(M)` on very large M.
#         if M >= k:
#             draws = min(M, max(k * 8, 64))
#             idx = torch.randint(0, M, (draws,))
#             picks = cands[idx].to(torch.int64)
#             picks = torch.unique(picks)
#             if int(picks.numel()) >= k:
#                 return picks[:k].to(torch.long).to(device)
#             # fallthrough: not enough unique, pad
#             pad = torch.full((k - int(picks.numel()),), anchor_local, dtype=torch.int64)
#             return torch.cat([picks, pad], dim=0).to(torch.long).to(device)

#         # M < k
#         pad = torch.full((k - M,), anchor_local, dtype=torch.int64)
#         out = torch.cat([cands.to(torch.int64), pad], dim=0)
#         return out.to(torch.long).to(device)

#     def get_contrastive_samples(
#         self,
#         z: torch.Tensor,
#         anchor_nodes: Optional[torch.Tensor] = None,
#         n_id: Optional[torch.Tensor] = None,
#     ):
#         device = z.device
#         B_rows, D = z.shape
#         k = self.k

#         if self._b_pos_key_u is None or self._b_neg_key_u is None:
#             empty = torch.empty(0, D, device=device)
#             empty_k = torch.empty(0, k, D, device=device)
#             return empty, empty_k, empty_k, empty_k, empty_k

#         if anchor_nodes is None or anchor_nodes.numel() == 0:
#             anchor_locals = torch.arange(B_rows, dtype=torch.long)
#         else:
#             # In full-graph mode, anchor_nodes are global ids == row ids.
#             # In neighbor mode, anchor_nodes are local row ids passed from trainer.
#             anchor_locals = anchor_nodes.detach().long().cpu().unique()

#         anchors_local_list: List[int] = []
#         shneg_idx, pos2neg_idx, neg2pos_idx, shpos_idx = [], [], [], []

#         for u_local in anchor_locals.tolist():
#             if u_local < 0 or u_local >= B_rows:
#                 continue

#             # resolve global id for pos-key lookup
#             if self._batch_global_for_row is None:
#                 u_global = int(u_local)
#             else:
#                 u_global = int(self._batch_global_for_row[u_local].item())

#             # POS keys
#             if self.global_pos_keys:
#                 pos_keys = self._get_pos_keys_global(u_global)
#             else:
#                 # low-memory mode: batch-derived
#                 if self._b_poskeys_by_inst is None:
#                     continue
#                 pos_keys = self._b_poskeys_by_inst.get(int(u_local), [])
#             if not pos_keys:
#                 continue

#             neg_keys = self._synth_neg_keys_from_pos(pos_keys)

#             cand_shpos = self._gather_from_csr(pos_keys, self._b_pos_key_u, self._b_pos_ptr, self._b_pos_inst_flat)
#             cand_shneg = self._gather_from_csr(neg_keys, self._b_neg_key_u, self._b_neg_ptr, self._b_neg_inst_flat)
#             cand_pos2neg = self._gather_from_csr(neg_keys, self._b_pos_key_u, self._b_pos_ptr, self._b_pos_inst_flat)
#             cand_neg2pos = self._gather_from_csr(pos_keys, self._b_neg_key_u, self._b_neg_ptr, self._b_neg_inst_flat)

#             # exclude anchor
#             if cand_shpos.numel() > 0:
#                 cand_shpos = cand_shpos[cand_shpos != u_local]
#             if cand_shneg.numel() > 0:
#                 cand_shneg = cand_shneg[cand_shneg != u_local]
#             if cand_pos2neg.numel() > 0:
#                 cand_pos2neg = cand_pos2neg[cand_pos2neg != u_local]
#             if cand_neg2pos.numel() > 0:
#                 cand_neg2pos = cand_neg2pos[cand_neg2pos != u_local]
#             if (cand_shpos.numel() == 0 and cand_shneg.numel() == 0 and
#                 cand_pos2neg.numel() == 0 and cand_neg2pos.numel() == 0):
#                 continue

#             anchors_local_list.append(int(u_local))
#             shpos_idx.append(self._sample_k(cand_shpos, u_local, k, device).unsqueeze(0))
#             shneg_idx.append(self._sample_k(cand_shneg, u_local, k, device).unsqueeze(0))
#             pos2neg_idx.append(self._sample_k(cand_pos2neg, u_local, k, device).unsqueeze(0))
#             neg2pos_idx.append(self._sample_k(cand_neg2pos, u_local, k, device).unsqueeze(0))

#         if not anchors_local_list:
#             empty = torch.empty(0, D, device=device)
#             empty_k = torch.empty(0, k, D, device=device)
#             return empty, empty_k, empty_k, empty_k, empty_k
#         anchors_local_t = torch.tensor(anchors_local_list, dtype=torch.long, device=device)
#         shneg_t = torch.cat(shneg_idx, dim=0)
#         pos2neg_t = torch.cat(pos2neg_idx, dim=0)
#         neg2pos_t = torch.cat(neg2pos_idx, dim=0)
#         shpos_t = torch.cat(shpos_idx, dim=0)
#         return (z[anchors_local_t], z[shneg_t], z[pos2neg_t], z[neg2pos_t], z[shpos_t])


ExternalStmt = Union[Tuple[int, int], Tuple[int, str, int], Tuple[int, int, int]]
class RandomInstanceSampler(NegativeInstanceSampler):
    """
    Wiki framework: random corruption-based instance sampler (TYPE-AWARE, FAST).

    Differences vs NegativeInstanceSampler:
      - Anchor negatives are NOT taken from graph.
      - For each anchor u and each base relation type r, we corrupt u's POS targets under r to
        synthetic NEG targets under the SAME base relation type r (no ontology expansion).
    Candidate retrieval is still based on the graph's real NEG edges:
      - shared_neg and neg_to_u_pos are populated using instances that have NEG statements
        of the SAME base relation type to those synthetic classes.
    Uses batch-local key->local-instance maps (no O(num_nodes) masks per batch).
    Statement polarity rules (Wiki):
      - POS: rel does NOT start with neg_prefix and rel != subclass_rel
      - NEG: rel DOES start with neg_prefix
    """

    def __init__(
        self,
        k: int = 1,
        instance_rel: str = "2",
        subclass_rel: str = "subclass_of",
        neg_prefix: str = "NOT_",
        external_negs: Optional[List[ExternalStmt]] = None,
        max_corrupt_per_anchor: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            k=k,
            subclass_rel=subclass_rel,
            neg_prefix=neg_prefix,
            instance_rel=instance_rel,
            neg_expansion_hops=0,
        )
        self.seed = seed
        self.external_negs = external_negs or []
        self.max_corrupt_per_anchor = max_corrupt_per_anchor

        # typed-key machinery
        self._stride: int = 0
        self._base_rel2id: Dict[str, int] = {}
        self._universe_by_rid: Dict[int, Tuple[int, ...]] = {}
        self._global_universe: Tuple[int, ...] = ()

        # per-node typed positive keys, plus per-node per-rel class sets for corruption
        self._pos_keys: List[List[int]] = []
        self._pos_by_rel: List[Dict[int, List[int]]] = []
        self._neg_keys_synth: List[List[int]] = []

        # Optional: external neg statements indexed by instance (for batch augmentation)
        self._ext_neg_by_inst: Dict[int, List[int]] = {}

        # Batch caches
        self._batch_global_for_row: Optional[torch.Tensor] = None   # CPU [B_rows]
        self._batch_pos_map: Optional[Dict[int, torch.Tensor]] = None  # key(int) -> local inst ids (CPU)
        self._batch_neg_map: Optional[Dict[int, torch.Tensor]] = None  # key(int) -> local inst ids (CPU)
        self._batch_pos_key_u: Optional[torch.Tensor] = None
        self._batch_pos_ptr: Optional[torch.Tensor] = None
        self._batch_pos_flat: Optional[torch.Tensor] = None
        self._batch_neg_key_u: Optional[torch.Tensor] = None
        self._batch_neg_ptr: Optional[torch.Tensor] = None
        self._batch_neg_flat: Optional[torch.Tensor] = None
        # Simple corruption-based sampling (batch-local)
        self._batch_pos_tails_by_inst: Optional[Dict[int, List[int]]] = None
        self._batch_class_universe: Optional[torch.Tensor] = None
        # Limit anchors per batch (ablation: fewer anchors)
        self.max_contrastive_anchors: int = 3072

        print("Using RandomInstanceSampler")

    # -----------------------
    # typed-key helpers
    # -----------------------
    def _base_rel(self, rel: str) -> str:
        r = str(rel)
        if r.startswith(self.neg_prefix):
            r = r[len(self.neg_prefix) :]
        elif self.neg_prefix in r:
            r = r.replace(self.neg_prefix, "", 1)
        if r.endswith("__cls"):
            r = r[: -len("__cls")]
        return r

    def _rel_id(self, base_rel: str) -> int:
        base_rel = str(base_rel)
        rid = self._base_rel2id.get(base_rel)
        if rid is None:
            rid = len(self._base_rel2id)
            self._base_rel2id[base_rel] = rid
        return rid

    def _mk_key(self, rid: int, cls: int) -> int:
        return int(rid) * int(self._stride) + int(cls)

    def _parse_external_item(self, item: ExternalStmt) -> Tuple[int, str, int]:
        if len(item) == 2:
            a, b = item  # type: ignore
            return int(a), "__external__", int(b)
        if len(item) == 3:
            a, rel, b = item  # type: ignore
            return int(a), str(rel), int(b)
        raise ValueError(f"Invalid external statement entry: {item}")

    # -----------------------
    # corruption
    # -----------------------
    def _corrupt_for_rel(self, rid: int, pos_set: set[int], n_draw: int) -> List[int]:
        uni = self._universe_by_rid.get(rid, ())
        if not uni:
            uni = self._global_universe
        if not uni:
            return []

        if self.max_corrupt_per_anchor is not None:
            n_draw = min(n_draw, int(self.max_corrupt_per_anchor))
        n_draw = max(1, int(n_draw))

        M = len(uni)
        if not pos_set:
            if n_draw <= M:
                return random.sample(list(uni), n_draw)
            return [uni[random.randrange(M)] for _ in range(n_draw)]

        out: set[int] = set()
        attempts = 0
        max_attempts = max(200, 50 + 20 * n_draw)
        while len(out) < n_draw and attempts < max_attempts:
            c = uni[random.randrange(M)]
            attempts += 1
            if int(c) in pos_set:
                continue
            out.add(int(c))
        return list(out)

    # -----------------------
    # global prep
    # -----------------------
    def prepare_global(self, full_g: HeteroData) -> None:
        if len(full_g.node_types) != 1:
            raise ValueError("RandomInstanceSampler assumes a single node type.")
        self.node_type = full_g.node_types[0]
        N = self._infer_num_nodes(full_g, self.node_type)
        self.num_nodes = N
        self._stride = N
        self._base_rel2id = {}

        # identify instances
        instance_nodes: set[int] = set()
        if self.instance_rel is not None:
            inst_key = self._find_edge_key(full_g, self.instance_rel)
            if inst_key is not None and "edge_index" in full_g[inst_key]:
                inst_src = full_g[inst_key].edge_index[0].detach().cpu().tolist()
                instance_nodes.update(int(x) for x in inst_src if 0 <= int(x) < N)

        if instance_nodes:
            is_instance = torch.zeros(N, dtype=torch.bool)
            is_instance[list(instance_nodes)] = True
            self._is_instance = is_instance
        else:
            self._is_instance = None

        # collect positive statements per node, per base-rel (typed)
        self._pos_keys = [[] for _ in range(N)]
        self._pos_by_rel = [dict() for _ in range(N)]
        anchor_sources: set[int] = set(instance_nodes) if instance_nodes else set()

        # build universes per base-rel from POS statements
        uni_tmp: Dict[int, set[int]] = {}

        for (_, rel, _), eidx in full_g.edge_index_dict.items():
            if eidx.numel() == 0 or rel == self.subclass_rel:
                continue

            rel_s = str(rel)
            is_neg = rel_s.startswith(self.neg_prefix)
            is_pos = (not is_neg) and (rel_s != self.subclass_rel)
            if not (is_neg or is_pos):
                continue

            base_rel = self._base_rel(rel_s)
            rid = self._rel_id(base_rel)

            src = eidx[0].detach().cpu().tolist()
            dst = eidx[1].detach().cpu().tolist()
            for a, b in zip(src, dst):
                if not (0 <= a < N and 0 <= b < N):
                    continue
                inst, cls = self._as_inst_class(int(a), int(b))
                if inst is None or cls is None:
                    continue

                anchor_sources.add(inst)
                if is_pos:
                    key = self._mk_key(rid, cls)
                    self._pos_keys[inst].append(key)
                    self._pos_by_rel[inst].setdefault(rid, []).append(int(cls))
                    uni_tmp.setdefault(rid, set()).add(int(cls))

        self.anchors = sorted(anchor_sources) if anchor_sources else list(range(N))

        # finalize universes
        self._universe_by_rid = {rid: tuple(sorted(list(s))) for rid, s in uni_tmp.items()}
        all_classes = set()
        for s in uni_tmp.values():
            all_classes |= s
        self._global_universe = tuple(sorted(list(all_classes)))

        # build synthetic negatives per node, per rel
        self._neg_keys_synth = [[] for _ in range(N)]
        for u in self.anchors:
            # corrupt per rel
            for rid, cls_list in self._pos_by_rel[u].items():
                pos_set = set(int(c) for c in cls_list)
                n_draw = len(pos_set) if len(pos_set) > 0 else 1
                neg_classes = self._corrupt_for_rel(rid, pos_set, n_draw)
                for c in neg_classes:
                    self._neg_keys_synth[u].append(self._mk_key(rid, int(c)))

        # Optional external neg statements (typed) indexed by instance for batch augmentation
        self._ext_neg_by_inst = {}
        warned_pairs = False
        for item in self.external_negs:
            a, rel, b = self._parse_external_item(item)
            if rel == "__external__":
                warned_pairs = True
            if not (0 <= a < N and 0 <= b < N):
                continue
            inst, cls = self._as_inst_class(int(a), int(b))
            if inst is None or cls is None:
                continue
            base_rel = self._base_rel(rel)
            rid = self._rel_id(base_rel)
            key = self._mk_key(rid, cls)
            self._ext_neg_by_inst.setdefault(int(inst), []).append(int(key))

        if warned_pairs and self.external_negs:
            print(
                "[RandomInstanceSampler] WARNING: external_negs provided as (src,dst) pairs. "
                "They will be treated as rel='__external__' and will NOT enforce matching against graph relation types. "
                "Provide triples (src, rel, dst) for full type-aware behavior."
            )

        # reset batch caches
        self._batch_global_for_row = None
        self._batch_pos_map = None
        self._batch_neg_map = None

    # -----------------------
    # batch-local map build (fast)
    # -----------------------
    def prepare_batch(self, batch: HeteroData) -> None:
        ntype = self.node_type or "node"
        if ntype not in batch.node_types:
            ntype = batch.node_types[0]

        if "n_id" in batch[ntype]:
            global_for_row = batch[ntype].n_id.detach().long().cpu()
        else:
            global_for_row = torch.arange(batch[ntype].num_nodes, dtype=torch.long)

        self._batch_global_for_row = global_for_row

        pos_keys: List[int] = []
        pos_inst: List[int] = []
        neg_keys: List[int] = []
        neg_inst: List[int] = []
        pos_tails_by_inst: Dict[int, List[int]] = {}
        class_universe: set[int] = set()

        # iterate batch edges
        for key, eidx in batch.edge_index_dict.items():
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            s, rel, d = key
            if s != ntype or d != ntype:
                continue

            rel_s = str(rel)
            if rel_s == self.subclass_rel:
                continue

            is_neg = rel_s.startswith(self.neg_prefix)
            is_pos = (not is_neg) and (rel_s != self.subclass_rel)
            if not (is_neg or is_pos):
                continue

            base_rel = self._base_rel(rel_s)
            rid = self._rel_id(base_rel)  # safe even if unseen
            src_l = eidx[0].detach().long().cpu()
            dst_l = eidx[1].detach().long().cpu()
            src_g = global_for_row[src_l]
            dst_g = global_for_row[dst_l]

            # orient inst/class
            if getattr(self, "_is_instance", None) is not None:
                src_is = self._is_instance[src_g]
                dst_is = self._is_instance[dst_g]
                ok = src_is ^ dst_is
                if ok.sum().item() == 0:
                    continue
                src_l = src_l[ok]
                dst_l = dst_l[ok]
                src_g = src_g[ok]
                dst_g = dst_g[ok]
                src_is = src_is[ok]
                inst_l = torch.where(src_is, src_l, dst_l)
                cls_g = torch.where(src_is, dst_g, src_g)
                cls_l = torch.where(src_is, dst_l, src_l)
            else:
                inst_l = src_l
                cls_g = dst_g
                cls_l = dst_l

            for c, il in zip(cls_g.tolist(), inst_l.tolist()):
                k_int = self._mk_key(rid, int(c))
                if is_neg:
                    neg_keys.append(int(k_int))
                    neg_inst.append(int(il))
                else:
                    pos_keys.append(int(k_int))
                    pos_inst.append(int(il))
            if is_pos:
                for cl, il in zip(cls_l.tolist(), inst_l.tolist()):
                    pos_tails_by_inst.setdefault(int(il), []).append(int(cl))
                    class_universe.add(int(cl))

        # add external neg statements for instances in batch (cheap)
        if self._ext_neg_by_inst:
            # map global->local for batch rows
            # (build a tiny dict, avoids O(num_nodes) tensors)
            g2l = {int(g): int(i) for i, g in enumerate(global_for_row.tolist())}
            for inst_g, keys in self._ext_neg_by_inst.items():
                il = g2l.get(int(inst_g))
                if il is None:
                    continue
                for k_int in keys:
                    neg_keys.append(int(k_int))
                    neg_inst.append(int(il))

        # Build CSR for pos/neg maps (key -> inst_local list)
        def _build_csr(keys: List[int], insts: List[int]):
            if not keys:
                key_u = torch.empty(0, dtype=torch.int64)
                ptr = torch.zeros(1, dtype=torch.int64)
                flat = torch.empty(0, dtype=torch.int64)
                return key_u, ptr, flat
            k = torch.tensor(keys, dtype=torch.int64)
            v = torch.tensor(insts, dtype=torch.int64)
            perm = torch.argsort(k)
            k = k[perm]
            v = v[perm]
            key_u, counts = torch.unique_consecutive(k, return_counts=True)
            ptr = torch.zeros(int(key_u.numel()) + 1, dtype=torch.int64)
            ptr[1:] = torch.cumsum(counts.to(torch.int64), dim=0)
            return key_u, ptr, v

        self._batch_pos_key_u, self._batch_pos_ptr, self._batch_pos_flat = _build_csr(pos_keys, pos_inst)
        self._batch_neg_key_u, self._batch_neg_ptr, self._batch_neg_flat = _build_csr(neg_keys, neg_inst)

        # keep dicts for fallback/debug; but avoid per-key unique in hot path
        self._batch_pos_map = None
        self._batch_neg_map = None
        # store corruption pools (local ids)
        self._batch_pos_tails_by_inst = pos_tails_by_inst
        if class_universe:
            self._batch_class_universe = torch.tensor(sorted(class_universe), dtype=torch.long)
        else:
            self._batch_class_universe = torch.empty(0, dtype=torch.long)

    # -----------------------
    # sampling (fast, type-aware)
    # -----------------------
    @staticmethod
    def _sample_k(cands: torch.Tensor, anchor_local: int, k: int, device: torch.device) -> torch.Tensor:
        """
        Fallback policy:
          - if empty -> all anchor
          - if <k -> take all unique then pad with anchor
          - if >=k -> sample without replacement
        """
        if cands.numel() == 0:
            return torch.full((k,), anchor_local, dtype=torch.long, device=device)

        # unique not strictly necessary if maps are unique; keep safe
        cands = torch.unique(cands)
        if cands.numel() >= k:
            idx = torch.randperm(cands.numel())[:k]
            out = cands[idx]
        else:
            pad = torch.full((k - cands.numel(),), anchor_local, dtype=torch.long)
            out = torch.cat([cands, pad], dim=0)
        return out.to(device)

    @staticmethod
    def _union_from_csr(keys: List[int], key_u: torch.Tensor, ptr: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        if key_u is None or ptr is None or flat is None:
            return torch.empty(0, dtype=torch.long)
        if key_u.numel() == 0 or not keys:
            return torch.empty(0, dtype=torch.long)
        parts: List[torch.Tensor] = []
        key_u_cpu = key_u
        for k in keys:
            k_int = int(k)
            idx = int(torch.searchsorted(key_u_cpu, torch.tensor(k_int, dtype=key_u_cpu.dtype)).item())
            if idx < int(key_u_cpu.numel()) and int(key_u_cpu[idx].item()) == k_int:
                start = int(ptr[idx].item())
                end = int(ptr[idx + 1].item())
                if end > start:
                    parts.append(flat[start:end])
        if not parts:
            return torch.empty(0, dtype=torch.long)
        if len(parts) == 1:
            return parts[0]
        return torch.unique(torch.cat(parts, dim=0))

    def get_contrastive_samples(self, z: torch.Tensor,
        anchor_nodes: Optional[torch.Tensor] = None,
        n_id: Optional[torch.Tensor] = None):
        device = z.device
        B_rows, D = z.shape
        k = self.k

        if self._batch_global_for_row is None or self._batch_pos_key_u is None or self._batch_neg_key_u is None:
            # If user forgot to call prepare_batch, return empty (no pooling/structure reuse).
            empty = torch.empty(0, z.size(1), device=z.device)
            empty_k = torch.empty(0, self.k, z.size(1), device=z.device)
            return empty, empty_k, empty_k, empty_k, empty_k

        # Use only random corruption-based negatives (no pooling/structure reuse)
        if self._batch_pos_tails_by_inst is not None and self._batch_class_universe is not None:
            universe = self._batch_class_universe
            universe_list = universe.tolist() if universe.numel() > 0 else []
            if anchor_nodes is None or anchor_nodes.numel() == 0:
                anchor_locals = torch.arange(B_rows, dtype=torch.long)
            else:
                anchor_locals = anchor_nodes.detach().long().cpu().unique()
            if self.max_contrastive_anchors and anchor_locals.numel() > self.max_contrastive_anchors:
                perm = torch.randperm(anchor_locals.numel())[: self.max_contrastive_anchors]
                anchor_locals = anchor_locals[perm]

            anchors_local_list: List[int] = []
            shneg_idx, pos2neg_idx, neg2pos_idx, shpos_idx = [], [], [], []

            for u_local in anchor_locals.tolist():
                if u_local < 0 or u_local >= B_rows:
                    continue
                pos_tails = self._batch_pos_tails_by_inst.get(int(u_local), [])
                if not pos_tails:
                    continue
                pos_set = set(int(x) for x in pos_tails)
                # build negatives by corrupting tails
                if universe_list:
                    neg_list = [c for c in universe_list if c not in pos_set]
                else:
                    neg_list = []
                if not neg_list:
                    continue

                cand_pos = torch.tensor(pos_tails, dtype=torch.long)
                cand_neg = torch.tensor(neg_list, dtype=torch.long)
                # exclude anchor if it appears (rare)
                cand_pos = cand_pos[cand_pos != u_local]
                cand_neg = cand_neg[cand_neg != u_local]
                if cand_pos.numel() == 0 or cand_neg.numel() == 0:
                    continue

                anchors_local_list.append(int(u_local))
                shpos_idx.append(self._sample_k(cand_pos, u_local, k, device).unsqueeze(0))
                shneg_idx.append(self._sample_k(cand_pos, u_local, k, device).unsqueeze(0))
                pos2neg_idx.append(self._sample_k(cand_neg, u_local, k, device).unsqueeze(0))
                neg2pos_idx.append(self._sample_k(cand_neg, u_local, k, device).unsqueeze(0))

            if not anchors_local_list:
                empty = torch.empty(0, D, device=device)
                empty_k = torch.empty(0, k, D, device=device)
                return empty, empty_k, empty_k, empty_k, empty_k

            anchors_local_t = torch.tensor(anchors_local_list, dtype=torch.long, device=device)
            shneg_t = torch.cat(shneg_idx, dim=0)
            pos2neg_t = torch.cat(pos2neg_idx, dim=0)
            neg2pos_t = torch.cat(neg2pos_idx, dim=0)
            shpos_t = torch.cat(shpos_idx, dim=0)

            return (z[anchors_local_t], z[shneg_t],
                z[pos2neg_t], z[neg2pos_t], z[shpos_t])

        # No structured pooling fallback
        empty = torch.empty(0, D, device=device)
        empty_k = torch.empty(0, k, D, device=device)
        return empty, empty_k, empty_k, empty_k, empty_k

        anchors_local_t = torch.tensor(anchors_local_list, dtype=torch.long, device=device)
        shneg_t = torch.cat(shneg_idx, dim=0)
        pos2neg_t = torch.cat(pos2neg_idx, dim=0)
        neg2pos_t = torch.cat(neg2pos_idx, dim=0)
        shpos_t = torch.cat(shpos_idx, dim=0)

        return (z[anchors_local_t], z[shneg_t],
            z[pos2neg_t], z[neg2pos_t], z[shpos_t])

# class RandomInstanceSampler(NegativeInstanceSampler):
#     """ For Wiki framework: random corruption-based instance sampler. Same instance-example groups 
#     as NegativeStatementSampler, but:
#       - Anchor negative classes are NOT taken from graph. Instead, for each anchor u, corrupts its 
#       positive classes to generate synthetic negatives: neg_classes(u) = random sample of classes 
#       from a global class universe excluding pos_classes(u)
#     The graph's negative statement edges are still used to find other instances that have negative
#     statements to those synthetic classes (so shared_neg and neg_to_u_pos can be populated).
#     """
#     def __init__(self, k: int = 1, instance_rel: str = "2", subclass_rel: str = "subclass_of",
#         neg_prefix: str = "NOT_", external_negs: Optional[List[Tuple[int, int]]] = None,
#         max_corrupt_per_anchor: Optional[int] = None, seed: Optional[int] = None):
#         super().__init__(k=k, subclass_rel=subclass_rel,
#             neg_prefix=neg_prefix, instance_rel=instance_rel,
#             neg_expansion_hops=0, seed=seed)
#         self.external_negs = external_negs or []
#         self.max_corrupt_per_anchor = max_corrupt_per_anchor
#         self._class_universe: List[int] = []
#         print("Using RandomInstanceSampler")
#     def _build_class_universe(self) -> None:
#         seen: set[int] = set()
#         for u in self.anchors:
#             for c in self.pos_classes[u]:
#                 seen.add(int(c))
#         if not seen:
#             for u in self.anchors:
#                 for c in self.neg_classes_direct[u]:
#                     seen.add(int(c))
#         self._class_universe = sorted(seen)

#     def _corrupt_classes(self, pos_set: set[int], n_draw: int) -> List[int]:
#         if not self._class_universe: return []
#         if self.max_corrupt_per_anchor is not None:
#             n_draw = min(n_draw, int(self.max_corrupt_per_anchor))
#         n_draw = max(1, int(n_draw))
#         neg: set[int] = set()
#         attempts = 0
#         max_attempts = max(200, 50 + 20 * n_draw)
#         while len(neg) < n_draw and attempts < max_attempts:
#             c = random.choice(self._class_universe)
#             attempts += 1
#             if c in pos_set: continue
#             neg.add(int(c))
#         return list(neg)

#     def prepare_global(self, full_g: HeteroData) -> None:
#         super().prepare_global(full_g)
#         if self.external_negs:
#             N = self.num_nodes
#             neg_key_list: List[int] = []
#             neg_val_list: List[int] = []

#             for cls, insts in self.class2neg_instances.items():
#                 for inst in insts.tolist():
#                     neg_key_list.append(int(cls))
#                     neg_val_list.append(int(inst))

#             for a, b in self.external_negs:
#                 if not (0 <= int(a) < N and 0 <= int(b) < N): continue
#                 inst, cls = self._as_inst_class(int(a), int(b))
#                 if inst is None or cls is None: continue
#                 self.neg_classes_direct[inst].append(int(cls))
#                 neg_key_list.append(int(cls))
#                 neg_val_list.append(int(inst))

#             self.neg_classes_direct = [list(set(xs)) for xs in self.neg_classes_direct]
#             self.class2neg_instances = (
#                 self._group_unique_by_key(torch.tensor(neg_key_list, dtype=torch.long),
#                                           torch.tensor(neg_val_list, dtype=torch.long))
#                 if neg_key_list else {})

#         self._build_class_universe()
#         N = self.num_nodes
#         self.neg_classes_expanded = [[] for _ in range(N)]
#         for u in self.anchors:
#             pos_set = set(int(c) for c in self.pos_classes[u])
#             n_draw = len(pos_set) if len(pos_set) > 0 else 1
#             self.neg_classes_expanded[u] = self._corrupt_classes(pos_set, n_draw)

#         empty = torch.empty(0, dtype=torch.long)
#         self.pool_shared_pos = [empty for _ in range(N)]
#         self.pool_shared_neg = [empty for _ in range(N)]
#         self.pool_pos_to_u_neg = [empty for _ in range(N)]
#         self.pool_neg_to_u_pos = [empty for _ in range(N)]

#         for u in self.anchors:
#             pos_cls = self.pos_classes[u]
#             neg_cls = self.neg_classes_expanded[u]
#             self.pool_shared_pos[u] = self._collect_instances(pos_cls, self.class2pos_instances, exclude=u)
#             self.pool_shared_neg[u] = self._collect_instances(neg_cls, self.class2neg_instances, exclude=u)
#             self.pool_pos_to_u_neg[u] = self._collect_instances(neg_cls, self.class2pos_instances, exclude=u)
#             self.pool_neg_to_u_pos[u] = self._collect_instances(pos_cls, self.class2neg_instances, exclude=u)

#         self._global2local_cpu = None
#         self._in_batch_mask_cpu = None
