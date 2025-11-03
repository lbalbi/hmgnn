# Sampler helpers
from torch_geometric.data import HeteroData
from typing import Tuple

def _find_key_by_rel(g: HeteroData, rel_name: str) -> Tuple[str, str, str]:
    key = next((et for et in g.edge_types if et[1] == rel_name), None)
    return key

def _num_nodes_of(g: HeteroData, ntype: str) -> int:
    explicit = getattr(g[ntype], "num_nodes", None)
    if explicit is not None: return int(explicit)
    if "x" in g[ntype] and g[ntype].x is not None: return int(g[ntype].x.size(0))
    max_id = -1
    for (s, _, d) in g.edge_types:
        ei = g[(s, _, d)].edge_index
        if s == ntype and ei.numel(): max_id = max(max_id, int(ei[0].max().item()))
        if d == ntype and ei.numel(): max_id = max(max_id, int(ei[1].max().item()))
    return max_id + 1 if max_id >= 0 else 0