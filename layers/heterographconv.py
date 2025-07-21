from functools import partial
import torch as th
import torch.nn as nn

class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.
    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.
    If the relation graph has no edge, the corresponding module will not be called.
    """

    def __init__(self, mods, aggregate="all"):
        super(HeteroGraphConv, self).__init__()
        self.mod_dict = mods
        mods = {str(k): v for k, v in mods.items()}
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            fn = getattr(v, "set_allow_zero_in_degree", None)
            if callable(fn):
                fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def _get_module(self, etype):
        mod = self.mod_dict.get(etype, None)
        if mod is not None:
            return mod
        if isinstance(etype, tuple):
            # etype is canonical
            _, etype, _ = etype
            return self.mod_dict[etype]
        raise KeyError(f"Cannot find module with edge type {etype}")

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}

        outputs = {nty: [] for nty in g.dsttypes}

        # Handle bipartite-block or full-graph uniformly
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in g.canonical_etypes:
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                rel_graph = g[stype, etype, dtype]

                # unwrap per‐relation if needed
                feat_src = src_inputs[stype]
                if isinstance(feat_src, dict):
                    feat_src = feat_src[etype]
                feat_dst = dst_inputs[dtype]
                if isinstance(feat_dst, dict):
                    feat_dst = feat_dst[etype]

                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (feat_src, feat_dst),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append((etype, dstdata))
        else:
            for stype, etype, dtype in g.canonical_etypes:
                if stype not in inputs:
                    continue
                rel_graph = g[stype, etype, dtype]

                # unwrap per‐relation if needed
                feat_src = inputs[stype]
                if isinstance(feat_src, dict):
                    feat_src = feat_src[etype]
                feat_dst = inputs[dtype]
                if isinstance(feat_dst, dict):
                    feat_dst = feat_dst[etype]

                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (feat_src, feat_dst),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append((etype, dstdata))

        # aggregate per‐relation results
        rsts = {}
        for nty, alist in outputs.items():
            if alist:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _stack_agg_func(inputs, dsttype):
    if not inputs:
        return None
    return th.stack([emb for _, emb in inputs], dim=1)


def _all_agg_func(inputs, dsttype):
    if not inputs:
        return {}
    return {etype: emb for etype, emb in inputs}


def _agg_func(inputs, dsttype, fn):
    if not inputs:
        return None
    stacked = th.stack([emb for _, emb in inputs], dim=0)
    return fn(stacked, dim=0)


def get_aggregate_fn(agg):
    if agg == "stack":
        return _stack_agg_func
    elif agg == "all":
        return _all_agg_func
    else:
        return partial(_agg_func, fn=agg)
