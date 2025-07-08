from functools import partial
import torch as th
import torch.nn as nn

from dgl.base import DGLError

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
            set_allow_zero_in_degree_fn = getattr(
                v, "set_allow_zero_in_degree", None
            )
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
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
        raise KeyError("Cannot find module with edge type %s" % etype)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {
                    k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
                }

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append((etype, dstdata))
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self._get_module((stype, etype, dtype))(
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {})
                )
                outputs[dtype].append((etype, dstdata))
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts

def _stack_agg_func(inputs, dsttype):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return th.stack(inputs, dim=1)

def _all_agg_func(inputs, dsttype):  # pylint: disable=unused-argument
    """
    inputs: a list of (edge_type, Tensor[N, D]) for this dsttype
    returns: { edge_type: Tensor[N, D], … }
    """
    if len(inputs) == 0:
        return {}
    out = {}
    for item in inputs:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise ValueError(f"_all_agg_func expected (etype, Tensor) tuples, " f"but got {item!r}")
        etype, emb = item
        out[etype] = emb
    return out


def _agg_func(inputs, dsttype, fn):  # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = th.stack(inputs, dim=0)
    return fn(stacked, dim=0)


def get_aggregate_fn(agg):
    if agg == "stack":
        return _stack_agg_func
    elif agg == "all":
        return _all_agg_func
    else:
        return partial(_agg_func, fn=fn)
