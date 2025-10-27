import sklearn.metrics, torch
import numpy as np

def evaluate_pyg(model, g, labels, npairs, e_types, mask_, device_, val=False, save=False, homo=False):
        model.eval()
        with torch.no_grad():
            logits = model(g.to(device_), npairs, e_types, mask_, val).detach().to("cpu")
            logits_ = torch.round(logits)
            acc = sklearn.metrics.accuracy_score(labels, logits_)
            f1 = sklearn.metrics.f1_score(labels, logits_, average='weighted')
            pr = sklearn.metrics.precision_score(labels, logits_, zero_division=0)
            re = sklearn.metrics.recall_score(labels, logits_, zero_division=0)
            roc_auc = sklearn.metrics.roc_auc_score(labels, logits, average='weighted')
            print(f'Accuracy: {acc}, F1: {f1}')
            if save:  torch.save(logits, 'predictions_{}.pth'.format(model.name[2:] if homo else model.name))
        return acc, f1, pr, re, roc_auc


def _to_long_1d(x):
    """Convert numpy/torch 1D-like column to 1D long tensor (allow empty)."""
    if x is None: return torch.empty(0, dtype=torch.long)
    t = torch.as_tensor(x, dtype=torch.long)
    return t.view(-1)

def _cat_or_empty(cols):
    """cols: list of 1D long tensors; returns concatenated 1D long tensor."""
    cols = [c for c in cols if c is not None and c.numel() > 0]
    if not cols: return torch.empty(0, dtype=torch.long)
    return torch.cat(cols, dim=0)

def _num_nodes_from_edges(src, dst):
    if src.numel() == 0 and dst.numel() == 0: return 0
    return int(torch.max(torch.stack([src.max() if src.numel() else torch.tensor(-1),
                                      dst.max() if dst.numel() else torch.tensor(-1)])).item()) + 1


        
def make_homographs(device, train_ppis, test_ppis, go_links, pos_annots, neg_annots,
                    rem_go=False, rem_pos=False, rem_neg=False):
    """
    Returns two HeteroData graphs with a single node type 'node' and relation 'has_edge'.
    """
    def build(ppis):
        src_list = [_to_long_1d(ppis.T[0])]
        dst_list = [_to_long_1d(ppis.T[1])]
        if not rem_go:
            src_list.append(_to_long_1d(go_links.T[0]))
            dst_list.append(_to_long_1d(go_links.T[1]))
        if not rem_pos:
            src_list.append(_to_long_1d(pos_annots.T[0]))
            dst_list.append(_to_long_1d(pos_annots.T[1]))
        if not rem_neg:
            src_list.append(_to_long_1d(neg_annots.T[0]))
            dst_list.append(_to_long_1d(neg_annots.T[1]))

        src = _cat_or_empty(src_list)
        dst = _cat_or_empty(dst_list)
        data = HeteroData()
        num_nodes = _num_nodes_from_edges(src, dst)
        data['node'].num_nodes = num_nodes
        data['node'].x = torch.randn(num_nodes, 128)
        data[('node', 'has_edge', 'node')].edge_index = torch.stack([src, dst], dim=0)
        return data.to(device)
            
    train_dict = build(train_ppis)
    test_dict = build(test_ppis)
    return train_dict, test_dict
                            

def make_bigraphs(device, train_ppis, test_ppis, go_links, pos_annots, neg_annots,
                  rem_go=False, rem_pos=False, rem_neg=False):
    """
    Returns train+test HeteroData graphs with 2 relation types (pos/neg).
    """
    def build(ppis):
        pos_src = [_to_long_1d(ppis.T[0])]
        pos_dst = [_to_long_1d(ppis.T[1])]
        if not rem_go:
            pos_src.append(_to_long_1d(go_links.T[0]))
            pos_dst.append(_to_long_1d(go_links.T[1]))
        if not rem_pos:
            pos_src.append(_to_long_1d(pos_annots.T[0]))
            pos_dst.append(_to_long_1d(pos_annots.T[1]))
        neg_src = []
        neg_dst = []
        if not rem_neg:
            neg_src.append(_to_long_1d(neg_annots.T[0]))
            neg_dst.append(_to_long_1d(neg_annots.T[1]))
        src_pos = _cat_or_empty(pos_src)
        dst_pos = _cat_or_empty(pos_dst)
        src_neg = _cat_or_empty(neg_src)
        dst_neg = _cat_or_empty(neg_dst)

        num_nodes = _num_nodes_from_edges(
            torch.cat([src_pos, src_neg], dim=0) if src_neg.numel() else src_pos,
            torch.cat([dst_pos, dst_neg], dim=0) if dst_neg.numel() else dst_pos)
        data = HeteroData()
        data['node'].num_nodes = num_nodes
        data['node'].x = torch.randn(num_nodes, 128)
        if src_pos.numel():
            data[('node', 'positive_link', 'node')].edge_index = torch.stack([src_pos, dst_pos], dim=0)
        if src_neg.numel():
            data[('node', 'negative_link', 'node')].edge_index = torch.stack([src_neg, dst_neg], dim=0)
        return data.to(device)
    train_dict = build(train_ppis)
    test_dict = build(test_ppis)
    return train_dict, test_dict


def make_heterographs(model, device, train_ppis, test_ppis, go_links, pos_annots, neg_annots):
    """
    Returns train+test hetero graphs with node types 'protein' and 'class' and optional relations.
    """
    def build(ppis):
        data = HeteroData()
        ppi_src = _to_long_1d(ppis.T[0])
        ppi_dst = _to_long_1d(ppis.T[1])
        data[('protein', 'has_ppi', 'protein')].edge_index = torch.stack([ppi_src, ppi_dst], dim=0)
            
        if 'has_link' in getattr(model, 'relations', []):
            cls_src = _to_long_1d(go_links.T[0])
            cls_dst = _to_long_1d(go_links.T[1])
            if cls_src.numel():
                data[('class', 'has_link', 'class')].edge_index = torch.stack([cls_src, cls_dst], dim=0)
        if 'has_+annotation' in getattr(model, 'relations', []):
            pa_src = _to_long_1d(pos_annots.T[0]); pa_dst = _to_long_1d(pos_annots.T[1])
            if pa_src.numel():
                data[('protein', 'has_+annotation', 'class')].edge_index = torch.stack([pa_src, pa_dst], dim=0)
        if 'has_-annotation' in getattr(model, 'relations', []):
            na_src = _to_long_1d(neg_annots.T[0]); na_dst = _to_long_1d(neg_annots.T[1])
            if na_src.numel():
                data[('protein', 'has_-annotation', 'class')].edge_index = torch.stack([na_src, na_dst], dim=0)

        prot_ids = [ppi_src, ppi_dst]
        if 'has_+annotation' in getattr(model, 'relations', []):
            prot_ids.append(_to_long_1d(pos_annots.T[0]))
        if 'has_-annotation' in getattr(model, 'relations', []):
            prot_ids.append(_to_long_1d(neg_annots.T[0]))
        prot_all = _cat_or_empty(prot_ids)
        n_prot = int(prot_all.max().item()) + 1 if prot_all.numel() else 0
        data['protein'].num_nodes = n_prot
        data['protein'].x = torch.randn(n_prot, 128)

        class_ids = []
        if 'has_link' in getattr(model, 'relations', []):
            class_ids += [cls_src, cls_dst]
        if 'has_+annotation' in getattr(model, 'relations', []):
            class_ids.append(_to_long_1d(pos_annots.T[1]))
        if 'has_-annotation' in getattr(model, 'relations', []):
            class_ids.append(_to_long_1d(neg_annots.T[1]))
        cls_all = _cat_or_empty(class_ids)
        n_cls = int(cls_all.max().item()) + 1 if cls_all.numel() else 0
        data['class'].num_nodes = n_cls
        data['class'].x = torch.randn(n_cls, 128)
        return data.to(device)
    train_dict = build(train_ppis)
    test_dict = build(test_ppis)
    return train_dict, test_dict


def masking(dict_edges, _idx, e_type, mask, device):
    """
    Set boolean edge mask on given relation of HeteroData graph (e.g., "has_edge")
    """
    et = ('node', e_type, 'node')
    E = dict_edges[et].edge_index.size(1)
    m = torch.zeros(E, dtype=torch.bool, device=device)
    m[_to_long_1d(_idx).to(device)] = True
    dict_edges[et][mask] = m
    return dict_edges


def mask_proteins(ppis, neg_ppis, pos_annots, neg_annots, partition = 0.85):
        
    import numpy
    proteins = []
    proteins.extend(numpy.unique(ppis).tolist())
    proteins.extend(numpy.unique(neg_ppis).tolist())
    if pos_annots != None: proteins.extend(numpy.unique(pos_annots.T[0]).tolist())
    if neg_annots != None: proteins.extend(numpy.unique(neg_annots.T[0]).tolist())
    proteins = list(set(proteins))

    train_proteins = torch.randperm(len(proteins))
    train_proteins = torch.tensor(proteins)[train_proteins].view(torch.tensor(proteins).size())
    train_proteins = train_proteins[:int(len(proteins)*partition)]
    torch.save(train_proteins, 'data/train_partition.pt')
    return train_proteins



def mask_edges(ppis, neg_ppis, partition=0.9):
    import numpy as np
    import torch

    num_ppis = len(ppis)
    num_neg_ppis = len(neg_ppis)
    ppi_perm = torch.randperm(num_ppis)
    neg_ppi_perm = torch.randperm(num_neg_ppis)

    split_ppi = int(num_ppis * partition)
    split_neg_ppi = int(num_neg_ppis * partition)

    train_ppis = ppis[ppi_perm[:split_ppi]]
    test_ppis = ppis[ppi_perm[split_ppi:]]
    train_neg_ppis = neg_ppis[neg_ppi_perm[:split_neg_ppi]]
    test_neg_ppis = neg_ppis[neg_ppi_perm[split_neg_ppi:]]

    torch.save({"train_ppis": train_ppis, "train_neg_ppis": train_neg_ppis},
        'data/train_edge_partition.pt')
    return train_ppis, test_ppis, train_neg_ppis, test_neg_ppis


def generate_negative_edges(pos_edges):
    """
    Generates negative edges based on the unique source node ids of a set of positive edges.
    Args: pos_edges: numpy array of shape (N, 2) with positive edges (source, target)
    Returns: neg_edges: numpy array of shape (N, 2) with negative edges
    """
    num_nodes = int(pos_edges.max()) + 1
    pos_edge_set = set(map(tuple, pos_edges.tolist()))
    pos_sources = np.unique(pos_edges[:, 0])
    
    rng = np.random.default_rng()
    neg_edges = set()

    # To guarantee at least one negative edge per source node
    for src in pos_sources:
        possible_dst = np.setdiff1d(np.arange(num_nodes), [src])
        dst = rng.choice(possible_dst)
        while (src, dst) in pos_edge_set:
            dst = rng.choice(possible_dst)
        neg_edges.add((src, dst))
    print("Initial negative edges generated: ", len(neg_edges), flush=True)

    needed = len(pos_edges) - len(neg_edges)
    while len(neg_edges) < len(pos_edges):
        src_batch = rng.integers(0, num_nodes, size=needed * 2)
        dst_batch = rng.integers(0, num_nodes, size=needed * 2)

        mask = src_batch != dst_batch
        candidates = zip(src_batch[mask], dst_batch[mask])

        for edge in candidates:
            if edge not in pos_edge_set and edge not in neg_edges:
                neg_edges.add(edge)
                if len(neg_edges) == len(pos_edges):
                    break

    return np.array(list(neg_edges))
