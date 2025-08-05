import sklearn.metrics, torch, dgl
import numpy as np

def evaluate_dgl(model, g, labels, npairs, e_types, mask_, device_, val=False, save=False, homo=False):

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



def make_homographs(device, train_ppis, test_ppis, go_links, pos_annots, neg_annots, rem_go = False, rem_pos=False, rem_neg=False):

    test_dict = dgl.heterograph({("node",'has_edge',"node"): (torch.cat((torch.tensor(test_ppis.T[0]), go_links.T[0] if rem_go == False else torch.tensor([]),
        pos_annots.T[0] if rem_pos == False else torch.tensor([]), neg_annots.T[0] if rem_neg == False else torch.tensor([]))).numpy(), 
        torch.cat((torch.tensor(test_ppis.T[1]), go_links.T[1] if rem_go == False else torch.tensor([]), pos_annots.T[1] if rem_pos == False else torch.tensor([]),
        neg_annots.T[1] if rem_neg == False else torch.tensor([]))).numpy())}).to(device)
    
    test_dict.nodes["node"].data["x"] =  torch.randn(test_dict.num_nodes("node"), 128).to(device)

    train_dict = dgl.heterograph({("node",'has_edge',"node"): (torch.cat((torch.tensor(train_ppis.T[0]), go_links.T[0] if rem_go == False else torch.tensor([]),
        pos_annots.T[0] if rem_pos == False else torch.tensor([]), neg_annots.T[0] if rem_neg == False else torch.tensor([]))).numpy(), 
        torch.cat((torch.tensor(train_ppis.T[1]), go_links.T[1] if rem_go == False else torch.tensor([]), pos_annots.T[1] if rem_pos == False else torch.tensor([]),
        neg_annots.T[1] if rem_neg == False else torch.tensor([]))).numpy())}).to(device)
    
    train_dict.nodes["node"].data["x"] =  torch.randn(train_dict.num_nodes("node"), 128).to(device)
    return train_dict, test_dict


def make_bigraphs(device, train_ppis, test_ppis, go_links, pos_annots, neg_annots, rem_go = False, rem_pos=False, rem_neg=False):

    test_dict = dgl.heterograph({
            ("node",'positive_link',"node"): (torch.cat((torch.tensor(test_ppis.T[0]), go_links.T[0] if rem_go == False else torch.tensor([]),
            pos_annots.T[0] if rem_pos == False else torch.tensor([]))).numpy(),  torch.cat((torch.tensor(test_ppis.T[1]),
            go_links.T[1] if rem_go == False else torch.tensor([]), pos_annots.T[1] if rem_pos == False else torch.tensor([]))).numpy()),
            **({("node",'negative_link',"node"): (neg_annots.T[0].numpy(), neg_annots.T[1].numpy())} if rem_neg == False else {})}).to(device)
    
    test_dict.nodes["node"].data["x"] =  torch.randn(test_dict.num_nodes("node"), 128).to(device)

    train_dict = dgl.heterograph({
            ("node",'positive_link',"node"): (torch.cat((torch.tensor(train_ppis.T[0]), go_links.T[0] if rem_go == False else torch.tensor([]),
            pos_annots.T[0] if rem_pos == False else torch.tensor([]))).numpy(),  torch.cat((torch.tensor(train_ppis.T[1]),
            go_links.T[1] if rem_go == False else torch.tensor([]), pos_annots.T[1] if rem_pos == False else torch.tensor([]))).numpy()),
            **({("node",'negative_link',"node"): (neg_annots.T[0].numpy(), neg_annots.T[1].numpy())} if rem_neg == False else {})}).to(device)
    
    train_dict.nodes["node"].data["x"] =  torch.randn(train_dict.num_nodes("node"), 128).to(device)

    return train_dict, test_dict


def make_heterographs(model, device, train_ppis, test_ppis, go_links, pos_annots, neg_annots):
    import dgl

    test_dict = dgl.heterograph({("protein",'has_ppi',"protein"): (test_ppis.T[0], test_ppis.T[1]),
                 **({("class",'has_link',"class"): (go_links.T[0].numpy(), go_links.T[1].numpy())} if 'has_link' in model.relations else {}),
                 **({("protein",'has_+annotation',"class"): (pos_annots.T[0].numpy(), pos_annots.T[1].numpy())} if 'has_+annotation' in model.relations else {}),
                 **({("protein",'has_-annotation',"class"): (neg_annots.T[0].numpy(), neg_annots.T[1].numpy())} if 'has_-annotation' in model.relations else {})}).to(device)
    
    test_dict.nodes["protein"].data["x"] =  torch.randn(test_dict.num_nodes("protein"), 128).to(device)
    test_dict.nodes["class"].data["x"]  = torch.randn(test_dict.num_nodes("class"), 128).to(device)

    train_dict = dgl.heterograph({("protein",'has_ppi',"protein"): (train_ppis.T[0], train_ppis.T[1]),
                 **({("class",'has_link',"class"): (go_links.T[0].numpy(), go_links.T[1].numpy())} if 'has_link' in model.relations else {}),
                 **({("protein",'has_+annotation',"class"): (pos_annots.T[0].numpy(), pos_annots.T[1].numpy())} if 'has_+annotation' in model.relations else {}),
                 **({("protein",'has_-annotation',"class"): (neg_annots.T[0].numpy(), neg_annots.T[1].numpy())} if 'has_-annotation' in model.relations else {})}).to(device)
    
    train_dict.nodes["protein"].data["x"] =  torch.randn(train_dict.num_nodes("protein"), 128).to(device)
    train_dict.nodes["class"].data["x"]  = torch.randn(train_dict.num_nodes("class"), 128).to(device) 
    return train_dict, test_dict



def masking(dict_edges, _idx, e_type, mask, device):

    dict_edges.edges[e_type].data[mask] = torch.Tensor([False]* len(dict_edges.edges(etype= e_type)[0])).to(device)
    dict_edges.edges[e_type].data[mask][_idx] = True
    dict_edges.edges[e_type].data[mask] = dict_edges.edges[e_type].data[mask].bool()

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