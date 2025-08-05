import torch, pandas, numpy, os
from models import *
from evaluation import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def train(model, device, train_ppis, train_neg_ppis, train_pos_labels, train_neg_labels, 
          test_ppis, test_neg_ppis, go_links, pos_annots, neg_annots, rem_go, rem_pos, rem_neg, epochs=300):
    print("Training model: ", model.name, flush=True)
    train_dict_, test_dict = make_homographs( device, train_ppis, 
                                          test_ppis, go_links, pos_annots, neg_annots, rem_go, rem_pos, rem_neg)
    best_f1 = 0
    optimizer = torch.optim.Adam(model.parameters())
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=10, shuffle=True, random_state=0)
    print("Using K-Fold Cross Validation with 10 folds", flush=True)
    partitions = [(train_idx, val_idx) for i, (train_idx, val_idx) in enumerate(skf.split(train_neg_ppis, train_neg_labels))]
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_ppis, train_pos_labels)):
        print("masking train and validation", flush=True)
        train_dict = masking(train_dict_, train_idx, "has_edge", "train", device)
        val_dict = masking(train_dict_, val_idx, "has_edge", "valid", device)
        train_npairs, val_npairs = train_neg_ppis[partitions[fold][0]], train_neg_ppis[partitions[fold][1]]
        train_pos_labels_, train_neg_labels_ = train_pos_labels[train_idx], train_neg_labels[partitions[fold][0]]
        val_pos_labels_, val_neg_labels_ = train_pos_labels[val_idx], train_neg_labels[partitions[fold][1]]

        print("On fold number: ", fold + 1, flush=True)
        model.reset_parameters()
        loss = torch.nn.BCELoss()
        for epoch in range(1, epochs +1):

            model.train()
            optimizer.zero_grad()
            logits = torch.squeeze(model(train_dict, train_npairs, e_types = ["has_edge"], mask = "train"))
            loss_ = loss(logits, torch.cat((train_pos_labels_,train_neg_labels_)).to(device))
            loss_.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss_.item()}', flush=True)

            if epoch % 5 == 0:
                acc, f1, pr, re, roc_auc = evaluate_dgl(model, val_dict, torch.cat((val_pos_labels_, val_neg_labels_)),
                                                     val_npairs, e_types = ["has_edge"], mask_ = "valid", device_=device)

                if f1 > best_f1 and epoch > (epochs+1)/3:
                    best_epoch = epoch
                    best_acc = acc
                    best_f1 = f1
                    torch.save(model.state_dict(), 'best_model_{}.pth'.format(model.name[2:]))
        print(f'Best Epoch: {best_epoch}, Best Accuracy: {best_acc}, Best F1: {best_f1}', flush=True)
        
    model.load_state_dict(torch.load('best_model_{}.pth'.format(model.name[2:])))
    model.train()

    print("Final Training  . . .", flush=True)
    for epoch in range(epochs//2):
        optimizer.zero_grad()
        all_idx = torch.cat((torch.tensor(train_idx).to(torch.int64), torch.tensor(val_idx).to(torch.int64)))
        train_dict_ = masking(train_dict_, all_idx, "has_edge", "all", device)
        logits = torch.squeeze(model(train_dict_.to(device), train_neg_ppis, e_types=["has_edge"], mask="all"))
        loss_ = loss(logits, torch.cat((train_pos_labels, train_neg_labels)).to(device))
        loss_.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print("Testing final model . . .", flush=True)
    test_dict = masking(test_dict, [i for i in range(len(test_ppis))], "has_edge", "all", device)
    test_acc, test_f1, test_pr, test_re, test_roc_auc = evaluate_dgl(model, test_dict, torch.cat((torch.ones(len(test_ppis)),
                        torch.zeros(len(test_neg_ppis)))), test_neg_ppis, e_types=["has_edge"], mask_="all", device_=device, save=True, homo=True)
    return test_acc, test_f1, test_pr, test_re, test_roc_auc


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=250)
    parser.add_argument('-go_links', type=str, default='melanogaster_data/idx_d_melanogaster_go_links.csv')
    parser.add_argument('-ppis', type=str, default='melanogaster_data/idx_d_melanogaster_filtered_d_melanogaster_ppi.csv')
    parser.add_argument('-positive_annotations', type=str, default='melanogaster_data/idx_d_melanogaster_filtered_d_melanogaster_pos.csv')
    parser.add_argument('-negative_annotations', type=str, default='melanogaster_data/idx_d_melanogaster_filtered_d_melanogaster_neg.csv')
    parser.add_argument('-direct_contradictions', type=str, default='data/idx_contradictions.csv')
    parser.add_argument('-hierarchical_contradictions', type=str, default='data/idx_hierarchical_contradictions.csv')
    parser.add_argument('-remove_contradictions', type=int, help='Remove contradictions: if =0 do not remove, if =1 '
    ' remove direct contradictions, if =2  remove hierarchical contradictions', default=0, choices=[0, 1, 2])
    parser.add_argument('-remove_negative_annotations', action='store_true', help='Remove negative annotations')
    parser.add_argument('-remove_positive_annotations', action='store_true', help='Remove positive annotations')
    parser.add_argument('-no_ontology', action='store_true', help='Do not use ontology "subclass of" relations')
    parser.add_argument('-model', type=str, default="GCN", choices=["GCN", "SAGE", "GAT"])
    args = parser.parse_args()
    args.model = eval(args.model + "_dgl")
    
    go_links = torch.tensor(pandas.read_csv(args.go_links, header=None, index_col=False).values)
    ppis = pandas.read_csv(args.ppis, header=None, index_col=False).values
    neg_ppis = generate_negative_edges(ppis)

    pos_annots = torch.tensor(pandas.read_csv(args.positive_annotations, header=None, index_col=False).values)
    neg_annots = torch.tensor(pandas.read_csv(args.negative_annotations, header=None, index_col=False).values)

    if args.remove_contradictions == 1: pos_annots = pos_annots[~numpy.isin(pos_annots, 
                        pandas.read_csv(args.direct_contradictions, header=None, index_col=False).values).all(1)]
    if args.remove_contradictions == 2: pos_annots = pos_annots[~numpy.isin(pos_annots, numpy.concatenate(
        (pandas.read_csv(args.direct_contradictions, header=None, index_col=False).values,
        pandas.read_csv(args.hierarchical_contradictions, header=None, index_col=False).values), axis=0)).all(1)]

    if os.path.exists('melanogaster_data/train_edge_partition.pt'):
        saved = torch.load('melanogaster_data/train_edge_partition.pt')
        train_ppis, train_neg_ppis = saved["train_ppis"], saved["train_neg_ppis"]
        used_ppis = set(map(tuple, train_ppis.tolist()))
        used_neg_ppis = set(map(tuple, train_neg_ppis.tolist()))
        test_ppis = numpy.array([e for e in ppis if tuple(e) not in used_ppis])
        test_neg_ppis = numpy.array([e for e in neg_ppis if tuple(e) not in used_neg_ppis])
    else: train_ppis, test_ppis, train_neg_ppis, test_neg_ppis = mask_edges(ppis, neg_ppis, partition=0.9)

    train_pos_labels, train_neg_labels = torch.ones(len(train_ppis)), torch.zeros(len(train_neg_ppis))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = args.model(relations=[("node", "has_edge","node")]).to(device)

    acc, f1, pr, re, roc_auc = train(model, device, train_ppis, train_neg_ppis, train_pos_labels, train_neg_labels, test_ppis, 
                test_neg_ppis, go_links, pos_annots, neg_annots, args.no_ontology, args.remove_positive_annotations, args.remove_negative_annotations, args.epochs)
    print(f'Final Accuracy: {acc}, Final F1: {f1}, Final Precision: {pr}, Final Recall: {re}, Final ROC AUC: {roc_auc}', flush=True)



if __name__ == '__main__':
    main()