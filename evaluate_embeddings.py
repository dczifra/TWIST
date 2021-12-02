import torch

def make_snn(embs, labs, temp=0.1):

    # --Normalize embeddings
    embs = embs.div(embs.norm(dim=1).unsqueeze(1)).detach_()

    softmax = torch.nn.Softmax(dim=1)

    def snn(h, h_train=embs, h_labs=labs):
        # -- normalize embeddings
        h = h.div(h.norm(dim=1).unsqueeze(1))
        return softmax(h @ h_train.T / temp) @ h_labs

    return snn

def make_knn(embs, labs, k=5, norm=False):
    assert k>0, 'K must be positive'
    #embs = embs
    def knn(h, h_train=embs, h_labs=labs):
        if norm:
            h = h.div(h.norm(dim=1).unsqueeze(1))
        h = h.unsqueeze(0)
        dist_matrix = torch.cdist(h, h_train, p=2)[0]
        closest_indices = torch.topk(-dist_matrix, k, dim=-1).indices
        hits = torch.zeros_like(dist_matrix, dtype=torch.float32)
        for i in range(len(hits)):
            hits[i, closest_indices[i]] = 1
        return hits @ h_labs 
    return knn

def evaluate_embeddings(
    device,
    data_loader,
    encoder,
    prototypes,
    labs,
    embs,
    num_classes,
    temp=0.1,
):
    ipe = len(data_loader)

    embs = embs.to(device)
    labs = labs.to(device)

    # if prototypes is not None:
    #     compute_cluster_stats(embs=embs, labs=labs, prototypes=prototypes)

    # -- make labels one-hot
    num_classes = num_classes
    labs = labs.long().view(-1, 1)
    labs = torch.full((labs.size()[0], num_classes), 0., device=device).scatter_(1, labs, 1.)

    knn_embs = embs.unsqueeze(0).detach()
    normed_embs = embs.div(embs.norm(dim=1, keepdim=True)).unsqueeze(0).detach()

    evaluators = {
        'snn' : make_snn(embs, labs, temp),
        'knn1' : make_knn(knn_embs, labs, k=1),
        'knn5' : make_knn(knn_embs, labs, k=5),
        'nknn1' : make_knn(normed_embs, labs, k=1, norm=True),
        'nknn5' : make_knn(normed_embs, labs, k=5, norm=True),
    }

    results = {n:{} for n in evaluators.keys()}

    tops = {name : {"top1" : 0, "top5" : 0} for name in evaluators.keys()}
    total = 0
    for itr, data in enumerate(data_loader):
        imgs, labels = data[0].to(device), data[1].to(device)

        z = encoder(imgs)
        total += imgs.shape[0]
        for name, func in evaluators.items():
            snn_probs = func(z)
            tops[name]['top5'] += float(snn_probs.topk(5, dim=1).indices.eq(labels.unsqueeze(1)).sum())
            tops[name]['top1'] += float(snn_probs.max(dim=1).indices.eq(labels).sum())
            top1_acc = 100. * tops[name]['top1'] / total
            top5_acc = 100. * tops[name]['top5'] / total
            results[name]['top1'] = top1_acc
            results[name]['top5'] = top5_acc

    print(('Final results: snn (%.2f%% %.2f%%) knn1 (%.2f%% %.2f%%) knn5 (%.2f%% %.2f%%) ' + \
                'nknn1 (%.2f%% %.2f%%) nknn5 (%.2f%% %.2f%%)') \
                         % (results['snn']['top1'], results['snn']['top5'],
                            results['knn1']['top1'], results['knn1']['top5'],
                            results['knn5']['top1'], results['knn5']['top5'],
                            #results['knn20']['top1'], results['knn20']['top5'],
                            results['nknn1']['top1'], results['nknn1']['top5'],
                            results['nknn5']['top1'], results['nknn5']['top5'],
                            #results['nknn20']['top1'], results['nknn20']['top5'],
                            ))

    return results