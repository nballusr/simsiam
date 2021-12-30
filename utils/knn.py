# from https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch import nn


def kNN(epoch, model, train_dataloader, val_dataloader, K, device, sigma=10):
    """Extract features from validation split and search on train split features."""
    model.eval()

    train_size = train_dataloader.dataset.__len__()
    feat_dim = 2048

    train_features = torch.zeros([feat_dim, train_size], device=device)
    current_idx = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # forward
            features = model(inputs)
            features = nn.functional.normalize(features)
            train_features[:, current_idx:current_idx + batch_size] = features.data.t()
            current_idx += batch_size

        train_labels = torch.LongTensor(train_dataloader.dataset.targets).cuda()
        C = train_labels.max() + 1

    total = 0
    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            targets = targets.cuda(non_blocking=True)
            batch_size = inputs.size(0)
            features = model(inputs.to(device))

            dist = torch.mm(features, train_features)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.resize_(batch_size * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)

            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(
                torch.mul(retrieval_one_hot.view(batch_size, -1, C), yd_transform.view(batch_size, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()

            total += targets.size(0)

        print('Epoch [{}]\t'
              'Top 1: {:02f} Top5: {:02f}'.format(epoch, top1 * 100. / total, top5 * 100. / total))
    model.train()
    return top1 * 100. / total, top5 * 100. / total
