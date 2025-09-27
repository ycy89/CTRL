from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from tqdm import tqdm


criterion_binary = torch.nn.BCELoss()


def accuracy(pred, target):
    """
    :param pred: list/np.array of predicted label (0, 1, 2, etc. depend on the number of classes)
    :param target: list/np.array of ground-truth label
    :return: the averaged accuracy
    """
    return (pred == target).mean()


def precision(pred, target):
    return average_precision_score(pred, target)


def f1_score_m(pred, target):
    return f1_score(pred, target)


def roc(pred, target):
    return roc_auc_score(pred, target)


def eval_one_epoch(model_t, data_eval, device, show_bar=False, test=False):
    val_acc, val_ap, val_f1, val_auc, val_loss = [], [], [], [], []
    weights_bs = []
    pbar = tqdm(enumerate(data_eval), disable=show_bar)
    if test:
        pbar.set_description(f"Test Epoch")
    else:
        pbar.set_description(f"Validation Epoch")

    with torch.no_grad():
        model_t = model_t.eval()
        for ind, data_batch in pbar:
            event = data_batch[0]
            event_neg = data_batch[1]
            ngb_batch = data_batch[2]
            for key in event:
                event[key] = event[key].to(device)
            for layer_k in ngb_batch:
                for key in ngb_batch[layer_k]:
                    for i in range(len(ngb_batch[layer_k][key])):
                        ngb_batch[layer_k][key][i] = ngb_batch[layer_k][key][i].to(device)
            for key in event_neg:
                event_neg[key] = event_neg[key].to(device)
            bs = event_neg['nid'].size()[0]
            # edge
            pos_score, neg_score = model_t(event, event_neg, ngb_batch)
            with torch.no_grad():
                pos_label = torch.ones(bs, dtype=torch.float, device=device)
                neg_label = torch.zeros(bs, dtype=torch.float, device=device)
            loss = criterion_binary(pos_score, pos_label)
            loss += criterion_binary(neg_score, neg_label)
            
            pbar.set_postfix(loss=loss.item())            
            pred_score = np.concatenate([pos_score.cpu().numpy(), neg_score.cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(bs), np.zeros(bs)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
            val_loss.append(loss.item())
            weights_bs.append(bs)
    acc = np.average(val_acc, weights=weights_bs)
    ap = np.average(val_ap, weights=weights_bs)
    f1 = np.average(val_f1, weights=weights_bs)
    auc = np.average(val_auc, weights=weights_bs)
    loss = np.average(val_loss, weights=weights_bs)
    return {"acc": acc, "ap": ap, "f1": f1, "auc": auc, "loss": loss}
