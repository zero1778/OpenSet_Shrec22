from pathlib import Path
from random import shuffle
import torch 
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

def split_trainval(data_root: str, train_ratio=0.8):
    train_root = Path(data_root) / "train"
    cates = sorted([d.stem for d in train_root.glob('*') if d.is_dir()])
    train_list, val_list = [], []
    for idx, cate in enumerate(cates):
        samples = [d for d in (train_root / cate).glob('*') if d.is_dir()]
        shuffle(samples)
        len_train = int(len(samples) * train_ratio)
        assert len_train > 0
        for _i, sample in enumerate(samples):
            if _i < len_train:
                train_list.append({'path': str(sample.absolute()),'label': idx})
            else:
                val_list.append({'path': str(sample.absolute()),'label': idx})
    return train_list, val_list


def res2tab(res: dict, n_palce=4):
    def dy_str(s, l):
        return  str(s) + ' '*(l-len(str(s)))
    min_size = 8
    k_str, v_str = '', ''
    for k, v in res.items():
        cur_len = max(min_size, len(k)+2)
        k_str += dy_str(f'{k}', cur_len) + '| '
        v_str += dy_str(f'{v:.4}', cur_len) + '| '
    return k_str, v_str

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


################################### metric #######################################
import scipy
import scipy.spatial
import numpy as np


def acc_score(y_true, y_pred, average="micro"):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if average == "micro": 
        # overall
        return np.mean(y_true == y_pred)
    elif average == "macro":
        # average of each class
        cls_acc = []
        for cls_idx in np.unique(y_true):
            cls_acc.append(np.mean(y_pred[y_true==cls_idx]==cls_idx))
        return np.mean(np.array(cls_acc))
    else:
        raise NotImplementedError


def map_score(dist_mat, lbl_a, lbl_b, metric='cosine'):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)


def map_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)


def nn_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        if lbl_a[i] == lbl_b[order[0]]:
            res.append(1)
        else:
            res.append(0)
    return np.mean(res)


def ndcg_score(dist_mat, lbl_a, lbl_b, k=100):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, n_b + 2)))
        dcg = np.cumsum([1.0/np.log2(idx+2) if lbl_a[i] == lbl_b[item] else 0.0 for idx, item in enumerate(order)])
        ndcg = (dcg/idcg)[k-1]
        res.append(ndcg)
    return np.mean(res)


def anmrr_score(dist_mat, lbl_a, lbl_b):
    # NG: number of ground truth images (target images) per query (vector)
    n_a, n_b = dist_mat.shape
    lbl_a, lbl_b = np.array(lbl_a), np.array(lbl_b)
    NG = np.array([(lbl_a[i]==lbl_b).sum() for i in range(lbl_a.shape[0])])
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        cur_NG = NG[i]
        K = min(4*cur_NG, 2*NG.max())
        order = s_idx[i]
        ARR = np.sum([(idx+1)/cur_NG if lbl_a[i] == lbl_b[order[idx]] else (K+1)/cur_NG for idx in range(cur_NG)])
        MRR = ARR - 0.5*cur_NG - 0.5
        NMRR = MRR / (K - 0.5*cur_NG + 0.5)
        res.append(NMRR)
    return np.mean(res)


if __name__ == "__main__":
    split_trainval('data/OS-MN40')
