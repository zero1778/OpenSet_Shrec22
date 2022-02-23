import os
import time
import json
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader

from models.uda import UniModel_cls, UniModel_base
from loaders import OSMN40_train
from utils import split_trainval, AverageMeter, res2tab, acc_score, map_score, op_copy
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

######### must config this #########
data_root = '/home/pbdang/Contest/SHREC22/OpenSet/data/OS-MN40'
####################################

# configure
n_class = 8
n_worker = 4
max_epoch = 150
batch_size = 4
learning_rate = 0.01
this_task = f"OS-MN40_{time.strftime('%Y-%m-%d-%H-%M-%S')}"

# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts_source'/this_task
save_dir.mkdir(parents=True, exist_ok=True)

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def setup_seed():
    seed = time.time() % 1000_000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def train(data_loader, netF, netC, criterion, optimizer, epoch, iter_num ,max_iter):
    print(f"Epoch {epoch}, Training...")

    netF.train()
    netC.train()
    loss_meter = AverageMeter()
    all_lbls, all_preds = [], []

    st = time.time()
    for i, (img, mesh, pt, vox, lbl) in enumerate(data_loader):
        iter_num += 1
        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, mesh, pt, vox)

        out = netC(netF(data))
        out_img, out_mesh, out_pt, out_vox = out
        # import pdb; pdb.set_trace()
        out_obj = (out_img + out_mesh + out_pt + out_vox)/4
        loss = criterion(out_obj, lbl)
        # loss = criterion(out_pt, lbl) + criterion(out_vox, lbl)
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        print(f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}")

    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return iter_num


def validation(data_loader, netF, netC, epoch):
    print(f"Epoch {epoch}, Validation...")

    netF.eval()
    netC.eval()
    all_lbls, all_preds = [], []
    fts_img, fts_mesh, fts_pt, fts_vox = [], [], [], []

    st = time.time()
    for img, mesh, pt, vox, lbl in data_loader:
        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, mesh, pt, vox)

        out, ft = netC(netF(data), global_ft=True)
        out_img, out_mesh, out_pt, out_vox = out
        ft_img, ft_mesh, ft_pt, ft_vox = ft
        out_obj = (out_img + out_mesh + out_pt + out_vox)/4

        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        fts_img.append(ft_img.detach().cpu().numpy())
        fts_mesh.append(ft_mesh.detach().cpu().numpy())
        fts_pt.append(ft_pt.detach().cpu().numpy())
        fts_vox.append(ft_vox.detach().cpu().numpy())

    fts_img = np.concatenate(fts_img, axis=0)
    fts_mesh = np.concatenate(fts_mesh, axis=0)
    fts_pt = np.concatenate(fts_pt, axis=0)
    fts_vox = np.concatenate(fts_vox, axis=0)
    fts_uni = np.concatenate((fts_img, fts_mesh, fts_pt, fts_vox), axis=1)
    dist_mat = scipy.spatial.distance.cdist(fts_uni, fts_uni, "cosine")
    map_s = map_score(dist_mat, all_lbls, all_lbls)
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma,
        "map": map_s
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return map_s, res


def save_checkpoint(val_state, res, name_model, net: nn.Module):
    state_dict = net.state_dict()
    ckpt = dict(
        val_state=val_state,
        res=res,
        net=state_dict,
    )
    fname = 'ckpt_' + name_model + '.pth'
    torch.save(ckpt, str(save_dir / fname))
    fname = 'ckpt_' + name_model + '.meta'
    with open(str(save_dir / fname), 'w') as fp:
        json.dump(res, fp)


def main():
    setup_seed()
    # init train_loader and val_loader
    print("Loader Initializing...\n")
    # import pdb; pdb.set_trace()
    train_list, val_list = split_trainval(data_root)
    train_data = OSMN40_train('train', train_list)
    val_data = OSMN40_train('val', val_list)
    print(f'train samples: {len(train_data)}')
    print(f'val samples: {len(val_data)}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                             num_workers=n_worker)
    print("Create new model")
    netF = UniModel_base(n_class)
    netF = netF.cuda()
    netF = nn.DataParallel(netF)

    netC = UniModel_cls(n_class)
    netC = netC.cuda()
    netC = nn.DataParallel(netC)

    param_group = []
    # learning_rate = 0.01
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    # optimizer = optim.SGD(par, 0.01, momentum=0.9, weight_decay=5e-4)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, 
                                                        # eta_min=1e-4)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_res, best_state = None, 0

    max_iter = max_epoch * len(train_loader)
    # interval_iter = max_iter // 10
    iter_num = 0

    for epoch in range(max_epoch):
        # train
        print(iter_num)
        iter_num = train(train_loader, netF, netC, criterion, optimizer, epoch, iter_num, max_iter)
        
        # lr_scheduler.step()
        # validation
        if epoch != 0 and epoch % 1 == 0:
            with torch.no_grad():
                val_state, res = validation(val_loader, netF, netC, epoch)
            # save checkpoint
            if val_state > best_state:
                print("saving model...")
                best_res, best_state = res, val_state
                save_checkpoint(val_state, res, 'F', netF.module)
                save_checkpoint(val_state, res, 'C', netC.module)

    print("\nTrain Finished!")
    tab_head, tab_data = res2tab(best_res)
    print(tab_head)
    print(tab_data)
    print(f'checkpoint can be found in {save_dir}!')
    return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
