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
import custom_loss as custom_loss
from utils import clip_gradient
from tqdm import tqdm

from models.uda_att_1024 import UniModel_cls, UniModel_att, UniModel_base
from loaders.source import OSMN40_train
from utils import split_trainval, AverageMeter, res2tab, acc_score, map_score, op_copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

######### must config this #########
data_root = '/home/pbdang/Contest/SHREC22/OpenSet/data/OS-MN40'
typedata = "full"
####################################

# configure
n_class = 8
n_worker = 2
max_epoch = 150
batch_size = 6
learning_rate = 0.01
this_task = f"OS-MN40_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
print("DEVICES =", device)


# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts_source_b1_att_both'/this_task
save_dir.mkdir(parents=True, exist_ok=True)
out_file = open(os.path.join(save_dir, 'log_model.txt'), 'w')

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


def train(data_loader, netC, netB, netF, criterion, optimizer, epoch, iter_num ,max_iter):
    Log_str = f"Epoch {epoch}, Training..."
    out_file.write(Log_str + '\n')
    out_file.flush()
    print(Log_str)

    optim_cls, optim_centers = optimizer
    crt_cls, crt_tlc, w1, w2 = criterion
    # netC, netF = model
 
    # netF.train()
    netB.train()
    netC.train()
    loss_meter = AverageMeter()
    tpl_losses = AverageMeter()

    all_lbls, all_preds = [], []

    st = time.time()
    for i, (img, mesh, pt, vox, _, lbl) in enumerate(data_loader):
        iter_num += 1

        img = img.to(device)
        mesh = [d.to(device) for d in mesh]
        pt = pt.to(device)
        vox = vox.to(device)
        lbl = lbl.to(device)
        data = (img, mesh, pt, vox)


        out = netC(netB(netF(data)))

        cls_loss = crt_cls(out, lbl)
        tpl_loss, _ = crt_tlc(out, lbl)

        loss = w1 * cls_loss + w2 * tpl_loss

        
        lr_scheduler(optim_cls, iter_num=iter_num, max_iter=max_iter)

        optim_cls.zero_grad()
        optim_centers.zero_grad()

        loss.backward()
        clip_gradient(optim_centers, 0.05)

        optim_cls.step()
        optim_centers.step()


        _, preds = torch.max(out, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        try:
            tpl_losses.update(tpl_loss.item(), lbl.shape[0])
        except:
            tpl_losses.update(tpl_loss, lbl.shape[0])
        Log_str = f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}"
        out_file.write(Log_str + '\n')
        out_file.flush()

        print(Log_str)

    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    Log_str = f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}, Tpl_Loss: {tpl_losses.avg:4f}"
    out_file.write(Log_str + '\n')
    out_file.flush()

    print(Log_str)

    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")

    out_file.write(tab_head + '\n')
    out_file.flush()
    out_file.write(tab_data + '\n')
    out_file.flush()
    out_file.write('This Epoch Done!' + '\n')
    out_file.flush()


    return iter_num


def validation(data_loader, netC, netB, netF, epoch):
    print(f"Epoch {epoch}, Validation...")

    # netC, netF = model
    # netF.eval()
    netB.eval()
    netC.eval()
    all_lbls, all_preds = [], []
    fts = []

    st = time.time()
    for img, mesh, pt, vox, _, lbl in tqdm(data_loader):
        img = img.to(device)
        mesh = [d.to(device) for d in mesh]
        pt = pt.to(device)
        vox = vox.to(device)
        lbl = lbl.to(device)
        data = (img, mesh, pt, vox)

        out, ft = netC(netB(netF(data)), global_ft=True)
        # out_img, out_mesh, out_pt, out_vox = out
        # ft_img, ft_mesh, ft_pt, ft_vox = ft
        # out_obj = (out_img + out_mesh + out_pt + out_vox)/4

        _, preds = torch.max(out, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        fts.append(ft.detach().cpu().numpy())

    fts = np.concatenate(fts, axis=0)
    dist_mat = scipy.spatial.distance.cdist(fts, fts, "cosine")
    map_s = map_score(dist_mat, all_lbls, all_lbls)
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")

    Log_str = f"Epoch: {epoch}, Time: {time.time()-st:.4f}s"
    out_file.write(Log_str + '\n')
    out_file.flush()

    print(Log_str)

    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma,
        "map": map_s,
        "epoch": epoch,
        "name": "model: resnet50"
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")

    out_file.write(tab_head + '\n')
    out_file.flush()
    out_file.write(tab_data + '\n')
    out_file.flush()
    out_file.write('This Epoch Done!' + '\n')
    out_file.flush()
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
    # import pdb; pdb.set_trace()
    train_data = OSMN40_train('train', train_list,typedata=typedata)
    val_data = OSMN40_train('val', val_list,typedata=typedata)
    print(f'train samples: {len(train_data)}')
    print(f'val samples: {len(val_data)}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                             num_workers=n_worker)

    ### MODEL 
    print("Create new model")
    netF = UniModel_base(n_class)
    # ckpt = torch.load('./cache/ckpts_source_b1_att/OS-MN40_2022-03-04-10-27-54_att_pre/ckpt_F.pth')
    # netF.load_state_dict(ckpt['net'])
    netF = netF.cuda()
    netF = nn.DataParallel(netF)
    netF.to(device)
    # netF.eval()

    netB = UniModel_att()
    netB = netB.cuda()
    netB = nn.DataParallel(netB)
    netB.to(device)

    netC = UniModel_cls(n_class)
    netC = netC.cuda()
    netC = nn.DataParallel(netC)
    netC.to(device)
    
    # model = (netC, netF)

    ### LOSS FUNCTION
    # classification loss 
    crt_cls = nn.CrossEntropyLoss().cuda()
    # triplet center loss 
    crt_tlc = custom_loss.TripletCenterLoss(num_classes=n_class).cuda()
    crt_tlc = torch.nn.utils.weight_norm(crt_tlc, name='centers')
    criterion = [crt_cls, crt_tlc, 1, 0.3]

    ### OPTIMIZER
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]   
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}] 

    optim_cls = optim.AdamW(param_group)
    optim_cls = op_copy(optim_cls)
    optim_centers = optim.AdamW(crt_tlc.parameters(), lr=0.1)
    optimizer = (optim_cls, optim_centers)


    best_res, best_state = None, 0

    max_iter = max_epoch * len(train_loader)
    # interval_iter = max_iter // 10
    iter_num = 0
   
    for epoch in range(max_epoch):
        # train
        iter_num = train(train_loader, netC, netB, netF, criterion, optimizer, epoch, iter_num, max_iter)
        
        # lr_scheduler.step()
        # validation
        if epoch != 0 and epoch % 1 == 0:
            with torch.no_grad():
                val_state, res = validation(val_loader, netC, netB, netF, epoch)
            # save checkpoint
            if val_state > best_state:
                print("saving model...")
                best_res, best_state = res, val_state
                save_checkpoint(val_state, res, 'F', netF.module)
                save_checkpoint(val_state, res, 'B', netB.module)
                save_checkpoint(val_state, res, 'C', netC.module)

    print("\nTrain Finished!")
    tab_head, tab_data = res2tab(best_res)
    print(tab_head)
    print(tab_data)
    print(f'checkpoint can be found in {save_dir}!')

    out_file.write(tab_head + '\n')
    out_file.flush()
    out_file.write(tab_data + '\n')
    out_file.flush()
    out_file.write(f'checkpoint can be found in {save_dir}!' + '\n')
    out_file.flush()

    out_file.close()
    return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
