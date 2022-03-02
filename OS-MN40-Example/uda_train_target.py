from cgi import test
import os
import time
import json
from cv2 import _InputArray_STD_ARRAY
import torch
import random
import numpy as np
import random
import pickle
import scipy.spatial
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import custom_loss as custom_loss
from scipy.spatial.distance import cdist
from utils import clip_gradient, cal_acc

from models.uda_efficient import UniModel_cls, UniModel_base
from loaders.target import OSMN40_train
from utils import split_trainval, AverageMeter, res2tab, acc_score, map_score, op_copy, Entropy
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

######### must config this #########
data_root = '/home/pbdang/Contest/SHREC22/OpenSet/data/OS-MN40'
typedata = "full"
output_dir_src = "./cache/ckpts_source_b1/OS-MN40_2022-02-28-13-14-02/"
####################################

# configure
n_class = 8
n_worker = 4
max_epoch = 10
batch_size = 2
lr = 0.01
lr_decay1 = 0.1
epsilon = 1e-5
w1 = 0.3 #cls_par
w2 = 0.2
ent_par = 1.0
# cls_par = 0.3

this_task = f"OS-MN40_{time.strftime('%Y-%m-%d-%H-%M-%S')}"

# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts_target_test'/this_task
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

def obtain_label(loader, netF, netC):
    start_test = True
    fts_img, fts_mesh, fts_pt, fts_vox = [], [], [], []
    all_label, all_output = [], []
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in tqdm(range(len(loader))):
            inputs_test = iter_test.next()
            img, mesh, pt, vox, _, lbl, _ = inputs_test

            img = img.cuda()
            mesh = [d.cuda() for d in mesh]
            pt = pt.cuda()
            vox = vox.cuda()
            lbl = lbl.cuda()
            data = (img, mesh, pt, vox)

            inputs = data
            labels = lbl

            out, feas = netC(netF(inputs), global_ft=True)

            out_img, out_mesh, out_pt, out_vox = out
            ft_img, ft_mesh, ft_pt, ft_vox = feas
            out_obj = (out_img + out_mesh + out_pt + out_vox)/4

            # _, preds = torch.max(out_obj, 1)
            # all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
            all_label.extend(lbl.squeeze().detach().cpu().numpy().tolist())
            all_output.extend(out_obj.squeeze().detach().cpu().numpy().tolist())

            fts_img.append(ft_img.detach().cpu().numpy())
            fts_mesh.append(ft_mesh.detach().cpu().numpy())
            fts_pt.append(ft_pt.detach().cpu().numpy())
            fts_vox.append(ft_vox.detach().cpu().numpy())
            
            # if start_test:
            #     all_fea = feas.float().cpu()
            #     all_output = outputs.float().cpu()
            #     all_label = labels.float()
            #     start_test = False
            # else:
            #     all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
            #     all_output = torch.cat((all_output, outputs.float().cpu()), 0)
            #     all_label = torch.cat((all_label, labels.float()), 0)

    fts_img = np.concatenate(fts_img, axis=0)
    fts_mesh = np.concatenate(fts_mesh, axis=0)
    fts_pt = np.concatenate(fts_pt, axis=0)
    fts_vox = np.concatenate(fts_vox, axis=0)
    all_fea = np.concatenate((fts_img, fts_mesh, fts_pt, fts_vox), axis=1)

    all_output = torch.FloatTensor(all_output)
    all_label = torch.FloatTensor(all_label)
    all_fea = torch.FloatTensor(all_fea)
    
    # import pdb; pdb.set_trace()

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1) / np.log(n_class)
    ent = ent.float().cpu()

    from sklearn.cluster import KMeans
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    all_fea = all_fea[known_idx,:]
    all_output = all_output[known_idx,:]
    predict = predict[known_idx]
    all_label_idx = all_label[known_idx]
    ENT_THRESHOLD = (kmeans.cluster_centers_).mean()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], "cosine")
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], "cosine")
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = n_class * np.ones(len(all_label), )
    guess_label[known_idx] = pred_label

    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)
    log_str = 'Threshold = {:.2f}, Accuracy = {:.2f}% -> {:.2f}%'.format(ENT_THRESHOLD, accuracy*100, acc*100)
    print(log_str)
    # import pdb; pdb.set_trace()
    return guess_label.astype('int'), ENT_THRESHOLD

    
def train(target_loader, netC, netF, criterion, optimizer, iter_num ,max_iter, mem_label):
    # print(f"Epoch {epoch}, Training...")

    optim_cls, optim_centers = optimizer
    crt_cls, crt_tlc, w1, w2 = criterion
    # netC, netF = model
 
    netF.train()
    netC.train()
    loss_meter = AverageMeter()
    # tpl_losses = AverageMeter()

    all_lbls, all_preds = [], []

    st = time.time()
    idx = 0
    tt = 0

    for i, (img, mesh, pt, vox, _, lbl, tar_idx) in enumerate(target_loader):
    # while iter_num < max_iter:
    #     try:
    #         inputs_test = iter_test.next()
    #     except:
    #         iter_test = iter(target_loader)
    #         inputs_test = iter_test.next()
        

        # if iter_num % interval_iter == 0:
        
        idx += 1
        iter_num += 1
        
        # img, mesh, pt, vox, _, lbl, tar_idx = inputs_test

        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, mesh, pt, vox)

        lr_scheduler(optim_cls, iter_num=iter_num, max_iter=max_iter)

        pred = mem_label[tar_idx]
        features_test = netF(data)
        out = netC(features_test)

        out_img, out_mesh, out_pt, out_vox = out
        # import pdb; pdb.set_trace()
        out_obj = (out_img + out_mesh + out_pt + out_vox)/4
        # loss = criterion(out_obj, lbl)
        # import pdb; pdb.set_trace()
        # cls_loss = crt_cls(out_obj, lbl)
        # tpl_loss, _ = crt_tlc(out_obj, lbl)

        # loss = w1 * cls_loss + w2 * tpl_loss

    
        softmax_out = nn.Softmax(dim=1)(out_obj)
        outputs_test_known = out_obj[pred < n_class, :]
        pred = pred[pred < n_class]

        
        if len(pred) == 0:
            print(tt)
            del features_test
            del out
            tt += 1
            continue

        loss = nn.CrossEntropyLoss()(outputs_test_known, pred)
        loss *= w1
        
        softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
        entropy_loss = torch.mean(Entropy(softmax_out_known))
        # if args.gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
        loss += entropy_loss * ent_par


    
        optim_cls.zero_grad()
        # optim_centers.zero_grad()

        loss.backward()
        # clip_gradient(optim_centers, 0.05)

        optim_cls.step()
        # optim_centers.step()


        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        # try:
        #     tpl_losses.update(tpl_loss.item(), lbl.shape[0])
        # except:
        #     tpl_losses.update(tpl_loss, lbl.shape[0])
        print(f"\t[{idx}/{len(target_loader)}], Loss {loss.item():.4f}")

        # if iter_num % interval_iter == 0 or iter_num == max_iter:
        #     netF.eval()
        #     acc_os1, acc_os2, acc_unknown, map_s = cal_acc(test_loader, netF, netC, out_file=out_file, flag=True, threshold=ENT_THRESHOLD)            
        #     log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%; mAP = {:.2f}%'.format("OpenSet", iter_num, max_iter, acc_os2, acc_os1, acc_unknown, map_s)
        #     out_file.write(log_str + '\n')
        #     out_file.flush()
        #     print(log_str+'\n')
        #     netF.train()

    # acc_mi = acc_score(all_lbls, all_preds, average="micro")
    # acc_ma = acc_score(all_lbls, all_preds, average="macro")
    # print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}, Tpl_Loss: {tpl_losses.avg:4f}")
    # res = {
    #     "overall acc": acc_mi,
    #     "meanclass acc": acc_ma
    # }
    # tab_head, tab_data = res2tab(res)
    # print(tab_head)
    # print(tab_data)
    # print("This Epoch Done!\n")
    return iter_num


def validation(data_loader, model, epoch):
    print(f"Epoch {epoch}, Validation...")

    netC, netF = model
    netF.eval()
    netC.eval()
    all_lbls, all_preds = [], []
    fts_img, fts_mesh, fts_pt, fts_vox = [], [], [], []

    st = time.time()
    for img, mesh, pt, vox, _, lbl in data_loader:
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
        "map": map_s,
        "epoch": epoch,
        "name": "model: EfficientNetB2"
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return map_s, res


def save_checkpoint(name_model, net: nn.Module):
    state_dict = net.state_dict()
    ckpt = dict(
        # val_state=val_state,
        # res=res,
        net=state_dict,
    )
    fname = 'ckpt_target_' + name_model + '.pth'
    torch.save(ckpt, str(save_dir / fname))
    # fname = 'ckpt_target_' + name_model + '.meta'
    # with open(str(save_dir / fname), 'w') as fp:
    #     json.dump(res, fp)


def main():
    setup_seed()
    # init train_loader and val_loader
    print("Loader Initializing...\n")
    # import pdb; pdb.set_trace()
    with open('/home/pbdang/Contest/SHREC22/OpenSet/data/OS-MN40/target_data.pickle', 'rb') as handle:
        test_list = pickle.load(handle) 
    

    # test_list = random.sample(test_list, 16)

    target_data = OSMN40_train('target', test_list,typedata=typedata)
    test_data = OSMN40_train('test', test_list,typedata=typedata)
    print(f'test samples: {len(test_data)}')
    print(f'target samples: {len(target_data)}')

    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size*4, shuffle=False,
                                             num_workers=n_worker)

    ### MODEL 
    print("Loading source model")
    netF = UniModel_base(n_class)
    modelpath = output_dir_src + '/ckpt_F.pth'   
    netF.load_state_dict(torch.load(modelpath)['net'])
    netF = netF.cuda()
    netF = nn.DataParallel(netF)

    netC = UniModel_cls(n_class)
    modelpath = output_dir_src + '/ckpt_C.pth' 
    netC.load_state_dict(torch.load(modelpath)['net'])
    netC = netC.cuda()
    netC = nn.DataParallel(netC)
    netC.eval() # Freeze netC

    model = (netC, netF)

    ### LOSS FUNCTION
    # classification loss 
    crt_cls = nn.CrossEntropyLoss().cuda()
    # triplet center loss 
    crt_tlc = custom_loss.TripletCenterLoss(num_classes=n_class).cuda()
    crt_tlc = torch.nn.utils.weight_norm(crt_tlc, name='centers')
    criterion = [crt_cls, crt_tlc, w1, w2]

    ############ OPTIMIZER ##################
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if lr_decay1 > 0:
            param_group += [{'params': v, 'lr': lr * lr_decay1}]
        else:
            v.requires_grad = False   
    #########################################

    optim_cls = optim.SGD(param_group)
    optim_cls = op_copy(optim_cls)
    optim_centers = optim.SGD(crt_tlc.parameters(), lr=0.1)
    optimizer = (optim_cls, optim_centers)


    best_res, best_state = None, 0

    max_iter = max_epoch * len(target_loader)
    interval_iter = max_iter // 15
    iter_num = 0

    out_file = open(os.path.join(save_dir, 'log_model.txt'), 'w')
    # out_file.write(print_args(args)+'\n')
    # out_file.flush()
    # for epoch in range(max_epoch):
        # train
    # iter_num = train(target_loader, test_loader, model, criterion, optimizer, iter_num, max_iter, interval_iter, out_file)
        
        # lr_scheduler.step()
        # validation
        # if epoch != 0 and epoch % 1 == 0:
        #     with torch.no_grad():
        #         val_state, res = validation(test_loader, model, epoch)
        #     # save checkpoint
        #     if val_state > best_state:
        #         print("saving model...")
                # best_res, best_state = res, val_state
    for epoch in range(max_epoch):

        netF.eval()
        mem_label, ENT_THRESHOLD = obtain_label(test_loader, netF, netC)
        mem_label = torch.from_numpy(mem_label).cuda()
        netF.train()
        # train
        iter_num = train(target_loader, netC, netF, criterion, optimizer, iter_num, max_iter, mem_label=mem_label)
        
        # lr_scheduler.step()
        # validation
        if epoch != 0 and epoch % 1 == 0:
            with torch.no_grad():
                netF.eval()
                acc_os1, acc_os2, acc_unknown, map_s = cal_acc(test_loader, netC, netF, out_file=out_file, flag=True, threshold=ENT_THRESHOLD)            
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}%; mAP = {:.2f}%'.format("OpenSet", iter_num, max_iter, acc_os2, acc_os1, acc_unknown, map_s)
                out_file.write(log_str + '\n')
                out_file.flush()
                print(log_str+'\n')
                netF.train()

            # save checkpoint
            # if val_state > best_state:
                print("saving model...")
                # best_res, best_state = res, val_state
                # save_checkpoint(val_state, res, 'F', netF.module)
                # save_checkpoint(val_state, res, 'C', netC.module)

                save_checkpoint('F', netF.module)
                save_checkpoint('C', netC.module)

    

    print("\nTrain Finished!")
    # # tab_head, tab_data = res2tab(best_res)
    # print(tab_head)
    # print(tab_data)
    print(f'checkpoint can be found in {save_dir}!')
    # return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
