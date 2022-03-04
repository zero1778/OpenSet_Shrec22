from re import T
import torch
import torch.nn as nn
from .image import MVCNN_feat, MVCNN_cls
from .mesh import MeshNet_feat, MeshNet_cls
from .voxel import VoxNet_feat, VoxNet_cls
from .pointcloud import PointNetCls_feat, PointNetCls_cls
from .pointmlp import PointMLP
import torch.nn.utils.weight_norm as weightNorm
from utils import init_weights


class UniModel_base(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img_feat = MVCNN_feat(n_view=24, pretrained=True)
        self.model_mesh_feat = MeshNet_feat(n_class)
        self.model_pt_feat = PointNetCls_feat()
        self.model_vox_feat = VoxNet_feat()

    def forward(self, data):
        img, mesh, pt, vox = data
        # out_img, out_mesh, out_pt, out_vox = None, None, None, None

        # if num_obj[0] == 1:
        out_img = self.model_img_feat(img)

    # if num_obj[1] == 1:
        out_mesh = self.model_mesh_feat(mesh)

    # if num_obj[2] == 1:
        out_pt = self.model_pt_feat(pt)

    # if num_obj[3] == 1:
        out_vox = self.model_vox_feat(vox)

        return (out_img, out_mesh, out_pt, out_vox)

class UniModel_att(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = nn.MultiheadAttention(1024, 2, dropout=0.5)
        
    def forward(self, data):
        img, mesh, pt, vox  = data # B, D

        # 1, B, D
        img = img.unsqueeze(1).transpose(1,0)
        mesh = mesh.unsqueeze(1).transpose(1,0)
        pt = pt.unsqueeze(1).transpose(1,0)
        vox = vox.unsqueeze(1).transpose(1,0) 

        # import pdb; pdb.set_trace()
        # print("img = ",img.shape)
        # print("mesh = ",mesh.shape)
        # print("pt = ",pt.shape)
        # print("vox = ",vox.shape)
               
        # 4, B, D
        inps = torch.vstack((img, mesh))
        inps = torch.vstack((inps, pt))
        inps = torch.vstack((inps, vox))

        inps = inps.transpose(1,0) # B, 4, D
        x, _ = self.att(inps, inps, inps) # # B, 4, D
        x = x.mean(1) # B, D
        return x

            
class UniModel_cls(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.cls_net = weightNorm(nn.Linear(1024, n_class), name="weight")
        self.cls_net.apply(init_weights)

    def forward(self, data, global_ft=False):
        x = self.cls_net(data)
        if global_ft:
            return x, data
        else:
            return x


if __name__ == "__main__":
    pass
