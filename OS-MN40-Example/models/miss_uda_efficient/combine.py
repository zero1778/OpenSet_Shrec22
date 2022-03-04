from re import T
import torch
import torch.nn as nn
from .image import MVCNN_feat, MVCNN_cls
from .mesh import MeshNet_feat, MeshNet_cls
from .voxel import VoxNet_feat, VoxNet_cls
from .pointcloud import PointNetCls_feat, PointNetCls_cls
from .pointmlp import PointMLP


class UniModel_base(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img_feat = MVCNN_feat(n_view=24, pretrained=True)
        self.model_mesh_feat = MeshNet_feat(n_class)
        self.model_pt_feat = PointNetCls_feat()
        self.model_vox_feat = VoxNet_feat()

    def forward(self, data):
        img, mesh, pt, vox, num_obj  = data
        out_img, out_mesh, out_pt, out_vox = None, None, None, None

        # if num_obj[0] == 1:
        out_img = self.model_img_feat(img)

    # if num_obj[1] == 1:
        out_mesh = self.model_mesh_feat(mesh)

    # if num_obj[2] == 1:
        out_pt = self.model_pt_feat(pt)

    # if num_obj[3] == 1:
        out_vox = self.model_vox_feat(vox)

        return (out_img, out_mesh, out_pt, out_vox)
            
class UniModel_cls(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img_cls = MVCNN_cls(n_class)
        self.model_mesh_cls = MeshNet_cls(n_class)
        self.model_pt_cls = PointNetCls_cls(n_class)
        self.model_vox_cls = VoxNet_cls(n_class)

    def forward(self, data, global_ft=False):
        img, mesh, pt, vox = data

        # import pdb; pdb.set_trace()
        # bz = mesh.shape[0]

        # out_img, out_mesh, out_pt, out_vox = torch.zeros((bz, 8)), torch.zeros((bz, 8)), torch.zeros((bz, 8)), torch.zeros((bz, 8))
        if global_ft:
            # ft_img, ft_mesh, ft_pt, ft_vox = torch.zeros((bz, 256)), torch.zeros((bz, 256)), torch.zeros((bz, 256)), torch.zeros((bz, 256))
            # if num_obj[0] == 1:
            out_img, ft_img = self.model_img_cls(img, global_ft)

        # if num_obj[1] == 1:
            out_mesh, ft_mesh = self.model_mesh_cls(mesh, global_ft)

        # if num_obj[2] == 1:
            out_pt, ft_pt = self.model_pt_cls(pt, global_ft)

        # if num_obj[3] == 1:
            out_vox, ft_vox = self.model_vox_cls(vox, global_ft)

            return (out_img, out_mesh, out_pt, out_vox), (ft_img, ft_mesh, ft_pt, ft_vox)
        else:
            # if num_obj[0] == 1:
            out_img = self.model_img_cls(img)

        # if num_obj[1] == 1:
            out_mesh = self.model_mesh_cls(mesh)

        # if num_obj[2] == 1:
            out_pt = self.model_pt_cls(pt)

        # if num_obj[3] == 1:
            out_vox = self.model_vox_cls(vox)
                
            return (out_img, out_mesh, out_pt, out_vox)


if __name__ == "__main__":
    pass
