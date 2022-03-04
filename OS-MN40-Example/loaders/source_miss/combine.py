import torch
from pathlib import Path
from torch.utils.data import Dataset
from .image import load_img
from .voxel import load_vox
from .mesh import load_mesh
from loaders.source import load_pt
import numpy as np


class OSMN40_train(Dataset):
    def __init__(self, phase, object_list, typedata="full"):
        super().__init__()
        assert phase in ('train', 'val', 'target', 'test')
        self.phase = phase
        self.object_list = object_list
        self.typedata = typedata

    def __getitem__(self, index):
        p = Path(self.object_list[index]['path'])
        lbl = self.object_list[index]['label']

        num_obj = np.loadtxt(p/'mask2.txt')

        img, mesh, pt, vox = torch.zeros((24, 3, 224, 224)), (torch.zeros((15, 500)), torch.zeros((500, 3), dtype=torch.int64)), torch.zeros((3, 2048)), torch.zeros((1, 64, 64, 64))
        
        if num_obj[0] == 1:
            # image
            img = load_img(p/'image', self.phase in ['train', 'target'], n_view=24)
            # print(img.shape)
        
        if num_obj[1] == 1:
            # mesh
            mesh = load_mesh(p/'mesh', self.phase in ['train', 'target'], typedata=self.typedata)

        if num_obj[2] == 1:
            # point cloud
            # print(p)
            pt = load_pt(p/'pointcloud',self.phase in ['train', 'target'], resolution=2048)
            # print(pt.shape)

        if num_obj[3] == 1:
            # voxel
            vox = load_vox(p/'voxel', self.phase in ['train', 'target'], resolution=64)
            # print(vox.shape)

        # print(img.shape, type(mesh), pt.shape, vox.shape)
        return img, mesh, pt, vox, num_obj, lbl

    def __len__(self):
        return len(self.object_list)


class OSMN40_retrive(Dataset):
    def __init__(self, object_list, typedata="full"):
        super().__init__()
        self.object_list = object_list
        self.typedata = typedata

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        
        p = Path(self.object_list[index])

        img, mesh, pt, vox = torch.zeros((24, 3, 224, 224)), (torch.zeros((15, 500)),torch.zeros((500, 3), dtype=torch.int64)), torch.zeros((3, 2048)), torch.zeros((1, 64, 64, 64))

        num_obj = np.loadtxt(p/'mask2.txt')

        if num_obj[0] == 1:
            # image
            img = load_img(p/'image', n_view=24)
            # print(img.shape)
        
        if num_obj[1] == 1:
            # mesh
            mesh = load_mesh(p/'mesh', typedata=self.typedata)

        if num_obj[2] == 1:
            # point cloud
            pt = load_pt(p/'pointcloud',resolution=2048)

        if num_obj[3] == 1:
            # voxel
            vox = load_vox(p/'voxel', resolution=64)

        # print(img.shape, type(mesh), pt.shape, vox.shape)
        return img, mesh, pt, vox, num_obj
    def __len__(self):
        return len(self.object_list)


if __name__ == '__main__':
    pass

