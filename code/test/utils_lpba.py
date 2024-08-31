import torch
import nibabel as nib
from itertools import permutations
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib.colors import LinearSegmentedColormap
"""----------------------------------DataLoader----------------------------------"""
class torch_Dataset_LPBA40(Dataset):
    def __init__(self, dataset_img, dataset_seg, mode):
        self.mode = mode
        '''读取OASIS数据集，每个配准pair，前一张图当作moving，后一张图当作fixed'''
        self.training_img_pair = list(permutations(dataset_img[0:40], 2))  # 1~255的组合
        self.training_seg_pair = list(permutations(dataset_seg[0:40], 2))  # 1~255的组合
        self.testing_img_pair = list((moving, atlas) for moving in dataset_img[0:-5] for atlas in dataset_img[-5:])
        self.testing_seg_pair = list((moving, atlas) for moving in dataset_seg[0:-5] for atlas in dataset_seg[-5:])
    def __len__(self):
        if self.mode == 'train':
            assert len(self.training_img_pair) == len(self.training_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.training_img_pair)
        elif self.mode == 'test':
            assert len(self.testing_img_pair) == len(self.testing_seg_pair), 'RaiseError: Img-pair number should be equal to Seg-pair number'
            return len(self.testing_img_pair)
    def __getitem__(self, item):
        if self.mode == 'train':
            self.moving_img = torch.from_numpy(nib.load(self.training_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            self.fixed_img = torch.from_numpy(nib.load(self.training_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            self.moving_seg = torch.from_numpy(nib.load(self.training_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
            self.fixed_seg = torch.from_numpy(nib.load(self.training_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
            pair = (self.training_img_pair[item][0][39:-7], self.training_img_pair[item][1][39:-7])
            return pair, self.moving_img.float(), self.fixed_img.float(), self.moving_seg.float(), self.fixed_seg.float()
        elif self.mode == 'test':
            self.moving_img = torch.from_numpy(nib.load(self.testing_img_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            self.fixed_img = torch.from_numpy(nib.load(self.testing_img_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29] / 255.0)
            self.moving_seg = torch.from_numpy(nib.load(self.testing_seg_pair[item][0]).get_fdata()[48:-48, 31:-33, 3:-29])
            self.fixed_seg = torch.from_numpy(nib.load(self.testing_seg_pair[item][1]).get_fdata()[48:-48, 31:-33, 3:-29])
            pair = (self.testing_img_pair[item][0][39:-7], self.testing_img_pair[item][1][39:-7])
            return pair, self.moving_img.float(), self.fixed_img.float(), self.moving_seg.float(), self.fixed_seg.float()

def torch_DataLoader_LPBA40(dataset_img, dataset_seg, mode, batch_size, random_seed=None):
    Dataset_LPBA40 = torch_Dataset_LPBA40(dataset_img, dataset_seg, mode)
    if random_seed is None:
        loader = DataLoader(dataset=Dataset_LPBA40,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False)
    else:
        g = torch.Generator()
        g.manual_seed(random_seed)
        '''或者也可以这样写'''
        # torch.manual_seed(random_seed)
        # g = torch.Generator()
        loader = DataLoader(dataset=Dataset_LPBA40,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False,
                            generator=g)
    return loader
"""----------------------------------DataLoader----------------------------------"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size=(160, 192, 224), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def dice_coef(x, y, labels):
    dices = np.zeros(shape=len(labels))
    for id, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(x == label, y == label))
        bottom = np.sum(x == label) + np.sum(y == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)
        dice = top/bottom
        dices[id] = dice
    return np.mean(dices)


def topology_change(x, y, labels):
    tcs = np.zeros(shape=len((labels)))
    for id, label in enumerate(labels):
        m1 = np.sum(x == label)
        m2 = np.sum(y == label)
        tcs[id] = m2/m1
    return np.mean(tcs)

def JacboianDet(y_pred, sample_grid):  # 雅各比
    save_path = "/public/ZYF/original_DIR/visual_new/"
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2
    Jdet11 = Jdet.cuda().data.cpu().numpy()

    flow_jc = Jdet11[0, :, :, :]
    mat_a = flow_jc[:, :, 112].transpose(1, 0)
    mat_a = cv2.flip(mat_a, -1)
    mat_s = flow_jc[80, :, :].transpose(1, 0)
    mat_s = cv2.flip(mat_s, -1)
    mat_c = flow_jc[:, 96, :].transpose(1, 0)
    mat_c = cv2.flip(mat_c, -1)
    clist = ['red', 'white', 'blue']
    nodes = [0, 0.28, 1]
    newcmap = LinearSegmentedColormap.from_list('chaos', list(zip(nodes, clist)))

    # plt.imshow(mat_a, vmin=-1, vmax=6, cmap=newcmap, interpolation='bicubic')
    # plt.colorbar()
    # plt.savefig(os.path.join(save_path, 'mat_a.png'), dpi=300)
    # plt.show()
    #
    # plt.imshow(mat_c, vmin=-1, vmax=6, cmap=newcmap, interpolation='bicubic')
    # plt.colorbar()
    # plt.savefig(os.path.join(save_path, 'mat_c.png'), dpi=300)
    # plt.show()
    #
    # plt.imshow(mat_s, vmin=-1, vmax=6, cmap=newcmap, interpolation='bicubic')
    # plt.colorbar()
    # plt.savefig(os.path.join(save_path, 'mat_s.png'), dpi=300, bbox_inches='tight')
    # plt.show()

    q = len(np.reshape(Jdet11, (1, -1))[0, :])
    score = np.sum(np.where(Jdet11 <= 0, 1, 0)) / q

    return score

# def save_img(I_img, savename):
#     affine = np.diag([1, 1, 1, 1])
#     new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
#     nib.save(new_img, savename)
#
# def save_flow(I_img, savename):
#     affine = np.diag([1, 1, 1, 1])
#     new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
#     nib.save(new_img, savename)
