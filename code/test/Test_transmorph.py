import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser
import time
import torch
import os, utils, glob
import sys
import numpy as np
from tqdm import tqdm
from networks import model
from utils import torch_DataLoader_OASIS, dice_coef, JacboianDet, topology_change, generate_grid, save_flow, save_img
from thop import profile
from thop import clever_format
import nibabel as nib

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default="/public/ZYF/original_DIR/experleakyrlu+layer3/stage_120000.8003787293020918_best.pth",
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default="/public/ZYF/original_DIR/visualization1/",
                    help="path for saving images")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default="/public/wlj/datasets/Brain_MRI/affine_img_ordered/",
                    help="data path for training images")
parser.add_argument("--datapath_mask", type=str,
                    dest="datapath_mask",
                    default="/public/wlj/datasets/Brain_MRI/affine_seg_ordered/",
                    help="data path for training masks")
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--reproducible_seed', default=None)
opt = parser.parse_args()
savepath = opt.savepath
dataset_img = glob.glob(opt.datapath + '*.nii.gz')
dataset_img.sort(key=lambda x: int(x[54:-7]))  # 405张图，从1-405排序好了
dataset_seg = glob.glob(opt.datapath_mask + '*.nii.gz')
dataset_seg.sort(key=lambda x: int(x[54:-7]))  # 405张图，从1-405排序好了
assert len(dataset_seg) == len(dataset_img), 'Image number != Segmentation number'

def model_structure(model):
    imgsize = (160,192,224)
    x_in = torch.randn(1, 2, imgsize[0], imgsize[1], imgsize[2]).to('cuda')
    model.to('cuda')
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    flops, params = profile(model, inputs=(x_in,))
    flops, params = clever_format([flops, params], "%.3f")
    print('The total number of parameters: ' + str(num_para))

    print(f"FLOPs: {flops}, Params: {params}")
    print('-' * 90)

testing_loader = torch_DataLoader_OASIS(dataset_img, dataset_seg, 'test', opt.batch_size, random_seed=opt.reproducible_seed)
if not os.path.isdir(savepath):
    os.mkdir(savepath)
model = model().cuda()
model_structure(model)

def save_img(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)

def save_flow(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)

def test():
    print(model.state_dict().keys())
    model.load_state_dict(torch.load(opt.modelpath,map_location='cuda:0'))
    model.eval()
    reg_model = utils.register_model((160, 192, 224), 'nearest')
    reg_model.cuda()
    grid1 = generate_grid(imgshape)
    grid1 = torch.from_numpy(np.reshape(grid1, (1,) + grid1.shape)).cuda().float()

    # labels = [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162, 163, 164, 165, 166]
    labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]

    runtime = []
    jc = []
    TC = []
    so_dice = []
    re_dice = []
    for pair, moving_img, fixed_img, moving_seg, fixed_seg in tqdm(testing_loader, desc='当前测试进度'):
        with torch.no_grad():
            moving_img = moving_img.unsqueeze(0).cuda()
            fixed_img = fixed_img.unsqueeze(0).cuda()
            moving_seg = moving_seg.unsqueeze(0).cuda()
            fixed_seg = fixed_seg.unsqueeze(0).cuda()
            x_in = torch.cat((moving_img, fixed_img), dim=1)
            start_time = time.time()
            output = model(x_in)
            end_time = time.time()
            warped_moving = reg_model([moving_img.cuda().float(), output[1].cuda()]).data.cpu().numpy()[0, 0, :, :, :]
            flow_norm = output[1].permute(0, 2, 3, 4, 1)
            jc.append(JacboianDet(flow_norm, grid1))
            runtime.append(end_time - start_time)
            warped_moving_mask = reg_model([moving_seg.cuda().float(), output[1].cuda()]).data.cpu().numpy()[0, 0, :, :, :]
            flow_cpu = output[1].data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            TC.append(topology_change(moving_seg.data.cpu().numpy()[0, 0, :, :, :], warped_moving_mask, labels))
            so_dice.append(
                dice_coef(moving_seg.data.cpu().numpy()[0, 0, :, :, :], fixed_seg.data.cpu().numpy()[0, 0, :, :, :],
                          labels))
            re_dice.append(dice_coef(warped_moving_mask, fixed_seg.data.cpu().numpy()[0, 0, :, :, :], labels))
            print(
                "\r" + pair[0][0] + '->' + pair[1][0] + ' - Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}" - TC "{3:.9f}" -Time "{4:.9f}"'.format(
                    so_dice[-1], re_dice[-1], jc[-1], TC[-1], runtime[-1]))
    aver_runtime = np.mean(runtime)
    aver_jc = np.mean(jc)
    aver_TC = np.mean(TC)
    aver_so_dice = np.mean(so_dice)
    aver_re_dice = np.mean(re_dice)
    print("============================================Final result=================================================")
    sys.stdout.write(
        "\r" + 'Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}" - TC "{3:.9f}" -Time "{4:.9f}"'.format(
            aver_so_dice, aver_re_dice, aver_jc, aver_TC, aver_runtime))
    save_flow(flow_cpu, savepath + '/flow' + '.nii')
    save_img(warped_moving, savepath + '/warped_moving'  + '.nii')
    save_img(warped_moving_mask, savepath + '/warped_moving_mask'  + '.nii')
    save_img(fixed_img.data.cpu().numpy()[0, 0, :, :, :], savepath + '/fixed_' + '.nii')
    save_img(moving_img.data.cpu().numpy()[0, 0, :, :, :], savepath + '/moving' + '.nii')
    save_img(fixed_seg.data.cpu().numpy()[0, 0, :, :, :], savepath + '/fixed_mask' + '.nii')
    save_img(moving_seg.data.cpu().numpy()[0, 0, :, :, :], savepath + '/moving_mask' + '.nii')
if __name__ == '__main__':
    imgshape = (160, 192, 224)
    range_flow = 0.4
    test()
