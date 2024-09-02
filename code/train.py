import sys
import time
import os
import glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from networks import model
from utils import torch_DataLoader_OASIS, dice_coef, JacboianDet, topology_change, generate_grid
import loss, utils
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(description='param')
parser.add_argument('--img_dir',
                    default="data_dir/", type=str)
parser.add_argument('--seg_dir',
                    default="data_dir/", type=str)
parser.add_argument('--GPU_id', default='0', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--reproducible_seed', default=None)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--smooth_loss', default=0.002, type=float)
args = parser.parse_args()
warnings.filterwarnings('ignore')

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

def train():
    dataset_img = glob.glob(args.img_dir + '*.nii.gz')
    dataset_img.sort(key=lambda x: int(x[54:-7]))  # 405张图，从1-405排序好了
    dataset_seg = glob.glob(args.seg_dir + '*.nii.gz')
    dataset_seg.sort(key=lambda x: int(x[54:-7]))  # 405张图，从1-405排序好了
    assert len(dataset_seg) == len(dataset_img), 'Image number != Segmentation number'

    training_loader = torch_DataLoader_OASIS(dataset_img, dataset_seg, 'train', args.batch_size, random_seed=args.reproducible_seed)
    testing_loader = torch_DataLoader_OASIS(dataset_img, dataset_seg, 'test', args.batch_size, random_seed=args.reproducible_seed)
    labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
    grid1 = generate_grid([160, 192, 224])
    grid1 = torch.from_numpy(np.reshape(grid1, (1,) + grid1.shape)).cuda().float()
    best_dice = 0
    model = model().cuda()
    model_structure(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    criterion = torch.nn.MSELoss()
    criterions = [criterion]
    criterions += [loss.Grad3d(penalty='l2')]
    weights = [1, args.smooth_loss]
    current_iter = 0
    load_model = True
    if load_model is True:
        model_path = "./experleakyrlu+layer3/stage_120000.8003787293020918_best.pth"
        print("Loading weight: ", model_path)
        current_iter = 62000
        model.load_state_dict(torch.load(model_path))
    loss_all = utils.AverageMeter()

    for pair, moving_img, fixed_img, moving_seg, fixed_seg in tqdm(training_loader, colour = 'blue', desc='当前配准进度'):
        model.train()
        pair = tuple(int(x[0]) for x in pair)
        moving_img = moving_img.unsqueeze(0).cuda()
        fixed_img = fixed_img.unsqueeze(0).cuda()
        input_img = torch.cat([moving_img, fixed_img], 1)
        time1_start = time.time()
        output = model(input_img)
        time1 = time.time() - time1_start
        reg_model = utils.register_model((160, 192, 224), 'nearest')
        reg_model.cuda()
        loss = 0
        loss_vals = []

        for n, loss_function in enumerate(criterions):
            curr_loss = loss_function(output[n], fixed_img) * weights[n]
            loss_vals.append(curr_loss)
            loss += curr_loss
        loss_all.update(loss.item(), fixed_img.numel())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input_img
        del output
        # flip fixed and moving images
        loss = 0
        input_img = torch.cat([fixed_img, moving_img], dim=1)
        time2_start = time.time()
        output = model(input_img)
        time2 = time.time() - time1_start
        for n, loss_function in enumerate(criterions):
            curr_loss = loss_function(output[n], moving_img) * weights[n]
            if (n == 0):
                curr_loss = curr_loss
            loss_vals[n] += curr_loss
            loss += curr_loss
        loss_all.update(loss.item(), moving_img.numel())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_dir = './experleakyrlu+layer3'
        print('\n' + str(pair[0]) + '->' + str(pair[1]) + '  loss {:.6f}, Img Sim: {:.6f}, Reg: {:.6f}, time: {:.6f}'.format(loss.item(), loss_vals[0].item() / 2,
                                                               loss_vals[1].item() / 2, (time1 + time2) / 2))
        if (current_iter % 2000 == 0 and current_iter != 0):
            modelname = model_dir + '/' + "stage_" + str(current_iter) + '.pth'
            torch.save(model.state_dict(), modelname)
            runtime = []
            jc = []
            TC = []
            so_dice = []
            re_dice = []
            for pair, moving_img, fixed_img, moving_seg, fixed_seg in tqdm(testing_loader, colour = 'green',desc='当前测试进度'):
                with torch.no_grad():
                    moving_img = moving_img.unsqueeze(0).cuda()
                    fixed_img = fixed_img.unsqueeze(0).cuda()
                    moving_seg = moving_seg.unsqueeze(0).cuda()
                    fixed_seg = fixed_seg.unsqueeze(0).cuda()
                    x_in = torch.cat((moving_img, fixed_img), dim=1)
                    start_time = time.time()
                    output = model(x_in)
                    end_time = time.time()
                    X_Y = reg_model([moving_img.cuda().float(), output[1].cuda()]).data.cpu().numpy()[0, 0, :, :, :]
                    flow_norm = output[1].permute(0, 2, 3, 4, 1)
                    jc.append(JacboianDet(flow_norm, grid1))
                    runtime.append(end_time - start_time)
                    X_Y1 = reg_model([moving_seg.cuda().float(), output[1].cuda()]).data.cpu().numpy()[0, 0, :, :, :]
                    F_X_Y_cpu = output[1].data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
                    TC.append(topology_change(moving_seg.data.cpu().numpy()[0, 0, :, :, :], X_Y1, labels))
                    so_dice.append(dice_coef(moving_seg.data.cpu().numpy()[0, 0, :, :, :], fixed_seg.data.cpu().numpy()[0, 0, :, :, :], labels))
                    re_dice.append(dice_coef(X_Y1, fixed_seg.data.cpu().numpy()[0, 0, :, :, :], labels))
                    sys.stdout.write(
                        "\r" + pair[0][0] + '->' + pair[1][0] + ' - Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}" - TC "{3:.9f}" -Time "{4:.9f}"'.format(
                            so_dice[-1] , re_dice[-1], jc[-1], TC[-1], runtime[-1]))
            aver_runtime = np.mean(runtime)
            aver_jc = np.mean(jc)
            aver_TC = np.mean(TC)
            aver_so_dice = np.mean(so_dice)
            aver_re_dice = np.mean(re_dice)
            if aver_re_dice > best_dice:
                best_dice = aver_re_dice
                modelname = model_dir + '/' + "stage_" + str(current_iter) + str(aver_re_dice) + "_best" + '.pth'
                torch.save(model.state_dict(), modelname)
            print(
                "============================================Final result of number " + str(current_iter) + "=================================================")
            print(
                "\r" + 'Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}" - TC "{3:.9f}" -Time "{4:.9f}"'.format(
                    aver_so_dice, aver_re_dice, aver_jc, aver_TC, aver_runtime))
        current_iter += 1

if __name__ == '__main__':
    train()