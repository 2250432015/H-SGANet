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
from utils import torch_DataLoader_OASIS, dice_coef, JacboianDet, topology_change, generate_grid
from thop import profile
from thop import clever_format
import nibabel as nib
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default="/public/ZYF/original_DIR/experleakyrlu+layer3/stage_120000.8003787293020918_best.pth",
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default="/public/ZYF/original_DIR/visualization1/",
                    help="path for saving images")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/public/wlj/datasets/Brain_MRI/affine_img_ordered/',
                    help="data path for training images")
parser.add_argument("--datapath_mask", type=str,
                    dest="datapath_mask",
                    default='/public/wlj/datasets/Brain_MRI/affine_seg_ordered/',
                    help="data path for training masks")
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--reproducible_seed', default=None)
parser.add_argument("--save_path", type=str,
                    dest="save_path", default="/public/ZYF/original_DIR/visual_new/",
                    help="path for saving images")
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

# def save_flow(I_img, savename):
#     affine = np.diag([1, 1, 1, 1])
#     new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
#     nib.save(new_img, savename)

def visulaize_deformation_field(flow_norm, save_path):
    # print(flow.shape)
    deformation_field = flow_norm.cpu()

    # 移除批次大小的维度
    deformation_field = deformation_field.squeeze(0)

    # 将通道维度移动到最后，以匹配NIfTI的期望格式 (X, Y, Z, 3)
    deformation_field = deformation_field.permute(1, 2, 3, 0).numpy()

    affine = np.eye(4)

    # 创建NIfTI图像对象
    nifti_img = nib.Nifti1Image(deformation_field, affine)

    # 保存为NIfTI文件
    nifti_img.to_filename(os.path.join(save_path, 'deformation_field.nii'))


    slice_100 = flow_norm[:,:,100,:,:].squeeze(0)

    slice_100 = (slice_100 - slice_100.min()) / (slice_100.max() - slice_100.min()) * 255
    # 创建一个PIL图像并保存
    slice_100_np = slice_100.cpu().numpy()
    slice_100_np = slice_100_np.astype('uint8')
    # print(slice_100_np.shape)
    # print(slice_100_np.dtype)
    image = Image.fromarray(slice_100_np)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(5.0)
    image.save(os.path.join(save_path, 'flow_100.jpg'))

def encode_deformation_field(deformation_field):
    """
    对整个形变场进行颜色编码并保存为NIfTI文件。

    Parameters:
        deformation_field (numpy.ndarray): 形变场数据，形状为 (depth, height, width, 3)。
        output_file (str): 输出的NIfTI文件路径。

    Returns:
        None
    """
    def encode_deformation_field(deformation_field):
        # 归一化形变场，使其范围在[0, 1]内

        deformation_field = deformation_field[0].detach().permute(1, 2, 3, 0).cpu().numpy()

        # 计算在 x 方向上的最大和最小值
        x_displacements = deformation_field[..., 0]
        max_x_displacement = round(np.max(x_displacements),2)
        min_x_displacement = round(np.min(x_displacements),2)

        # 计算在 y 方向上的最大和最小值
        y_displacements = deformation_field[..., 1]
        max_y_displacement = round(np.max(y_displacements),2)
        min_y_displacement = round(np.min(y_displacements),2)

        # 计算在 z 方向上的最大和最小值
        z_displacements = deformation_field[..., 2]
        max_z_displacement = round(np.max(z_displacements),2)
        min_z_displacement = round(np.min(z_displacements),2)

        # 计算在 x、y、z 方向上的平均位移
        mean_x_displacement = np.mean(x_displacements)
        mean_x_displacement = round(mean_x_displacement,2)
        mean_y_displacement = np.mean(y_displacements)
        mean_y_displacement = round(mean_y_displacement,2)
        mean_z_displacement = np.mean(z_displacements)
        mean_z_displacement = round(mean_z_displacement,2)
        # 计算在 x、y、z 方向上的位移标准差
        std_x_displacement = np.std(x_displacements)
        std_x_displacement = round(std_x_displacement,2)
        std_y_displacement = np.std(y_displacements)
        std_y_displacement = round(std_y_displacement,2)
        std_z_displacement = np.std(z_displacements)
        std_z_displacement = round(std_z_displacement,2)
        # 打印结果
        print("在 x 方向上的位移:[",  min_x_displacement,',', max_x_displacement, ']')
        print("在 y 方向上的位移:[",  min_y_displacement,',' ,max_y_displacement, ']')
        print("在 z 方向上的位移:[",  min_z_displacement, ',',max_z_displacement, ']')


        print("在 x 方向上的平均位移:", mean_x_displacement,"$\pm$",std_x_displacement)
        print("在 y 方向上的平均位移:", mean_y_displacement,"$\pm$",std_y_displacement)
        print("在 z 方向上的平均位移:", mean_z_displacement,"$\pm$",std_z_displacement)


        deformation_magnitude = np.linalg.norm(deformation_field, axis=3)

        print("位移平均值：", round(np.mean(deformation_magnitude),2),"$\pm$", round(np.std(deformation_magnitude),2))
        max_magnitude = np.max(deformation_magnitude)
        min_magnitude = np.min(deformation_magnitude)
        print("位移：[", round(min_magnitude,2),",", round(max_magnitude,2),']')

    encode_deformation_field(deformation_field)

def color_boundaries(segmentation):
    # 创建一个彩色图像
    colored_image = cv2.cvtColor(np.uint8(segmentation), cv2.COLOR_GRAY2BGR)

    # 获取唯一的标签值
    unique_labels = np.unique(segmentation)
    # print(unique_labels)
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    # 对每个标签的边界进行检测并用不同颜色标记
    for i, label in enumerate(unique_labels):
        if label == 0:  # 忽略背景
            continue

        mask = (segmentation == label).astype(np.uint8)
        # print(mask)
        edges = cv2.Canny(mask, threshold1=0, threshold2=0.9)  # 使用Canny边缘检测
        color = colors[i % len(colors)]  # 按顺序选择颜色
        colored_image[edges > 0] = color

    return colored_image

# vis_orginal函数的主要目的是将不同图像和分割结果以及它们的叠加效果保存为图像文件
def vis_orginal(fixed_img,moving_img,warped_moving,moving_seg,fixed_seg,flow_norm,save_path):
    # print(fixed_img.shape)
    fixed_img = fixed_img.detach().cpu().numpy()[0, 0, 100, :, :]
    moving_img = moving_img.detach().cpu().numpy()[0, 0, 100, :, :]
    warped_moving = warped_moving[100, :, :]

    moving_seg = moving_seg.detach().cpu().numpy()[0, 0, 100, :, :]
    unique_elements_x, counts = np.unique(moving_seg, return_counts=True)
    element_count_dict = dict(zip(unique_elements_x, counts))
    # print(element_count_dict)

    fixed_seg = fixed_seg.detach().cpu().numpy()[0, 0, 100, :, :]
    unique_elements_y, counts = np.unique(fixed_seg, return_counts=True)
    element_count_dict = dict(zip(unique_elements_y, counts))
    # print(element_count_dict)

    desire_values = np.union1d(unique_elements_x, unique_elements_y)
    flow_norm = flow_norm.detach().cpu().numpy()[0, 0, 100, :, :]
    closest_indices = np.argmin(np.abs(flow_norm[..., np.newaxis] - desire_values), axis=-1)
    def_out = desire_values[closest_indices]

    unique_elements_out, counts = np.unique(def_out, return_counts=True)
    element_count_dict = dict(zip(unique_elements_out, counts))
    # print(element_count_dict)

    # 处理分割结果
    desired_labels = [ 51, 54, 49]
    moving_seg[~np.isin(moving_seg,desired_labels)] = 0
    fixed_seg[~np.isin(fixed_seg,desired_labels)] = 0
    x_seg = cv2.medianBlur(moving_seg.astype(np.uint8), 3)
    y_seg = cv2.medianBlur(fixed_seg.astype(np.uint8), 3)

    # 处理位移场
    out = np.zeros((def_out.shape[0], def_out.shape[1]))

    for i in desired_labels:
        middle = def_out.copy()
        middle[middle != i] = 0
        middle = cv2.medianBlur(middle.astype(np.uint8), 5) # 使用5x5的中值滤波器
        out = np.where(middle != 0, middle, out)



    # 可视化固定图像
    seg_fix = color_boundaries(y_seg)
    plt.imshow(fixed_img,cmap='gray')
    plt.imshow(seg_fix, cmap='viridis', alpha=0.6)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'fix.png'),dpi=320)

    # 可视化移动图像
    seg_moving = color_boundaries(x_seg)
    plt.imshow(moving_img,cmap='gray')
    plt.imshow(seg_moving, cmap='viridis', alpha=0.6)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'moving.png'),dpi=320)

    # # 可视化移动后的图像
    # seg_moved = color_boundaries(out)
    # plt.imshow(warped_moving,cmap='gray')
    # plt.imshow(seg_moved, cmap='viridis', alpha=0.6)
    # plt.axis('off')
    # plt.savefig(os.path.join(save_path, 'moved.png'),dpi=320)

    # 可视化真值分割
    plt.imshow(y_seg,cmap='viridis')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'ground_truth.png'),dpi=320)

    # 可视化输入分割
    plt.imshow(x_seg,cmap='viridis')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'input.png'),dpi=320)

    # 可视化预测分割
    # plt.imshow(out,cmap='viridis')
    # plt.axis('off')
    # plt.savefig(os.path.join(save_path, 'pred.png'),dpi=320)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def comput_fig(img,save_path):
    img = img.detach().cpu().numpy()[0, 0, 100, :, :]

    plt.figure(dpi=200)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_path,'grid.png'), dpi=320)

def addimage(img_1, img_2, img_3,save_path):
    print(img_1.shape)
    img_1 = img_1[0, 0, 100, :, :].cpu()
    img_2 = img_2[0, 0, 100, :, :].cpu()
    img_3 = img_3[100, :, :]
    # 将灰度图转换为彩图  底用蓝色，顶用红色
    img = np.zeros((192, 224, 3))
    img[:, :, 0] = img_1*3 # 蓝色
    img1 = img
    img = np.zeros((192, 224, 3))
    img[:, :, 2] = img_2*3
    img2 = img  # 红色的浮动图像
    # 固定和浮动
    overlap1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    # 固定和配准后
    img[:, :, 2] = img_3*3
    img3 = img
    overlap2 = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)
    plt.imshow(overlap1)
    plt.axis('off')

    plt.savefig(os.path.join(save_path, 'overlap1'),dpi=320)
    plt.imshow(overlap2)
    plt.axis('off')

    plt.savefig(os.path.join(save_path, 'overlap2'),dpi=320)

def test():
    img_size = (160, 192, 224)
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
            # print(flow_norm.shape)
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
                "\r" + pair[0][0] + '->' + pair[1][0] + ' - Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}" - TC "{3:.9f}" - Regist_minus_Source "{4:.9f}" - Time "{5:.9f}"'.format(so_dice[-1], re_dice[-1], jc[-1], TC[-1], re_dice[-1] - so_dice[-1], runtime[-1]))
    aver_runtime = np.mean(runtime)
    aver_jc = np.mean(jc)
    aver_TC = np.mean(TC)
    aver_so_dice = np.mean(so_dice)
    aver_re_dice = np.mean(re_dice)
    print("============================================Final result=================================================")
    sys.stdout.write(
        "\r" + 'Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}" - TC "{3:.9f}" -Time "{4:.9f}"'.format(
            aver_so_dice, aver_re_dice, aver_jc, aver_TC, aver_runtime))
    visulaize_deformation_field(flow_norm, opt.save_path)
    vis_orginal(fixed_img,moving_img,warped_moving,moving_seg,fixed_seg,flow_norm,opt.save_path)
    addimage(fixed_img, moving_img, warped_moving, opt.save_path)

    grid_img = mk_grid_img(8, 1, img_size)
    def_grid = reg_model([grid_img.float(), flow_norm.permute(0, 4, 1, 2, 3).cuda()])
    comput_fig(def_grid, opt.save_path)

    encode_deformation_field(flow_norm)


    # save_flow(flow_cpu, savepath + '/flow' + '.nii')
    # save_img(warped_moving, savepath + '/warped_moving'  + '.nii')
    # save_img(warped_moving_mask, savepath + '/warped_moving_mask'  + '.nii')
    # save_img(fixed_img.data.cpu().numpy()[0, 0, :, :, :], savepath + '/fixed_' + '.nii')
    # save_img(moving_img.data.cpu().numpy()[0, 0, :, :, :], savepath + '/moving' + '.nii')
    # save_img(fixed_seg.data.cpu().numpy()[0, 0, :, :, :], savepath + '/fixed_mask' + '.nii')
    # save_img(moving_seg.data.cpu().numpy()[0, 0, :, :, :], savepath + '/moving_mask' + '.nii')
if __name__ == '__main__':
    imgshape = (160, 192, 224)
    range_flow = 0.4
    test()
