import os
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from sewar import rmse, ssim, sam, psnr
from pathlib import Path
from torch.nn.functional import interpolate
import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import PatchSet, load_image_pair, transform_image
from models.mynet import FusionNet, NLayerDiscriminator, GANLoss
from utils import AverageMeter, log_csv
from ssim import msssim


def uiqi(im1, im2, block_size=64, return_map=False):
    if len(im1.shape)==3:
        return np.array([uiqi(im1[:,:,i], im2[:,:,i], block_size, return_map=return_map) for i in range(im1.shape[2])])
    delta_x = np.std(im1, ddof=1)
    delta_y = np.std(im2, ddof=1)
    delta_xy = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / (im1.shape[0] * im1.shape[1] - 1)
    mu_x = np.mean(im1)
    mu_y = np.mean(im2)
    q1 = delta_xy / (delta_x * delta_y)
    q2 = 2 * mu_x * mu_y / (mu_x ** 2 + mu_y ** 2)
    q3 = 2 * delta_x * delta_y / (delta_x ** 2 + delta_y ** 2)
    q = q1 * q2 * q3
    return q


def test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE):
    cur_result = {}
    model.eval()

    PATCH_STRIDE = PATCH_SIZE // 2
    end_h = (IMAGE_SIZE[0] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    end_w = (IMAGE_SIZE[1] - PATCH_STRIDE) // PATCH_STRIDE * PATCH_STRIDE
    h_index_list = [i for i in range(0, end_h, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, end_w, PATCH_STRIDE)]
    if (IMAGE_SIZE[0] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if (IMAGE_SIZE[1] - PATCH_STRIDE) % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    final_ssim = 0.0
    for cur_date in test_dates:
        cur_day = int(cur_date.split('_')[1])
        if cur_day == 347:
            for ref_date in test_dates:
                ref_day = int(ref_date.split('_')[1])
                if ref_day != cur_day:
                    images = load_image_pair(opt.root_dir, cur_date, ref_date)

                    output_image = np.zeros(images[1].shape)
                    image_mask = np.ones(images[1].shape)
                    for i in range(4):
                        negtive_mask = np.where(images[i] < 0)
                        inf_mask = np.where(images[i] > 10000.)
                        image_mask[negtive_mask] = 0
                        image_mask[inf_mask] = 0

                    for i in range(len(h_index_list)):
                        for j in range(len(w_index_list)):
                            h_start = h_index_list[i]
                            w_start = w_index_list[j]

                            input_lr = images[0][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            target_hr = images[1][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            ref_lr = images[2][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]
                            ref_hr = images[3][:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE]

                            flip_num = 0
                            rotate_num0 = 0
                            rotate_num = 0
                            input_lr, im_mask = transform_image(input_lr, flip_num, rotate_num0, rotate_num)
                            ref_lr, im_mask = transform_image(ref_lr, flip_num, rotate_num0, rotate_num)
                            ref_hr, im_mask = transform_image(ref_hr, flip_num, rotate_num0, rotate_num)

                            input_lr = input_lr.unsqueeze(0).cuda()
                            ref_lr = ref_lr.unsqueeze(0).cuda()
                            ref_hr = ref_hr.unsqueeze(0).cuda()

                            output = model(ref_lr, ref_hr, input_lr)
                            output = output.squeeze()

                            # 确定填补图像的四个坐标
                            h_end = h_start + PATCH_SIZE
                            w_end = w_start + PATCH_SIZE
                            cur_h_start = 0
                            cur_h_end = PATCH_SIZE
                            cur_w_start = 0
                            cur_w_end = PATCH_SIZE

                            if i != 0:
                                h_start = h_start + PATCH_SIZE // 4
                                cur_h_start = PATCH_SIZE // 4

                            if i != len(h_index_list) - 1:
                                h_end = h_end - PATCH_SIZE // 4
                                cur_h_end = cur_h_end - PATCH_SIZE // 4

                            if j != 0:
                                w_start = w_start + PATCH_SIZE // 4
                                cur_w_start = PATCH_SIZE // 4

                            if j != len(w_index_list) - 1:
                                w_end = w_end - PATCH_SIZE // 4
                                cur_w_end = cur_w_end - PATCH_SIZE // 4

                            output_image[:, h_start: h_end, w_start: w_end] = \
                                output[:, cur_h_start: cur_h_end, cur_w_start: cur_w_end].cpu().detach().numpy()  #重叠部分各取一半

                    real_im = images[1] * 0.0001 * image_mask
                    real_output = (output_image + 1) * 0.5 * image_mask  #这是什么意思

                    for real_predict in [real_output]:
                        cur_result['rmse'] = []
                        cur_result['ssim'] = []
                        cur_result['cc'] = []
                        cur_result['uiqi'] = []
                        cur_result['ergas'] = 0

                        for i in range(6):
                            cur_result['rmse'].append(rmse(real_im[i], real_predict[i]))
                            cur_result['ssim'].append(ssim(real_im[i], real_predict[i], MAX=1.0)[0])
                            cur_result['uiqi'].append(uiqi(real_im[i], real_predict[i]))
                            cur_cc = np.sum((real_im[i] - np.mean(real_im[i])) * (real_predict[i] - np.mean(real_predict[i]))) / \
                                     np.sqrt((np.sum(np.square(real_im[i] - np.mean(real_im[i])))) * np.sum(
                                         np.square(real_predict[i] - np.mean(real_predict[i]))) + 1e-100)
                            cur_result['cc'].append(cur_cc)

                            cur_result['ergas'] += rmse(real_im[i], real_predict[i]) ** 2 / (np.mean(real_im[i]) ** 2 + 1e-100)

                        cur_result['ergas'] = np.sqrt(cur_result['ergas'] / 6.) * 6
                        cur_result['psnr'] = psnr(real_im, real_predict, MAX=1.0)

                        cur_im = real_im * 10000.
                        cur_predict = real_predict * 10000.

                        cur_result['sam'] = sam(cur_im.transpose(1, 2, 0), cur_predict.transpose(1, 2, 0)) * 180 / np.pi
                        print('[%s/%s] RMSE: %.4f SSIM: %.4f UIQI: %.4f CC: %.4f ERGAS: %.4f SAM: %.4f PSNR: %.4f' % (
                            cur_date, ref_date, np.mean(np.array(cur_result['rmse'])),
                            np.mean(np.array(cur_result['ssim'])), np.mean(np.array(cur_result['uiqi'])),
                            np.mean(np.array(cur_result['cc'])), cur_result['ergas'], cur_result['sam'],
                            cur_result['psnr']))
                        if ref_day == 331:
                            final_ssim = np.mean(np.array(cur_result['ssim']))
                            final_rmse = np.mean(np.array(cur_result['rmse']))
                            final_cc = np.mean(np.array(cur_result['cc']))
                            final_uiqi = np.mean(np.array(cur_result['uiqi']))
                            final_ergas = cur_result['ergas']
                            final_sam = cur_result['sam']

    return final_ssim, final_rmse, final_cc, final_uiqi, final_ergas, final_sam


def train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE, patch_stride, a=0.01, b=1, c=1, d=1):

    train_set = PatchSet(opt.train_dir, train_dates, IMAGE_SIZE, PATCH_SIZE, patch_stride)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=8, shuffle=True)

    model = FusionNet()
    discriminator = NLayerDiscriminator(input_nc=12,  getIntermFeat=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('There are %d trainable parameters for generator.' % n_params)
    n_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print('There are %d trainable parameters for discriminator.' % n_params)


    pd_loss = GANLoss()

    model.cuda()
    discriminator.cuda()
    pd_loss.cuda()


    g_optimizer = optim.Adam(model.parameters(), lr=2e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4)

    def lambda_rule(epoch):
        lr_l = 1.0
        return lr_l

    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_rule)
    pd_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_rule)

    best_ssim = 0.0
    best_epoch = -1

    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    history = save_dir + '/history.csv'
    history = Path(history)

    for epoch in tqdm(range(opt.num_epochs)):
        g_scheduler.step()
        pd_scheduler.step()
        model.train()
        discriminator.train()

        g_loss, d_loss, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
        batches = len(train_loader)

        for item, (data, target, ref_lr, ref_target, gt_mask) in enumerate(train_loader):
            t_start = timer()

            data = data.cuda()
            target = target.cuda()
            ref_lr = ref_lr.cuda()
            ref_target = ref_target.cuda()
            gt_mask = gt_mask.float().cuda()
            # data = interpolate(interpolate(data, scale_factor=1 / 16), scale_factor=16, mode='bicubic')
            # ref_lr = interpolate(interpolate(ref_lr, scale_factor=1 / 16), scale_factor=16, mode='bicubic')

            prediction = model(ref_lr,  ref_target, data)
            discriminator.zero_grad()
            model.zero_grad()
            ############################
            # (1) Update D network
            ###########################
            pred_fake = discriminator(torch.cat((prediction.detach(), data), dim=1))
            pred_real1 = discriminator(torch.cat((target, data), dim=1))
            l_d_loss = (pd_loss(pred_fake, False) +
                       pd_loss(pred_real1, True)) * 0.5
            l_d_loss.backward()
            d_optimizer.step()
            ############################
            # (2) Update G network
            ###########################
            prediction = model(ref_lr,  ref_target, data)
            pred_fake = discriminator(torch.cat((prediction, data), dim=1))

            loss_G_GAN = pd_loss(pred_fake, True) * a

            loss_G_l1 = F.l1_loss(prediction, target) * b \
                        + (1.0 - msssim(prediction, target, normalize=True)) * d\
                        + (1.0 - torch.mean(F.cosine_similarity(prediction, target, 1))) * c

            l_total = loss_G_l1 + loss_G_GAN

            l_total.backward()
            g_optimizer.step()
            ###########################
            d_loss.update(l_d_loss.cpu().item())
            g_loss.update(l_total.cpu().item())

            t_end = timer()
            batch_time.update(round(t_end - t_start, 4))


            if item % 10 == 0:
                print('[%d/%d][%d/%d] G-Loss: %.4f D-loss: %.4f Batch_Time: %.4f' % (
                    epoch + 1, opt.num_epochs, item + 1, batches, g_loss.avg, d_loss.avg, batch_time.avg,
                ))
        print('[%d/%d] G-Loss: %.4f Batch_Time: %.4f' % (
            epoch + 1, opt.num_epochs, g_loss.avg, batch_time.avg,
        ))

        final_ssim, final_rmse, final_cc, final_uiqi, final_ergas, final_sam = test(opt, model, test_dates, IMAGE_SIZE, PATCH_SIZE)
        csv_header = ['epoch', 'train_loss', 'd_loss', 'test_ssim', 'test_rmse', 'test_cc', 'test_uiqi', 'test_ergas', 'test_sam']
        csv_values = [epoch+1, g_loss.avg, d_loss.avg, final_ssim, final_rmse, final_cc, final_uiqi, final_ergas, final_sam]
        log_csv(history, csv_values, header=csv_header)

        if final_ssim > best_ssim:
            best_ssim = final_ssim
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir + '/epoch_best.pth')

        torch.save(model.state_dict(), save_dir + '/epoch_%d.pth' % (epoch + 1))
        print('Best Epoch is %d' % (best_epoch + 1), 'SSIM is %.4f' % best_ssim)
        print('------------------')


def main():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[2720, 3200], type=int, help='the image size (height, width)')
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--patch_stride', default=200, type=int)
    parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
    parser.add_argument('--root_dir', default='/data/ZDD/LGC_npy/', help='Datasets root directory')   #原始数据
    parser.add_argument('--train_dir', default='/data/ZDD/input3/LGC_npy_train/', help='Datasets train directory') #训练数据
    parser.add_argument('--save_dir',  default='result_lgc', help='save_dir')

    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    PATCH_STRIDE = opt.patch_stride
    print('The current path of the training data is: ', opt.train_dir)
    print('IMAGE_SIZE: ', IMAGE_SIZE)
    print('save_dir:',  opt.save_dir)

    # Loading Datasets
    train_dates = []
    test_dates = []
    for dir_name in os.listdir(opt.root_dir):
        cur_day = int(dir_name.split('_')[1])

        if cur_day not in [331, 347, 363]:
            train_dates.append(dir_name)
        if cur_day in [331, 347]:
            test_dates.append(dir_name)

    train(opt, train_dates, test_dates, IMAGE_SIZE, PATCH_SIZE, PATCH_STRIDE)


if __name__ == '__main__':
    main()


