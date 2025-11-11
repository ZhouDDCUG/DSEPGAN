import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch


def main():
    # 设置随机数种子
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('--image_size', default=[2040, 1720], type=int, help='the image size (height, width)') # [2720,3200]
    parser.add_argument('--patch_size', default=256, type=int, help='training images crop size')
    parser.add_argument('--patch_stride', default=200, type=int)
    parser.add_argument('--root_dir', default='/home/wk001/ZDD/SwinSTFM/CIA_npy', help='Datasets root directory')
    parser.add_argument('--save_dir', default='/home/wk001/ZDD/SwinSTFM/CIA_npy_train', help='Datasets train directory')


    opt = parser.parse_args()
    IMAGE_SIZE = opt.image_size
    PATCH_SIZE = opt.patch_size
    PATCH_STRIDE = opt.patch_stride

    # train_dates = ['2005_093_Apr03', '2005_045_Feb14',
    #                '2005_029_Jan29', '2004_123_May02',
    #                '2004_299_Oct25', '2005_013_Jan13',
    #                '2004_235_Aug22', '2004_107_Apr16',
    #                '2004_187_Jul05', '2005_061_Mar02',
    #                '2004_219_Aug06']


    train_dates = ['2001_281_08oct', '2001_290_17oct', '2001_306_02nov',
                   '2001_313_09nov', '2001_338_04dec', '2002_005_05jan',
                   '2002_044_13feb', '2002_069_10mar',
                   '2002_076_17mar', '2002_092_02apr', '2002_101_11apr',
                   '2002_108_18apr', '2002_117_27apr', '2002_124_04may']

    # split the whole image into several patches
    max_end_h = IMAGE_SIZE[0] - PATCH_SIZE  #最大下标是可以取到值的
    max_end_w = IMAGE_SIZE[1] - PATCH_SIZE
    h_index_list = [i for i in range(0, max_end_h+1, PATCH_STRIDE)]
    w_index_list = [i for i in range(0, max_end_w+1, PATCH_STRIDE)]

    if max_end_h % PATCH_STRIDE != 0:
        h_index_list.append(IMAGE_SIZE[0] - PATCH_SIZE)
    if max_end_w % PATCH_STRIDE != 0:
        w_index_list.append(IMAGE_SIZE[1] - PATCH_SIZE)

    total_index = 0
    # path where the training images saved in
    output_dir = opt.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save all the train images into one numpy array
    # 有len(train_dates)个训练日期，每个日期中有2个数据集，每个数据集中有6波段，每张图像大小为IMAGE_SIZE
    total_original_images = np.zeros((len(train_dates), 2, 6, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.int16)
    for k in tqdm(range(len(train_dates))):
        cur_date = train_dates[k]
        target_dir = opt.root_dir + '/' + cur_date
        for filename in os.listdir(target_dir):
            #如果是MODIS数据集，就将图像存储到total_original_images[k, 1]中
            #如果是landsat数据集，就将图像存储到total_original_images[k, 0] 中
            if filename[:3] != 'MOD':
                path = os.path.join(target_dir, filename)
                total_original_images[k, 1] = np.load(path)
            else:
                path = os.path.join(target_dir, filename)
                total_original_images[k, 0] = np.load(path)
    #
    # 这段代码用于将训练集中的每个图像裁剪成多个小图像，并将这些小图像保存到一个numpy数组中。
    # 对于每个训练日期，代码会遍历所有的裁剪起始位置，然后随机选择一个不同于当前日期的参考日期。
    # 接下来，代码会从该日期下的MODIS和landsat数据集以及参考日期下的MODIS和landsat数据集中，
    # 分别取出对应的图像，并将它们裁剪成大小为PATCH_SIZE的小图像。这些小图像会被拼接成一个输入图像，然后将其保存到output_dir目录下。
    # 最后，total_index加1，表示已经处理完一个小图像。
    # 这样，代码就可以将训练集中的所有图像都裁剪成多个小图像，并将它们保存到一个numpy数组中，以便后续的模型训练。
    for k in tqdm(range(len(train_dates))):
        for i in range(len(h_index_list)):
            for j in range(len(w_index_list)):
                h_start = h_index_list[i]
                w_start = w_index_list[j]

                ref_index = k
                while ref_index == k:
                    ref_index = np.random.choice(len(train_dates))

                #顺序是预测日期landsat  modis，参考日期landsat  modis
                images = []
                images.append(total_original_images[k, 0])
                images.append(total_original_images[k, 1])
                images.append(total_original_images[ref_index, 0])
                images.append(total_original_images[ref_index, 1])

                input_images = []
                for im in images:
                    input_images.append(im[:, h_start: h_start + PATCH_SIZE, w_start: w_start + PATCH_SIZE])

                #图像被拼接了
                input_images = np.concatenate(input_images, axis=0)
                # save the patch for training
                np.save(os.path.join(output_dir, str(total_index) + '.npy'), input_images)

                total_index += 1

    assert total_index == len(train_dates) * len(h_index_list) * len(w_index_list)


if __name__ == '__main__':
    main()