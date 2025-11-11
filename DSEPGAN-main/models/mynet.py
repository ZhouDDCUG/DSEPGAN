import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchgan.layers import SpectralNorm2d
import enum
import numpy as np
# from ssim import msssim
# from normalization import SwitchNorm2d
from torch.autograd import Variable
# from torchvision.models.vgg import vgg19
# from torchsummary import Summary
from thop import profile



class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


NUM_BANDS = 6
PATCH_SIZE = 256
SCALE_FACTOR = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FusionNet(nn.Module):
    def __init__(self, dim=48, kernel_size=[13,13,13,13], mlp_ratio=[4,4,4,4], INN=True, time_pre=True, spa_pre=False):
        super(FusionNet, self).__init__()
        print('inn-gan')
        self.INN = INN
        # self.conv_f = nn.Sequential(nn.Conv2d(NUM_BANDS, dim, 3, 1, 1),nn.LeakyReLU(negative_slope=0.2, inplace=True))
        if (self.INN):
            self.detail_ext = nn.Sequential(nn.Conv2d(NUM_BANDS, dim, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), DetailFeatureExtraction(dim=dim))
        else:
            print('注意注意，当前没有INN模块！！！')
            self.detail_ext = ChangeFeatureExtraction(out_dim=dim)

        self.change_ext = ChangeFeatureExtraction(out_dim=dim)

        self.conv_pre =fusionBlock_same_level_ALL(in_ch=dim*4)
        if(not time_pre):
            print("当前仅有空间预测")
            self.conv_pre = fusionBlock_same_level_spa(in_ch=dim*4)

        if (not spa_pre):
            print("当前仅有时间预测")
            self.conv_pre =fusionBlock_same_level_time(in_ch=dim * 4, kernel_size=kernel_size[0], mlp_ratio=mlp_ratio[0])

        self.fusion1 = fusionblock(dim*4, dim*2, kernel_size=kernel_size[1], mlp_ratio=mlp_ratio[1], time_pre=time_pre, spa_pre=spa_pre)

        self.fusion2 = fusionblock(dim*2, dim, kernel_size=kernel_size[2], mlp_ratio=mlp_ratio[2], time_pre=time_pre, spa_pre=spa_pre)

        self.fusion3 = fusionblock(dim, dim//2, kernel_size=kernel_size[3], mlp_ratio=mlp_ratio[3], time_pre=time_pre, spa_pre=spa_pre)

        self.recons = reconstruction(dim//2)


    def forward(self, ref_lr, ref_target, data):
        f_c1 = self.change_ext(ref_lr)
        f_c2 = self.change_ext(data)
        f_change = []
        for i in range(4):
            f_change.append(f_c2[i] - f_c1[i])
        f_f = self.detail_ext(ref_target)
        f_fusion = self.conv_pre(f_c1[3], f_f[3], f_change[3])    #32   128

        f_fusion = self.fusion1(f_c1[2], f_f[2], f_change[2], f_fusion)   #64 128

        f_fusion = self.fusion2(f_c1[1], f_f[1], f_change[1], f_fusion)   #128   64

        f_fusion = self.fusion3(f_c1[0], f_f[0], f_change[0], f_fusion)   #256   32

        return self.recons(f_fusion)


############################################################################################
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, in_channel):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        half_ch = in_channel//2
        self.theta_phi = InvertedResidualBlock(inp=half_ch, oup=half_ch, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=half_ch, oup=half_ch, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=half_ch, oup=half_ch, expand_ratio=2)
        self.shffleconv = nn.Conv2d(in_channel, in_channel, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtractionBlock(nn.Module):
    def __init__(self, channel, num_layers):
        super(DetailFeatureExtractionBlock, self).__init__()
        INNmodules = [DetailNode(in_channel=channel) for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


############################################################
class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, out_dim, 1)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        x = self.reduction(x)
        return x


###########################################################################################
class ChangeFeatureExtraction(nn.Module):
    def __init__(self, in_dim=NUM_BANDS, out_dim=64):
        super(ChangeFeatureExtraction, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(out_dim, out_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv3 = nn.Conv2d(out_dim*4, out_dim*4, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x0, x1, x2, x3


##############################################################################################
class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3, dim=64):
        super(DetailFeatureExtraction, self).__init__()
        self.block0 = DetailFeatureExtractionBlock(dim, num_layers)
        self.block1 = nn.Sequential(PatchMerging(dim, dim*2),
                                    DetailFeatureExtractionBlock(dim*2, num_layers))
        self.block2 = nn.Sequential(PatchMerging(dim*2, dim*4),
                                    DetailFeatureExtractionBlock(dim*4, num_layers))
        self.block3 = nn.Sequential(PatchMerging(dim*4, dim*4),
                                    DetailFeatureExtractionBlock(dim*4, num_layers))


    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return x0, x1, x2, x3


##############################################################################################
class fusionblock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=11, mlp_ratio=4, time_pre=True, spa_pre=True):
        super(fusionblock, self).__init__()
        # self.up_flag = up
        self.up = nn.Sequential(nn.Conv2d(in_ch, 4 * in_ch, 3, 1, 1), nn.PixelShuffle(2))
        self.fusion = fusionBlock_same_level_ALL(in_ch, kernel_size=11, mlp_ratio=4)
        if(not time_pre):
            # print("当前仅有空间预测")
            self.fusion = fusionBlock_same_level_spa(in_ch)

        if(not spa_pre):
            # print("当前仅有时间预测")
            self.fusion = fusionBlock_same_level_time(in_ch, kernel_size, mlp_ratio)

        self.proj = nn.Sequential(nn.Conv2d(in_ch*3, out_ch, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.fusion2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.fusion2 = fusionBlock_same_level_time(out_ch, kernel_size, mlp_ratio)

    def forward(self, f_c0, f_f, f_change, f_fusion_last):
        f_fusion_up = self.up(f_fusion_last)
        f_fusion = self.fusion(f_c0, f_f, f_change)
        x = torch.cat((f_c0, f_fusion, f_fusion_up), dim=1)
        x = self.proj(x)
        x = self.fusion2(x)
        return x

class fusionBlock_same_level_ALL(nn.Module):
    def __init__(self, in_ch, kernel_size=11, mlp_ratio=4):
        super(fusionBlock_same_level_ALL, self).__init__()
        self.fusion_time = fusionBlock_same_level_time(in_ch, kernel_size, mlp_ratio)
        self.fusion_spa = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.fusion = nn.Conv2d(in_ch*2, in_ch, kernel_size=1)

    def forward(self, f_c0, f_f, f_change):
        # 当前级特征时间预测
        f_fusion_time = self.fusion_time(f_c0, f_f, f_change)
        # 当前级特征空间预测
        f_fusion_spa = self.fusion_spa(f_change * f_f) + f_c0
        fusion = self.fusion(torch.cat((f_fusion_time, f_fusion_spa), dim=1))

        return fusion


class fusionBlock_same_level_time(nn.Module):
    def __init__(self, in_ch, kernel_size=13, mlp_ratio=4):
        super(fusionBlock_same_level_time, self).__init__()
        # 特征光谱聚合
        self.down =nn.Conv2d(2 * in_ch, in_ch, kernel_size=1)
        self.bkconv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,groups=in_ch),
                                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1))

        self.convFFN = nn.Sequential(nn.Conv2d(in_ch, in_ch*mlp_ratio, 1),
                                     nn.GELU(),
                                     nn.Conv2d(in_ch*mlp_ratio, in_ch, 1)
        )


    def forward(self, f_c0, f_f, f_change):
        # 当前级特征时间预测
        f_fusion = self.down(torch.cat((f_f, f_change), dim=1))
        f_fusion = f_fusion + self.bkconv(f_fusion)
        f_fusion = f_fusion + self.convFFN(f_fusion)

        return f_fusion + f_f


class fusionBlock_same_level_spa(nn.Module):
    def __init__(self, in_ch):
        super(fusionBlock_same_level_spa, self).__init__()
        self.fusion_spa = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1),
                                        nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, f_c0, f_f, f_change):
        # 当前级特征空间预测
        f_fusion_spa = self.fusion_spa(f_change * f_f) + f_c0
        return f_fusion_spa


class reconstruction(nn.Module):
    def __init__(self, dim):
        super(reconstruction, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim//2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim//2, NUM_BANDS, kernel_size=1)
        )

    def forward(self, x):
        return self.proj(x)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)



class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)



# if __name__ == '__main__':
    # 生成器测试
    # x1 = torch.randn(8, 6, 256, 256)
    # x2 = torch.randn(8, 6, 256, 256)
    # x3 = torch.randn(8, 6, 256, 256)
    # model = FusionNet()
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('There are %d trainable parameters for generator.' % n_params)
    # flops, params = profile(model, inputs=(x1, x2, x3))
    # print(params)
    # print(flops)
    # y = model(x1, x2, x3)
    # print(y.shape)




    #判别器测试
    # x1 = torch.randn(2, 6, 256, 256)
    # x2 = torch.randn(2, 6, 256, 256)
    # model = NLayerDiscriminator(input_nc=12,  getIntermFeat=True)
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('There are %d trainable parameters for generator.' % n_params)
    # y = model(torch.cat((x1, x2), dim=1))
    # print(len(y))
    # print(y[0].shape)
    # print(y[1].shape)
    # print(y[2].shape)
    # print(y[3].shape)
    # print(y[4].shape)
    #
    # print(isinstance(y[0], list))



    # model = fusionblock(in_ch=16, out_ch=8, time_pre=True, spa_pre=False)
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('There are %d trainable parameters for generator.' % n_params)
