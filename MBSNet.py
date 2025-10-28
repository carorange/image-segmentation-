from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.data
import torch
from torchvision.models.segmentation import fcn_resnet50
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat

##############################
# Basic
##############################

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch, scale_factor=2, mode='bilinear'):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=64):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 用全连接代替1x1的卷积
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class DilatedConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(DilatedConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


##############################
# Basic module
##############################

class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


class Spatial_only_branch(nn.Module):

    """
        SAM Block
    """

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_q1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.conv_q2 = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(1)

        self.conv_v = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # Q branch
        atten_q = self.agp(x)
        atten_q = self.conv_q1(atten_q)
        atten_q = self.act(atten_q)
        atten_q = self.conv_q2(atten_q)
        atten_q = self.softmax(atten_q)
        atten_q = atten_q.permute(0,2,3,1).reshape(b, 1, c)

        # V branch
        atten_v = self.conv_v(x)
        atten_v = atten_v.reshape(b, c, -1)
        out = torch.matmul(atten_q, atten_v)
        out = out.reshape(b, 1, h, w)
        # out = self.sigmoid(out)
        out = torch.add(out, x)

        return out


# MaxPooling + DWcONV + Res
class PRMModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(PRMModule, self).__init__()

        self.conv_maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
            DWConv(dim=out_chan),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

        self.fuConv = nn.Sequential(
            nn.Conv2d(out_chan * 2, out_chan, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_chan),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # out1 MaxPooling
        out2 = self.conv_maxpool(x)

        # out2 DWConv
        out3 = self.dwconv(x)

        # Fusion
        out = torch.cat((out2, out3), dim=1)
        out = self.fuConv(out)

        out = self.act(torch.add(out, x))

        return out


# Dilated convolution block
class Bottleneck(nn.Module):

    def __init__(self, in_chan, out_chan, rate):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, ),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

        self.dilat = DilatedConv(out_chan, out_chan, rate)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, ),
            nn.BatchNorm2d(out_chan),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dilat(x)
        x = self.conv2(x)
        x = self.act(x)

        return x


##############################
# MBSNet
##############################

class UNET(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, if_adb=False):
        super(UNET, self).__init__()

        self.if_adb = if_adb
        self.out_ch = out_ch
        # [16, 32, 64, 128, 256]
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.adjust_conv = nn.Conv2d(17, 19, kernel_size=1, stride=1, padding=0)

        multiGrid = [1, 2, 4]
        # rates = 2 * multiGrid
        rates = [2, 4, 8]

        # ----------------------------- L
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.maxDw1 = PRMModule(filters[0], filters[0])
        self.maxDw2 = PRMModule(filters[1], filters[1])
        self.maxDw3 = PRMModule(filters[2], filters[2])
        self.maxDw4 = PRMModule(filters[3], filters[3])
        self.maxDw5 = PRMModule(filters[4], filters[4])

        self.se = SEBlock(filters[4], filters[4])

        # ----------------------------- G
        self.D_catConv1 = conv_block(filters[1], filters[0])
        self.D_catConv2 = conv_block(filters[2], filters[1])
        self.D_catConv3 = conv_block(filters[2] * 2, filters[2])

        self.D_bott1 = Bottleneck(in_ch, filters[0], rates[0])
        self.D_bott2 = Bottleneck(filters[0], filters[1], rates[1])
        self.D_bott3 = Bottleneck(filters[1], filters[2], rates[2])

        self.D_Avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.D_Avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.D_Avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.attn = Spatial_only_branch(filters[2], filters[2])

        # ------------------------------ F
        self.up_conv2 = up_conv(filters[4], filters[3])
        self.up_conv3 = up_conv(filters[3], filters[2])

        self.fu_Conv2 = conv_block(filters[4], filters[3])
        self.fu_Conv3 = conv_block(filters[3], filters[2])

        self.intere1_1 = conv_block(filters[2], filters[1])
        self.fu_Conv4 = conv_block(filters[2], filters[1])
        self.drop1 = nn.Dropout(0.1)

        self.intere1_2 = conv_block(filters[2], filters[1])
        self.D_Conv3 = conv_block(filters[2], filters[1])
        self.drop2 = nn.Dropout(0.1)

        self.intere2_1 = conv_block(filters[1], filters[0])
        self.fu_Conv5 = conv_block(filters[1], filters[0])
        self.drop3 = nn.Dropout(0.1)

        self.inter2_2 = conv_block(filters[1], filters[0])
        self.D_Conv4 = conv_block(filters[0] + 3, filters[0])
        self.drop4 = nn.Dropout(0.1)

        # --------------------------------
        self.ConvOut1 = nn.Conv2d(filters[0] + filters[0], filters[0], kernel_size=1, stride=1, padding=0)
        self.ConvOut2 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # -----L Branch-----------------------------
        e1 = self.Conv1(x)
        e1 = self.maxDw1(e1)
        print(f"After Conv1 + maxDw1: {e1.shape}")

        e2 = self.Maxpool1(e1)
        print(f"After Maxpool1: {e2.shape}")
        e2 = self.Conv2(e2)
        e2 = self.maxDw2(e2)
        print(f"After Conv2 + maxDw2: {e2.shape}")

        e3 = self.Maxpool2(e2)
        print(f"After Maxpool2: {e3.shape}")
        e3 = self.Conv3(e3)
        e3 = self.maxDw3(e3)
        print(f"After Conv3 + maxDw3: {e3.shape}")

        e4 = self.Maxpool3(e3)
        print(f"After Maxpool3: {e4.shape}")
        e4 = self.Conv4(e4)
        e4 = self.maxDw4(e4)
        print(f"After Conv4 + maxDw4: {e4.shape}")

        e5 = self.Maxpool4(e4)
        print(f"After Maxpool4: {e5.shape}")
        e5 = self.Conv5(e5)
        e5 = self.maxDw5(e5)
        print(f"After Conv5 + maxDw5: {e5.shape}")

        e5 = self.se(e5)
        print(f"After SE Block: {e5.shape}")

        # -----G Branch-----------------------------
        D_e1 = self.D_bott1(x)
        D_e1 = torch.cat((D_e1, e1), dim=1)
        print(f"After concat (D_bott1 + e1): {D_e1.shape}")
        D_e1 = self.D_catConv1(D_e1)
        print(f"After D_catConv1: {D_e1.shape}")

        D_e2 = self.D_Avgpool1(D_e1)
        print(f"After D_Avgpool1: {D_e2.shape}")
        D_e2 = self.D_bott2(D_e2)
        D_e2 = torch.cat((D_e2, e2), dim=1)
        print(f"After concat (D_bott2 + e2): {D_e2.shape}")
        D_e2 = self.D_catConv2(D_e2)
        print(f"After D_catConv2: {D_e2.shape}")

        D_e3 = self.D_Avgpool2(D_e2)
        print(f"After D_Avgpool2: {D_e3.shape}")
        D_e3 = self.D_bott3(D_e3)
        D_e3 = torch.cat((D_e3, e3), dim=1)
        print(f"After concat (D_bott3 + e3): {D_e3.shape}")
        D_e3 = self.D_catConv3(D_e3)
        print(f"After D_catConv3: {D_e3.shape}")

        D_d3 = self.attn(D_e3)
        print(f"After Spatial Attention: {D_d3.shape}")

        # -----Fusion------------------------------
        d4 = self.up_conv2(e5)
        print(f"After up_conv2 (e5): {d4.shape}")

        e4 = F.interpolate(d4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        print(f"After interpolate (e4 to match d4): {e4.shape}")

        d4 = torch.cat((e4, d4), dim=1)
        print(f"After concat (e4 + d4): {d4.shape}")
        d4 = self.fu_Conv2(d4)
        print(f"After fu_Conv2: {d4.shape}")

        d3 = self.up_conv3(d4)
        print(f"After up_conv3: {d3.shape}")

        e3 = F.interpolate(d3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        print(f"After interpolate (e3 to match d3): {e3.shape}")
        d3 = torch.cat((e3, d3), dim=1)
        print(f"After concat (e3 + d3): {d3.shape}")
        d3 = self.fu_Conv3(d3)
        print(f"After fu_Conv3: {d3.shape}")

        # 融合
        d2 = F.interpolate(self.intere1_1(D_d3), scale_factor=2, mode='bilinear', align_corners=False)
        print(f"After interpolate (intere1_1 D_d3): {d2.shape}")
        d2 = torch.cat((d2, e2), dim=1)
        print(f"After concat (d2 + e2): {d2.shape}")
        d2 = self.fu_Conv4(d2)
        d2 = self.drop1(d2)
        print(f"After fu_Conv4 + drop1: {d2.shape}")

        D_d2 = F.interpolate(self.intere1_2(d3), scale_factor=2, mode='bilinear', align_corners=False)
        print(f"After interpolate (intere1_2 d3): {D_d2.shape}")
        D_e2 = F.interpolate(D_d2, size=D_d2.shape[2:], mode='bilinear', align_corners=False)
        print(f"After interpolate (D_e2 to match D_d2): {D_e2.shape}")

        D_d2 = torch.cat((D_d2, D_e2), dim=1)
        print(f"After concat (D_d2 + D_e2): {D_d2.shape}")
        D_d2 = self.D_Conv3(D_d2)
        D_d2 = self.drop2(D_d2)
        print(f"After D_Conv3 + drop2: {D_d2.shape}")

        d1 = F.interpolate(self.intere2_1(D_d2), scale_factor=2, mode='bilinear', align_corners=False)
        print(f"After interpolate (intere2_1 D_d2): {d1.shape}")
        e1 = F.interpolate(d1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        print(f"After interpolate (e1 to match d1): {e1.shape}")
        d1 = torch.cat((d1, e1), dim=1)
        print(f"After concat (d1 + e1): {d1.shape}")
        d1 = self.fu_Conv5(d1)
        d1 = self.drop3(d1)
        print(f"After fu_Conv5 + drop3: {d1.shape}")

        D_d1 = F.interpolate(self.inter2_2(d2), scale_factor=2, mode='bilinear', align_corners=False)
        print(f"After interpolate (inter2_2 d2): {D_d1.shape}")
        D_d1 = torch.cat((D_d1, x), dim=1)
        print(f"After concat (D_d1 + x): {D_d1.shape}")
        D_d1 = self.adjust_conv(D_d1)
        D_d1 = self.D_Conv4(D_d1)
        D_d1 = self.drop4(D_d1)
        print(f"After adjust_conv + D_Conv4 + drop4: {D_d1.shape}")
        D_d1 = F.interpolate(D_d1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        print(f"After interpolate (D_d1 to match d1): {D_d1.shape}")

        out = torch.cat((d1, D_d1), dim=1)
        print(f"After concat (d1 + D_d1): {out.shape}")

        out = self.ConvOut1(out)
        out = self.ConvOut2(out)
        print(f"After ConvOut1 + ConvOut2: {out.shape}")

        return out


