import math

import torch
import torch.nn as nn

from .block import DFL
from .conv import autopad
from .head import Detect, dist2bbox, make_anchors

# ---------------------------------------------------------------------#
#   Horizon X3  VarGNet modules
# ---------------------------------------------------------------------#

__all__ = [
    'X3Conv', 'X3SGConv', 'X3SPPF', 'X3VarGBlock', 'X3C2f', 'X3ConvTranspose', 'X3Detect', 'X3Proto', 'X3Segment',
    'X3Pose']


class X3Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class X3SGConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, c3=None, factor=1.0, act=True):
        super().__init__()
        if c3 is None:
            c3 = c1
        tmp = int(c3 * factor)
        self.dw = X3Conv(c1, tmp, k, s, p, g, d, act)
        self.pw = X3Conv(tmp, c2, 1, 1, None, 1, 1, True)

    def forward(self, x):
        return self.pw(self.dw(x))


class X3SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = X3Conv(c1, c_, 1, 1)
        self.cv2 = X3Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class X3VarGBlock(nn.Module):

    def __init__(self, c1, c2, shortcut=True, group_base=8, k=(3, 3), e=0.5, factor=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        g = c1 // group_base
        self.cv1 = X3SGConv(c1, c_, k[0], 1, g=g, factor=factor)
        self.cv2 = X3SGConv(c_, c2, k[1], 1, g=g, factor=factor)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class X3C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=8, e=0.5, factor=2.0):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = X3Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = X3Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            X3VarGBlock(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, factor=factor) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class X3ConvTranspose(nn.Module):
    # Convolution transpose 2d layer
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        return self.act(self.conv_transpose(x))


class X3Detect(nn.Module):
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    x3pi = False

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(X3Conv(x, c2, 3), X3Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(X3Conv(x, c3, 3), X3Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        if self.x3pi:
            return self.forward_x3pi(x)
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def forward_x3pi(self, x):
        res = []
        for i in range(self.nl):
            dfl = self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()
            cls = self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()
            res.append(cls)
            res.append(dfl)
        return tuple(res)

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class X3Proto(nn.Module):
    # YOLOv8 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = X3Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = X3Conv(c_, c_, k=3)
        self.cv3 = X3Conv(c_, c2)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            self.cv3.act = nn.Identity()
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class X3Segment(X3Detect):
    # YOLOv8 Segment head for segmentation models
    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = X3Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = X3Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(X3Conv(x, c4, 3), X3Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        if self.x3pi:
            results = self.forward_seg_x3pi(x)
            results.append(p)
            return tuple(results)
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

    def forward_seg_x3pi(self, x):
        results = []
        for i in range(self.nl):
            dfl = self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()
            cls = self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()
            mcoef = self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous()
            results.append(cls)
            results.append(dfl)
            results.append(mcoef)
        return results


class X3Pose(X3Detect):
    # YOLOv8 Pose head for keypoints models
    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(X3Conv(x, c4, 3), X3Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        if self.x3pi:
            results = self.forward_pose_x3pi(x)
            return tuple(results)
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, kpts):
        ndim = self.kpt_shape[1]
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::3].sigmoid_()  # inplace sigmoid
        y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
        return y

    def forward_pose_x3pi(self, x):
        results = []
        for i in range(self.nl):
            dfl = self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()
            cls = self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()
            kpt = self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous()
            results.append(cls)
            results.append(dfl)
            results.append(kpt)
        return results
