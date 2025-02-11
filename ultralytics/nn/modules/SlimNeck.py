import torch
import torch.nn as nn
import math


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=16):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)

# class GSConv(nn.Module):
#     # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
#         super().__init__()
#         c_ = c2 // 2
#         self.cv1 = Conv(c1, c_, k, s, None, g, act)
#         self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         b, n, h, w = x2.data.size()
#         b_n = b * n // 2
#         y = x2.reshape(b_n, 2, h * w)
#         y = y.permute(1, 0, 2)
#         y = y.reshape(2, -1, n // 2, h, w)
#
#         return torch.cat((y[0], y[1]), 1)


# class GSConvns(GSConv):
#     # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
#         super().__init__(c1, c2, k=1, s=1, g=1, act=True)
#         c_ = c2 // 2
#         self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         x2 = torch.cat((x1, self.cv2(x1)), 1)
#         # normative-shuffle, TRT supported
#         return nn.ReLU(self.shuf(x2))
#
#
# class GSBottleneck(nn.Module):
#     # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1, e=0.5):
#         super().__init__()
#
#         c_ = int(c2 * e)
#         # self.cv2 = DualConv(c2, c_)
#         # for lighting
#         self.conv_lighting = nn.Sequential(
#             GSConv(c1, c_, 1, 1),
#             GSConv(c_, c2, 3, 1, act=False))
#             # DualConv(c1, c_),
#             # DualConv(c_, c2))
#         self.shortcut = Conv(c1, c2, 1, 1, act=False)
#
#     def forward(self, x):
#         return self.conv_lighting(x) + self.shortcut(x)

class DualBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()

        c_ = int(c2 * e)
        # self.cv2 = DualConv(c2, c_)
        # for lighting
        self.conv_lighting = nn.Sequential(
            # GSConv(c1, c_, 1, 1),
            # GSConv(c_, c2, 3, 1, act=False))
            DualConv(c1, c_),
            DualConv(c_, c2))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        # print("dual bottletneck")
        return self.conv_lighting(x) + self.shortcut(x)

# class DWConv(Conv):
#     # Depth-wise convolution class
#     def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


# class GSBottleneckC(GSBottleneck):
#     # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
#     def __init__(self, c1, c2, k=3, s=1):
#         super().__init__(c1, c2, k, s)
#         self.shortcut = DWConv(c1, c2, k, s, act=False)


# class VoVGSCSP(nn.Module):
#     # VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
#         self.res = Conv(c_, c_, 3, 1, act=False)
#         self.cv3 = Conv(2 * c_, c2, 1)  #
#
#     def forward(self, x):
#         x1 = self.gsb(self.cv1(x))
#         y = self.cv2(x)
#         return self.cv3(torch.cat((y, x1), dim=1))

class VoVDualCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(DualBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

#
# class VoVGSCSPC(VoVGSCSP):
#     # cheap VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2)
#         c_ = int(c2 * 0.5)  # hidden channels
#         self.gsb = GSBottleneckC(c_, c_, 1, 1)


# class C2f_Dual(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.m = nn.Sequential(*(DualBottleneck(self.c, self.c, e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

if __name__ == "__main__":
    # Generating Sample image
    image_size = (3, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    # model = VoVGSCSP(64, 64)
    model = VoVDualCSP(64, 64)
    print(model)

    out = model(image)
    print(out.size())