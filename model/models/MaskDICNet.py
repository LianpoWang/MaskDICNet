import torch
import torch.nn
import torch.nn.functional
from torch.nn.init import kaiming_normal_, constant_
import sys
from torchvision import ops

sys.path.append('../correlation_package')
from correlation import Correlation

__all__ = ['MaskDICNet']

backwarp_tenGrid = {}
backwarp_tenPartial = {}


def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(
            1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(
            1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
            [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.shape)]], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput,
                                                grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3,
                                                                                                              1),
                                                mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask


class MaskStrainNetV1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.correlation_compute = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1,
                                               corr_multiply=1)
        self.leakyRELU = torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)

        class MaskSegmentor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.MaxPool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

                self.maskEncoderOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskEncoderTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskEncoderThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskEncoderFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskEncoderFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(1024),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(1024),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )

                self.upConvFiv = torch.nn.Sequential(
                    torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskDecoderFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(512),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.upConvFou = torch.nn.Sequential(
                    torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskDecoderFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(256),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.upConvThr = torch.nn.Sequential(
                    torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskDecoderThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.upConvTwo = torch.nn.Sequential(
                    torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )
                self.maskDecoderTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
                )

                self.conv_out = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
                )

            def forward(self, input_tensor):
                x1 = self.maskEncoderOne(input_tensor)

                x2 = self.MaxPool(x1)
                x2 = self.maskEncoderTwo(x2)

                x3 = self.MaxPool(x2)
                x3 = self.maskEncoderThr(x3)

                x4 = self.MaxPool(x3)
                x4 = self.maskEncoderFou(x4)

                x5 = self.MaxPool(x4)
                x5 = self.maskEncoderFiv(x5)

                m5 = self.upConvFiv(x5)
                m5 = torch.cat([x4, m5], dim=1)
                m5 = self.maskDecoderFiv(m5)

                m4 = self.upConvFou(m5)
                m4 = torch.cat([x3, m4], dim=1)
                m4 = self.maskDecoderFou(m4)

                m3 = self.upConvThr(m4)
                m3 = torch.cat([x2, m3], dim=1)
                m3 = self.maskDecoderThr(m3)

                m2 = self.upConvTwo(m3)
                m2 = torch.cat([x1, m2], dim=1)
                m2 = self.maskDecoderTwo(m2)

                m1 = self.conv_out(m2)

                return m1

        self.maskSegment = MaskSegmentor()

        self.extractorZero = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.extractorOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.extractorTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.extractorThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.conv1x1_l3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.conv1x1_l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        # self.conv1x1_l1 = torch.nn.Sequential(
        #    torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        #    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        # )

        self.feature_fusion = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.conv1x1_l3_mask = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.conv1x1_l2_mask = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        # self.conv1x1_l1_mask = torch.nn.Sequential(
        #    torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
        #    torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        # )

        self.decode3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode3_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1,
                            bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode3_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1,
                            bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode3_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1,
                            bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.predict_flow_3 = torch.nn.Conv2d(in_channels=1540, out_channels=2,
                                              kernel_size=3, stride=1, padding=1, bias=True)

        self.decode2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode2_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1,
                            bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode2_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1,
                            bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decode2_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=196 + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1,
                            bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.predict_flow_2 = torch.nn.Conv2d(in_channels=1540, out_channels=2,
                                              kernel_size=3, stride=1, padding=1, bias=True)

        self.decodef3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef3_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=608, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef3_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=768, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef3_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=864, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.decodef2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef2_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=608, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef2_4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=768, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.decodef2_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=864, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.offsetEstimate1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=18, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.offsetEstimate2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=18, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.offsetEstimate3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=18, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.deform3_1 = ops.DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform3_2 = ops.DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform3_3 = ops.DeformConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform3_4 = ops.DeformConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform3_5 = ops.DeformConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                                          bias=True)

        self.deform2_1 = ops.DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform2_2 = ops.DeformConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform2_3 = ops.DeformConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform2_4 = ops.DeformConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.deform2_5 = ops.DeformConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
                                          bias=True)

        self.mask_feature = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.mask_aggregation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.refine_flow1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=194, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.refine_flow2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=194 + 128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.refine_flow3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=194 + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.refine_flow4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=194 + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.refine_flow5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=194 + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        self.refine_flow6 = torch.nn.Conv2d(in_channels=642, out_channels=2, kernel_size=3, stride=1, padding=1,
                                            bias=True)

        self.deform_last = ops.DeformConv2d(in_channels=642, out_channels=642, kernel_size=3, stride=1, padding=1,
                                            bias=False)

        self.predict_flow = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=642, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        )

        self.upres = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1540, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        self.convTran4x4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m,
                                                                                                       ops.DeformConv2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, tenOne, tenTwo):
        mask1 = self.maskSegment(tenOne)
        mask2 = self.maskSegment(tenTwo)

        mask_def = self.sigmoid(mask1)
        mask_ref = self.sigmoid(mask2)

        img1 = tenOne
        img2 = tenTwo

        mask_feature1 = self.mask_feature(mask_def)
        mask_feature2 = self.mask_feature(mask_ref)

        feature1_l0 = self.extractorZero(img1)
        feature1_l1 = self.extractorOne(torch.cat([feature1_l0, mask_feature1], 1))
        feature1_l2 = self.extractorTwo(feature1_l1)
        feature1_l3 = self.extractorThr(feature1_l2)

        feature2_l0 = self.extractorZero(img2)
        feature2_l1 = self.extractorOne(torch.cat([feature2_l0, mask_feature2], 1))
        feature2_l2 = self.extractorTwo(feature2_l1)
        feature2_l3 = self.extractorThr(feature2_l2)

        b_l3, _, h_l3, w_l3 = feature1_l3.size()
        b_l2, _, h_l2, w_l2 = feature1_l2.size()
        b_l1, _, h_l1, w_l1 = feature1_l1.size()

        offset1 = self.offsetEstimate1(feature1_l1)
        offset2 = self.offsetEstimate2(feature1_l2)
        offset3 = self.offsetEstimate3(feature1_l3)

        mask_defH = torch.round(mask_def).detach()
        mask_refH = torch.round(mask_ref).detach()

        mask_def_copy = mask_defH.clone()
        mask_ref_copy = mask_refH.clone()
        mask_def_2 = torch.nn.functional.interpolate(mask_def_copy, size=(h_l2, w_l2), mode='bilinear')
        mask_def_3 = torch.nn.functional.interpolate(mask_def_copy, size=(h_l3, w_l3), mode='bilinear')
        mask_ref_2 = torch.nn.functional.interpolate(mask_ref_copy, size=(h_l2, w_l2), mode='bilinear')
        mask_ref_3 = torch.nn.functional.interpolate(mask_ref_copy, size=(h_l3, w_l3), mode='bilinear')
        mask_def_2H = (mask_def_2 > 0.0).float()

        mask_def_1L = mask_def_copy.long()
        mask_def_2L = (mask_def_2 > 0.0).long()
        mask_def_3L = (mask_def_3 > 0.0).long()
        mask_ref_1L = mask_ref_copy.long()
        mask_ref_2L = (mask_ref_2 > 0.0).long()
        mask_ref_3L = (mask_ref_3 > 0.0).long()

        b_init, _, h_init, w_init = feature1_l3.size()
        init_dtype = feature1_l3.dtype
        init_device = feature1_l3.device
        flow_init = torch.zeros(b_init, 2, h_init, w_init, dtype=init_dtype, device=init_device).float()

        correlation3 = self.correlation_compute(feature1_l3, feature2_l3)
        correlation3 = self.leakyRELU(correlation3)
        feature1_l3_1x1 = self.conv1x1_l3(feature1_l3)

        feature1_l3_1x1_mask = self.conv1x1_l3_mask(feature1_l3)
        mask_def_3H_oh = torch.nn.functional.one_hot(mask_def_3L)
        mask_pooled_value1 = torch.amax(mask_def_3H_oh * feature1_l3_1x1_mask[..., None], dim=(2, 3))
        mask_feat1 = (mask_def_3H_oh * mask_pooled_value1[:, :, None, None, :]).sum(dim=-1)

        feature2_l3_1x1_mask = self.conv1x1_l3_mask(feature2_l3)
        mask_ref_3H_oh = torch.nn.functional.one_hot(mask_ref_3L)
        mask_pooled_value2 = torch.amax(mask_ref_3H_oh * feature2_l3_1x1_mask[..., None], dim=(2, 3))
        mask_feat2 = (mask_ref_3H_oh * mask_pooled_value2[:, :, None, None, :]).sum(dim=-1)

        x_mask_feat1 = self.mask_aggregation(torch.concat([feature1_l3_1x1_mask, mask_feat1], 1))
        x_mask_feat2 = self.mask_aggregation(torch.concat([feature2_l3_1x1_mask, mask_feat2], 1))

        correlation3_mask = self.correlation_compute(x_mask_feat1, x_mask_feat2)
        correlation3_mask = self.leakyRELU(correlation3_mask)

        x = torch.cat([correlation3, correlation3_mask, feature1_l3_1x1, flow_init], 1)
        xdc = self.decode3_1(x)
        xdf = self.leakyRELU(self.deform3_1(xdc, offset3))
        xdfc = self.decodef3_1(xdf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xdf], 1)
        xdc = self.decode3_2(x)
        xdf = self.leakyRELU(self.deform3_2(xdc, offset3))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef3_2(xf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xf], 1)
        xdc = self.decode3_3(x)
        xdf = self.leakyRELU(self.deform3_3(xdc, offset3))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef3_3(xf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xf], 1)
        xdc = self.decode3_4(x)
        xdf = self.leakyRELU(self.deform3_4(xdc, offset3))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef3_4(xf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xf], 1)
        xdc = self.decode3_5(x)
        xdf = self.leakyRELU(self.deform3_5(xdc, offset3))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef3_5(xf)
        x = torch.cat([xdc, x, xdfc, xf], 1)
        flow_res3 = self.predict_flow_3(x)
        flow3 = flow_init + flow_res3

        flow2 = torch.nn.functional.interpolate(flow3, size=(h_l2, w_l2), mode="bilinear", align_corners=False)
        warp2 = backwarp(feature2_l2, flow2 * 0.5)
        warp2 = warp2 * mask_def_2H + feature1_l2 * (1 - mask_def_2H)
        correlation2 = self.correlation_compute(feature1_l2, warp2)
        correlation2 = self.leakyRELU(correlation2)
        feature1_l2_1x1 = self.conv1x1_l2(feature1_l2)

        feature1_l2_1x1_mask = self.conv1x1_l2_mask(feature1_l2)
        mask_def_2H_oh = torch.nn.functional.one_hot(mask_def_2L)
        mask_pooled_value3 = torch.amax(mask_def_2H_oh * feature1_l2_1x1_mask[..., None], dim=(2, 3))
        mask_feat3 = (mask_def_2H_oh * mask_pooled_value3[:, :, None, None, :]).sum(dim=-1)

        feature2_l2_1x1_mask = self.conv1x1_l2_mask(feature2_l2)
        mask_ref_2H_oh = torch.nn.functional.one_hot(mask_ref_2L)
        mask_pooled_value4 = torch.amax(mask_ref_2H_oh * feature2_l2_1x1_mask[..., None], dim=(2, 3))
        mask_feat4 = (mask_ref_2H_oh * mask_pooled_value4[:, :, None, None, :]).sum(dim=-1)

        x_mask_feat3 = self.mask_aggregation(torch.concat([feature1_l2_1x1_mask, mask_feat3], 1))
        x_mask_feat4 = self.mask_aggregation(torch.concat([feature2_l2_1x1_mask, mask_feat4], 1))

        x_mask_feat4_warp = backwarp(x_mask_feat4, flow2 * 0.5)
        x_mask_feat4_warp = x_mask_feat4_warp * mask_def_2H + x_mask_feat3 * (1 - mask_def_2H)
        correlation2_mask = self.correlation_compute(x_mask_feat3, x_mask_feat4_warp)
        correlation2_mask = self.leakyRELU(correlation2_mask)

        x = torch.cat([correlation2, correlation2_mask, feature1_l2_1x1, flow2], 1)
        xdc = self.decode2_1(x)
        xdf = self.leakyRELU(self.deform2_1(xdc, offset2))
        xdfc = self.decodef2_1(xdf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xdf], 1)
        xdc = self.decode2_2(x)
        xdf = self.leakyRELU(self.deform2_2(xdc, offset2))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef2_2(xf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xf], 1)
        xdc = self.decode2_3(x)
        xdf = self.leakyRELU(self.deform2_3(xdc, offset2))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef2_3(xf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xf], 1)
        xdc = self.decode2_4(x)
        xdf = self.leakyRELU(self.deform2_4(xdc, offset2))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef2_4(xf)
        x = torch.cat([xdc, x], 1)
        xf = torch.cat([xdfc, xf], 1)
        xdc = self.decode2_5(x)
        xdf = self.leakyRELU(self.deform2_5(xdc, offset2))
        xf = torch.cat([xdf, xf], 1)
        xdfc = self.decodef2_5(xf)
        x_out2 = torch.cat([xdc, x, xdfc, xf], 1)
        flow_res2 = self.predict_flow_2(x_out2)
        flow2 = flow2 + flow_res2

        flow1 = torch.nn.functional.interpolate(flow2, size=(h_l1, w_l1), mode="bilinear", align_corners=False)
        res = self.upres(x_out2)
        res_4x4 = self.convTran4x4(res)
        warp1 = backwarp(tenTwo, flow1)
        warp1 = warp1 * mask_def_copy + tenOne * (1 - mask_def_copy)
        feature_fusion = self.feature_fusion(torch.cat([tenOne, warp1], 1))
        # dif = (tenOne - warp1) * mask_defH
        x = torch.cat([feature_fusion, res_4x4, flow1], 1)
        x = torch.cat([self.refine_flow1(x), x], 1)
        x = torch.cat([self.refine_flow2(x), x], 1)
        x = torch.cat([self.refine_flow3(x), x], 1)
        x = torch.cat([self.refine_flow4(x), x], 1)
        x = torch.cat([self.refine_flow5(x), x], 1)
        deform_feat = self.leakyRELU(self.deform_last(x, offset1))
        flow1 = flow1 + self.predict_flow(deform_feat)

        return flow1, flow2, flow3, mask_def, mask_ref

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def MaskDICNet(data=None):
    model = MaskStrainNetV1()

    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
