"""
Video analysis network based on existing backbone networks.
"""


import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, original_resnet):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(*list(original_resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1))
        return x


class ResNetFC(nn.Module):
    def __init__(self, original_resnet, fc_vis=16):
        super(ResNetFC, self).__init__()

        # use resnet18 architecture before the avgpool layer, i.e., all convolutional layers
        self.features = nn.Sequential(*list(original_resnet.children())[:-2])

        # # final visual feature map: 1 x 1
        # self.fc = nn.Sequential(
        #     nn.Conv2d(512, fc_vis, kernel_size=7, stride=1, padding=0),
        #     nn.BatchNorm2d(fc_vis),
        #     nn.LeakyReLU(0.2)
        #     )

        # final visual feature map: 2 x 2
        self.fc = nn.Sequential(
            nn.Conv2d(512, fc_vis, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(fc_vis),
            nn.LeakyReLU(0.2)
        )

        # # final visual feature map: 4 x 4
        # self.fc = nn.Sequential(
        #     nn.Conv2d(512, fc_vis, kernel_size=4, stride=1, padding=0),
        #     nn.BatchNorm2d(fc_vis),
        #     nn.LeakyReLU(0.2)
        # )

        # # final visual feature map: 7 x 7
        # self.fc = nn.Sequential(
        #     nn.Conv2d(512, fc_vis, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(fc_vis),
        #     nn.LeakyReLU(0.2)
        # )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.detach())
        return x

    def forward_multiframe(self, x):

        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        # print('size of visual feature map from ResNetFC: ', x.size())
        # torch.Size([30, 512, 7, 7])

        x = self.fc(x.detach())
        # print('size of visual feature map from fc layer: ', x.size())
        # h=w=1: torch.Size([30, 16, 1, 1])
        # h=w=2: torch.Size([30, 16, 2, 2])
        # h=w=4: torch.Size([30, 16, 4, 4])
        # h=w=7: torch.Size([30, 16, 7, 7])

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.mean(dim=1)
        # print('size of visual feature map after viewing: ', x.size())
        # h=w=1: torch.Size([10, 16, 1, 1])
        # h=w=2: torch.Size([10, 16, 2, 2])
        # h=w=4: torch.Size([10, 16, 4, 4])
        # h=w=7: torch.Size([10, 16, 7, 7])

        return x


class ResNetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, fc_vis=512):
        super(ResNetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # use resnet18 architecture before the avgpool layer, i.e., all convolutional layers
        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Sequential(
            nn.Conv2d(512, fc_vis, kernel_size=12, stride=2, padding=0),
            nn.BatchNorm2d(fc_vis),
            nn.LeakyReLU(0.2)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.detach())
        return x

    def forward_multiframe(self, x):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        x = self.fc(x.detach())

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.mean(dim=1)

        return x
