"""
Construct the complete network architecture based on existing network classes.
"""


import torch
import torchvision
import torch.nn.functional as F

from .vision_net import ResNetFC, ResNetDilated
from .audio_net import PCNetLR
from .criterion import BCELoss, L1Loss, L2Loss, KLDivLoss


def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'leaky_relu':
        return F.leaky_relu(x, 0.2)
    elif activation == 'elu':
        return F.elu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():

    # build for vision
    def build_frame(self, arch='resnet18', fc_vis=512, weights=''):
        pretrained = True
        if arch == 'resnet18fc':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResNetFC(original_resnet, fc_vis=fc_vis)
        elif arch == 'resnet18dilated':
            original_resnet = torchvision.models.resnet18(pretrained)
            net = ResNetDilated(original_resnet, fc_vis=fc_vis)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_frame')
            # net.load_state_dict(torch.load(weights))

            net_dict = net.state_dict()
            pretrained_dict = torch.load(weights)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)

        for param in net.features.parameters():
            param.requires_grad = False

        return net

    # build for audio
    def build_sound(self, arch='pcnetlr', weights='', cyc_in=4, ngf=64, fc_vis=16, n_fm_out=1):
        if arch == 'pcnetlr':
            net_sound = PCNetLR(cycs_in=cyc_in, ngf=ngf, fc_vis=fc_vis, n_fm_out=n_fm_out)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            print('Loading weights for net_sound')
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    # specify loss function for reconstructing target mask
    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Loss function undefined!')

        return net
