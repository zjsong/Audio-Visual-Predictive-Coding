"""
Sound separation network.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


######################################
# PCNet-LR
######################################
# feedforward submodule 1
class DownC(nn.Module):
    def __init__(self, inchan, outchan, downsample=False):
        super(DownC, self).__init__()
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)   # retain fm size
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.downsample:
            x = self.Downsample(x)
        return x


# feedback submodule
class UpC(nn.Module):
    def __init__(self, inchan, outchan, upsample=False, dropout=False):
        super(UpC, self).__init__()
        self.convtrans2d = nn.ConvTranspose2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)    # retain fm size
        self.upsample = upsample
        if self.upsample:
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = dropout
        if self.dropout:
            self.Dropout = nn.Dropout(0.5)

    def forward(self, x):
        if self.upsample:
            x = self.Upsample(x)
        x = self.convtrans2d(x)
        if self.dropout:
            x = self.Dropout(x)
        return x


# visual feature maps guided, no feedback from r_mask
class PCNetLR(nn.Module):
    def __init__(self, cycs_in=5, ngf=64, fc_vis=16, n_fm_out=1):
        super(PCNetLR, self).__init__()
        self.cycs_in = cycs_in

        # final visual feature map: 1 x 1, 2 x 2
        self.in_channels =  [fc_vis,  ngf * 8, ngf * 4, ngf * 4, ngf * 2, ngf * 2, ngf, ngf]
        self.out_channels = [ngf * 8, ngf * 4, ngf * 4, ngf * 2, ngf * 2, ngf,     ngf, n_fm_out]
        # # final visual feature map: 4 x 4
        # self.in_channels =  [fc_vis,  ngf * 4, ngf * 4, ngf * 2, ngf * 2, ngf, ngf]
        # self.out_channels = [ngf * 4, ngf * 4, ngf * 2, ngf * 2, ngf,     ngf, n_fm_out]
        # # final visual feature map: 7 x 7
        # self.in_channels =  [fc_vis,  ngf * 4, ngf * 2, ngf * 2, ngf, ngf]
        # self.out_channels = [ngf * 4, ngf * 2, ngf * 2, ngf,     ngf, n_fm_out]

        self.num_layers = len(self.in_channels)

        # ----------------
        # conv layers from mixture spec to middle repres
        # ----------------
        # # final visual feature map: 1 x 1
        # DownC_first = [DownC(self.out_channels[0], self.in_channels[0], downsample=True)]
        # final visual feature map: 2 x 2
        DownC_first = [nn.Conv2d(self.out_channels[0], self.in_channels[0], kernel_size=1, stride=1, bias=False)]
        # # final visual feature map: 4 x 4
        # DownC_first = [nn.Conv2d(self.out_channels[0], self.in_channels[0], kernel_size=1, stride=1, bias=False)]
        # # final visual feature map: 7 x 7
        # DownC_first = [nn.Conv2d(self.out_channels[0], self.in_channels[0], kernel_size=2, stride=1, bias=False)]

        DownC_last = [DownC(1, self.in_channels[-1], downsample=True)]
        self.DownConvs = nn.ModuleList(DownC_first + [DownC(self.out_channels[i], self.in_channels[i], downsample=True)
                                                      for i in range(1, self.num_layers - 1)] + DownC_last)

        # ----------------
        # conv layers from middle repres to mask repres
        # ----------------
        # # final visual feature map: 1 x 1
        # UpC_first = [UpC(self.in_channels[0], self.out_channels[0], upsample=True)]
        # final visual feature map: 2 x 2
        UpC_first = [nn.Conv2d(self.in_channels[0], self.out_channels[0], kernel_size=1, stride=1, bias=False)]
        # # final visual feature map: 4 x 4
        # UpC_first = [nn.Conv2d(self.in_channels[0], self.out_channels[0], kernel_size=1, stride=1, bias=False)]
        # # final visual feature map: 7 x 7
        # UpC_first = [nn.ConvTranspose2d(self.in_channels[0], self.out_channels[0], kernel_size=2, stride=1, bias=False)]

        UpC_last = [UpC(self.in_channels[-1], self.out_channels[-1], upsample=True)]
        self.UpConvs = nn.ModuleList(UpC_first + [UpC(self.in_channels[i], self.out_channels[i], upsample=True)
                                                  for i in range(1, self.num_layers - 1)] + UpC_last)

        # ----------------
        # update rates of two repres flows
        # ----------------
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.out_channels[i], 1, 1) + 1.0)
                                    for i in range(self.num_layers - 1)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.in_channels[i], 1, 1) + 0.5)
                                    for i in range(1, self.num_layers)])

        # ----------------
        # batch norm
        # ----------------
        self.BNUp = nn.ModuleList([nn.BatchNorm2d(self.out_channels[i]) for i in range(self.num_layers - 1)])
        BNDown_ = [nn.BatchNorm2d(1)]  # normalize input log_mag_mix
        self.BNDown = nn.ModuleList([nn.BatchNorm2d(self.in_channels[i]) for i in range(1, self.num_layers)] + BNDown_)

        # ----------------
        # BN for each time-step
        # ----------------
        BNUp_step = []
        BNDown_step = []
        for t in range(cycs_in):
            BNUp_step = BNUp_step + [nn.BatchNorm2d(self.out_channels[i]) for i in range(self.num_layers - 1)]
            BNDown_step = BNDown_step + [nn.BatchNorm2d(self.in_channels[i]) for i in range(1, self.num_layers - 1)]
        self.BNUp_step = nn.ModuleList(BNUp_step)
        self.BNDown_step = nn.ModuleList(BNDown_step)

    def forward(self, x, fm_vis):

        x = self.BNDown[-1](x)

        # ----------------
        # top-down process from spec to middle repres
        # ----------------
        # init r_down[-1]
        r_down = [F.leaky_relu(self.BNDown[-2](self.DownConvs[-1](x)), 0.2)]
        # init r_down[i] (i<-1)
        for i in range(self.num_layers - 2, 0, -1):
            r_down = [F.leaky_relu(self.BNDown[i - 1](self.DownConvs[i](r_down[0])), 0.2)] + r_down
        
        # ----------------
        # predict fc_vis
        # ----------------
        fc_vis_pred = F.leaky_relu(self.DownConvs[0](r_down[0]), 0.2)

        # ----------------
        # bottom-up process which updates repres with prediction error
        # ----------------
        # update r_up[0]
        fc_vis_pred_err = fm_vis - fc_vis_pred
        a0 = F.relu(self.a0[0]).expand_as(r_down[0])
        r_up = [F.leaky_relu(self.BNUp[0](r_down[0] + a0 * self.UpConvs[0](fc_vis_pred_err)), 0.2)]
        # update r_up[i] (i>0)
        for i in range(1, self.num_layers - 1):
            pred_err = r_up[i - 1] - r_down[i - 1]
            a0 = F.relu(self.a0[i]).expand_as(r_down[i])
            r_up.append(F.leaky_relu(self.BNUp[i](r_down[i] + a0 * self.UpConvs[i](pred_err)), 0.2))

        for t in range(self.cycs_in):

            # ----------------
            # top-down process which updates repres with prediction
            # ----------------
            # update r_up[-1]
            b0 = F.relu(self.b0[-1]).expand_as(r_up[-1])
            r_up[-1] = F.leaky_relu((1 - b0) * r_up[-1] + b0 * r_down[-1], 0.2)
            # update r_down[i] and r_up[i] (i<-1)
            for i in range(self.num_layers - 2, 0, -1):
                r_down[i - 1] = self.DownConvs[i](r_up[i])
                b0 = F.relu(self.b0[i - 1]).expand_as(r_up[i - 1])
                r_up[i - 1] = F.leaky_relu(self.BNDown_step[(self.num_layers-2)*t+i-1]((1 - b0) * r_up[i - 1] +
                                           b0 * r_down[i - 1]), 0.2)
            # ----------------
            # predict fc_vis
            # ----------------
            fc_vis_pred = F.leaky_relu(self.DownConvs[0](r_up[0]), 0.2)

            # ----------------
            # bottom-up process which updates repres with prediction error
            # ----------------
            # update r_up[0]
            fc_vis_pred_err = fm_vis - fc_vis_pred
            a0 = F.relu(self.a0[0]).expand_as(r_up[0])
            r_up[0] = F.leaky_relu(self.BNUp_step[(self.num_layers-1)*t](r_up[0] +
                                                                         a0 * self.UpConvs[0](fc_vis_pred_err)), 0.2)
            # update r_up[i] (i>0)
            for i in range(1, self.num_layers - 1):
                pred_err = r_up[i - 1] - r_down[i - 1]
                a0 = F.relu(self.a0[i]).expand_as(r_up[i])
                r_up[i] = F.leaky_relu(self.BNUp_step[(self.num_layers-1)*t+i](r_up[i] +
                                       a0 * self.UpConvs[i](pred_err)), 0.2)

        # ----------------
        # infer mask repres
        # ----------------
        r_mask = self.UpConvs[-1](r_up[-1])

        return r_mask


    def forward_test_stage(self, x, fm_vis, cycs_in_test):

        x = self.BNDown[-1](x)

        # ----------------
        # top-down process from spec to middle repres
        # ----------------
        # init r_down[-1]
        r_down = [F.leaky_relu(self.BNDown[-2](self.DownConvs[-1](x)), 0.2)]
        # init r_down[i] (i<-1)
        for i in range(self.num_layers - 2, 0, -1):
            r_down = [F.leaky_relu(self.BNDown[i - 1](self.DownConvs[i](r_down[0])), 0.2)] + r_down

        # ----------------
        # predict fc_vis
        # ----------------
        fc_vis_pred = F.leaky_relu(self.DownConvs[0](r_down[0]), 0.2)

        # ----------------
        # bottom-up process which updates repres with prediction error
        # ----------------
        # update r_up[0]
        fc_vis_pred_err = fm_vis - fc_vis_pred
        a0 = F.relu(self.a0[0]).expand_as(r_down[0])
        r_up = [F.leaky_relu(self.BNUp[0](r_down[0] + a0 * self.UpConvs[0](fc_vis_pred_err)), 0.2)]
        # update r_up[i] (i>0)
        for i in range(1, self.num_layers - 1):
            pred_err = r_up[i - 1] - r_down[i - 1]
            a0 = F.relu(self.a0[i]).expand_as(r_down[i])
            r_up.append(F.leaky_relu(self.BNUp[i](r_down[i] + a0 * self.UpConvs[i](pred_err)), 0.2))

        for t in range(cycs_in_test):

            # ----------------
            # top-down process which updates repres with prediction
            # ----------------
            # update r_up[-1]
            b0 = F.relu(self.b0[-1]).expand_as(r_up[-1])
            r_up[-1] = F.leaky_relu((1 - b0) * r_up[-1] + b0 * r_down[-1], 0.2)
            # update r_down[i] and r_up[i] (i<-1)
            for i in range(self.num_layers - 2, 0, -1):
                r_down[i - 1] = self.DownConvs[i](r_up[i])
                b0 = F.relu(self.b0[i - 1]).expand_as(r_up[i - 1])
                if t < self.cycs_in:
                    BNDown_step_idx = (self.num_layers - 2) * t + i - 1
                else:
                    BNDown_step_idx = (self.num_layers - 2) * (self.cycs_in - 1) + i - 1
                r_up[i - 1] = F.leaky_relu(self.BNDown_step[BNDown_step_idx]((1 - b0) * r_up[i - 1] +
                                           b0 * r_down[i - 1]), 0.2)
            
            # ----------------
            # predict fc_vis
            # ----------------
            fc_vis_pred = F.leaky_relu(self.DownConvs[0](r_up[0]), 0.2)

            # ----------------
            # bottom-up process which updates repres with prediction error
            # ----------------
            # update r_up[0]
            fc_vis_pred_err = fm_vis - fc_vis_pred
            a0 = F.relu(self.a0[0]).expand_as(r_up[0])
            if t < self.cycs_in:
                BNUp_step0_idx = (self.num_layers - 1) * t
            else:
                BNUp_step0_idx = (self.num_layers - 1) * (self.cycs_in - 1)
            r_up[0] = F.leaky_relu(self.BNUp_step[BNUp_step0_idx](r_up[0] +
                                                                  a0 * self.UpConvs[0](fc_vis_pred_err)), 0.2)
            # update r_up[i] (i>0)
            for i in range(1, self.num_layers - 1):
                pred_err = r_up[i - 1] - r_down[i - 1]
                a0 = F.relu(self.a0[i]).expand_as(r_up[i])
                if t < self.cycs_in:
                    BNUp_step_idx = (self.num_layers - 1) * t + i
                else:
                    BNUp_step_idx = (self.num_layers - 1) * (self.cycs_in - 1) + i
                r_up[i] = F.leaky_relu(self.BNUp_step[BNUp_step_idx](r_up[i] +
                                       a0 * self.UpConvs[i](pred_err)), 0.2)

        # ----------------
        # infer mask repres
        # ----------------
        r_mask = self.UpConvs[-1](r_up[-1])

        return r_mask
