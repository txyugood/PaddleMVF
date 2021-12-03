"""
Code for "MVFNet: Multi-View Fusion Network for Efficient Video Recognition"
https://arxiv.org/pdf/2012.06977.pdf

Authors: Wenhao Wu (wuwenhao17@mails.ucas.edu.cn)
"""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class HardSigmoid(nn.Layer):
    """h_sigmoid"""
    def __init__(self):
        super(HardSigmoid, self).__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        """forward"""
        return self.relu(x + 3) / 6

class HardSwish(nn.Layer):
    """h_swish"""
    def __init__(self):
        super(HardSwish, self).__init__()
        self.sigmoid = HardSigmoid()

    def forward(self, x):
        """forward"""
        return x * self.sigmoid(x)



def make_multi_view_fusion(
        net, n_segment, alpha, mvf_freq=(1, 1, 1, 1),
        use_hs=True, share=False, mode='THW'):
    """Insert MVF module to ResNet"""
    n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    print('=> n_segment per stage: {}'.format(n_segment_list))

    n_round = 1
    if len(list(net.layer3.children())) >= 23:  # R101 & R152
        # n_round = 2
        print('=> Using n_round {} to insert MVF module'.
                format(n_round))

    def make_block_MVF(stage, this_segment):
        """build MVF Block"""
        blocks = list(stage.children())
        print('=> Processing stage with {} {} blocks residual'.format(len(blocks), mode))
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = MVF(
                    b.conv1, this_segment, blocks[i].conv1._in_channels, alpha, use_hs, share, mode)
        return nn.Sequential(*blocks)

    net.layer1 = make_block_MVF(
        net.layer1, n_segment_list[0]) if mvf_freq[0] else net.layer1
    net.layer2 = make_block_MVF(
        net.layer2, n_segment_list[1]) if mvf_freq[1] else net.layer2
    net.layer3 = make_block_MVF(
        net.layer3, n_segment_list[2]) if mvf_freq[2] else net.layer3
    net.layer4 = make_block_MVF(
        net.layer4, n_segment_list[3]) if mvf_freq[3] else net.layer4



class MVF(nn.Layer):
    """MVF Module"""
    def __init__(self, net, n_segment, in_channels, alpha=0.5, use_hs=True, share=False, mode='THW'):
        super(MVF, self).__init__()
        self.net = net
        self.n_segment = n_segment
        num_shift_channel = int(in_channels * alpha)
        self.num_shift_channel = num_shift_channel
        self.share = share
        if self.num_shift_channel != 0:
            self.split_sizes = [num_shift_channel, in_channels - num_shift_channel]

            self.shift_conv = nn.Conv3D(
                num_shift_channel, num_shift_channel, [3, 1, 1], stride=1,
                padding=[1, 0, 0], groups=num_shift_channel, bias_attr=False)

            self.bn = nn.BatchNorm3D(num_shift_channel)
            self.use_hs = use_hs
            self.activation = HardSwish() if use_hs else nn.ReLU()
            self.mode = mode

            if not self.share:
                if self.mode == 'THW':
                    self.h_conv = nn.Conv3D(
                        num_shift_channel, num_shift_channel, [1, 3, 1], stride=1,
                        padding=[0, 1, 0], groups=num_shift_channel, bias_attr=False)
                    self.w_conv = nn.Conv3D(
                        num_shift_channel, num_shift_channel, [1, 1, 3], stride=1,
                        padding=[0, 0, 1], groups=num_shift_channel, bias_attr=False)
                elif self.mode == 'T':
                    pass
                elif self.mode == 'TH':
                    self.h_conv = nn.Conv3D(
                        num_shift_channel, num_shift_channel, [1, 3, 1], stride=1,
                        padding=[0, 1, 0], groups=num_shift_channel, bias_attr=False)
            self._initialize_weights()
        print('=> Using Multi-view Fusion...')

    def _initialize_weights(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3D):
        #         # kaiming_init(m)
        #         n = m.kernel_size[0] * m.kernel_size[1] * \
        #             m.kernel_size[2] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm3D):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        """forward"""
        nt, c, h, w = x.shape
        n_batch = nt // self.n_segment
        if self.num_shift_channel != 0:
            # x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
            x = x.reshape([n_batch, self.n_segment, c, h, w])
            x = x.transpose([0, 2, 1, 3, 4])
            x = list(x.split(self.split_sizes, axis=1))

            if self.mode == 'THW':
                # get H & W
                if self.share:
                    tmp_h = self.shift_conv(x[0].transpose(2, 3)).transpose(2, 3)
                    tmp_w = self.shift_conv(x[0].permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
                else:
                    tmp_h = self.h_conv(x[0])
                    tmp_w = self.w_conv(x[0])
                x[0] = self.shift_conv(x[0]) + tmp_h + tmp_w
            elif self.mode == 'T':
                x[0] = self.shift_conv(x[0])
            elif self.mode == 'TH':
                # get H & W
                if self.share:
                    tmp_h = self.shift_conv(x[0].transpose(2, 3)).transpose(2, 3)
                else:
                    tmp_h = self.h_conv(x[0])
                x[0] = self.shift_conv(x[0]) + tmp_h

            if self.use_hs:
                # add bn and activation
                x[0] = self.bn(x[0])
                x[0] = self.activation(x[0])
            x = paddle.concat(x, axis=1)  # n, c, t, h, w

            # x = x.transpose(1, 2).contiguous().view(nt, c, h, w)
            x = x.transpose([0, 2, 1, 3, 4])
            x = x.reshape([nt, c, h, w])
        return self.net(x)
