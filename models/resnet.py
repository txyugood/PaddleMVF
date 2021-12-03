"""resnet"""
import paddle
import paddle.nn as nn




def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias_attr=False)


class BasicBlock(nn.Layer):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)

        self.bn1 = nn.BatchNorm2D(planes)

        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2D(planes)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp


    def forward(self, x):
        """forward"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers
        stride (int): Spatial stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        downsample (obj): Downsample layer. Default None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 avd=False,
                 avd_first=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.avd = avd and stride > 1
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2D(3, stride, padding=1)
            stride = 1


        self.conv1_stride = 1
        self.conv2_stride = stride

        self.conv1 = nn.Conv2D(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias_attr=False)
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias_attr=False)

        self.bn1 = nn.BatchNorm2D(planes)
        self.bn2 = nn.BatchNorm2D(planes)

        self.conv3 = nn.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias_attr=False)

        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x):
        """forward"""
        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            if self.avd and self.avd_first:
                out = self.avd_layer(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            if self.avd and not self.avd_first:
                out = self.avd_layer(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)
        out = self.relu(out)
        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   norm_cfg=None,
                   with_cp=False,
                   avg_down=False,
                   avd=False,
                   avd_first=False):
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Default 1.
        dilation (int): Spacing between kernel elements. Default 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default 'pytorch'.
        norm_cfg (dict): Config for norm layers. required keys are `type`,
            Default None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.

    Returns:
        A residual layer for the given config.
    """
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:

        down_layers = []
        if avg_down:
            if dilation == 1:
                down_layers.append(
                    nn.AvgPool2D(kernel_size=stride,
                                 stride=stride,
                                 ceil_mode=True))
            else:
                down_layers.append(
                    nn.AvgPool2D(kernel_size=1,
                                 stride=1,
                                 ceil_mode=True))
            down_layers.append(
                nn.Conv2D(inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias_attr=False))
        else:
            down_layers.append(
                nn.Conv2D(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False))
        down_layers.append(nn.BatchNorm2D(planes * block.expansion))
        downsample = nn.Sequential(*down_layers)

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
            avd=avd,
            avd_first=avd_first))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation,
                  norm_cfg=norm_cfg, with_cp=with_cp,
                  avd=avd, avd_first=avd_first))

    return nn.Sequential(*layers)


class ResNet(nn.Layer):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str): Name of pretrained model. Default None.
        num_stages (int): Resnet stages. Default 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default `pytorch`.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default -1.
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default True.
        bn_frozen (bool): Whether to freeze weight and bias of BN layersn
            Default False.
        partial_bn (bool): Whether to freeze weight and bias of **all
            but the first** BN layersn Default False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default False.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 norm_frozen=False,
                 partial_norm=False,
                 with_cp=False,
                 avg_down=False,
                 avd=False,
                 avd_first=False,
                 deep_stem=False,
                 stem_width=64):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.norm_frozen = norm_frozen
        self.partial_norm = partial_norm
        self.with_cp = with_cp

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_width * 2 if deep_stem else 64

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2D(3, stem_width, kernel_size=3, stride=2,
                          padding=1, bias_attr=False),
                nn.BatchNorm2D(stem_width),
                nn.ReLU(),
                nn.Conv2D(stem_width, stem_width, kernel_size=3,
                          stride=1, padding=1, bias_attr=False),
                nn.BatchNorm2D(stem_width),
                nn.ReLU(),
                nn.Conv2D(stem_width, stem_width * 2, kernel_size=3,
                          stride=1, padding=1, bias_attr=False),
            )
        else:
            self.conv1 = nn.Conv2D(
                3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)

        self.bn1 = nn.BatchNorm2D(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                avg_down=avg_down,
                avd=avd,
                avd_first=avd_first)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_sublayer(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1)


    def init_weights(self):
        pass

    def forward(self, x):
        """forward"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)



if __name__ == '__main__':
    net = ResNet(depth=50, out_indices=(3,),norm_eval=False, partial_norm=False)
    net.train()
    img = paddle.rand([1,3,224,224])
    out = net(img)
    print(out.shape)
    pass
