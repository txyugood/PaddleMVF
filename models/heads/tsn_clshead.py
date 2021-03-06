"""cls head for 2D input (B C H W)"""
import paddle
import paddle.nn as nn
from models.heads.base import BaseHead
from models.heads.segmental_consensuses import SimpleConsensus

class TSNClsHead(BaseHead):
    """ cls head for 2D input"""

    def __init__(
        self,
        spatial_type='avg',
        spatial_size=7,
        consensus_cfg=dict(type='avg', dim=1),
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.8,
        in_channels=1024,
        num_classes=101,
        init_std=0.001,
        fcn_testing=False,
        extract_feat=False,
    ):
        super(TSNClsHead, self).__init__(spatial_size, dropout_ratio,
                                         in_channels, num_classes, init_std,
                                         extract_feat)
        self.spatial_type = spatial_type
        self.consensus_type = consensus_cfg['type']
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.cls_pool_size = (self.temporal_feature_size,
                              self.spatial_feature_size,
                              self.spatial_feature_size)
        self.with_avg_pool = with_avg_pool


        self.segmental_consensus = SimpleConsensus(
            self.consensus_type, consensus_cfg['dim'])

        if self.spatial_size == -1:
            self.pool_size = (1, 1)
            if self.spatial_type == 'avg':
                self.Logits = nn.AdaptiveAvgPool2D(self.pool_size)
            if self.spatial_type == 'max':
                self.Logits = nn.AdaptiveMaxPool2D(self.pool_size)
        else:
            self.pool_size = self.spatial_size
            if self.spatial_type == 'avg':
                self.Logits = nn.AvgPool2D(
                    self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.Logits = nn.MaxPool2D(
                    self.pool_size, stride=1, padding=0)
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3D(self.cls_pool_size)
        if self.consensus_type in ['TRN', 'TRNmultiscale']:
            img_feature_dim = 256
            self.new_fc = nn.Linear(self.in_channels, img_feature_dim)
        else:
            self.new_fc = nn.Linear(self.in_channels, self.num_classes)
        self.fcn_testing = fcn_testing
        self.new_cls = None

    def forward(self, x, num_seg):
        """forward"""
        if not self.fcn_testing:
            # [4*3*10 2048 7 7]
            x = self.Logits(x)
            # [4*3*10 2048 1 1]
            if x.ndimension() == 4:
                x = x.unsqueeze(2)
                # [8*10 2048 1 1 1]
            assert x.shape[1] == self.in_channels
            assert x.shape[2] == self.temporal_feature_size
            assert x.shape[3] == self.spatial_feature_size
            assert x.shape[4] == self.spatial_feature_size
            if self.with_avg_pool:
                x = self.avg_pool(x)
            if self.dropout is not None:
                x = self.dropout(x)
            # x = x.view(x.size(0), -1)  # [4*3*10 2048]
            x = x.reshape([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]])
            if self.extract_feat:
                cls_score = x  # [4*3*10 2048]
            else:
                cls_score = self.new_fc(x)  # [4*3*10 400]

            cls_score = cls_score.reshape(
                [-1, num_seg] + list(cls_score.shape[1:]))  # [3*10 4 400]
            cls_score = self.segmental_consensus(cls_score)  # [3*10 1 400]
            cls_score = cls_score.squeeze(1)  # [30 400]
            return cls_score
        else:
            # [3*10 2048 4 8 8]
            if self.new_cls is None:
                self.new_cls = nn.Conv3D(
                    self.in_channels,
                    self.num_classes,
                    1, 1, 0)
                weight = self.new_fc.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                weight = paddle.transpose(weight, [1, 0, 2, 3, 4])
                initializer = paddle.nn.initializer.Assign(weight)
                initializer(self.new_cls.weight)
                initializer = paddle.nn.initializer.Assign(self.new_fc.bias)
                initializer(self.new_cls.bias)
                # self.new_cls.load_state_dict(
                #     {'weight': self.new_fc.weight.unsqueeze(-1).unsqueeze(
                #         -1).unsqueeze(-1),
                #      'bias': self.new_fc.bias})
            if self.extract_feat:
                feature = x.mean([2, 3, 4])  # [3*10 2048]
                return feature
            else:
                class_map = self.new_cls(x)
                # [3*10 400 4 8 8]
                cls_score = class_map.mean([2, 3, 4])  # [3*10 400]
                return cls_score

    def init_weights(self):
        """init weight"""
        pass
        # nn.init.normal_(self.new_fc.weight, 0, self.init_std)
        # nn.init.constant_(self.new_fc.bias, 0)

if __name__ == '__main__':
    head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                      with_avg_pool=False,
                      temporal_feature_size=1,
                      spatial_feature_size=1,
                      dropout_ratio=0.5,
                      in_channels=2048,
                      init_std=0.01,
                      num_classes=400)
    img = paddle.rand([1,2048,7,7])
    out = head(img, 1)
    print(out.shape)
