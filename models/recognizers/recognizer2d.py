"""recognizer2d"""
import paddle
import paddle.nn as nn
from models.recognizers.base import BaseRecognizer


class Recognizer2D(BaseRecognizer):
    """class for recognizer2d"""

    def __init__(self,
                 modality='RGB',
                 backbone='BNInception',
                 cls_head='TSNClsHead',
                 fcn_testing=False,
                 module_cfg=None,
                 nonlocal_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Recognizer2D, self).__init__(backbone, cls_head)
        self.fcn_testing = fcn_testing
        self.modality = modality
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.module_cfg = module_cfg

        # insert module into backbone
        if self.module_cfg:
            self._prepare_base_model(backbone, self.module_cfg, nonlocal_cfg)

        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.in_channels = 3

    def _prepare_base_model(self, backbone, module_cfg, nonlocal_cfg):
        # module_cfg, example:
        # tsm: dict(type='tsm', n_frames=8 , n_div=8,
        #           shift_place='blockres',
        #           temporal_pool=False, two_path=False)
        # nolocal: dict(n_segment=8)
        backbone_name = 'ResNet'
        module_name = module_cfg.pop('type')
        self.module_name = module_name
        if backbone_name == 'ResNet':
            # Add module for 2D backbone
            if module_name == 'MVF':
                print('Adding MVF module...')
                from models.recognizers.MVF import make_multi_view_fusion
                make_multi_view_fusion(self.backbone, **module_cfg)

    def forward_train(self, imgs, labels, **kwargs):
        """train"""
        #  [B S C H W]
        #  [BS C H W]
        num_batch = imgs.shape[0]
        imgs = imgs.reshape([-1, self.in_channels] + imgs.shape[3:])
        num_seg = imgs.shape[0] // num_batch

        x = self.extract_feat(imgs)  # 64 2048 7 7
        losses = dict()
        if self.with_cls_head:
            temporal_pool = imgs.shape[0] // x.shape[0]
            cls_score = self.cls_head(x, num_seg // temporal_pool)
            gt_label = labels.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, return_numpy, **kwargs):
        """test"""
        #  imgs: [B tem*crop*clip C H W]
        #  imgs: [B*tem*crop*clip C H W]
        num_batch = imgs.shape[0]
        imgs = imgs.reshape([-1, self.in_channels] + imgs.shape[3:])
        num_frames = imgs.shape[0] // num_batch
        x = self.extract_feat(imgs)
        if self.with_cls_head:
            temporal_pool = imgs.shape[0] // x.shape[0]
            if self.module_cfg:
                if self.fcn_testing:
                    # view to 3D, [120, 2048, 8, 8] -> [30, 4, 2048, 8, 8]
                    x = x.reshape(
                        (-1, self.module_cfg['n_segment'] // temporal_pool) + x.shape[1:])
                    x = x.transpose(1, 2)  # [30, 2048, 4, 8, 8]
                    cls_score = self.cls_head(
                        x, self.module_cfg['n_segment'] // temporal_pool)  # [30 400]
                else:
                    # [120 2048 8 8] ->  [30 400]
                    cls_score = self.cls_head(
                        x, self.module_cfg['n_segment'] // temporal_pool)
            else:
                cls_score = self.cls_head(x, num_frames // temporal_pool)
            cls_score = self.average_clip(cls_score)
        if return_numpy:
            return cls_score.cpu().numpy()
        else:
            return cls_score


from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
import numpy as np
if __name__ == '__main__':
    backbone = ResNet(depth=50, out_indices=(3,), norm_eval=False, partial_norm=False)
    head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                      with_avg_pool=False,
                      temporal_feature_size=1,
                      spatial_feature_size=1,
                      dropout_ratio=0.5,
                      in_channels=2048,
                      init_std=0.01,
                      num_classes=400)
    net = Recognizer2D(backbone=backbone, cls_head=head,
                       module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'))

    net.train()
    img = paddle.rand([2,16,3,224,224])
    label = np.random.randn(2).astype('int64')
    output = net(img, paddle.to_tensor(label))
    pass


