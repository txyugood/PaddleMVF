"""base recognizer"""
from abc import abstractmethod

import paddle.nn as nn
import paddle.nn.functional as F


class BaseRecognizer(nn.Layer):
    """Abstract base class for recognizers"""

    def __init__(self, backbone, cls_head):
        super(BaseRecognizer, self).__init__()
        self.fp16_enabled = False
        self.backbone = backbone
        if cls_head is not None:
            self.cls_head = cls_head
        self.init_weights()

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @abstractmethod
    def forward_train(self, imgs, label, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, imgs, **kwargs):
        pass

    def init_weights(self):
        self.backbone.init_weights()
        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def average_clip(self, cls_score):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.

        return:
            torch.Tensor: Averaged class score.
        """
        if self.test_cfg is None:
            self.test_cfg = {}
            self.test_cfg['average_clips'] = None

        if 'average_clips' not in self.test_cfg.keys():
            # self.test_cfg['average_clips'] = None
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, axis=1).mean(axis=0, keepdim=True)
        elif average_clips == 'score':
            cls_score = cls_score.mean(axis=0, keepdim=True)
        return cls_score

    def forward(self, img_group, label, return_loss=True,
                return_numpy=True, **kwargs):
        if return_loss:
            return self.forward_train(img_group, label, **kwargs)
        else:
            return self.forward_test(img_group, return_numpy, **kwargs)
