# !/usr/bin/env python3
from abc import ABCMeta, abstractmethod

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res

class BaseHead(nn.Layer):
    def __init__(
        self,
        spatial_size=7,
        dropout_ratio=0.8,
        in_channels=1024,
        num_classes=101,
        init_std=0.001,
        extract_feat=False,
        ls_eps=0.1
    ):
        super(BaseHead, self).__init__()
        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = (spatial_size, spatial_size)
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.Logits = None
        self.extract_feat = extract_feat
        self.ls_eps = ls_eps

    @abstractmethod
    def forward(self, x):
        pass

    def init_weights(self):
        pass

    def loss(self, cls_score, labels):
        losses = dict()
        # if labels.shape == torch.Size([]):
        #     labels = labels.unsqueeze(0)

        if cls_score.shape != labels.shape:
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = paddle.to_tensor(
                top_k_acc[0])
            losses['top5_acc'] = paddle.to_tensor(
                top_k_acc[1])
        if self.ls_eps == 0:
            losses['loss_cls'] = F.cross_entropy(cls_score, labels)
        else:
            losses['loss_cls'] = self.label_smooth_loss(cls_score, labels)

        return losses

    def label_smooth_loss(self, scores, labels):
        labels = F.one_hot(labels, self.num_classes)
        labels = F.label_smooth(labels, epsilon=self.ls_eps)
        labels = paddle.squeeze(labels, axis=1)
        # loss = self.loss_func(scores, labels, soft_label=True, **kwargs)
        loss = F.cross_entropy(scores, labels, soft_label=True)
        return loss