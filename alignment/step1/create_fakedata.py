import sys
import random
from collections import OrderedDict
sys.path.append('.')

import numpy as np
import paddle
import torch
from mmcv import Config

from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
from models.recognizers.recognizer2d import Recognizer2D
from alignment.torch.codes.models import build_recognizer

paddle.seed(0)
np.random.seed(0)
random.seed(0)

img = paddle.rand([2, 16, 3, 224, 224])
img = img.numpy()

np.save("alignment/step1/data/fake_data.npy", img)

target = paddle.randint(low=0, high=400, shape=[2, 1])
target = target.numpy()
np.save("alignment/step1/data/fake_label.npy", target)

results = []
for i in range(4):
    results.extend(np.random.randn(1, 400))

np.save("alignment/step2/data/results.npy", results)

cfg = Config.fromfile('alignment/torch/configs/MVFNet/K400/mvf_kinetics400_2d_rgb_r50_dense.py')

torch_model = build_recognizer(cfg.model)

backbone = ResNet(depth=50, out_indices=(3,), norm_eval=False, partial_norm=False)
head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                  with_avg_pool=False,
                  temporal_feature_size=1,
                  spatial_feature_size=1,
                  dropout_ratio=0.5,
                  in_channels=2048,
                  init_std=0.01,
                  num_classes=101)
paddle_model = Recognizer2D(backbone=backbone, cls_head=head,
                     module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'))

torch_state_dict = torch_model.state_dict()
paddle_state_dict = paddle_model.state_dict()
torch.save(torch_state_dict, "alignment/step1/data/fake_torch_weights.pth")

paddle_list = paddle_state_dict.keys()
paddle_state_dict = OrderedDict()

torch_list = torch_state_dict.keys()
torch_list = [l for l in list(torch_list) if "num_batches_tracked" not in l]

for i, p in enumerate(paddle_list):

    p = p.strip()
    t = p
    if "mean" in p:
        t = p.replace("_mean", "running_mean")
    if "variance" in p:
        t = p.replace("_variance", "running_var")
    if "fc" not in p:
        paddle_state_dict[p] = torch_state_dict[list(torch_list)[i]].detach().cpu().numpy()
    else:
        paddle_state_dict[p] = torch_state_dict[list(torch_list)[i]].detach().cpu().numpy().T


f = open("alignment/step1/data/fake_paddle_weights.pdparams", 'wb')
import pickle
pickle.dump(paddle_state_dict, f)
f.close()


