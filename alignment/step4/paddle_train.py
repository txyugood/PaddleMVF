import sys
sys.path.append('.')

import paddle
import numpy as np


from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
from models.recognizers.recognizer2d import Recognizer2D

from reprod_log import ReprodLogger
from utils import load_pretrained_model

reprod_logger = ReprodLogger()

backbone = ResNet(depth=50, out_indices=(3,), norm_eval=False, partial_norm=False)
head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                  with_avg_pool=False,
                  temporal_feature_size=1,
                  spatial_feature_size=1,
                  dropout_ratio=0.5,
                  in_channels=2048,
                  init_std=0.01,
                  num_classes=400)
model = Recognizer2D(backbone=backbone, cls_head=head,
                     module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'))


load_pretrained_model(model, 'alignment/step1/data/fake_paddle_weights.pdparams')
model.eval()

fake_data = np.load('alignment/step1/data/fake_data.npy')
fake_data = paddle.to_tensor(fake_data)

fake_label = np.load('alignment/step1/data/fake_label.npy')
fake_label = paddle.to_tensor(fake_label)
loss_list = []

model.eval()
max_iters = 5

optimizer = paddle.optimizer.Momentum(learning_rate=1e-3, weight_decay=5e-4, parameters=model.parameters(),
                                      momentum=0.9)

for idx in range(max_iters):

    out = model(fake_data, fake_label, return_loss=True)
    loss = out['loss_cls']
    loss.backward()
    optimizer.step()
    model.clear_gradients()
    loss_list.append(loss)

reprod_logger = ReprodLogger()
for idx, loss in enumerate(loss_list):
    reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
reprod_logger.save("alignment/step4/bp_align_paddle.npy")

