import sys
sys.path.append('.')

import torch
import numpy as np

from mmcv import Config
from reprod_log import ReprodLogger

from alignment.torch.codes.models import build_recognizer


reprod_logger = ReprodLogger()

cfg = Config.fromfile('alignment/torch/configs/MVFNet/K400/mvf_kinetics400_2d_rgb_r50_dense.py')

model = build_recognizer(cfg.model)
model.eval()
state_dict = torch.load('alignment/step1/data/fake_torch_weights.pth')
model.load_state_dict(state_dict)

fake_data = np.load('alignment/step1/data/fake_data.npy')
fake_data = torch.from_numpy(fake_data)

fake_label = np.load('alignment/step1/data/fake_label.npy')
fake_label = torch.from_numpy(fake_label)
loss_list = []

optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
max_iters = 5
model.eval()
for idx in range(max_iters):
    out = model(fake_data, fake_label, return_loss=True, return_numpy=False)
    loss = out['loss_cls']
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss)

reprod_logger = ReprodLogger()
for idx, loss in enumerate(loss_list):
    reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
reprod_logger.save("alignment/step4/bp_align_torch.npy")