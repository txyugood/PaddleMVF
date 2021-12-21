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
out = model(fake_data, None, return_loss=False, return_numpy=False)

reprod_logger.add("logits", out.detach().numpy())
reprod_logger.save("alignment/step1/forward_torch.npy")