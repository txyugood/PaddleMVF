import os
import sys
sys.path.append('.')
import argparse

import torch
import numpy as np
from reprod_log import ReprodLogger
from mmcv import Config
from mmcv.runner import obj_from_dict

from alignment.torch.codes import datasets

from alignment.torch.codes.core import (get_dist_info, init_dist, mean_class_accuracy,
                        multi_gpu_test, single_gpu_test, top_k_accuracy)


parser = argparse.ArgumentParser(description='Model training')
# params of training

parser.add_argument(
    '--dataset_root',
    dest='dataset_root',
    help='The path of dataset root',
    type=str,
    default=None)

args = parser.parse_args()

cfg = Config.fromfile('alignment/torch/configs/MVFNet/K400/mvf_kinetics400_2d_rgb_r50_dense.py')
cfg.data.test.ann_file = os.path.join(args.dataset_root, 'ucf101_val_split_1_rawframes.txt')
cfg.data.test.data_root = os.path.join(args.dataset_root,'rawframes')
dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))


gt_labels = []
for i in range(len(dataset)):
    ann = dataset.video_infos[i]
    gt_labels.append(ann['label'])

results = np.load("alignment/step2/data/results.npy", allow_pickle=True)

new_results = []
for i in range(results.shape[0]):
    new_results.append(results[i])

results = new_results

top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
mean_acc = mean_class_accuracy(results, gt_labels)

reprod_logger = ReprodLogger()
reprod_logger.add("top1_acc", np.array([top1]))
reprod_logger.add('mean_class_accuracy', np.array([mean_acc]))
reprod_logger.save("metric_torch.npy")