import sys
sys.path.insert(0, '.')
import os
import argparse

import paddle
import numpy as np
from reprod_log import ReprodLogger

from datasets import SampleFrames, RawFrameDecode, Resize, RandomResizedCrop, CenterCrop, Flip, Normalize, FormatShape, Collect
from datasets import RawframeDataset

parser = argparse.ArgumentParser(description='Model training')
# params of training

parser.add_argument(
    '--dataset_root',
    dest='dataset_root',
    help='The path of dataset root',
    type=str,
    default=None)

args = parser.parse_args()

val_tranforms = [
    SampleFrames(clip_len=16, frame_interval=4, num_clips=1, test_mode=True),
    RawFrameDecode(),
    Resize(scale=(np.Inf, 256), keep_ratio=True),
    CenterCrop(crop_size=224),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
    FormatShape(input_format='NCHW'),
    Collect(keys=['imgs', 'label'], meta_keys=[])
]
val_dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, f'ucf101_val_split_1_rawframes.txt'),
                              pipeline=val_tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"),
                              test_mode=True)

batch_size = 1
val_loader = paddle.io.DataLoader(val_dataset,
                                  batch_size=batch_size, shuffle=False, drop_last=False, return_list=True)

results = np.load("alignment/step2/data/results.npy", allow_pickle=True)

new_results = []
for i in range(results.shape[0]):
    new_results.append(results[i])

results = new_results

key_score = val_dataset.evaluate(results, metrics=['top_k_accuracy', 'mean_class_accuracy'])

reprod_logger = ReprodLogger()
reprod_logger.add("top1_acc", np.array([key_score['top1_acc']]))
reprod_logger.add('mean_class_accuracy', np.array([key_score['mean_class_accuracy']]))
reprod_logger.save("alignment/step2/metric_paddle.npy")