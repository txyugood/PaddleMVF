import os
import argparse

import numpy as np
import paddle

from datasets import SampleFrames, RawFrameDecode, Resize, ThreeCrop, Normalize, FormatShape, Collect
from datasets import RawframeDataset
from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
from models.recognizers.recognizer2d import Recognizer2D
from utils import load_pretrained_model
from progress_bar import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/Users/alex/baidu/mmaction2/data/ucf101/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)
    
    parser.add_argument(
        '--split',
        dest='split',
        help='split',
        type=int,
        default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tranforms = [
        SampleFrames(clip_len=16, frame_interval=4, num_clips=10, test_mode=True),
        RawFrameDecode(),
        Resize(scale=(np.Inf, 256), keep_ratio=True),
        ThreeCrop(crop_size=256),
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
        FormatShape(input_format='NCHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, f'ucf101_val_split_{args.split}_rawframes.txt'),
                              pipeline=tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"))

    loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        return_list=True,
    )

    backbone = ResNet(depth=50, out_indices=(3,), norm_eval=False, partial_norm=False)
    head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                      with_avg_pool=False,
                      temporal_feature_size=1,
                      spatial_feature_size=1,
                      dropout_ratio=0.5,
                      in_channels=2048,
                      init_std=0.01,
                      num_classes=101,
                      fcn_testing=True)
    model = Recognizer2D(backbone=backbone, cls_head=head,fcn_testing=True,
                         module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'),
                         test_cfg=dict(average_clips='prob'))
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    model.eval()
    results = []
    prog_bar = ProgressBar(len(dataset))
    for batch_id, data in enumerate(loader):
        with paddle.no_grad():
            imgs = data['imgs']
            label = data['label']
            result = model(imgs, label, return_loss=False)
        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    eval_res = dataset.evaluate(results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')
