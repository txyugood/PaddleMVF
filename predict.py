import os
import argparse
import glob
import json

import numpy as np
import paddle

from datasets import SampleFrames, RawFrameDecode, Resize, ThreeCrop, Normalize, FormatShape, Collect, Compose
from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
from models.recognizers.recognizer2d import Recognizer2D
from utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model predict')

    parser.add_argument(
        '--video',
        dest='video',
        help='The path of video',
        type=str,
        default=None)

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

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

    tranforms = Compose(tranforms)
    video_info = {'frame_dir': args.video}
    total_frames = len(glob.glob(os.path.join(args.video, "*.jpg")))
    video_info['total_frames'] = total_frames
    video_info['start_index'] = 1
    video_info['filename_tmpl'] = 'img_{:05}.jpg'
    video_info['modality'] = 'RGB'
    video_info['label'] = -1
    results = tranforms(video_info)

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
    model = Recognizer2D(backbone=backbone, cls_head=head, fcn_testing=True,
                         module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'),
                         test_cfg=dict(average_clips='prob'))
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)
    with open('labels.json', 'r') as f:
        label_name = json.load(f)

    model.eval()

    with paddle.no_grad():
        imgs = results['imgs']
        imgs = paddle.to_tensor(imgs)
        imgs = paddle.unsqueeze(imgs, axis=0)
        prob = model(imgs, None, return_loss=False, return_numpy=False)
        top1 = paddle.argmax(prob, axis=-1)
        top1 = top1.detach().numpy()[0]
        prob = prob.detach().numpy()[0][top1]
        print("Top1 class:{} prob:{:.6f}".format(label_name[str(top1)], prob))
