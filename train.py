import os
import time
import argparse
import random

import paddle
import numpy as np

from datasets import SampleFrames, RawFrameDecode, Resize, RandomResizedCrop, CenterCrop, Flip, Normalize, FormatShape, Collect
from datasets import RawframeDataset
from timer import TimeAverager, calculate_eta
from utils import load_pretrained_model
from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
from models.recognizers.recognizer2d import Recognizer2D
from custom_lr import CustomWarmupCosineDecay
from precise_bn import do_preciseBN


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
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=2
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=100
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=1234
    )

    parser.add_argument(
        '--split',
        dest='split',
        help='split annotation of ucf101',
        type=int,
        default=1
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tranforms = [
        SampleFrames(clip_len=16, frame_interval=4, num_clips=1),
        RawFrameDecode(),
        Resize(scale=(-1, 256)),
        RandomResizedCrop(),
        Resize(scale=(224, 224), keep_ratio=False),
        Flip(flip_ratio=0.5),
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
        FormatShape(input_format='NCHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, f'ucf101_train_split_{args.split}_rawframes.txt'),
                              pipeline=tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"))

    val_tranforms = [
        SampleFrames(clip_len=16, frame_interval=4, num_clips=1, test_mode=True),
        RawFrameDecode(),
        Resize(scale=(np.Inf, 256), keep_ratio=True),
        CenterCrop(crop_size=224),
        Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
        FormatShape(input_format='NCHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    val_dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, f'ucf101_val_split_{args.split}_rawframes.txt'),
                                  pipeline=val_tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"),
                                  test_mode=True)

    backbone = ResNet(depth=50, out_indices=(3,), norm_eval=False, partial_norm=False)
    head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                      with_avg_pool=False,
                      temporal_feature_size=1,
                      spatial_feature_size=1,
                      dropout_ratio=0.5,
                      in_channels=2048,
                      init_std=0.01,
                      num_classes=101)
    model = Recognizer2D(backbone=backbone, cls_head=head,
                         module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'))

    head.new_fc.bias.optimize_attr['learning_rate'] = 1.0
    head.new_fc.weight.optimize_attr['learning_rate'] = 1.0

    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    batch_size = args.batch_size
    train_loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        return_list=True,
    )

    iters_per_epoch = len(train_loader)
    val_loader = paddle.io.DataLoader(val_dataset,
                                      batch_size=batch_size, shuffle=False, drop_last=False, return_list=True)

    max_epochs = args.max_epochs

    lr = CustomWarmupCosineDecay(warmup_start_lr=0,
                                 cosine_base_lr=0.00125,
                                 warmup_epochs=5,
                                 max_epoch=max_epochs,
                                 num_iters=len(train_loader))

    grad_clip = paddle.nn.ClipGradByNorm(40)
    optimizer = paddle.optimizer.Momentum(learning_rate=lr, weight_decay=5e-4, parameters=model.parameters(),
                                          momentum=0.9, use_nesterov=True,
                                          grad_clip=grad_clip)

    epoch = 1

    log_iters = args.log_iters
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()

    iters = iters_per_epoch * max_epochs
    iter = 0
    batch_start = time.time()
    best_accuracy = -0.01
    while epoch <= max_epochs:
        total_loss = 0.0
        total_acc = 0.0
        model.train()
        for batch_id, data_batch in enumerate(train_loader):
            reader_cost_averager.record(time.time() - batch_start)
            iter += 1

            imgs = data_batch['imgs']
            label = data_batch['label']
            outputs = model(imgs, label, return_loss=True)
            loss = outputs['loss_cls']
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            lr.step()
            total_loss += loss.numpy()[0]
            total_acc += outputs['top1_acc'].numpy()[0]

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            if iter % log_iters == 0:
                avg_loss = total_loss / (batch_id + 1)
                avg_acc = total_acc / (batch_id + 1)
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)

                print(
                    "[TRAIN] epoch={}, batch_id={}, loss={:.6f}, lr={:.6f},acc={:.3f}, "
                    "avg_reader_cost: {:.3f} sec, avg_batch_cost: {:.3f} sec, avg_samples: {}, avg_ips: {:.3f} images/sec  ETA {}"
                        .format(epoch, batch_id + 1,
                                avg_loss, optimizer.get_lr(), avg_acc,
                                avg_train_reader_cost, avg_train_batch_cost,
                                batch_size, batch_size / avg_train_batch_cost,
                                eta))
                reader_cost_averager.reset()
                batch_cost_averager.reset()
            batch_start = time.time()
        if epoch % 5 == 0:
            do_preciseBN(
                model, train_loader, False,
                min(200, len(train_loader)))

        model.eval()
        results = []
        total_val_avg_loss = 0.0
        total_val_avg_acc = 0.0
        for batch_id, data in enumerate(val_loader):
            with paddle.no_grad():
                # outputs = model.val_step(data, optimizer)
                imgs = data['imgs']
                label = data['label']
                result = model(imgs, label, return_loss=False)
            results.extend(result)
        print(f"[EVAL] epoch={epoch}")
        key_score = val_dataset.evaluate(results, metrics=['top_k_accuracy', 'mean_class_accuracy'])

        if key_score['mean_class_accuracy'] > best_accuracy:
            print("Save best model.")
            best_accuracy = key_score['mean_class_accuracy']
            current_save_dir = os.path.join("output", f'best_model_split_{args.split}')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        epoch += 1
