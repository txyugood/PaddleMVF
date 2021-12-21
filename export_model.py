import os
import argparse

import paddle

from models.resnet import ResNet
from models.heads.tsn_clshead import TSNClsHead
from models.recognizers.recognizer2d import Recognizer2D
from utils import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):

    backbone = ResNet(depth=50, out_indices=(3,), norm_eval=False, partial_norm=False)
    head = TSNClsHead(spatial_size=-1, spatial_type='avg',
                      with_avg_pool=False,
                      temporal_feature_size=1,
                      spatial_feature_size=1,
                      dropout_ratio=0.5,
                      in_channels=2048,
                      init_std=0.01,
                      num_classes=101)
    net = Recognizer2D(backbone=backbone, cls_head=head,
                         module_cfg=dict(type='MVF', n_segment=16, alpha=0.125, mvf_freq=(0, 0, 1, 1), mode='THW'))

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        print('Loaded trained params of model successfully.')


    shape = [-1, 16, 3, 224, 224]

    new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32'), None, False, False])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    # yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    # with open(yml_file, 'w') as file:
    #     transforms = cfg.export_config.get('transforms', [{
    #         'type': 'Normalize'
    #     }])
    #     data = {
    #         'Deploy': {
    #             'transforms': transforms,
    #             'model': 'model.pdmodel',
    #             'params': 'model.pdiparams'
    #         }
    #     }
    #     yaml.dump(data, file)

    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)