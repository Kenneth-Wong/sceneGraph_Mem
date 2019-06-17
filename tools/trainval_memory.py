from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb
from model.train_val_memory import train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from roi_data_layer.roidb import prepare_roidb
import nets.ISGG as ISGG
import nets.memory_ISGG as memory_ISGG
import nets.ISGG_iscbox as ISGG_iscbox
import nets.ISGG_relmix as ISGG_relmix
import nets.memory_ISGG_relmix as memory_ISGG_relmix
import nets.memory_ISGG_iscbox as memory_ISGG_iscbox
import nets.attend_memory_ISGG_relmix as attend_memory_ISGG_relmix
from datasets.factory import get_imdb, get_vg_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a region classification network with memory')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn/experiments/cfgs/vgg16.yml',
                        type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        default='/home/wangwenbin/Method/scene_graph/mem-sg-full/data/frcnn_weights/vgg16_vg_35-49k/vgg16_iter_900000.ckpt',
                        #default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn-relPN/data/frcnn_weights/coco_vgg16_faster_rcnn_final.npy',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='visual_genome_train_rel', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validate on',
                        default='visual_genome_val_rel', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default="exp_debug", type=str)
    parser.add_argument('--extra_tag', dest='extra_tag',
                        help='extra tag of the model',
                        default=None, type=str)
    parser.add_argument('--net_base', dest='net_base',
                        help='vgg16, res50, res101, res152, mobile',
                        default='vgg16', type=str)
    parser.add_argument('--net_arch', dest='net_arch',
                        help='ISGG, BM_ISGG,'
                             'ISGG_iscbox, ISGG_relmix, memory_ISGG_relmix, attend_memory_ISGG_relmix,'
                             'memory_ISGG_iscbox,', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--train_size', dest='train_size',
                        default=-1, type=int)
    parser.add_argument('--val_size', dest='val_size',
                        default=10, type=int)

    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(split, task, num_im, num_val_im):
    """
    Combine multiple roidbs
    """
    imdb = get_vg_imdb(split, task, num_im, num_val_im)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set
    split, task = args.imdb_name.split('_')[2:]
    imdb, roidb = combined_roidb(split, task, num_im=args.train_size, num_val_im=args.val_size)

    print('Loaded imdb `{:s}` for training'.format(args.imdb_name))
    print('{:d} roidb entries'.format(len(roidb)))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    split, task = args.imdbval_name.split('_')[2:]
    valimdb, valroidb = combined_roidb(split, task, num_im=args.val_size, num_val_im=-1)
    cfg.TRAIN.USE_FLIPPED = orgflip
    print('Loaded imdb `{:s}` for validating'.format(args.imdbval_name))
    print('{:d} validation roidb entries'.format(len(valroidb)))

    if args.net_arch == 'ISGG':
        arch = ISGG
    elif args.net_arch == 'BM_ISGG':
        arch = memory_ISGG
    elif args.net_arch == 'ISGG_iscbox':
        arch = ISGG_iscbox
    elif args.net_arch == 'ISGG_relmix':
        arch = ISGG_relmix
    elif args.net_arch == 'memory_ISGG_relmix':
        arch = memory_ISGG_relmix
    elif args.net_arch == 'memory_ISGG_iscbox':
        arch = memory_ISGG_iscbox
    elif args.net_arch == 'attend_memory_ISGG_relmix':
        arch = attend_memory_ISGG_relmix
    else:
        raise NotImplementedError

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag, args.extra_tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag, args.extra_tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # load network
    if args.net_base == 'vgg16':
        net = arch.vgg16()
    else:
        raise NotImplementedError
    """
    elif args.net_base == 'res50':
        net = arch.resnetv1(num_layers=50)
    #elif args.net_base == 'res101':
        net = arch.resnetv1(num_layers=101)
    elif args.net_base == 'res152':
        net = arch.resnetv1(num_layers=152)
    elif args.net_base == 'mobile':
        net = arch.mobilenetv1()
    """
    train_net(net, imdb, roidb, valimdb, valroidb, output_dir, tb_dir,
              pretrained_model=args.weight,
              max_iters=args.max_iters)
