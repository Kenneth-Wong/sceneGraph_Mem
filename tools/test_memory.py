from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net, draw_image, draw_gt
from model.config import cfg, cfg_from_file, cfg_from_list
import nets.ISGG as ISGG
import nets.memory_ISGG as memory_ISGG
import nets.ISGG_iscbox as ISGG_iscbox
import nets.ISGG_relmix as ISGG_relmix
import nets.memory_ISGG_relmix as memory_ISGG_relmix
import nets.memory_ISGG_iscbox as memory_ISGG_iscbox
import nets.attend_memory_ISGG_relmix as attend_memory_ISGG_relmix
from datasets.factory import get_imdb, get_vg_imdb
from roi_data_layer.roidb import prepare_roidb
import argparse
import pprint
import time, os, sys

import tensorflow as tf


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a region classification network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn/experiments/cfgs/vgg16.yml', type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        type=str)
    parser.add_argument('--visualize', dest='visualize', help='whether to show results',
                        action='store_true')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='visual_genome_test_rel', type=str)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
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
    parser.add_argument('--test_size', dest='test_size',
                        default=-1, type=int)
    parser.add_argument('--iter_test', dest='iter_test',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


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

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        #filename = os.path.splitext(os.path.basename(args.model))[0]
        filename = '_'.join(args.model.split('/')[-2:])+'_update'
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    split, task = args.imdb_name.split('_')[2:]
    imdb = get_vg_imdb(split, task, num_im=args.test_size, num_val_im=-1)
    roidb = imdb.roidb
    #draw_image(imdb, roidb)
    #draw_gt(imdb, roidb)
    print('Loaded imdb `{:s}` for training'.format(args.imdb_name))
    print('{:d} roidb entries'.format(len(roidb)))
    if cfg.TEST.USE_RPN_DB:
        imdb.add_rpn_rois(roidb, make_copy=False)
    prepare_roidb(roidb)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)

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


    # load model
    net.create_architecture("TEST", imdb.num_classes, imdb.num_predicates, tag='default')

    if args.model:
        print(('Loading model check point from {:s}').format(args.model))
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print('Loaded.')
    else:
        print(('Loading initial weights from {:s}').format(args.weight))
        sess.run(tf.global_variables_initializer())
        print('Loaded.')

    test_net(sess, net, imdb, roidb, filename, False, iter_test=args.iter_test, mode='all')
    print('model %s done' % args.model)
    sess.close()
