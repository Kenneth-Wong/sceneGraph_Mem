from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test_coco import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
import nets.base_memory as base_memory
import nets.attend_memory as attend_memory
import nets.attend_memory_ISGG as attend_memory_ISGG
import nets.attend_ISGG_iscbox as attend_ISGG_iscbox
import nets.ISGG as ISGG
import nets.memory_ISGG as memory_ISGG
import nets.memory_ISGG_v2 as memory_ISGG_v2
import nets.memory_ISGG_no_relmem as memory_ISGG_no_relmem
import nets.memory_ISGG_no_relmem_v2 as memory_ISGG_no_relmem_v2
import nets.memory_ISGG_no_objmem as memory_ISGG_no_objmem
import nets.alt_memory_ISGG as alt_memory_ISGG
import nets.ISGG_CBAM_iscbox as ISGG_CBAM_iscbox
import nets.ISGG_CBAM_v2 as ISGG_CBAM_v2
import nets.ISGG_CBAM_v3 as ISGG_CBAM_v3
import nets.ISGG_CBAM_v2_iscbox as ISGG_CBAM_v2_iscbox
import nets.ISGG_CBAM_v2_relmix as ISGG_CBAM_v2_relmix
import nets.ISGG_iscbox as ISGG_iscbox
import nets.ISGG_relmix as ISGG_relmix
import nets.memory_ISGG_relmix as memory_ISGG_relmix
import nets.memory_ISGG_iscbox as memory_ISGG_iscbox
import nets.memory_ISGG_relmix_v2 as memory_ISGG_relmix_v2
import nets.attend_memory_ISGG_relmix as attend_memory_ISGG_relmix
from datasets.coco17 import coco17
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
                        default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn-relPN/output/vgg16/visual_genome_train_rel/6a12_16_attend_MemLoss_iter3_VertEdgeGru_relmix_fr9k_rpn/vgg16_iter_80000.ckpt',
                        #default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn-relPN/output/vgg16/visual_genome_train_rel/10_20_LFFbaseline_iscbox_fr9k_rpn/vgg16_iter_80000.ckpt',
                        type=str)
    parser.add_argument('--visualize', dest='visualize', help='whether to show results',
                        action='store_true')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='coco_val', type=str)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net_base', dest='net_base',
                        help='vgg16, res50, res101, res152, mobile',
                        default='vgg16', type=str)
    parser.add_argument('--net_arch', dest='net_arch',
                        help='BM, ISGG, BM_ISGG, attend_memory_ISGG_relmix',
                        default='attend_memory_ISGG_relmix', type=str)
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

    split = args.imdb_name.split('_')[1]
    imdb = coco17(split, num_im=args.test_size)
    roidb = imdb.roidb
    #draw_image(imdb, roidb)
    #draw_gt(imdb, roidb)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)

    if args.net_arch == 'BM':
        arch = base_memory
    elif args.net_arch == 'ISGG':
        arch = ISGG
    elif args.net_arch == 'BM_ISGG':
        arch = memory_ISGG
    elif args.net_arch == 'attend_memory_ISGG_relmix':
        arch = attend_memory_ISGG_relmix

    else:
        raise NotImplementedError

    if args.net_base == 'vgg16':
        net = arch.vgg16()
    elif args.net_base == 'res50':
        net = arch.resnetv1(num_layers=50)
    elif args.net_base == 'res101':
        net = arch.resnetv1(num_layers=101)
    elif args.net_base == 'res152':
        net = arch.resnetv1(num_layers=152)
    elif args.net_base == 'mobile':
        net = arch.mobilenetv1()
    else:
        raise NotImplementedError

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

    test_net(sess, net, imdb, roidb, filename, 100, False)
    print('model %s done' % args.model)
    sess.close()
