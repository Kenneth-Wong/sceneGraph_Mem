# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test_frcnn import test_net, _get_blobs
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb, get_vg_imdb
from datasets.visualgenome import visual_genome
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
import numpy as np
import pickle
import h5py


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn/experiments/cfgs/vgg16.yml',
                        type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn-relPN/output/vgg16/visual_genome_train_det/35a70_90_frcnn/vgg16_iter_800000.ckpt',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='visual_genome_test_det', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--test_size', dest='test_size',
                        default=-1, type=int)

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
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    split, task = args.imdb_name.split('_')[2:]
    imdb = visual_genome('all', 'det', num_im=args.test_size, num_val_im=-1,
                         filter_empty_rels=False,
                         filter_duplicate_rels=False, filter_empty_boxes=False
                         )

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    # load model
    net.create_architecture("TEST", imdb.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    if args.model:
        print(('Loading model check point from {:s}').format(args.model))
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print('Loaded.')
    else:
        print(('Loading initial weights from {:s}').format(args.weight))
        sess.run(tf.global_variables_initializer())
        print('Loaded.')

    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    imdb_boxes = [[] for _ in range(num_images)]
    imdb_scores = [[] for _ in range(num_images)]

    h5_file = h5py.File(os.path.join(cfg.VG_DIR, "vg_rpn_proposals.h5"), 'w')
    box_dset = h5_file.create_dataset('rpn_rois', (num_images*cfg.TEST.RPN_POST_NMS_TOP_N, 4), dtype=np.float32)
    score_dset = h5_file.create_dataset('rpn_scores', (num_images*cfg.TEST.RPN_POST_NMS_TOP_N, 1), dtype=np.float32)
    im_to_roi_idx = []
    num_rois = []
    start_idx = 0
    for i in range(num_images):
        im = imdb.im_getter(i)
        print(i)
        blobs, im_scales = _get_blobs(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs['data']
        blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
        rpn_boxes, rpn_scores = net.get_rpn(sess, blobs)
        scale = blobs['im_info'][2]
        boxes = rpn_boxes[:, 1:].copy() / scale
        scores = rpn_scores.copy()
        num_boxes = boxes.shape[0]
        box_dset[start_idx:(start_idx+num_boxes), :] = boxes
        score_dset[start_idx:(start_idx+num_boxes), :] = scores
        im_to_roi_idx.append(start_idx)
        num_rois.append(num_boxes)
        start_idx += num_boxes
    im_to_roi_idx = np.array(im_to_roi_idx, dtype=np.int32)
    num_rois = np.array(num_rois, dtype=np.int32)
    h5_file.create_dataset('im_to_roi_idx', data=im_to_roi_idx)
    h5_file.create_dataset('num_rois', data=num_rois)

    sess.close()
