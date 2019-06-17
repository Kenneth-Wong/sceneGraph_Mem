import os
import os.path as osp
import pickle
import json
import cv2
import h5py
import numpy as np
import copy

from datasets.imdb import imdb
from model.config import cfg
from utils.cython_bbox import bbox_overlaps
from collections import defaultdict
import scipy.sparse


class coco17(imdb):
    def __init__(self, mode, num_im=-1):
        imdb.__init__(self, 'coco17_%s' % mode)
        if mode not in ('train', 'val'):
            raise ValueError("Mode must be in train, or val. Supplied {}".format(mode))
        self.im_h5 = h5py.File(osp.join(cfg.COCO_DIR, 'imdb_%s_%d.h5' % (mode, cfg.IMG_SCALE)), 'r')
        self._image_set = mode
        self._info = json.load(open(osp.join(cfg.VG_DIR, 'VG-SGG-dicts.json'), 'r'))
        self._roidb_handler = self.rpn_roidb
        self.im_refs = self.im_h5['images']
        self.im_names = self.im_h5['image_names']
        self.num_im = len(self.im_names) if num_im <= 0 else num_im

        print('mode==%s' % mode)
        self._image_index = np.arange(len(self.im_names))[:self.num_im]
        self.im_sizes = np.vstack([self.im_h5['image_widths'][:self.num_im],
                                   self.im_h5['image_heights'][:self.num_im]]).transpose()
        self.original_sizes = np.vstack([self.im_h5['original_widths'][:self.num_im],
                                   self.im_h5['original_heights'][:self.num_im]]).transpose()
        if cfg.TEST.USE_RPN_DB:
            self.rpn_h5 = h5py.File(osp.join(cfg.COCO_DIR, 'coco_' + mode + '_rpn_proposals.h5'), 'r')
            self.rpn_rois = self.rpn_h5['rpn_rois']
            self.rpn_scores = self.rpn_h5['rpn_scores']
            self.rpn_im_to_roi_idx = np.array(self.rpn_h5['im_to_roi_idx'][:self.num_im])
            self.rpn_num_rois = np.array(self.rpn_h5['num_rois'][:self.num_im])

        self._info['label_to_idx']['__background__'] = 0
        self._class_to_ind = self._info['label_to_idx']
        self._classes = sorted(self._class_to_ind, key=lambda k: self._class_to_ind[k])
        cfg.ind_to_class = self._classes
        self._predicate_to_ind = self._info['predicate_to_idx']
        self._predicate_to_ind['__background__'] = 0
        self._predicates = sorted(self._predicate_to_ind, key=lambda k: self._predicate_to_ind[k])
        cfg.ind_to_predicates = self._predicates

    def im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w]
        im = im.transpose((1, 2, 0))
        return im

    def get_image_name(self, idx):
        return self.im_names[idx]

    def get_scale(self, idx):
        ori_w, ori_h = self.original_sizes[idx]
        scale = 1024 / max(ori_w, ori_h)
        return scale

    def _get_widths(self):
        return self.im_sizes[:, 0]

    def rpn_roidb(self):
        roidb = []
        for i in range(self.num_im):
            im_rois = self.rpn_rois[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i] + self.rpn_num_rois[i],
                      :].copy()
            roi_scores = self.rpn_scores[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i] + self.rpn_num_rois[i],
                         0].copy()
            roidb.append({'boxes': im_rois, 'image': lambda im_i=i: self.im_getter(im_i), 'flipped': False,
                          'width': self.im_sizes[i][0], 'height': self.im_sizes[i][1],
                          'roi_scores': roi_scores})
        return roidb
