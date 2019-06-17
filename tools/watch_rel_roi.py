import _init_paths
from model.config import cfg
from model.train_val import filter_roidb, SolverWrapper
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer

from model.train_val import get_training_roidb
from model.train_val_memory import train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import nets.attend_memory as attend_memory
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
from trainval_memory import combined_roidb
from utils.cython_bbox import bbox_overlaps
import os.path as osp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


bbox_dist = np.load(osp.join(cfg.VG_DIR, cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE), encoding='latin1').item()
bbox_means = bbox_dist['means']
bbox_stds = bbox_dist['stds']

cfg.TRAIN.USE_FLIPPED = False
imdb, roidb = combined_roidb('train', 'rel', num_im=-1, num_val_im=-1)
num_images = len(roidb)
data_layer = RoIDataLayer(imdb, roidb, bbox_means, bbox_stds)

epoch = 10
thresh = 0.8
fg_bg = AverageMeter()
print_freq = 100
for i in range(num_images):
    db = roidb[i]
    gt_relations = db['gt_relations']
    gt_boxes = db['boxes']

    blobs = data_layer.forward()
    predicates = blobs['predicates']
    gt_relaitons = blobs['gt_relations']
    rel_rois = blobs['rel_rois']
    fg_rel_inds = np.where(predicates)[0]
    bg_rel_inds = np.where(predicates==0)[0]
    fg_rel_rois = rel_rois[fg_rel_inds, 1:]
    bg_rel_rois = rel_rois[bg_rel_inds, 1:]
    fg_bg_overlaps = bbox_overlaps(fg_rel_rois, bg_rel_rois)
    fg_fg_overlaps = bbox_overlaps(fg_rel_rois, fg_rel_rois)

    fg_inds, bg_inds = np.where(fg_bg_overlaps > thresh)
    num_fg_bg_pair = len(fg_rel_inds) * len(bg_rel_inds)
    num = len(fg_inds)
    fg_bg.update(num / num_fg_bg_pair, num_fg_bg_pair)
    if i > 0 and i % print_freq == 0:
        print('(fg/bg)Val: {fg_bg.val:.3f}| (fg/bg)Avg: {fg_bg.avg:.3f}'.format(fg_bg=fg_bg))
print('epoch {0}: (fg/bg)Val: {fg_bg.val:.3f}| (fg/bg)Avg: {fg_bg.avg:.3f}'.format(e, fg_bg=fg_bg))


