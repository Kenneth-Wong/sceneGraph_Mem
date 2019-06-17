import _init_paths
from datasets.factory import get_imdb, get_vg_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import os.path as osp
import pickle
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from trainval_memory import combined_roidb

if __name__ == '__main__':
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb = combined_roidb('train', 'rel', num_im=-1, num_val_im=-1)
    cfg.TRAIN.USE_FLIPPED = orgflip
    num_classes = len(cfg.ind_to_class) - 1
    num_predicates = len(cfg.ind_to_predicates) - 1

    X = np.zeros((num_classes, num_classes, num_predicates), dtype=np.float32)
    for item in roidb:
        gt_relations = item['gt_relations']
        gt_classes = item['gt_classes']
        for gt_rel in gt_relations:
            sub_cls, obj_cls, pred = gt_classes[gt_rel[0]] - 1, gt_classes[gt_rel[1]] - 1, gt_rel[2] - 1
            X[sub_cls, obj_cls, pred] += 1
    X = X + 1 # smooth
    X = X / np.sum(X, axis=2, keepdims=True)
    with open(osp.join(cfg.DATA_DIR, 'predicate_distribution.pkl'), 'wb') as f:
        pickle.dump(X, f)

