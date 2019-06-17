"""
A helper class for evaluating scene graph prediction tasks
"""

import numpy as np
from .sg_eval import eval_relation_recall, eval_predicate_recall, eval_predicate_recall_v2
from model.config import cfg
import os.path as osp

class SceneGraphEvaluator:
    def __init__(self, imdb, mode):
        self.imdb = imdb
        self.roidb = imdb.roidb
        self.result_dict = {}
        self.mode = mode
        self.result_dict = {}
        self.img_result_dict = {}
        if self.mode == 'pred_eval':
            self.result_dict[self.mode+'_recall'] = {pred:[0., 0.] for pred in cfg.ind_to_predicates}
        else:
            self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
            self.img_result_dict[self.mode + '_recall'] = {}

    def evaluate_scene_graph_entry(self, sg_entry, im_idx, iou_thresh):
        pred_triplets, triplet_boxes = \
            eval_relation_recall(sg_entry, self.roidb[im_idx],
                                 self.result_dict,
                                 self.mode,
                                 iou_thresh=iou_thresh)
        self.img_result_dict[self.mode + '_recall'][im_idx] = [self.result_dict[self.mode + '_recall'][20][-1],
                                                               self.result_dict[self.mode + '_recall'][50][-1],
                                                               self.result_dict[self.mode + '_recall'][100][-1]]

        return pred_triplets, triplet_boxes

    def evaluate_predicate_cls_entry(self, sg_entry, im_idx):
        eval_predicate_recall(sg_entry, self.roidb[im_idx], self.result_dict, self.mode)

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('======================' + self.mode + '============================')
        if self.mode == 'pred_eval':
            avg = 0.
            for k, v in self.result_dict[self.mode + '_recall'].items():
                if v[0] == 0:
                    rec = 0.
                else:
                    rec = v[1] / v[0]
                print('%s: %f' % (k, rec))
                avg += rec
            print(avg / len(self.result_dict[self.mode+'_recall']))
        else:
            for k, v in self.result_dict[self.mode + '_recall'].items():
                print('R@%i: %f' % (k, np.mean(v)))

    def save_stats(self, output_dir):
        print('writing ' + self.mode)
        with open(osp.join(output_dir, self.mode+'.txt'), 'w') as f:
            for im_idx, item in self.img_result_dict[self.mode + '_recall'].items():
                res = str(im_idx) + ' ' + ' '.join(list(map(str, item))) + '\n'
                f.write(res)


