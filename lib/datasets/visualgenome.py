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
from .vg_voc_eval import voc_eval
from collections import defaultdict
import scipy.sparse


class visual_genome(imdb):
    def __init__(self, mode, task='det', num_im=10, num_val_im=1, filter_empty_rels=True,
                 filter_duplicate_rels=True, filter_empty_boxes=True):
        imdb.__init__(self, 'visual_genome_%s_%s' % (mode, task))
        if mode not in ('train', 'test', 'val', 'all'):
            raise ValueError("Mode must be in test, train, all, or val. Supplied {}".format(mode))
        assert task in ('det', 'rel')
        self.im_h5 = h5py.File(osp.join(cfg.VG_DIR, 'imdb_%d.h5' % cfg.IMG_SCALE), 'r')
        self.roi_h5 = h5py.File(osp.join(cfg.VG_DIR, 'VG-SGG.h5'), 'r')
        self._info = json.load(open(osp.join(cfg.VG_DIR, 'VG-SGG-dicts.json'), 'r'))
        self._image_set = mode
        self.task = task
        self.im_refs = self.im_h5['images']
        self._roidb_handler = self.gt_roidb
        self.filter_duplicate_rels = filter_duplicate_rels
        self.num_im = num_im
        self.num_val_im = num_val_im

        print('mode==%s' % mode)
        data_split = self.roi_h5['split'][:]
        if filter_empty_boxes:
            valid_mask = self.roi_h5['img_to_first_box'][:] >= 0
        else:
            valid_mask = self.roi_h5['img_to_first_box'][:] >= -1
        if mode != 'all':
            split = 2 if mode == 'test' else 0
            split_mask = data_split == split
            valid_mask = np.bitwise_and(split_mask, valid_mask)

        if filter_empty_rels:
            valid_mask &= self.roi_h5['img_to_first_rel'][:] >= 0
        self._image_index = np.where(valid_mask)[0]

        if num_im > -1:
            self._image_index = self._image_index[:num_im]
        if num_val_im > 0:
            if mode == 'train':
                self._image_index = self._image_index[num_val_im:]
            elif mode == 'val':
                self._image_index = self._image_index[:num_val_im]

        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[self.image_index] = True
        self.im_sizes = np.vstack([self.im_h5['image_widths'][split_mask],
                                   self.im_h5['image_heights'][split_mask]]).transpose()
        if cfg.TRAIN.USE_RPN_DB:
            self.rpn_h5 = h5py.File(osp.join(cfg.VG_DIR, 'vg_rpn_proposals.h5'), 'r')
            self.rpn_rois = self.rpn_h5['rpn_rois']
            self.rpn_scores = self.rpn_h5['rpn_scores']
            self.rpn_im_to_roi_idx = np.array(self.rpn_h5['im_to_roi_idx'][split_mask])
            self.rpn_num_rois = np.array(self.rpn_h5['num_rois'][split_mask])

        self.im_to_first_box = self.roi_h5['img_to_first_box'][split_mask]
        self.im_to_last_box = self.roi_h5['img_to_last_box'][split_mask]
        self.all_boxes = self.roi_h5['boxes_%d' % cfg.BOX_SCALE][:]
        self.labels = self.roi_h5['labels'][:, 0]
        assert (np.all(self.all_boxes[:, :2] >= 0))
        assert (np.all(self.all_boxes[:, 2:] > 0))

        self.all_boxes[:, :2] = self.all_boxes[:, :2] - np.floor(self.all_boxes[:, 2:] / 2)
        self.all_boxes[:, 2:] = self.all_boxes[:, :2] + self.all_boxes[:, 2:] - 1

        self._info['label_to_idx']['__background__'] = 0
        self._class_to_ind = self._info['label_to_idx']
        self._classes = sorted(self._class_to_ind, key=lambda k: self._class_to_ind[k])
        cfg.ind_to_class = self._classes

        self.im_to_first_rel = self.roi_h5['img_to_first_rel'][split_mask]
        self.im_to_last_rel = self.roi_h5['img_to_last_rel'][split_mask]
        self.relations = self.roi_h5['relationships'][:]
        self.relation_predicates = self.roi_h5['predicates'][:, 0]
        assert (self.im_to_first_rel.shape[0] == self.im_to_last_rel.shape[0])
        assert (self.relations.shape[0] == self.relation_predicates.shape[0])
        self._predicate_to_ind = self._info['predicate_to_idx']
        self._predicate_to_ind['__background__'] = 0
        self._predicates = sorted(self._predicate_to_ind, key=lambda k: self._predicate_to_ind[k])
        cfg.ind_to_predicates = self._predicates

        # for voc-eval
        self._devkit_path = cfg.VG_DIR
        self.config = {}
        self.config['cleanup'] = False
        self.config['use_diff'] = True

    def load_image_filenames(self, image_dir):
        corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
        fns = []
        for i, img in enumerate(self._image_refs):
            name = '{}.jpg'.format(img['image_id'])
            if name in corrupted_ims:
                continue
            filename = os.path.join(image_dir, '/'.join(img['url'].split('/')[-2:]))
            if os.path.exists(filename):
                fns.append(filename)
        assert len(fns) == 108073
        return fns

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        pass

    def im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w]
        im = im.transpose((1, 2, 0))
        return im

    def _get_widths(self):
        return self.im_sizes[:, 0]

    def gt_roidb(self):
        # get gt roidb and reldb
        gt_roidb = []
        for i in range(self.num_images):
            assert (self.im_to_first_box[i] >= 0)
            boxes = self.all_boxes[self.im_to_first_box[i]:self.im_to_last_box[i] + 1, :]
            # clip boxes
            w, h = self.im_sizes[i]
            boxes[:, 0] = np.maximum(np.minimum(boxes[:, 0], w - 1), 0)
            boxes[:, 1] = np.maximum(np.minimum(boxes[:, 1], h - 1), 0)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], w - 1), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], h - 1), 0)

            gt_classes = self.labels[self.im_to_first_box[i]:self.im_to_last_box[i] + 1]

            gt_relations = []
            if self.im_to_first_rel[i] >= 0:
                predicates = self.relation_predicates[self.im_to_first_rel[i]:self.im_to_last_rel[i] + 1]
                obj_idx = self.relations[self.im_to_first_rel[i]:self.im_to_last_rel[i] + 1]
                obj_idx = obj_idx - self.im_to_first_box[i]
                assert (np.all(obj_idx >= 0) and np.all(obj_idx < boxes.shape[0]))
                for j, p in enumerate(predicates):
                    gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])
                gt_relations = np.array(gt_relations)
            else:
                gt_relations = np.zeros((0, 3), dtype=np.int32)

            if self.filter_duplicate_rels:
                assert self._image_set == 'train'
                all_rel_sets = defaultdict(list)
                for (s, o, r) in gt_relations:
                    all_rel_sets[(s, o)].append(r)
                gt_relations = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
                gt_relations = np.array(gt_relations)

            overlaps = np.zeros((boxes.shape[0], self.num_classes), dtype=np.float32)
            overlaps[np.arange(boxes.shape[0], dtype=np.int32), gt_classes] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)
            seg_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
            gt_roidb.append({'boxes': boxes, 'gt_classes': gt_classes,
                             'gt_overlaps': overlaps, 'seg_areas': seg_areas,
                             'flipped': False, 'gt_relations': gt_relations,
                             'db_idx': i, 'image': lambda im_i=i: self.im_getter(im_i),
                             'width': self.im_sizes[i][0], 'height': self.im_sizes[i][1],
                             'roi_scores': np.ones(boxes.shape[0])})
        return gt_roidb

    def add_rpn_rois(self, gt_roidb_batch, make_copy=True):
        """
        Load precomputed RPN proposals
        """
        gt_roidb = copy.deepcopy(gt_roidb_batch) if make_copy else gt_roidb_batch
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_gt_rpn_roidb(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        box_list = []
        score_list = []
        for entry in gt_roidb:
            i = entry['db_idx']
            im_rois = self.rpn_rois[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i] + self.rpn_num_rois[i],
                      :].copy()
            roi_scores = self.rpn_scores[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i] + self.rpn_num_rois[i],
                         0].copy()
            box_list.append(im_rois)
            score_list.append(roi_scores)
        roidb = self.create_roidb_from_box_list(box_list, gt_roidb)
        for i, rdb in enumerate(roidb):
            rdb['roi_scores'] = score_list[i]
        return roidb

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == np.array([]):
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, self._image_set, self.roidb, self.image_index, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
            aps += [ap]
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
