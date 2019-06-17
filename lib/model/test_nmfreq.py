from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import os.path as osp

from model.config import cfg, get_output_dir
from utils.timer import Timer
from utils.cython_bbox import bbox_overlaps
from utils.visualization import draw_predicted_boxes_test
from datasets.evaluator import SceneGraphEvaluator
from model.nms_wrapper import nms
from roi_data_layer.minibatch import get_test_minibatch
from model.bbox_transform import bbox_transform_inv, clip_boxes
from datasets.eval_utils import ground_predictions
from datasets.viz import viz_scene_graph, draw_scene_graph


def non_gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    non_gt_inds = np.setdiff1d(np.arange(overlaps.shape[0]), gt_inds)
    rois = roidb['boxes'][non_gt_inds]
    scores = roidb['roi_scores'][non_gt_inds]
    return rois, scores


def gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    rois = roidb['boxes'][gt_inds]
    return rois


def im_detect(sess, net, im, boxes, use_gt_cls, bbox_reg, roidb, pred_prior):
    blobs = get_test_minibatch(roidb, boxes)
    num_roi = blobs['num_roi']
    relations = blobs['relations']
    gt_classes = roidb[0]['gt_classes']
    _, cls_probs, bbox_delta = net.test_image(sess, blobs)
    rel_probs = np.zeros([num_roi, num_roi, pred_prior.shape[2]])
    if use_gt_cls: # pred_cls
        for i, rel in enumerate(relations):
            rel_probs[rel[0], rel[1], :] = pred_prior[gt_classes[rel[0]], gt_classes[rel[1]], :]
    else:
        pred_classes = np.argmax(cls_probs, axis=1)
        for i, rel in enumerate(relations):
            rel_probs[rel[0], rel[1], :] = pred_prior[pred_classes[rel[0]], pred_classes[rel[1]], :]

    if bbox_reg:
        bbox_stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (cls_probs.shape[1]))
        bbox_means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (cls_probs.shape[1]))
        bbox_delta *= bbox_stds
        bbox_means += bbox_means
        pred_boxes = bbox_transform_inv(boxes, bbox_delta)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        pred_boxes = np.tile(boxes, (1, cls_probs.shape[1]))

    out_dict = {'scores': cls_probs.copy(),
                'boxes': pred_boxes.copy(),
                'relations': rel_probs.copy()}

    return out_dict


def im_detect_iter(sess, imdb, net, roidb, iter):
    blobs = imdb.minibatch(roidb, is_training=False)
    _, cls_prob, bbox_pred, _, rel_cls_prob = net.test_image_iter(sess, blobs, iter)

    return scores, blobs


def test_net(sess, net, imdb, roidb, weights_filename, visualize=False, iter_test=False, mode='all'):
    if iter_test:
        for iter in range(cfg.MEM.ITER):
            test_net_memory(sess, net, imdb, roidb, weights_filename, visualize, iter, mode)
    else:
        test_net_base(sess, net, imdb, roidb, weights_filename, visualize, mode)


def test_net_base(sess, net, imdb, roidb, weights_filename, visualize=False, mode='all'):
    with open(osp.join(cfg.DATA_DIR, 'predicate_distribution.pkl'), 'rb') as f:
        pred_prior = pickle.load(f)
    nc = pred_prior.shape[0]
    npred = pred_prior.shape[2]
    pred_prior = np.concatenate((np.zeros((nc, nc, 1)), pred_prior), axis=2) # 150, 150, 51
    pred_prior = np.concatenate((np.zeros((1, nc, npred+1)), pred_prior), axis=0) # 151, 150, 51
    pred_prior = np.concatenate((np.zeros((nc+1, 1, npred+1)), pred_prior), axis=1)

    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(roidb)
    output_dir = get_output_dir(imdb, weights_filename)
    output_dir_image = os.path.join(output_dir, 'images')
    if visualize and not os.path.exists(output_dir_image):
        os.makedirs(output_dir_image)
    all_scores = [[] for _ in range(num_images)]

    # timers
    _t = {'im_detect': Timer(), 'evaluate': Timer(), 'draw': Timer()}

    if mode == 'all':
        eval_modes = ['pred_cls', 'sg_cls', 'sg_det']
    else:
        eval_modes = [mode]
    print('EVAL MODES = ')
    print(eval_modes)

    evaluators = {}
    for m in eval_modes:
        evaluators[m] = SceneGraphEvaluator(imdb, mode=m)

    not_test = 0
    max_gt = 0
    min_gt = 1e5
    for i in range(num_images):
        im = imdb.im_getter(i)
        for mode in eval_modes:
            bbox_reg = True
            use_gt_cls = True if mode == 'pred_cls' or mode == 'pred_eval' else False
            if mode == 'pred_cls' or mode == 'sg_cls' or mode == 'pred_eval':
                bbox_reg = False
                box_proposals = gt_rois(roidb[i])
                max_gt = max(max_gt, box_proposals.shape[0])
                min_gt = min(min_gt, box_proposals.shape[0])
                if box_proposals.shape[0] > cfg.TEST.NUM_PROPOSALS:
                    not_test += 1
                    continue
                if box_proposals.size == 0 or box_proposals.shape[0] < 2:
                    continue
            elif mode == 'sg_det':
                box_proposals, roi_scores = non_gt_rois(roidb[i])
                roi_scores = np.expand_dims(roi_scores, axis=1)
                nms_keep = nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                               cfg.TEST.NMS)
                nms_keep = np.array(nms_keep)
                num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
                keep = nms_keep[:num_proposal]
                box_proposals = box_proposals[keep, :]
                if box_proposals.size == 0 or box_proposals.shape[0] < 2:
                    continue
            if mode in ['pred_cls', 'sg_cls', 'sg_det', 'pred_eval']:
                _t['im_detect'].tic()
                out_dict = im_detect(sess, net, im, box_proposals, use_gt_cls, bbox_reg, [roidb[i]], pred_prior)
                _t['im_detect'].toc()

                _t['evaluate'].tic()
                sg_entry = out_dict
                if mode == 'pred_eval':
                    evaluators[mode].evaluate_predicate_cls_entry(sg_entry, i)
                else:
                    evaluators[mode].evaluate_scene_graph_entry(sg_entry, i, iou_thresh=0.5)
                _t['evaluate'].toc()
            else: # vis gt
                sg_entry = {}
                sg_entry['boxes'] = np.tile(gt_rois(roidb[i]), (1, imdb.num_classes))
                gt_inds = np.where(roidb[i]['max_overlaps'] == 1)[0]

                gt_classes = roidb[i]['gt_classes'][gt_inds]
                # print gt_inds
                # print gt_classes
                # for c in gt_classes:
                #    print cfg.ind_to_class[c]
                prob = np.zeros((len(gt_inds), imdb.num_classes), dtype=np.float32)
                prob[range(len(gt_inds)), gt_classes] = 1.
                sg_entry['scores'] = prob.copy()
                gt_relations = roidb[i]['gt_relations']
                num_gt_box = gt_inds.shape[0]
                rel_prob = np.zeros((num_gt_box, num_gt_box, imdb.num_predicates))
                for rel in gt_relations:
                    rel_prob[rel[0], rel[1], rel[2]] = 1
                sg_entry['relations'] = rel_prob.copy()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['evaluate'].average_time))

    for mode in eval_modes:
        evaluators[mode].print_stats()
    print("actual test: ", num_images - not_test / 2)
    print("max: ", max_gt)
    print("min: ", min_gt)
