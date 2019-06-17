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
import matplotlib.pyplot as plt


def draw_graph_pred(im, im_i, boxes, cls_score, rel_score, gt_to_pred, roidb, output_dir=None, mode=None):
    """
    Draw a predicted scene graph. To keep the graph interpretable, only draw
    the node and edge predictions that have correspounding ground truth
    labels.
    args:
        im: image
        boxes: prediceted boxes
        cls_score: object classification scores
        rel_score: relation classification scores
        gt_to_pred: a mapping from ground truth box indices to predicted box indices
        idx: for saving
        roidb: roidb
    """
    gt_relations = roidb['gt_relations']
    im = im[:, :, (2, 1, 0)].copy()
    if mode == 'pred_cls':
        gt_inds = np.where(roidb['max_overlaps'] == 1)[0]
        cls_pred = roidb['gt_classes'][gt_inds]
    else:
        cls_pred = np.argmax(cls_score, 1)
    #print cls_pred
    rel_pred_mat = np.argmax(rel_score, 2)
    #print rel_pred_mat
    rel_pred = []
    all_rels = []

    for i in range(rel_pred_mat.shape[0]):
        for j in range(rel_pred_mat.shape[1]):
            # find graph predictions (nodes and edges) that have
            # correspounding ground truth annotations
            # ignore nodes that have no edge connections
            for rel in gt_relations:
                if rel[0] not in gt_to_pred or rel[1] not in gt_to_pred:
                    continue
                # discard duplicate grounding
                if [i, j] in all_rels:
                    continue
                if i == gt_to_pred[rel[0]] and j == gt_to_pred[rel[1]]:
                    rel_pred.append([i, j, rel_pred_mat[i,j], 1])
                    all_rels.append([i, j])

    rel_pred = np.array(rel_pred)
    if rel_pred.size == 0:
        return

    # indices of predicted boxes
    pred_inds = rel_pred[:, :2].ravel()

    # draw graph predictions
    graph_dict = draw_scene_graph(cls_pred, pred_inds, rel_pred, im_i, output_dir)
    missed_gt_boxes = None
    missed_gt_classes = None
    if mode != 'vis_gt':
        gt_inds = np.where(roidb['max_overlaps'] == 1)[0]
        gt_boxes = roidb['boxes'][gt_inds]
        gt_classes = roidb['gt_classes'][gt_inds]
        if len(gt_to_pred) < len(gt_inds):
            missed_idx = []
            for idx in gt_inds:
                if idx not in gt_to_pred:
                    missed_idx.append(idx)
            missed_gt_boxes = gt_boxes[np.array(missed_idx), :]
            missed_gt_classes = gt_classes[np.array(missed_idx)]
    viz_scene_graph(im, im_i, boxes, cls_pred, pred_inds, rel_pred, missed_gt_classes, missed_gt_boxes,
                    preprocess=False, output_dir=output_dir)


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


def im_obj_detect(sess, net, im, boxes, roidb):
    blobs = get_test_minibatch(roidb, boxes)
    num_roi = blobs['num_roi']
    _, cls_probs, bbox_delta, _, _ = net.test_image(sess, blobs)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        bbox_dist = np.load(osp.join(cfg.VG_DIR, cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE),
                            encoding='latin1').item()
        bbox_means = bbox_dist['means']
        bbox_stds = bbox_dist['stds']
        for cls in range(1, len(cfg.ind_to_class)):
            bbox_delta[range(num_roi), cls * 4:(cls + 1) * 4] = \
                bbox_delta[range(num_roi), cls * 4:(cls + 1) * 4] * bbox_stds[cls, :] + bbox_means[cls, :]
    pred_boxes = bbox_transform_inv(boxes, bbox_delta)
    pred_boxes = clip_boxes(pred_boxes, im.shape)
    return cls_probs, pred_boxes


def im_detect(sess, net, im, boxes, bbox_reg, roidb):
    blobs = get_test_minibatch(roidb, boxes)
    num_roi = blobs['num_roi']
    num_rel = blobs['num_rel']
    relations = blobs['relations']
    _, cls_probs, bbox_delta, _, rel_cls_prob = net.test_image(sess, blobs)
    rel_probs = None
    rel_probs_flat = rel_cls_prob
    rel_probs = np.zeros([num_roi, num_roi, rel_probs_flat.shape[1]])
    for i, rel in enumerate(relations):
        rel_probs[rel[0], rel[1], :] = rel_probs_flat[i, :]

    if bbox_reg:
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            bbox_dist = np.load(osp.join(cfg.VG_DIR, cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE),
                                encoding='latin1').item()
            bbox_means = bbox_dist['means']
            bbox_stds = bbox_dist['stds']
            for cls in range(1, len(cfg.ind_to_class)):
                bbox_delta[range(num_roi), cls*4:(cls+1)*4] = \
                    bbox_delta[range(num_roi), cls*4:(cls+1)*4] * bbox_stds[cls, :] + bbox_means[cls, :]
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
        if mode == 'obj_det':
            test_net_detection(sess, net, imdb, roidb, weights_filename)
        else:
            test_net_base(sess, net, imdb, roidb, weights_filename, visualize, mode)


def test_net_detection(sess, net, imdb, roidb, weights_filename, max_per_image=100, thresh=0.):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #  all_boxes[cls][image] = N x 5 array of detections in
    #  (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in range(num_images):
        im = imdb.im_getter(i)

        box_proposals, roi_scores = non_gt_rois(roidb[i])
        roi_scores = np.expand_dims(roi_scores, axis=1)
        nms_keep = nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                       cfg.TEST.NMS)
        nms_keep = np.array(nms_keep)
        num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
        keep = nms_keep[:num_proposal]
        box_proposals = box_proposals[keep, :]
        if box_proposals.size == 0:
            continue
        _t['im_detect'].tic()
        scores, boxes = im_obj_detect(sess, net, im, box_proposals, [roidb[i]])
        _t['im_detect'].toc()

        _t['misc'].tic()

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)


def test_net_base(sess, net, imdb, roidb, weights_filename, visualize=False, mode='all'):
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
                out_dict = im_detect(sess, net, im, box_proposals, bbox_reg, [roidb[i]])
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

            if visualize:
                this_output_dir = osp.join(output_dir_image, mode)
                _t['draw'].tic()
                gt_to_pred = ground_predictions(sg_entry, roidb[i], 0.5)
                draw_graph_pred(im, i, sg_entry['boxes'], sg_entry['scores'], sg_entry['relations'], gt_to_pred,
                                roidb[i], this_output_dir, mode)

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['evaluate'].average_time))

    for mode in eval_modes:
        evaluators[mode].print_stats()
        evaluators[mode].save_stats(output_dir)
    print("actual test: ", num_images - not_test / 2)
    print("max: ", max_gt)
    print("min: ", min_gt)


def draw_image(imdb, roidb):
    num_images = len(roidb)
    output_dir_image = get_output_dir(imdb, 'default/gt_images')
    for i in range(num_images):
        if i >= 2000:
            print(i)
            im = imdb.im_getter(i)
            im = im[:, :, (2, 1, 0)].copy()
            plt.imsave(osp.join(output_dir_image, '%d.png'%i), im)
            plt.close('all')


def draw_gt(imdb, roidb):
    num_images = len(roidb)
    output_dir_image = get_output_dir(imdb, 'default/vis_gt/images')
    for i in range(num_images):
        if i < 2000:
            continue
        print(i)
        im = imdb.im_getter(i)
        sg_entry = {}
        sg_entry['boxes'] = np.tile(roidb[i]['boxes'], (1, imdb.num_classes))
        gt_inds = np.arange(roidb[i]['boxes'].shape[0])

        gt_classes = roidb[i]['gt_classes']
        prob = np.zeros((len(gt_inds), imdb.num_classes), dtype=np.float32)
        prob[range(len(gt_inds)), gt_classes] = 1.
        sg_entry['scores'] = prob.copy()
        gt_relations = roidb[i]['gt_relations']
        num_gt_box = gt_inds.shape[0]
        rel_prob = np.zeros((num_gt_box, num_gt_box, imdb.num_predicates))
        for rel in gt_relations:
            rel_prob[rel[0], rel[1], rel[2]] = 1
        sg_entry['relations'] = rel_prob.copy()

        this_output_dir = osp.join(output_dir_image, 'vis_gt')
        gt_to_pred = {i:i for i in gt_inds}
        draw_graph_pred(im, i, sg_entry['boxes'], sg_entry['scores'], sg_entry['relations'], gt_to_pred,
                        roidb[i], this_output_dir, 'vis_gt')


def test_net_memory(sess, net, imdb, roidb, weights_filename, visualize=False, iter=0, mode='all'):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(roidb)
    output_dir = get_output_dir(imdb, weights_filename + "_iter%02d" % iter)
    output_dir_image = os.path.join(output_dir, 'images')
    if visualize and not os.path.exists(output_dir_image):
        os.makedirs(output_dir_image)
    all_scores = [[] for _ in range(num_images)]

    # timers
    _t = {'score': Timer()}

    for i in range(num_images):
        _t['score'].tic()
        all_scores[i], blobs = im_detect_iter(sess, imdb, net, [roidb[i]], iter)
        _t['score'].toc()

        print('score: {:d}/{:d} {:.3f}s' \
              .format(i + 1, num_images, _t['score'].average_time))

        if visualize and i % 10 == 0:
            basename = os.path.basename(imdb.image_path_at(i)).split('.')[0]
            im_vis, wrong = draw_predicted_boxes_test(blobs['data'], all_scores[i], blobs['gt_boxes'])
            if wrong:
                out_image = os.path.join(output_dir_image, basename + '.jpg')
                print(out_image)
                cv2.imwrite(out_image, im_vis)

    res_file = os.path.join(output_dir, 'results.pkl')
    with open(res_file, 'wb') as f:
        pickle.dump(all_scores, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    mcls_sc, mcls_ac, mcls_ap, mins_sc, mins_ac, mins_ap = imdb.evaluate(all_scores, output_dir)
    eval_file = os.path.join(output_dir, 'results.txt')
    with open(eval_file, 'w') as f:
        f.write('{:.3f} {:.3f} {:.3f} {:.3f}'.format(mins_ap, mins_ac, mcls_ap, mcls_ac))
