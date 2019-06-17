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
from datasets.sg_eval import _triplet
import matplotlib.pyplot as plt
import json
import pickle
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def draw_graph_pred(im, im_i, classes, boxes, relations, triplet_scores, output_dir=None):
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
    im = im[:, :, (2, 1, 0)].copy()

    # indices of predicted boxes
    pred_inds = relations[:10, :2].ravel()

    # draw graph predictions
    graph_dict = draw_scene_graph(classes, pred_inds, relations[:10], im_i, output_dir)
    viz_scene_graph(im, im_i, boxes, classes, pred_inds, relations[:10], preprocess=False, output_dir=output_dir)


def im_detect(sess, net, im, boxes, roidb):
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

    out_dict = {'scores': cls_probs.copy(),
                'boxes': pred_boxes.copy(),
                'relations': rel_probs.copy()}

    return out_dict


def sg_result(sg_entry):
    box_preds = sg_entry['boxes']
    predicate_preds = sg_entry['relations']
    class_preds = sg_entry['scores']

    classes = np.argmax(class_preds, 1)
    class_scores = class_preds.max(axis=1)
    # drop bg box
    fg_inds = np.where(classes > 0)[0]
    classes = classes[fg_inds]
    class_scores = class_scores[fg_inds]
    box_preds = box_preds[fg_inds]
    predicate_preds = predicate_preds[fg_inds][:, fg_inds]
    num_boxes = len(fg_inds)
    boxes = []
    for i, c in enumerate(classes):
        boxes.append(box_preds[i, c * 4:(c + 1) * 4])
    boxes = np.vstack(boxes)

    # no bg
    predicate_preds = predicate_preds[:, :, 1:]
    predicates = np.argmax(predicate_preds, 2).ravel() + 1
    predicate_scores = predicate_preds.max(axis=2).ravel()
    relations = []
    keep = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j:
                keep.append(num_boxes * i + j)
                relations.append([i, j])
    # take out self relations
    predicates = predicates[keep]
    predicate_scores = predicate_scores[keep]

    relations = np.array(relations)
    assert (relations.shape[0] == num_boxes * (num_boxes - 1))
    assert (predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]

    pred_triplets, pred_triplet_boxes, relation_scores, _ = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)
    sorted_inds = np.argsort(relation_scores)[::-1]

    # box1, box2, predicate, triplet_score
    sorted_relations = np.hstack((relations[sorted_inds], pred_triplets[sorted_inds][:, 1][:, None]))
    sorted_triplet_scores = relation_scores[sorted_inds]
    return classes, boxes, sorted_relations, sorted_triplet_scores


def write_det(det, file):
    with open(file, 'wb') as f:
        pickle.dump(det, f)


def test_net(sess, net, imdb, roidb, weights_filename, topN=100, visualize=False):
    np.random.seed(cfg.RNG_SEED)
    """Test a Fast R-CNN network on an image database."""
    num_images = len(roidb)
    output_dir = get_output_dir(imdb, weights_filename)
    output_dir_image = os.path.join(output_dir, 'images')
    if visualize and not os.path.exists(output_dir_image):
        os.makedirs(output_dir_image)

    _t = {'im_detect': Timer(), 'draw': Timer()}
    det_info = []
    for i in range(num_images):
        im = imdb.im_getter(i)
        im_name = imdb.get_image_name(i)
        scale = imdb.get_scale(i)

        box_proposals, roi_scores = roidb[i]['boxes'], roidb[i]['roi_scores']
        roi_scores = np.expand_dims(roi_scores, axis=1)
        nms_keep = nms(np.hstack((box_proposals, roi_scores)).astype(np.float32),
                       cfg.TEST.NMS)
        nms_keep = np.array(nms_keep)
        num_proposal = min(cfg.TEST.NUM_PROPOSALS, nms_keep.shape[0])
        keep = nms_keep[:num_proposal]
        box_proposals = box_proposals[keep, :]

        _t['im_detect'].tic()
        out_dict = im_detect(sess, net, im, box_proposals, [roidb[i]])
        _t['im_detect'].toc()

        object_classes, object_boxes, relations, triplet_scores = sg_result(out_dict)

        if visualize:
            _t['draw'].tic()
            draw_graph_pred(im, i, object_classes, object_boxes, relations, triplet_scores, output_dir_image)
            _t['draw'].toc()

        # write detection result
        det = {'name': im_name, 'classes': object_classes, 'bboxes': object_boxes / scale, 'relations': relations[:topN],
               'triplet_scores': triplet_scores[:topN]}
        det_info.append(det)
        if i > 0 and (i + 1) % 10000 == 0:
            write_det(det_info, osp.join(output_dir, 'sg_coco_'+imdb.name.split('_')[1]+'_'+str(i//10000)+'.pkl'))
            det_info = []
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, _t['im_detect'].average_time,
                                                            _t['draw'].average_time))
    if len(det_info) > 0:
        write_det(det_info, osp.join(output_dir, 'sg_coco_'+imdb.name.split('_')[1]+'_'+str((num_images-1)//10000)+'.pkl'))
