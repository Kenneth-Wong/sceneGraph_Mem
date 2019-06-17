# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from utils.isgg_utils import create_graph_data


def get_minibatch(roidb, num_classes, is_rel=True):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    assert num_images == 1, 'only support single image per batch.'

    rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int32)
    rels_per_image = int(cfg.TRAIN.REL_BATCH_SIZE / num_images)
    pos_rels_per_image = np.round(cfg.TRAIN.POS_REL_FRACTION * rels_per_image).astype(np.int32)

    rels_per_image = int(cfg.TRAIN.REL_BATCH_SIZE / num_images)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}
    if not is_rel:
        if cfg.TRAIN.USE_ALL_GT:
            # Include all ground truth boxes
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        else:
            # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
            gt_inds = np.where(roidb[0]['gt_classes'] != 0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[
                0]
        gt_boxes = np.empty((len(gt_inds), 4), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
        blobs['gt_labels'] = roidb[0]['gt_classes'][gt_inds].reshape((-1, 1))
        return blobs

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    gt_blob = np.zeros((0, 4), dtype=np.float32)
    gt_labels_blob = np.zeros((0), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    rels_blob = np.zeros((0, 3), dtype=np.int32)
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    all_overlaps = []
    box_idx_offset = 0

    for im_i in range(num_images):
        # sample graph
        roi_inds, rels = _sample_graph_v2(roidb[im_i],
                                          fg_rois_per_image,
                                          rois_per_image,
                                          pos_rels_per_image,
                                          rels_per_image)
        if rels.size == 0:
            print('batch skipped')
            return None

        # gather all samples based on the sampled graph
        rels, labels, overlaps, im_rois, bbox_targets, bbox_inside_weights = \
            _gather_samples(roidb[im_i], roi_inds, rels, num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])

        batch_ind = im_i * np.ones((rois.shape[0], 1))  # im id for roi_pooling
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))
        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
        all_overlaps = np.hstack((all_overlaps, overlaps))

        # Add gt boxes
        gt_inds = np.where(roidb[im_i]['max_overlaps'] == 1)[0]
        gt_boxes = roidb[im_i]['boxes'][gt_inds]
        gt_boxes = _project_im_rois(gt_boxes, im_scales[im_i])
        gt_labels = roidb[im_i]['max_classes'][gt_inds]
        gt_blob = np.vstack((gt_blob, gt_boxes))
        gt_labels_blob = np.hstack((gt_labels_blob, gt_labels))

        # offset the relationship reference idx the number of previously
        # added box
        rels_offset = rels.copy()
        rels_offset[:, :2] += box_idx_offset
        rels_blob = np.vstack([rels_blob, rels_offset])
        box_idx_offset += rois.shape[0]

        # viz_inds = np.where(overlaps == 1)[0] # ground truth
        # viz_inds = npr.choice(np.arange(rois.shape[0]), size=50, replace=False) # random sample
        # viz_inds = np.where(overlaps > cfg.TRAIN.FG_THRESH)[0]  # foreground
        # viz_scene_graph(im_blob[im_i], rois, labels, viz_inds, rels)

    blobs['rois'] = rois_blob.copy()
    blobs['labels'] = labels_blob.copy().astype(np.int32).reshape((-1, 1))
    blobs['gt_boxes'] = gt_blob.copy()
    blobs['gt_labels'] = gt_labels_blob.copy().astype(np.int32).reshape((-1, 1))
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    blobs['relations'] = rels_blob[:, :2].copy().astype(np.int32)
    blobs['predicates'] = rels_blob[:, 2].copy().astype(np.int32).reshape((-1, 1))
    blobs['bbox_targets'] = bbox_targets_blob.copy()
    blobs['bbox_inside_weights'] = bbox_inside_blob.copy()
    blobs['bbox_outside_weights'] = \
        np.array(bbox_inside_blob > 0).astype(np.float32).copy()

    num_roi = rois_blob.shape[0]
    num_rel = rels_blob.shape[0]
    blobs['rel_rois'] = _compute_rel_rois(num_rel, rois_blob, rels_blob)
    blobs['isc_rois'] = _compute_intersect_rel_rois(num_rel, rois_blob, rels_blob, ratio=cfg.RATIO)
    blobs['num_roi'] = num_roi
    blobs['num_rel'] = num_rel
    blobs['memory_size'] = np.ceil(blobs['im_info'][:2] / cfg.BOTTLE_SCALE).astype(np.int32)

    graph_dict = create_graph_data(num_roi, num_rel, rels_blob[:, :2])
    for k in graph_dict:
        blobs[k] = graph_dict[k]
    return blobs


def get_test_minibatch(roidb, boxes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    assert num_images == 1, 'only support single image per batch.'

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    rois = _project_im_rois(boxes, im_scales[0])
    rois = np.hstack((np.zeros((rois.shape[0], 1)), rois))
    blobs = {'data': im_blob, 'rois': rois}
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    relations = []
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[0]):
            if i != j:
                relations.append([i, j])
    relations = np.array(relations, dtype=np.int32)
    blobs['relations'] = relations
    blobs['num_roi'] = rois.shape[0]
    blobs['num_rel'] = relations.shape[0]
    blobs['rel_rois'] = _compute_rel_rois(relations.shape[0], rois, relations)
    blobs['isc_rois'] = _compute_intersect_rel_rois(relations.shape[0], rois, relations, ratio=cfg.RATIO)
    blobs['memory_size'] = np.ceil(blobs['im_info'][:2] / cfg.BOTTLE_SCALE).astype(np.int32)

    graph_dict = create_graph_data(rois.shape[0], relations.shape[0], relations)
    for k in graph_dict:
        blobs[k] = graph_dict[k]
    return blobs


def _gather_samples(roidb, roi_inds, rels, num_classes):
    """
    join all samples and produce sampled items
    """
    rois = roidb['boxes']
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']

    # decide bg rois
    bg_inds = np.where(overlaps < cfg.TRAIN.FG_THRESH)[0]

    labels = labels.copy()
    labels[bg_inds] = 0
    labels = labels[roi_inds]
    # print('num bg = %i' % np.where(labels==0)[0].shape[0])

    # rois and bbox targets
    overlaps = overlaps[roi_inds]
    rois = rois[roi_inds]

    # convert rel index
    roi_ind_map = {}
    for i, roi_i in enumerate(roi_inds):
        roi_ind_map[roi_i] = i
    for i, rel in enumerate(rels):
        rels[i] = [roi_ind_map[rel[0]], roi_ind_map[rel[1]], rel[2]]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
        roidb['bbox_targets'][roi_inds, :], num_classes)

    return rels, labels, overlaps, rois, bbox_targets, bbox_inside_weights


def _sample_graph(roidb, num_fg_rois, num_rois, num_neg_rels=128):
    """
    Sample a graph from the foreground rois of an image
    :param:
    roidb: roidb of an image
    rois_per_image: maximum number of rois per image

    :return:
    roi_inds: 1d-array, the indexes of rois that are considered in the sampled graph.
                fg:bg ~ 1:3, fg may less than num_fg_rois(32)
    rels: (N, 3)-array for (sub, obj, rel), N is not certain. negative rel is no more than num_neg_rels
    """

    gt_rels = roidb['gt_relations']
    # index of assigned gt box for foreground boxes
    fg_gt_ind_assignments = roidb['fg_gt_ind_assignments']

    # find all fg proposals that are mapped to a gt
    gt_to_fg_roi_inds = {}
    all_fg_roi_inds = []
    for ind, gt_ind in fg_gt_ind_assignments.items():
        if gt_ind not in gt_to_fg_roi_inds:
            gt_to_fg_roi_inds[gt_ind] = []
        gt_to_fg_roi_inds[gt_ind].append(ind)
        all_fg_roi_inds.append(ind)

    # print('gt rois = %i' % np.where(roidb['max_overlaps']==1)[0].shape[0])
    # print('assigned gt = %i' % len(gt_to_fg_roi_inds.keys()))
    # dedup the roi inds
    all_fg_roi_inds = np.array(list(set(all_fg_roi_inds)))

    # find all valid relations in fg objects
    pos_rels = []
    for rel in gt_rels:
        for sub_i in gt_to_fg_roi_inds[rel[0]]:
            for obj_i in gt_to_fg_roi_inds[rel[1]]:
                pos_rels.append([sub_i, obj_i, rel[2]])

    # print('num fg rois = %i' % all_fg_roi_inds.shape[0])

    rels = []
    rels_inds = []
    roi_inds = []

    if len(pos_rels) > 0:
        # de-duplicate the relations
        _, indices = np.unique(["{} {}".format(i, j) for i, j, k in pos_rels], return_index=True)
        pos_rels = np.array(pos_rels)[indices, :]
        pos_inds = pos_rels[:, :2].tolist()
        # print('num pos rels = %i' % pos_rels.shape[0])

        # construct graph based on valid relations
        for rel in pos_rels:
            roi_inds += rel[:2].tolist()
            roi_inds = list(set(roi_inds))  # keep roi inds unique
            rels.append(rel)
            rels_inds.append(rel[:2].tolist())

            if len(
                    roi_inds) >= num_fg_rois:  # or len(rels_inds) >= rels_per_image: # here it usually limit the num of pos rel
                break

    # print('sampled rels = %i' % len(rels))

    roi_candidates = np.setdiff1d(all_fg_roi_inds, roi_inds)
    num_rois_to_sample = min(num_fg_rois - len(roi_inds), len(roi_candidates))
    # if not enough rois, sample fg rois
    if num_rois_to_sample > 0:
        roi_sample = npr.choice(roi_candidates.astype(np.int32), size=num_rois_to_sample,
                                replace=False)
        roi_inds = np.hstack([roi_inds, roi_sample])
        # print('sampled fg rois = %i' % num_rois_to_sample)

    # sample background relations
    sample_rels = []
    sample_rels_inds = []
    for i in roi_inds:
        for j in roi_inds:
            if i != j and [i, j] not in rels_inds:
                sample_rels.append([i, j, 0])
                sample_rels_inds.append([i, j])
    # print('background rels= %i' % len(sample_rels))

    if len(sample_rels) > 0:
        # randomly sample negative edges to prevent no edges
        num_neg_rels = np.minimum(len(sample_rels), num_neg_rels)
        # fprint('sampled background rels= %i' % num_neg_rels)
        inds = npr.choice(np.arange(len(sample_rels)), size=num_neg_rels, replace=False)
        rels += [sample_rels[i] for i in inds]
        rels_inds += [sample_rels_inds[i] for i in inds]

    # if still not enough rois, sample bg rois
    num_rois_to_sample = num_rois - len(roi_inds)
    if num_rois_to_sample > 0:
        bg_roi_inds = _sample_bg_rois(roidb, num_rois_to_sample)
        roi_inds = np.hstack([roi_inds, bg_roi_inds])

    roi_inds = np.array(roi_inds).astype(np.int64)
    # print('sampled rois = %i' % roi_inds.shape[0])
    return roi_inds.astype(np.int64), np.array(rels).astype(np.int64)


def _sample_graph_v2(roidb, num_fg_rois, num_rois, num_pos_rels, num_rels):
    gt_rels = roidb['gt_relations']
    # index of assigned gt box for foreground boxes
    fg_gt_ind_assignments = roidb['fg_gt_ind_assignments']

    # find all fg proposals that are mapped to a gt
    gt_to_fg_roi_inds = {}
    all_fg_roi_inds = []
    for ind, gt_ind in fg_gt_ind_assignments.items():
        if gt_ind not in gt_to_fg_roi_inds:
            gt_to_fg_roi_inds[gt_ind] = []
        gt_to_fg_roi_inds[gt_ind].append(ind)
        all_fg_roi_inds.append(ind)
    all_fg_roi_inds = np.array(list(set(all_fg_roi_inds)))

    # find all valid relations in fg objects
    pos_rels = []
    for rel in gt_rels:
        for sub_i in gt_to_fg_roi_inds[rel[0]]:
            for obj_i in gt_to_fg_roi_inds[rel[1]]:
                pos_rels.append([sub_i, obj_i, rel[2]])
    if len(pos_rels) > 0:
        # de-duplicate the relations
        _, indices = np.unique(["{} {}".format(i, j) for i, j, k in pos_rels], return_index=True)
        pos_rels = np.array(pos_rels)[indices, :]

    # print('num fg rois = %i' % all_fg_roi_inds.shape[0])

    # get adj list and degrees
    adj_list = [[] for _ in all_fg_roi_inds]
    map_id = dict(zip(all_fg_roi_inds.tolist(), np.arange(len(all_fg_roi_inds)).tolist()))
    for i, p_rel in enumerate(pos_rels):
        adj_list[map_id[p_rel[0]]].append([i, map_id[p_rel[1]]])
        adj_list[map_id[p_rel[1]]].append([i, map_id[p_rel[0]]])
    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    rels = []
    rels_inds = []
    roi_inds = []
    sample_counts = np.array([d for d in degrees])
    picked = np.zeros(pos_rels.shape[0])
    seen = np.array([False for _ in degrees])

    while np.sum(picked) < len(picked):
        weights = sample_counts * seen
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0
        probabilities = weights / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_id = chosen_edge[0]

        while picked[edge_id]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_id = chosen_edge[0]

        rels.append(pos_rels[edge_id])
        roi_inds += pos_rels[edge_id][:2].tolist()
        roi_inds = list(set(roi_inds))
        rels_inds.append(pos_rels[edge_id][:2].tolist())

        other_vertex = chosen_edge[1]
        picked[edge_id] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

        if len(rels) >= num_pos_rels:
            break

    # print('sampled rels = %i' % len(rels))

    roi_candidates = np.setdiff1d(all_fg_roi_inds, roi_inds)
    num_rois_to_sample = min(num_fg_rois - len(roi_inds), len(roi_candidates))
    # if not enough rois, sample fg rois
    if num_rois_to_sample > 0:
        roi_sample = npr.choice(roi_candidates.astype(np.int32), size=num_rois_to_sample,
                                replace=False)
        roi_inds = np.hstack([roi_inds, roi_sample])
        # print('sampled fg rois = %i' % num_rois_to_sample)

    # sample background relations
    sample_rels = []
    sample_rels_inds = []
    for i in roi_inds:
        for j in roi_inds:
            if i != j and [i, j] not in rels_inds:
                sample_rels.append([i, j, 0])
                sample_rels_inds.append([i, j])
    # print('background rels= %i' % len(sample_rels))

    if len(sample_rels) > 0:
        # randomly sample negative edges to prevent no edges
        num_neg_rels = np.minimum(len(sample_rels), num_rels - num_pos_rels)
        # fprint('sampled background rels= %i' % num_neg_rels)
        inds = npr.choice(np.arange(len(sample_rels)), size=num_neg_rels, replace=False)
        rels += [sample_rels[i] for i in inds]
        rels_inds += [sample_rels_inds[i] for i in inds]

    # if still not enough rois, sample bg rois
    num_rois_to_sample = num_rois - len(roi_inds)
    if num_rois_to_sample > 0:
        bg_roi_inds = _sample_bg_rois(roidb, num_rois_to_sample)
        roi_inds = np.hstack([roi_inds, bg_roi_inds])

    roi_inds = np.array(roi_inds).astype(np.int64)
    # print('sampled rois = %i' % roi_inds.shape[0])
    return roi_inds.astype(np.int64), np.array(rels).astype(np.int64)


def _sample_bg_rois(roidb, num_bg_rois):
    """
    Sample rois from background
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']

    bg_inds = np.where(((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                        (overlaps >= cfg.TRAIN.BG_THRESH_LO)) |
                       (labels == 0))[0]
    bg_rois_per_this_image = np.minimum(num_bg_rois, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
    return bg_inds


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois


def _compute_rel_rois(num_rel, rois, relations):
    """
    union subject boxes and object boxes given a set of rois and relations
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert (sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                     np.minimum(sub_roi[1], obj_roi[1]),
                     np.maximum(sub_roi[2], obj_roi[2]),
                     np.maximum(sub_roi[3], obj_roi[3])]
        rel_rois[i, 1:] = union_roi

    return rel_rois


def _clip(a, b):
    return np.array([np.maximum(a[0], b[0]), np.maximum(a[1], b[1]),
                     np.minimum(a[2], b[2]), np.minimum(a[3], b[3])])


def _compute_intersect_rel_rois(num_rel, rois, relations, ratio):
    """
    intersection boxes for given set of rois and relations
    :param num_rel:
    :param rois:
    :param relations:
    :return:
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert (sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]

        # judge whether the two rois intersect with each other or not
        x11, y11, x12, y12 = sub_roi
        x21, y21, x22, y22 = obj_roi
        # initialize the intersect box as union box
        union_box = np.array([np.minimum(x11, x21), np.minimum(y11, y21),
                              np.maximum(x12, x22), np.maximum(y12, y22)])
        intersect_box = union_box.copy()
        iw = np.minimum(x12, x22) - np.maximum(x11, x21) + 1
        ih = np.minimum(y12, y22) - np.maximum(y11, y21) + 1
        # intersect
        if iw > 1 and ih > 1:
            intersect_box = np.array([np.maximum(x11, x21), np.maximum(y11, y21),
                                      np.minimum(x12, x22), np.minimum(y12, y22)])
            w = intersect_box[2] - intersect_box[0] + 1
            h = intersect_box[3] - intersect_box[1] + 1
            nh = h * np.sqrt(ratio)
            nw = w * np.sqrt(ratio)
            deltax = (nw - w) / 2
            deltay = (nh - h) / 2
            intersect_box[0] -= deltax
            intersect_box[1] -= deltay
            intersect_box[2] += deltax
            intersect_box[3] += deltay
            intersect_box = _clip(intersect_box, union_box)
        else:
            w1 = x12 - x11 + 1
            h1 = y12 - y11 + 1
            w2 = x22 - x21 + 1
            h2 = y22 - y21 + 1
            xc1 = (x11 + x12) / 2
            yc1 = (y11 + y12) / 2
            xc2 = (x21 + x22) / 2
            yc2 = (y21 + y22) / 2
            if np.abs(xc1 - xc2) + 1 >= w1 / 2 + w2 / 2 and np.abs(yc1 - yc2) + 1 >= h1 / 2 + h2 / 2:
                intersect_box = np.array([np.minimum(xc1, xc2), np.minimum(yc1, yc2),
                                          np.maximum(xc1, xc2), np.maximum(yc1, yc2)])
            else:
                if np.abs(xc1 - xc2) + 1 < w1 / 2 + w2 / 2:
                    intersect_box = np.array([np.minimum(x11, x21), np.minimum(yc1, yc2),
                                              np.maximum(x12, x22), np.maximum(yc1, yc2)])
                elif np.abs(yc1 - yc2) + 1 < h1 / 2 + h2 / 2:
                    intersect_box = np.array([np.minimum(xc1, xc2), np.minimum(y11, y21),
                                              np.maximum(xc1, xc2), np.maximum(y12, y22)])
        rel_rois[i, 1:] = intersect_box
    return rel_rois


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind].astype(np.int64)
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = roidb[i]['image']()
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
