from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from utils.cython_bbox import bbox_overlaps

try:
    import cPickle as pickle
except ImportError:
    import pickle


# Just return the ground truth boxes for a single image
def compute_target(feat_size, rois, feat_stride):
    factor_h = (feat_size[0] - 1.) * feat_stride
    factor_w = (feat_size[1] - 1.) * feat_stride
    num_roi = rois.shape[0]

    x1 = rois[:, [1]] / factor_w
    y1 = rois[:, [2]] / factor_h
    x2 = rois[:, [3]] / factor_w
    y2 = rois[:, [4]] / factor_h

    n_rois = np.hstack((y1, x1, y2, x2))
    batch_ids = np.zeros((num_roi), dtype=np.int32)
    # overlap to regions of interest
    #roi_overlaps = np.ones((num_gt), dtype=np.float32)
    #labels = np.array(gt_boxes[:, 4], dtype=np.int32)

    return n_rois, batch_ids


# Also return the reverse index of rois
def compute_target_memory(memory_size, rois, feat_stride):
    """

    :param memory_size: [H/16, W/16], shape of memory
    :param rois: [N, 5], for (batch_id, x1, y1, x2, y2)
    :param labels: [N,], roi labels
    :param feat_stride: 16
    :return:
    """
    minus_h = memory_size[0] - 1.
    minus_w = memory_size[1] - 1.
    num_roi = rois.shape[0]
    assert np.all(rois[:, 0] == 0), 'only support single image per batch.'

    x1 = rois[:, [1]] / feat_stride
    y1 = rois[:, [2]] / feat_stride
    x2 = rois[:, [3]] / feat_stride
    y2 = rois[:, [4]] / feat_stride

    # h, w, h, w
    n_rois = np.hstack((y1, x1, y2, x2))
    n_rois[:, 0::2] /= minus_h
    n_rois[:, 1::2] /= minus_w
    batch_ids = np.zeros(num_roi, dtype=np.int32)

    # h, w, h, w
    inv_rois = np.empty_like(n_rois)
    inv_rois[:, 0:2] = 0.
    inv_rois[:, 2] = minus_h
    inv_rois[:, 3] = minus_w
    inv_rois[:, 0::2] -= y1
    inv_rois[:, 1::2] -= x1

    # normalize coordinates
    inv_rois[:, 0::2] /= np.maximum(y2 - y1, cfg.EPS)
    inv_rois[:, 1::2] /= np.maximum(x2 - x1, cfg.EPS)

    inv_batch_ids = np.arange(num_roi, dtype=np.int32)

    return n_rois, batch_ids, inv_rois, inv_batch_ids


def compute_rel_rois(num_rel, rois, relations):
    """
    union subject boxes and object boxes given a set of rois and relations
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert(sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                    np.minimum(sub_roi[1], obj_roi[1]),
                    np.maximum(sub_roi[2], obj_roi[2]),
                    np.maximum(sub_roi[3], obj_roi[3])]
        rel_rois[i, 1:] = union_roi

    return rel_rois

def compute_hole_region(union_rois, isc_rois, crop_size):
    """
    compute the relative position of isc_rois to the union rois under the scale of crop_size
    :param num_rel:
    :param union_rois: [N, 5]
    :param isc_rois: [N, 5]
    :param crop_size: int, default is 7
    :return: [N, 4], under the scale of crop_size
    """
    hole_region = np.zeros([union_rois.shape[0], 5], dtype=np.float32)
    union_w = union_rois[:, 3] - union_rois[:, 1] + 1
    union_h = union_rois[:, 4] - union_rois[:, 2] + 1
    assert (isc_rois[:, 1] >= union_rois[:, 1]).all()
    assert (isc_rois[:, 2] >= union_rois[:, 2]).all()
    assert (isc_rois[:, 3] <= union_rois[:, 3]).all()
    assert (isc_rois[:, 4] <= union_rois[:, 4]).all()
    relative_isc_rois = isc_rois.copy()
    relative_isc_rois[:, 1::2] -= union_rois[:, 0:1]
    relative_isc_rois[:, 2::2] -= union_rois[:, 1:2]

    hole_region[:, 0] = union_rois[:, 0]
    hole_region[:, 1] = crop_size * relative_isc_rois[:, 1] / union_w
    hole_region[:, 2] = crop_size * relative_isc_rois[:, 2] / union_h
    hole_region[:, 3] = crop_size * relative_isc_rois[:, 3] / union_w
    hole_region[:, 4] = crop_size * relative_isc_rois[:, 4] / union_h
    hole_region = np.floor(hole_region).astype(np.int32)
    return hole_region

def sub_hole_region(rel_union_pool5, hole_region):
    """
    :param rel_union_pool5: [N, 7, 7, 512]
    :param hole_region: [N, 5]
    :return:
    """
    num_roi = hole_region.shape[0]
    sub_isc_pool5 = rel_union_pool5.copy()
    for i in range(num_roi):
        sub_isc_pool5[hole_region[i, 0], hole_region[i, 2]:(hole_region[i, 4]+1), hole_region[i, 1]:(hole_region[i, 3]+1), :] = 0
    return sub_isc_pool5

def select_map_by_class(weighted_map, label):
    """

    :param weighted_map: [N, num_classes, 7, 7]
    :param label: [N, ]
    :return:
    """
    selected_map = weighted_map[range(weighted_map.shape[0]), label]
    map = selected_map - np.min(selected_map, axis=(1,2), keepdims=True)
    map = map / np.max(map, axis=(1,2), keepdims=True)
    map = map[:, :, :, None] # N*7*7*1
    inv_batch_ids = np.arange(weighted_map.shape[0], dtype=np.int32)
    return map, inv_batch_ids


def compute_target_map(memory_size, relations, predicates, rois, labels, crop_size, feat_stride):

    num_rel = relations.shape[0]
    num_roi = rois.shape[0]

    new_rois = rois[:, 1:].copy()
    new_rois = new_rois.astype(np.float32)
    new_rois /= feat_stride
    new_rois = new_rois.astype(np.int32)
    target_map = np.zeros((num_rel, memory_size[0], memory_size[1], 1), dtype=np.float32)
    for i, rel in enumerate(relations):
        if predicates[i, 0] == 0:
            continue
        sub_roi = new_rois[rel[0]]
        obj_roi = new_rois[rel[1]]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                     np.minimum(sub_roi[1], obj_roi[1]),
                     np.maximum(sub_roi[2], obj_roi[2]),
                     np.maximum(sub_roi[3], obj_roi[3])]
        target_map[i, sub_roi[1]:sub_roi[3] + 1, sub_roi[0]:sub_roi[2] + 1, :] = 1.
        target_map[i, obj_roi[1]:obj_roi[3] + 1, obj_roi[0]:obj_roi[2] + 1, :] = 1.
        if bbox_overlaps(sub_roi[None], obj_roi[None]) > 0.:
            target_map[i, sub_roi[1]:sub_roi[3] + 1, sub_roi[0]:sub_roi[2] + 1, :] = \
                1. - target_map[i, sub_roi[1]:sub_roi[3] + 1, sub_roi[0]:sub_roi[2] + 1, :]
            target_map[i, obj_roi[1]:obj_roi[3] + 1, obj_roi[0]:obj_roi[2] + 1, :] = \
                1. - target_map[i, obj_roi[1]:obj_roi[3] + 1, obj_roi[0]:obj_roi[2] + 1, :]
        else:
            target_map[i, union_roi[1]:union_roi[3] + 1, union_roi[0]:union_roi[2] + 1, :] = \
                1. - target_map[i, union_roi[1]:union_roi[3] + 1, union_roi[0]:union_roi[2] + 1, :]

    #target_map = np.zeros((num_rel, memory_size[0], memory_size[1], 1), dtype=np.float32)
    obj_target_map = np.zeros((num_roi, crop_size, crop_size, 1), dtype=np.float32)
    obj_target_map[np.where(labels > 0)[0]] = 1.
    return target_map, obj_target_map


# Update weights for the target
def update_weights(labels, cls_prob):
    num_gt = labels.shape[0]
    index = np.arange(num_gt)
    cls_score = cls_prob[index, labels]
    big_ones = cls_score >= 1. - cfg.MEM.BETA
    # Focus on the hard examples
    weights = 1. - cls_score
    weights[big_ones] = cfg.MEM.BETA
    weights /= np.maximum(np.sum(weights), cfg.EPS)

    return weights
