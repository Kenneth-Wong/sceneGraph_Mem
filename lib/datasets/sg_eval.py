import numpy as np
from utils.cython_bbox import bbox_overlaps
from model.config import cfg


# only used in predciate cls mode, gt boxes == box_preds
def eval_predicate_recall(sg_entry, roidb_entry, result_dict, mode):
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    box_preds = gt_boxes
    num_boxes = box_preds.shape[0]
    predicate_preds = sg_entry['relations']
    predicate_preds = predicate_preds.reshape(num_boxes, num_boxes, -1)
    # predicate_preds_top1 = np.argmax(predicate_preds[:, :, 1:], axis=2) + 1
    predicate_preds_top5 = np.argsort(-predicate_preds[:, :, 1:], axis=2)[:, :, :5] + 1
    for rel in gt_relations:
        result_dict[mode + '_recall'][cfg.ind_to_predicates[rel[2]]][0] += 1
        # if rel[2] == predicate_preds_top1[rel[0], rel[1]]:
        if rel[2] in predicate_preds_top5[rel[0], rel[1], :]:
            result_dict[mode + '_recall'][cfg.ind_to_predicates[rel[2]]][1] += 1


def eval_predicate_recall_v2(sg_entry, roidb_entry, result_dict, mode):
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _, gt_union_boxes = _triplet(gt_relations[:, 2],
                                                                gt_relations[:, :2],
                                                                gt_classes,
                                                                gt_boxes,
                                                                gt_predicate_scores,
                                                                gt_class_scores)
    box_preds = sg_entry['boxes']
    overlaps = bbox_overlaps(gt_boxes, box_preds.reshape(-1, 4).astype(np.float64))

    num_boxes = box_preds.shape[0]
    predicate_preds = sg_entry['relations']
    class_preds = sg_entry['scores']
    predicate_preds = predicate_preds.reshape(num_boxes, num_boxes, -1)

    # no bg
    predicate_preds = predicate_preds[:, :, 1:]
    predicates = np.argmax(predicate_preds, 2).ravel() + 1
    predicate_scores = predicate_preds.max(axis=2).ravel()
    predicates = []
    predicate_preds = predicate_preds.reshape((-1, 50))
    for i, p in enumerate(predicate_preds):
        index = np.where(p == predicate_scores[i])[0]
        if len(index) > 1:
            pred = np.random.choice(index, 1)[0]
        else:
            pred = index[0]
        predicates.append(pred + 1)
    predicates = np.array(predicates)
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

    classes = gt_classes
    class_scores = gt_class_scores
    boxes = gt_boxes

    pred_triplets, pred_triplet_boxes, relation_scores, pred_union_boxes = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)

    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    keep_inds = sorted_inds[:100]
    recall = _predicate_recall(gt_triplets,
                               pred_triplets[keep_inds, :],
                               gt_triplet_boxes,
                               pred_triplet_boxes[keep_inds, :])
    for pred in recall:
        result_dict[mode + '_recall'][pred][0] += recall[pred][0]
        result_dict[mode + '_recall'][pred][1] += recall[pred][1]


def _predicate_recall(gt_triplets, pred_triplets,
                      gt_boxes, pred_boxes, iou_thresh=1.):
    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    pred_recall = {p:[0,0] for p in cfg.ind_to_predicates[1:]}

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        pred_recall[cfg.ind_to_predicates[gt[1]]][0] += 1
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] \
                    and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep, :]
        sub_iou = iou(gt_box[:4], boxes[:, :4])
        obj_iou = iou(gt_box[4:], boxes[:, 4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            pred_recall[cfg.ind_to_predicates[gt[1]]][1] += 1
    return pred_recall


def eval_relation_recall(sg_entry,
                         roidb_entry,
                         result_dict,
                         mode,
                         iou_thresh):
    # gt
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _, gt_union_boxes = _triplet(gt_relations[:, 2],
                                                                gt_relations[:, :2],
                                                                gt_classes,
                                                                gt_boxes,
                                                                gt_predicate_scores,
                                                                gt_class_scores)

    # pred
    box_preds = sg_entry['boxes']
    overlaps = bbox_overlaps(gt_boxes, box_preds.reshape(-1, 4).astype(np.float64))

    num_boxes = box_preds.shape[0]
    predicate_preds = sg_entry['relations']
    class_preds = sg_entry['scores']
    predicate_preds = predicate_preds.reshape(num_boxes, num_boxes, -1)

    # no bg
    predicate_preds = predicate_preds[:, :, 1:]
    predicates = np.argmax(predicate_preds, 2).ravel() + 1
    predicate_scores = predicate_preds.max(axis=2).ravel()
    predicates = []
    predicate_preds = predicate_preds.reshape((-1, 50))
    for i, p in enumerate(predicate_preds):
        index = np.where(p == predicate_scores[i])[0]
        if len(index) > 1:
            pred = np.random.choice(index, 1)[0]
        else:
            pred = index[0]
        predicates.append(pred + 1)
    predicates = np.array(predicates)
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

    if mode == 'pred_cls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert (num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode == 'sg_cls':
        assert (num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = gt_boxes
    elif mode == 'sg_det':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = []
        for i, c in enumerate(classes):
            boxes.append(box_preds[i, c * 4:(c + 1) * 4])
        boxes = np.vstack(boxes)
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets, pred_triplet_boxes, relation_scores, pred_union_boxes = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)

    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    for k in result_dict[mode + '_recall']:
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds, :],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds, :],
                                  iou_thresh)
        result_dict[mode + '_recall'][k].append(recall)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores):
    # format predictions into triplets
    assert (predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    union_boxes = np.zeros([num_relations, 4]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in range(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i, :2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score = class_scores[sub_i]
        score *= class_scores[obj_i]
        score *= predicate_scores[i]
        triplet_scores[i] = score
        union_boxes[i, :] = np.array([np.minimum(boxes[sub_i, :][0], boxes[obj_i, :][0]),
                                      np.minimum(boxes[sub_i, :][1], boxes[obj_i, :][1]),
                                      np.maximum(boxes[sub_i, :][2], boxes[obj_i, :][2]),
                                      np.maximum(boxes[sub_i, :][3], boxes[obj_i, :][3]),
                                      ])
    return triplets, triplet_boxes, triplet_scores, union_boxes


def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh):
    # compute the R@K metric for a set of predicted triplets

    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] \
                    and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep, :]
        sub_iou = iou(gt_box[:4], boxes[:, :4])
        obj_iou = iou(gt_box[4:], boxes[:, 4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt) / float(num_gt)


def iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:, 0])
    iymin = np.maximum(gt_box[1], pred_boxes[:, 1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:, 2])
    iymax = np.minimum(gt_box[3], pred_boxes[:, 3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
           (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
           (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps
