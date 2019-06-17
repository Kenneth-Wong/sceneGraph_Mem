import tensorflow as tf
import numpy as np


def exp_average_summary(ops, dep_ops, decay=0.9, name='avg', scope_pfix='',
                        raw_pfix=' (raw)', avg_pfix=' (avg)'):
    averages = tf.train.ExponentialMovingAverage(decay, name=name)
    averages_op = averages.apply(ops)

    for op in ops:
        tf.scalar_summary(scope_pfix + op.name + raw_pfix, op)
        tf.scalar_summary(scope_pfix + op.name + avg_pfix,
                          averages.average(op))

    with tf.control_dependencies([averages_op]):
        for i, dep_op in enumerate(dep_ops):
            dep_ops[i] = tf.identity(dep_op, name=dep_op.name.split(':')[0])

    return dep_ops


def exp_average(vec, curr_avg, decay=0.9):
    vec_avg = tf.reduce_mean(vec, 0)
    avg = tf.assign(curr_avg, curr_avg * decay + vec_avg * (1 - decay))
    return avg


def gather_vec_pairs(vecs, gather_inds):
    """
    gather obj-subj feature pairs
    """
    vec_pairs = tf.gather(vecs, gather_inds, )
    vec_len = int(vec_pairs.get_shape()[2]) * 2
    vec_pairs = tf.reshape(vec_pairs, [-1, vec_len])
    return vec_pairs


def pad_and_gather(vecs, mask_inds, pad=None):
    """
    pad a vector with a zero row and gather with input inds
    """
    if pad is None:
        pad = tf.expand_dims(tf.zeros_like(vecs[0]), 0)
    else:
        pad = tf.expand_dims(pad, 0)
    vecs_padded = tf.concat([vecs, pad], axis=0)
    # flatten mask and edges
    vecs_gathered = tf.gather(vecs_padded, mask_inds)
    return vecs_gathered


def padded_segment_reduce(vecs, segment_inds, num_segments, reduction_mode):
    """
    Reduce the vecs with segment_inds and reduction_mode
    Input:
        vecs: A Tensor of shape (batch_size, vec_dim)
        segment_inds: A Tensor containing the segment index of each
        vec row, should agree with vecs in shape[0]
    Output:
        A tensor of shape (vec_dim)
    """
    if reduction_mode == 'max':
        print('USING MAX POOLING FOR REDUCTION!')
        vecs_reduced = tf.segment_max(vecs, segment_inds)
    elif reduction_mode == 'mean':
        print('USING AVG POOLING FOR REDUCTION!')
        vecs_reduced = tf.segment_mean(vecs, segment_inds)
    vecs_reduced.set_shape([num_segments, vecs.get_shape()[1]])
    return vecs_reduced


def create_graph_data(num_roi, num_rel, relations):
    """
    compute graph structure from relations
    """

    rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool)
    roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    for i, rel in enumerate(relations):
        rel_mask[rel[0], i] = True
        rel_mask[rel[1], i] = True
        roi_rel_inds[rel[0], rel[1]] = i

    rel_mask_inds = []
    rel_segment_inds = []
    for i, mask in enumerate(rel_mask):
        mask_inds = np.where(mask)[0].tolist() + [num_rel]
        segment_inds = [i for _ in mask_inds]
        rel_mask_inds += mask_inds
        rel_segment_inds += segment_inds

    # compute relation pair inds
    rel_pair_mask_inds = []  #
    roi_pair_mask_inds = []
    rel_pair_segment_inds = []  # for segment gather
    for i in range(num_roi):
        mask_inds = []
        roi_mask_inds = []
        for j in range(num_roi):
            out_inds = roi_rel_inds[i,j]
            in_inds = roi_rel_inds[j,i]
            #if out_inds >= 0 and in_inds >= 0:
            if out_inds >= 0 or in_inds >= 0:
                out_roi_inds = j if out_inds >= 0 else num_roi
                in_roi_inds = j if in_inds >= 0 else num_roi
                out_inds = out_inds if out_inds >=0 else num_rel
                in_inds = in_inds if in_inds >=0 else num_rel
                mask_inds.append([out_inds, in_inds])
                roi_mask_inds.append([out_roi_inds, in_roi_inds])

        mask_inds.append([num_rel, num_rel]) # pad with dummy edge ind
        roi_mask_inds.append([num_roi, num_roi])

        rel_pair_mask_inds += mask_inds
        roi_pair_mask_inds += roi_mask_inds
        rel_pair_segment_inds += [i for _ in mask_inds]

    # sanity check
    for i, inds in enumerate(rel_pair_mask_inds):
        if inds[0] < num_rel:
            assert(relations[inds[0]][0] == rel_pair_segment_inds[i])
        if inds[1] < num_rel:
            assert(relations[inds[1]][1] == rel_pair_segment_inds[i])

    output_dict = {
        'rel_mask_inds': np.array(rel_mask_inds).astype(np.int32),
        'rel_segment_inds': np.array(rel_segment_inds).astype(np.int32),
        'rel_pair_segment_inds': np.array(rel_pair_segment_inds).astype(np.int32),
        'rel_pair_mask_inds': np.array(rel_pair_mask_inds).astype(np.int32),
        'roi_pair_mask_inds': np.array(roi_pair_mask_inds).astype(np.int32)
    }

    return output_dict
