from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

from nets.network import Network
from nets.base_memory import BaseMemory
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
from utils.snippets import compute_target
from model.config import cfg
from utils.visualization import draw_predicted_boxes, draw_memory
from utils.isgg_utils import gather_vec_pairs, pad_and_gather


class DualGraph(BaseMemory, Network):
    def __init__(self):
        self._n_iter = 2
        self._predictions = {}
        self._predictions["cls_score"] = []
        self._predictions["cls_prob"] = []
        self._predictions["cls_pred"] = []
        self._predictions["bbox_pred"] = []
        self._predictions["rel_cls_score"] = []
        self._predictions["rel_cls_prob"] = []
        self._predictions["rel_cls_pred"] = []
        self._losses = {}
        self._targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = [[] for _ in range(self._n_iter)]
        self._event_summaries = {}
        self._variables_to_fix = {}

        self._vert_state_dim = 512
        self._edge_state_dim = 512

    def _target_layer(self, name):
        with tf.variable_scope(name):
            rois, batch_ids = tf.py_func(compute_target,
                                         [self._feat_size, self._rois,
                                          self._feat_stride[0]],
                                         [tf.float32, tf.int32],
                                         name="target_layer")

            labels = self._labels
            bbox_targets = self._bbox_targets
            bbox_inside_weights = self._bbox_inside_weights
            bbox_outside_weights = self._bbox_outside_weights

            rois.set_shape([None, 4])
            labels.set_shape([None, 1])

            bbox_targets.set_shape([None, 4 * self._num_classes])
            bbox_inside_weights.set_shape([None, 4 * self._num_classes])
            bbox_outside_weights.set_shape([None, 4 * self._num_classes])
            self._targets['rois'] = rois
            self._targets['batch_ids'] = batch_ids
            self._targets['labels'] = labels
            self._targets['bbox_targets'] = bbox_targets
            self._targets['bbox_inside_weights'] = bbox_inside_weights
            self._targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries[0].append(rois)
            self._score_summaries[0].append(labels)
            self._score_summaries[0].append(bbox_targets)
            self._score_summaries[0].append(bbox_inside_weights)
            self._score_summaries[0].append(bbox_outside_weights)

        return rois, batch_ids, bbox_targets, bbox_inside_weights

    def _target_rel_layer(self, name):
        with tf.variable_scope(name):
            rel_rois, rel_batch_ids = tf.py_func(compute_target,
                                                 [self._feat_size, self._rel_rois,
                                                  self._feat_stride[0]],
                                                 [tf.float32, tf.int32],
                                                 name="target_rel_layer")

            predicates = self._predicates

            rel_rois.set_shape([None, 4])
            predicates.set_shape([None, 1])
            self._targets['rel_rois'] = rel_rois
            self._targets['rel_batch_ids'] = rel_batch_ids
            self._targets['predicates'] = predicates

            self._score_summaries[0].append(rel_rois)
            self._score_summaries[0].append(predicates)

        return rel_rois, rel_batch_ids

    def _cells(self):
        """
        construct RNN cells and states
        """
        # intiialize lstms
        self.vert_rnn = tf.nn.rnn_cell.GRUCell(self._vert_state_dim, activation=tf.tanh)
        self.edge_rnn = tf.nn.rnn_cell.GRUCell(self._edge_state_dim, activation=tf.tanh)

        # lstm states
        self.vert_state = self.vert_rnn.zero_state(self._num_roi, tf.float32)
        self.edge_state = self.edge_rnn.zero_state(self._num_rel, tf.float32)

    def _vert_rnn_forward(self, vert_in, name, reuse=False):
        with tf.variable_scope(name):
            if reuse: tf.get_variable_scope().reuse_variables()
            (vert_out, self.vert_state) = self.vert_rnn(vert_in, self.vert_state)
        return vert_out

    def _edge_rnn_forward(self, edge_in, name, reuse=False):
        with tf.variable_scope(name):
            if reuse: tf.get_variable_scope().reuse_variables()
            (edge_out, self.edge_state) = self.edge_rnn(edge_in, self.edge_state)
        return edge_out

    def _fc_init(self, fc7, rel_fc7, is_training, name):
        xavier = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                trainable=is_training,
                                weights_initializer=xavier,
                                biases_initializer=tf.constant_initializer(0.0)):
                vert_unary, edge_unary = None, None
                if fc7 is not None:
                    vert_unary = slim.fully_connected(fc7, self._vert_state_dim, scope="vert_unary")
                    self._act_summaries.append(vert_unary)
                if rel_fc7 is not None:
                    edge_unary = slim.fully_connected(rel_fc7, self._edge_state_dim, scope="edge_unary")
                    self._act_summaries.append(edge_unary)

        return vert_unary, edge_unary

    def _fc_iter(self, w_input, is_training, reuse, name):
        xavier = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name, reuse=reuse):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                trainable=is_training,
                                weights_initializer=xavier,
                                biases_initializer=tf.constant_initializer(0.0)):
                w_fc = slim.fully_connected(w_input, 1, scope="w_fc")
                w_score = tf.nn.sigmoid(w_fc, name="w_score")
                self._act_summaries.append(w_score)
        return w_score

    def _cls_iter(self, input, is_training, reuse, prefix=None):
        out = self._num_classes if prefix is None else self._num_predicates
        prefix = '' if prefix is None else prefix + '_'
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        cls_score = slim.fully_connected(input, out,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         reuse=reuse,
                                         activation_fn=None, scope=prefix + 'cls_score')
        cls_prob = tf.nn.softmax(cls_score, name=prefix + "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name=prefix + "cls_pred")

        self._score_summaries[0].append(cls_score)
        self._score_summaries[0].append(cls_pred)
        self._score_summaries[0].append(cls_prob)

        self._predictions[prefix + 'cls_score'].append(cls_score)
        self._predictions[prefix + 'cls_prob'].append(cls_prob)
        self._predictions[prefix + 'cls_pred'].append(cls_pred)

    def _reg_iter(self, input, is_training, reuse):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        bbox_pred = slim.fully_connected(input, self._num_classes * 4,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         reuse=reuse,
                                         activation_fn=None, scope='bbox_pred')
        self._score_summaries[0].append(bbox_pred)
        self._predictions['bbox_pred'].append(bbox_pred)

    def _build_conv(self, is_training):
        # Get the head
        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope, self._scope):
            # get the region of interest
            rois, batch_ids, bbox_targets, bbox_inside_weights = self._target_layer("target")
            rel_rois, rel_batch_ids = self._target_rel_layer("rel_target")
            # region of interest pooling
            pool5 = self._crop_rois(net_conv, rois, batch_ids, "pool5")
            pool5_nb = tf.stop_gradient(pool5, name="pool5_nb")
            # region of relation interest pooling
            rel_pool5 = self._crop_rois(net_conv, rel_rois, rel_batch_ids, "rel_pool5")
            rel_pool5_nb = tf.stop_gradient(rel_pool5, name="rel_pool5_nb")

        fc7 = self._head_to_tail(pool5, is_training)
        rel_fc7 = self._head_to_tail(rel_pool5, is_training, prefix='rel')

        return fc7, rel_fc7

    def _compute_edge_context(self, vert_factor, edge_factor, is_training, reuse, atten=None):
        prefix = '' if atten is None else atten+'_'
        vert_pairs = gather_vec_pairs(vert_factor, self._relations)
        sub_vert, obj_vert = tf.split(vert_pairs, num_or_size_splits=2, axis=1)
        sub_vert_w_input = tf.concat([sub_vert, edge_factor], axis=1)
        obj_vert_w_input = tf.concat([obj_vert, edge_factor], axis=1)

        # compute compatibility scores
        sub_vert_w = self._fc_iter(sub_vert_w_input, is_training, reuse=reuse, name=prefix+"sub_vert")
        obj_vert_w = self._fc_iter(obj_vert_w_input, is_training, reuse=reuse, name=prefix+"obj_vert")

        weighted_sub = tf.multiply(sub_vert, sub_vert_w)
        weighted_obj = tf.multiply(obj_vert, obj_vert_w)
        edge_context = tf.add(weighted_sub, weighted_obj, name=prefix+"edge_context")
        return edge_context

    def _compute_vert_context(self, edge_factor, vert_factor, is_training, reuse, atten=None):
        prefix = '' if atten is None else atten + '_'
        out_edge = pad_and_gather(edge_factor, self._edge_pair_mask_inds[:, 0])
        in_edge = pad_and_gather(edge_factor, self._edge_pair_mask_inds[:, 1])
        vert_factor_gathered = tf.gather(vert_factor, self._edge_pair_segment_inds)

        out_edge_w_input = tf.concat([out_edge, vert_factor_gathered], axis=1)
        in_edge_w_input = tf.concat([in_edge, vert_factor_gathered], axis=1)

        out_edge_w = self._fc_iter(out_edge_w_input, is_training, reuse=reuse, name=prefix+"out_edge")
        in_edge_w = self._fc_iter(in_edge_w_input, is_training, reuse=reuse, name=prefix+"in_edge")

        out_edge_weighted = tf.multiply(out_edge, out_edge_w)
        in_edge_weighted = tf.multiply(in_edge, in_edge_w)

        edge_sum = out_edge_weighted + in_edge_weighted
        vert_context = tf.segment_sum(edge_sum, self._edge_pair_segment_inds, name=prefix+"vert_context")
        return vert_context

    def _update_inference(self, vert_factor, edge_factor, is_training, iter):
        reuse = iter > 0
        self._cls_iter(vert_factor, is_training, reuse)
        self._cls_iter(edge_factor, is_training, reuse, prefix='rel')
        self._reg_iter(vert_factor, is_training, reuse)

    def _build_iterate(self, fc7, rel_fc7, is_training):
        with tf.variable_scope('ISGG'):
            vert_unary, edge_unary = self._fc_init(fc7, rel_fc7, is_training, "unary")
            vert_factor = self._vert_rnn_forward(vert_unary, "vert_rnn", reuse=False)
            edge_factor = self._edge_rnn_forward(edge_unary, "edge_rnn", reuse=False)

            if self._n_iter:
                for i in range(self._n_iter):
                    reuse = i > 0
                    edge_ctx = self._compute_edge_context(vert_factor, edge_factor, is_training, reuse=reuse)
                    edge_factor = self._edge_rnn_forward(edge_ctx, "edge_rnn", reuse=True)

                    vert_ctx = self._compute_vert_context(edge_factor, vert_factor, is_training, reuse=reuse)
                    vert_factor = self._vert_rnn_forward(vert_ctx, "vert_rnn", reuse=True)
                    vert_in = vert_factor
                    edge_in = edge_factor

                    self._update_inference(vert_in, edge_in, is_training, i)

    def build_dual_graph(self, is_training, is_testing):
        fc7, rel_fc7 = self._build_conv(is_training)
        self._build_iterate(fc7, rel_fc7, is_training)

    def _add_losses(self, name):
        cross_entropy = []
        rel_cross_entropy = []
        bbox_loss = []
        # tag_loss = []
        assert len(self._predictions["cls_score"]) == self._n_iter
        assert len(self._predictions["rel_cls_score"]) == self._n_iter
        assert len(self._predictions["bbox_pred"]) == self._n_iter
        with tf.variable_scope(name):
            # load the groundtruth
            label = tf.reshape(self._targets["labels"], [-1])
            predicate = tf.reshape(self._targets["predicates"], [-1])
            bbox_targets = self._targets["bbox_targets"]
            bbox_inside_weights = self._targets["bbox_inside_weights"]
            bbox_outside_weights = self._targets["bbox_outside_weights"]

            for iter in range(self._n_iter):
                # RCNN, class loss
                cls_score = self._predictions["cls_score"][iter]
                ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score,
                                                                                   labels=label))
                cross_entropy.append(ce)

                # RCNN, bbox loss
                bbox_pred = self._predictions["bbox_pred"][iter]
                bbox_loss.append(
                    self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights))

                # relation loss
                rel_cls_score = self._predictions["rel_cls_score"][iter]
                rel_ce = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rel_cls_score, labels=predicate))
                rel_cross_entropy.append(rel_ce)

            self._losses['cross_entropy'] = tf.reduce_mean(cross_entropy, name='cross_entropy')
            self._losses['bbox_loss'] = tf.reduce_mean(bbox_loss, name='bbox_loss')
            self._losses['rel_cross_entropy'] = tf.reduce_mean(rel_cross_entropy, name='rel_cross_entropy')
            # regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = self._losses['cross_entropy'] + self._losses['bbox_loss'] \
                                         + self._losses['rel_cross_entropy']  # + regularization_loss

            self._event_summaries.update(self._losses)

    def _create_summary(self):
        """
        Note: The merge_all() function will merge all the tf.summary.xxx.
        So the summary appended after the val_summaries will also included in training summaries.
        """
        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_gt_image_summary())
            for iter in range(self._n_iter):
                # val_summaries.append(self._add_pred_memory_summary(iter))
                for var in self._score_summaries[iter]:
                    self._add_score_iter_summary(iter, var)
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for var in self._act_summaries:
                self._add_zero_summary(var)
        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(val_summaries)

    def create_architecture(self, mode, num_classes, num_predicates, n_iter=2, tag=None):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._rois = tf.placeholder(tf.float32, shape=[None, 5])  # including batch_id and coord
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 4])
        self._labels = tf.placeholder(tf.int32, shape=[None, 1])
        self._gt_labels = tf.placeholder(tf.int32, shape=[None, 1])
        self._rel_rois = tf.placeholder(tf.float32, shape=[None, 5])
        self._relations = tf.placeholder(tf.int32, shape=[None, 2])
        self._predicates = tf.placeholder(tf.int32, shape=[None, 1])
        self._bbox_targets = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
        self._bbox_inside_weights = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
        self._bbox_outside_weights = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
        self._num_roi = tf.placeholder(tf.int32, shape=[])
        self._num_rel = tf.placeholder(tf.int32, shape=[])
        self._feat_size = tf.placeholder(tf.int32, shape=[2])
        self._edge_mask_inds = tf.placeholder(tf.int32, shape=[None, ])
        self._edge_segment_inds = tf.placeholder(tf.int32, shape=[None, ])
        self._edge_pair_mask_inds = tf.placeholder(tf.int32, shape=[None, 2])
        self._edge_pair_segment_inds = tf.placeholder(tf.int32, shape=[None, ])
        self._tag = tag
        self._n_iter = n_iter

        self._num_classes = num_classes
        self._num_predicates = num_predicates
        self._mode = mode

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        self._cells()

        assert tag is not None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                             slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                            weights_regularizer=weights_regularizer,
                            biases_regularizer=biases_regularizer,
                            biases_initializer=tf.constant_initializer(0.0)):
            self.build_dual_graph(training, testing)

        layers_to_output = {}

        if not testing:
            self._add_losses("loss")
            layers_to_output.update(self._losses)
            self._create_summary()

        layers_to_output.update(self._predictions)

        return layers_to_output

    def _parse_dict(self, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._rois: blobs['rois'], self._rel_rois: blobs['rel_rois'],
                     self._gt_boxes: blobs['gt_boxes'], self._gt_labels: blobs['gt_labels'],
                     self._feat_size: blobs['memory_size'],
                     self._labels: blobs['labels'], self._relations: blobs['relations'],
                     self._predicates: blobs['predicates'], self._bbox_targets: blobs['bbox_targets'],
                     self._bbox_inside_weights: blobs['bbox_inside_weights'],
                     self._bbox_outside_weights: blobs['bbox_outside_weights'],
                     self._num_roi: blobs['num_roi'], self._num_rel: blobs['num_rel'],
                     self._edge_mask_inds: blobs['rel_mask_inds'],
                     self._edge_segment_inds: blobs['rel_segment_inds'],
                     self._edge_pair_mask_inds: blobs['rel_pair_mask_inds'],
                     self._edge_pair_segment_inds: blobs['rel_pair_segment_inds']}
        return feed_dict

    def _parse_test_dict(self, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._rois: blobs['rois'], self._rel_rois: blobs['rel_rois'],
                     self._feat_size: blobs['memory_size'],
                     self._relations: blobs['relations'],
                     self._num_roi: blobs['num_roi'], self._num_rel: blobs['num_rel'],
                     self._edge_mask_inds: blobs['rel_mask_inds'],
                     self._edge_segment_inds: blobs['rel_segment_inds'],
                     self._edge_pair_mask_inds: blobs['rel_pair_mask_inds'],
                     self._edge_pair_segment_inds: blobs['rel_pair_segment_inds']
                     }
        return feed_dict

    def debug_step(self, sess, blobs, tensors):
        if len(tensors) == 1:
            var = tensors[0]
        else:
            var = tensors
        output = sess.run(
            var,
            feed_dict=self._parse_dict(blobs))
        return

    def train_step(self, sess, blobs, train_op):
        loss_cls, loss_bbox, loss_rel, loss, _ = sess.run([self._losses["cross_entropy"],
                                                           self._losses['bbox_loss'],
                                                           self._losses['rel_cross_entropy'],
                                                           self._losses['total_loss'],
                                                           train_op],
                                                          feed_dict=self._parse_dict(blobs))
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_rel': loss_rel, 'loss': loss}

    def train_step_with_summary(self, sess, blobs, train_op, summary_grads):
        loss_cls, loss_bbox, loss_rel, loss, summary, gsummary, _ = sess.run(
            [self._losses["cross_entropy"],
             self._losses['bbox_loss'],
             self._losses['rel_cross_entropy'],
             self._losses['total_loss'],
             self._summary_op,
             summary_grads,
             train_op],
            feed_dict=self._parse_dict(blobs))
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox, 'loss_rel': loss_rel, 'loss': loss,
                'summary': summary, 'gsummary': gsummary}


        # take the last predicted output

    def test_image(self, sess, blobs):
        cls_score, cls_prob, bbox_pred, rel_cls_score, rel_cls_prob = sess.run([self._predictions["cls_score"][-1],
                                                                                self._predictions['cls_prob'][-1],
                                                                                self._predictions['bbox_pred'][-1],
                                                                                self._predictions['rel_cls_score'][-1],
                                                                                self._predictions['rel_cls_prob'][-1]],
                                                                               feed_dict=self._parse_test_dict(blobs))
        return cls_score, cls_prob, bbox_pred, rel_cls_score, rel_cls_prob

    # Test the base output
    def test_image_iter(self, sess, blobs, iter):
        cls_score, cls_prob, bbox_pred, rel_cls_score, rel_cls_prob = sess.run([self._predictions["cls_score"][iter],
                                                                                self._predictions['cls_prob'][iter],
                                                                                self._predictions['bbox_pred'][iter],
                                                                                self._predications['rel_cls_score'][
                                                                                    iter],
                                                                                self._predictions['rel_cls_prob'][
                                                                                    iter]],
                                                                               feed_dict=self._parse_test_dict(blobs))
        return cls_score, cls_prob, bbox_pred, rel_cls_score, rel_cls_prob


class vgg16(DualGraph, vgg16):
    def __init__(self):
        DualGraph.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'


class resnetv1(DualGraph, resnetv1):
    def __init__(self, num_layers=50):
        DualGraph.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._num_layers = num_layers
        self._scope = 'resnet_v1_%d' % num_layers
        resnetv1._decide_blocks(self)


class mobilenetv1(DualGraph, mobilenetv1):
    def __init__(self):
        DualGraph.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
        self._scope = 'MobilenetV1'
