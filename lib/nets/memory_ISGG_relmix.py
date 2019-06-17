from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

from nets.network import Network
from nets.base_memory import BaseMemory
from nets.ISGG import DualGraph

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

from model.config import cfg
from utils.snippets import compute_target_memory, compute_rel_rois, select_map_by_class, compute_target_map
from utils.visualization import draw_predicted_boxes, draw_memory, draw_CAM


class Memory_ISGG_relmix(DualGraph, Network):
    def __init__(self):
        self._predictions = {}
        self._predictions["cls_score"] = []
        self._predictions["cls_prob"] = []
        self._predictions["cls_pred"] = []
        self._predictions["bbox_pred"] = []
        self._predictions["rel_proposals"] = []
        self._predictions["rel_cls_score"] = []
        self._predictions["rel_cls_prob"] = []
        self._predictions["rel_cls_pred"] = []
        self._predictions["tag"] = []
        self._predictions["obj_pool_map"] = []
        self._predictions["fg_obj_pool_map"] = []
        self._predictions["fg_obj_global_map"] = []
        self._predictions["rel_pool_map"] = []
        self._predictions["fg_rel_pool_map"] = []
        self._predictions["fg_rel_global_map"] = []
        self._losses = {}
        self._targets = {}
        self._layers = {}
        self._gt_image = None
        self._mems = []
        self._rel_mems = []
        self._act_summaries = []
        self._score_summaries = [[] for _ in range(cfg.MEM.ITER)]
        self._event_summaries = {}
        self._variables_to_fix = {}

        self._vert_state_dim = 512
        self._edge_state_dim = 512

    def _add_score_iter_summary(self, iter, tensor):
        tf.summary.histogram('SCORE-%02d/' % iter + tensor.op.name, tensor)

    def _add_memory_summary(self, iter):
        image = tf.py_func(draw_memory,
                           [self._mems[iter]],
                           tf.float32,
                           name="memory")
        return tf.summary.image('MEM-%02d' % iter, image)

    def _add_rel_memory_summary(self, iter):
        image = tf.py_func(draw_memory,
                           [self._rel_mems[iter]],
                           tf.float32,
                           name="rel_memory")
        return tf.summary.image('REL-MEM-%02d' % iter, image)

    def _add_CAM_summary(self, iter):
        if self._gt_image is None:
            self._add_gt_image()
        # recover to BGR, since CAM use opencv to draw
        bgr_image = tf.reverse(self._gt_image, axis=[-1])
        image = tf.py_func(draw_CAM, [self._predictions['fg_obj_global_map'][iter], bgr_image], tf.float32,
                           name="CAM")
        return tf.summary.image('CAM-%02d' % iter, image)

    def _add_rel_CAM_summary(self, iter):
        if self._gt_image is None:
            self._add_gt_image()
        # recover to BGR, since CAM use opencv to draw
        bgr_image = tf.reverse(self._gt_image, axis=[-1])
        image = tf.py_func(draw_CAM, [self._predictions['fg_rel_global_map'][iter], bgr_image], tf.float32,
                           name="CAM")
        return tf.summary.image('REL_CAM-%02d' % iter, image)

    def _add_pred_memory_summary(self, iter):
        # also visualize the predictions of the network
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_predicted_boxes,
                           [self._gt_image,
                            self._predictions['cls_prob'][iter],
                            self._gt_boxes],
                           tf.float32, name="pred_boxes")
        return tf.summary.image('PRED-%02d' % iter, image)

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

    def _target_memory_layer(self, name):
        with tf.variable_scope(name):
            rois, batch_ids, inv_rois, inv_batch_ids = tf.py_func(compute_target_memory,
                                                                  [self._memory_size, self._rois,
                                                                   self._feat_stride[0]],
                                                                  [tf.float32, tf.int32, tf.float32,
                                                                   tf.int32],
                                                                  name="target_memory_layer")

            labels = self._labels
            bbox_targets = self._bbox_targets
            bbox_inside_weights = self._bbox_inside_weights
            bbox_outside_weights = self._bbox_outside_weights

            rois.set_shape([None, 4])
            labels.set_shape([None, 1])
            inv_rois.set_shape([None, 4])

            bbox_targets.set_shape([None, 4 * self._num_classes])
            bbox_inside_weights.set_shape([None, 4 * self._num_classes])
            bbox_outside_weights.set_shape([None, 4 * self._num_classes])
            self._targets['rois'] = rois
            self._targets['batch_ids'] = batch_ids
            self._targets['labels'] = labels
            self._targets['inv_rois'] = inv_rois
            self._targets['inv_batch_ids'] = inv_batch_ids
            self._targets['bbox_targets'] = bbox_targets
            self._targets['bbox_inside_weights'] = bbox_inside_weights
            self._targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries[0].append(rois)
            self._score_summaries[0].append(labels)
            self._score_summaries[0].append(inv_rois)
            self._score_summaries[0].append(bbox_targets)
            self._score_summaries[0].append(bbox_inside_weights)
            self._score_summaries[0].append(bbox_outside_weights)

        return rois, batch_ids, inv_rois, inv_batch_ids, bbox_targets, bbox_inside_weights

    def _target_rel_memory_layer(self, name):
        with tf.variable_scope(name):
            rel_rois, rel_batch_ids, inv_rel_rois, inv_rel_batch_ids = tf.py_func(compute_target_memory,
                                                                                  [self._rel_memory_size,
                                                                                   self._rel_rois,
                                                                                   self._feat_stride[0]],
                                                                                  [tf.float32, tf.int32, tf.float32,
                                                                                   tf.int32],
                                                                                  name="target_rel_memory_layer")
            relations = self._relations
            predicates = self._predicates

            rel_rois.set_shape([None, 4])
            inv_rel_rois.set_shape([None, 4])
            relations.set_shape([None, 2])
            predicates.set_shape([None, 1])

            self._targets['rel_rois'] = rel_rois
            self._targets['rel_batch_ids'] = rel_batch_ids
            self._targets['inv_rel_rois'] = inv_rel_rois
            self._targets['inv_rel_batch_ids'] = inv_rel_batch_ids
            self._targets['relations'] = relations
            self._targets['predicates'] = predicates

            self._score_summaries[0].append(rel_rois)
            self._score_summaries[0].append(inv_rel_rois)
            self._score_summaries[0].append(relations)
            self._score_summaries[0].append(predicates)

            return rel_rois, rel_batch_ids, inv_rel_rois, inv_rel_batch_ids

    def _target_isc_memory_layer(self, name):
        with tf.variable_scope(name):
            isc_rois, isc_batch_ids, inv_isc_rois, inv_isc_batch_ids = tf.py_func(compute_target_memory,
                                                                                  [self._rel_memory_size,
                                                                                   self._isc_rois,
                                                                                   self._feat_stride[0]],
                                                                                  [tf.float32, tf.int32, tf.float32,
                                                                                   tf.int32],
                                                                                  name="target_isc_memory_layer")

            isc_rois.set_shape([None, 4])
            inv_isc_rois.set_shape([None, 4])

            self._targets['isc_rois'] = isc_rois
            self._targets['isc_batch_ids'] = isc_batch_ids
            self._targets['inv_isc_rois'] = inv_isc_rois
            self._targets['inv_isc_batch_ids'] = inv_isc_batch_ids

            self._score_summaries[0].append(isc_rois)
            self._score_summaries[0].append(inv_isc_rois)

            return isc_rois, isc_batch_ids, inv_isc_rois, inv_isc_batch_ids

    def _rel_integrate_conv(self, all_rel_pool5, is_training):
        xavier = tf.contrib.layers.variance_scaling_initializer()
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            activation_fn=None,
                            trainable=is_training,
                            weights_initializer=xavier,
                            biases_initializer=tf.constant_initializer(0.0)):
            conv = cfg.MEM.CT_CONV
            num_layers = 2
            input = all_rel_pool5
            for i in range(num_layers):
                out = slim.conv2d(input, cfg.MEM.C, [conv, conv], scope='rel_mix%d'%i)
                input = out
            self._act_summaries.append(out)
            self._score_summaries[0].append(out)
        return out

    def compute_target_map_layer(self, name):
        with tf.variable_scope(name):
            rel_target, target_obj_map = tf.py_func(compute_target_map,
                                                    [self._memory_size, self._relations, self._predicates,
                                                     self._rois, self._labels, cfg.MEM.CROP_SIZE,
                                                     self._feat_stride[0]], [tf.float32, tf.float32],
                                                    name="target_map_layer")

            rel_target.set_shape([None, None, None, 1])
            target_rel_map = self._crop_rois(rel_target, self._targets["rel_rois"],
                                             self._targets["inv_rel_batch_ids"],
                                             name="target_rel_map")
            target_obj_map.set_shape([None, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
            self._targets["target_rel_map"] = target_rel_map
            self._targets["target_obj_map"] = target_obj_map
            self._score_summaries[0].append(target_rel_map)
            self._score_summaries[0].append(target_obj_map)

    def _crop_rois(self, bottom, rois, batch_ids, name, iter=0):
        with tf.variable_scope(name):
            crops = tf.image.crop_and_resize(bottom, rois, batch_ids,
                                             [cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE],
                                             name="crops")
            self._score_summaries[iter].append(crops)
        return crops

    def _inv_crops(self, pool5, inv_rois, inv_batch_ids, name):
        with tf.variable_scope(name):
            inv_crops = tf.image.crop_and_resize(pool5, inv_rois, inv_batch_ids, self._memory_size,
                                                 extrapolation_value=0,  # difference is 0 outside
                                                 name="inv_crops")
            # Add things up (make sure it is relu)
            inv_crop = tf.reduce_sum(inv_crops, axis=0, keep_dims=True, name="reduce_sum")
            return inv_crop, inv_crops

    # The initial classes, only use output from the conv features
    def _cls_init(self, fc7, is_training, prefix=None):
        out = self._num_classes if prefix is None else self._num_predicates
        prefix = '' if prefix is None else prefix + '_'
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        cls_score = slim.fully_connected(fc7, out,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope=prefix + 'cls_score')
        cls_prob = tf.nn.softmax(cls_score, name=prefix + "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name=prefix + "cls_pred")

        self._score_summaries[0].append(cls_score)
        self._score_summaries[0].append(cls_pred)
        self._score_summaries[0].append(cls_prob)

        return cls_score, cls_prob

    # The initial regressions
    def _reg_init(self, fc7, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')
        self._score_summaries[0].append(bbox_pred)

        return bbox_pred

    def _mem_init(self, is_training, name):
        mem_initializer = tf.constant_initializer(0.0)
        # Kinda like bias
        if cfg.TRAIN.BIAS_DECAY:
            mem_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        else:
            mem_regularizer = tf.no_regularizer

        with tf.variable_scope(self._scope):
            with tf.variable_scope(name):
                mem_init = tf.get_variable('mem_init',
                                           [1, cfg.MEM.INIT_H, cfg.MEM.INIT_W, cfg.MEM.C],
                                           initializer=mem_initializer,
                                           trainable=is_training,
                                           regularizer=mem_regularizer)
                self._score_summaries[0].append(mem_init)
                # resize it to the image-specific size
                mem_init = tf.image.resize_bilinear(mem_init, self._memory_size,
                                                    name="resize_init")

        return mem_init

    def _rel_mem_init(self, is_training, name):
        mem_initializer = tf.constant_initializer(0.0)
        if cfg.TRAIN.BIAS_DECAY:
            mem_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        else:
            mem_regularizer = tf.no_regularizer
        with tf.variable_scope(self._scope):
            with tf.variable_scope(name):
                rel_mem_init = tf.get_variable('rel_mem_init',
                                               [1, cfg.MEM.INIT_H, cfg.MEM.INIT_W, cfg.MEM.C],
                                               initializer=mem_initializer,
                                               trainable=is_training,
                                               regularizer=mem_regularizer)
                self._score_summaries[0].append(rel_mem_init)
                rel_mem_init = tf.image.resize_bilinear(rel_mem_init, self._rel_memory_size, name='rel_resize_init')
        return rel_mem_init

    def _context_conv(self, net, conv, scope):
        net = slim.conv2d(net, cfg.MEM.C, [conv, conv], scope=scope)
        return net

    def _context(self, net, is_training, name, iter):
        num_layers = cfg.MEM.CT_L
        xavier = tf.contrib.layers.variance_scaling_initializer()

        assert num_layers % 2 == 1
        conv = cfg.MEM.CT_CONV
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                activation_fn=None,
                                trainable=is_training,
                                weights_initializer=xavier,
                                biases_initializer=tf.constant_initializer(0.0)):
                net = self._context_conv(net, cfg.MEM.CT_FCONV, "conv1")
                for i in range(2, num_layers + 1, 2):
                    net1 = tf.nn.relu(net, name="relu%02d" % (i - 1))
                    self._act_summaries.append(net1)
                    self._score_summaries[iter].append(net1)
                    net1 = self._context_conv(net1, conv, "conv%02d" % i)
                    net2 = tf.nn.relu(net1, name="relu%02d" % i)
                    self._act_summaries.append(net2)
                    self._score_summaries[iter].append(net2)
                    net2 = self._context_conv(net2, conv, "conv%02d" % (i + 1))
                    net = tf.add(net, net2, "residual%02d" % i)

        return net

    def _rel_context(self, net, is_training, name, iter):
        # design for relation memory context specifically
        pass

    def _local_fc_iter(self, mem_pool5, is_training, name, iter):
        xavier = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(name):
            mem_fc7 = slim.flatten(mem_pool5, scope='flatten')
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=tf.nn.relu,
                                trainable=is_training,
                                weights_initializer=xavier,
                                biases_initializer=tf.constant_initializer(0.0)):
                for i in range(cfg.MEM.FC_L):
                    mem_fc7 = slim.fully_connected(mem_fc7,
                                                   cfg.MEM.FC_C,
                                                   scope="mem_fc%d" % (i + 6))
                    self._act_summaries.append(mem_fc7)
                    self._score_summaries[iter].append(mem_fc7)

        return mem_fc7

    def _cls_iter(self, mem_fc7, is_training, name, iter):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.C_STD)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            cls_score_mem = slim.fully_connected(mem_fc7, self._num_classes,
                                                 weights_initializer=initializer,
                                                 activation_fn=None,
                                                 trainable=is_training,
                                                 biases_initializer=tf.constant_initializer(0.0),
                                                 scope="cls_score_mem")
            self._score_summaries[iter].append(cls_score_mem)

        return cls_score_mem

    def _bbox_iter(self, mem_fc7, is_training, name, iter):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.C_STD)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            bbox_pred_mem = slim.fully_connected(mem_fc7, self._num_classes * 4,
                                                 weights_initializer=initializer,
                                                 activation_fn=None,
                                                 trainable=is_training,
                                                 biases_initializer=tf.constant_initializer(0.0),
                                                 scope="bbox_pred_mem")
            self._score_summaries[iter].append(bbox_pred_mem)
        return bbox_pred_mem

    def _rel_cls_iter(self, mem_fc7, is_training, name, iter):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.C_STD)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            rel_cls_score = slim.fully_connected(mem_fc7, self._num_predicates,
                                                 weights_initializer=initializer,
                                                 activation_fn=None,
                                                 trainable=is_training,
                                                 biases_initializer=tf.constant_initializer(0.0),
                                                 scope="rel_cls_score")
            self._score_summaries[iter].append(rel_cls_score)
        return rel_cls_score

    def _comb_conv_mem(self, cls_score_conv, cls_score_mem, bbox_pred_conv, bbox_pred_mem, name, iter):
        with tf.variable_scope(name):
            # take the output directly from each iteration
            if iter == 0:
                cls_score = cls_score_conv
                bbox_pred = bbox_pred_conv
            else:
                cls_score = cls_score_mem
                bbox_pred = bbox_pred_mem

            cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

            self._predictions['cls_score'].append(cls_score)
            self._predictions['cls_pred'].append(cls_pred)
            self._predictions['cls_prob'].append(cls_prob)
            self._predictions['bbox_pred'].append(bbox_pred)

            self._score_summaries[iter].append(cls_score)
            self._score_summaries[iter].append(cls_pred)
            self._score_summaries[iter].append(cls_prob)
            self._score_summaries[iter].append(bbox_pred)

        return cls_score, cls_prob, cls_pred, bbox_pred

    def _rel_comb_conv_mem(self, rel_score_conv, rel_score_mem, name, iter):
        with tf.variable_scope(name):
            if iter == 0:
                rel_cls_score = rel_score_conv
            else:
                rel_cls_score = rel_score_mem

            rel_cls_prob = tf.nn.softmax(rel_cls_score, name="rel_cls_prob")
            rel_cls_pred = tf.argmax(rel_cls_score, axis=1, name="rel_cls_pred")

            self._predictions['rel_cls_score'].append(rel_cls_score)
            self._predictions['rel_cls_prob'].append(rel_cls_prob)
            self._predictions['rel_cls_pred'].append(rel_cls_pred)

            self._score_summaries[iter].append(rel_cls_score)
            self._score_summaries[iter].append(rel_cls_prob)
            self._score_summaries[iter].append(rel_cls_pred)

        return rel_cls_score, rel_cls_prob, rel_cls_pred

    def _bottomtop(self, pool5, cls_prob, bbox_pred, is_training, name, iter):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD)
        initializer_fc = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD * cfg.MEM.FP_R)
        with tf.variable_scope(name):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=None,
                                trainable=is_training):
                # just make the representation more dense
                map_prob = slim.fully_connected(cls_prob,
                                                cfg.MEM.C,
                                                weights_initializer=initializer_fc,
                                                biases_initializer=None,
                                                scope="map_prob")
                map_bbox = slim.fully_connected(bbox_pred,
                                                cfg.MEM.C,
                                                weights_initializer=initializer_fc,
                                                biases_initializer=None,
                                                scope="map_bbox")

                map_comp = tf.reshape(map_prob, [-1, 1, 1, cfg.MEM.C], name="map_comp")
                map_bbox = tf.reshape(map_bbox, [-1, 1, 1, cfg.MEM.C], name="map_bbox")

                pool5_comp = slim.conv2d(pool5,
                                         cfg.MEM.C,
                                         [1, 1],
                                         weights_initializer=initializer,
                                         biases_initializer=tf.constant_initializer(0.0),
                                         scope="pool5_comp")

                pool5_comb = tf.add(pool5_comp, map_comp, name="addition")
                pool5_comb = tf.add(pool5_comb, map_bbox, name="addition2")
                pool5_comb = tf.nn.relu(pool5_comb, name="pool5_comb")

                self._score_summaries[iter].append(map_prob)
                self._score_summaries[iter].append(map_bbox)
                self._score_summaries[iter].append(pool5_comp)
                self._score_summaries[iter].append(pool5_comb)
                self._act_summaries.append(pool5_comb)

        return pool5_comb

    def _rel_bottomtop(self, rel_pool5_nb, rel_cls_score_nb, is_training, name, iter):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD)
        initializer_fc = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD * cfg.MEM.FP_R)
        with tf.variable_scope(name):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=None,
                                trainable=is_training):
                # just make the representation more dense
                map_prob = slim.fully_connected(rel_cls_score_nb,
                                                cfg.MEM.C,
                                                weights_initializer=initializer_fc,
                                                biases_initializer=None,
                                                scope="rel_map_prob")

                map_comp = tf.reshape(map_prob, [-1, 1, 1, cfg.MEM.C], name="rel_map_comp")

                pool5_comp = slim.conv2d(rel_pool5_nb,
                                         cfg.MEM.C,
                                         [1, 1],
                                         weights_initializer=initializer,
                                         biases_initializer=tf.constant_initializer(0.0),
                                         scope="convolution")
                pool5_comb = tf.add(map_comp, pool5_comp, name="addition")
                pool5_comb = tf.nn.relu(pool5_comb, name="pool5_comb")

                self._score_summaries[iter].append(map_prob)
                self._score_summaries[iter].append(pool5_comp)
                self._score_summaries[iter].append(pool5_comb)
                self._act_summaries.append(pool5_comb)

        return pool5_comb

    def _bottom(self, pool5, is_training, name, iter):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=cfg.MEM.STD)
        with tf.variable_scope(name):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=None,
                                trainable=is_training):
                # just make the representation more dense
                pool5_comp = slim.conv2d(pool5,
                                         cfg.MEM.C,
                                         [1, 1],
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=initializer,
                                         biases_initializer=tf.constant_initializer(0.0),
                                         scope="pool5_comp")
                self._score_summaries[iter].append(pool5_comp)
                self._act_summaries.append(pool5_comp)

        return pool5_comp

    def _topprob(self, cls_prob, is_training, name, iter):
        initializer_fc = tf.random_normal_initializer(mean=0.0,
                                                      stddev=cfg.MEM.STD * cfg.MEM.FP_R)
        with tf.variable_scope(name):
            # just make the representation more dense
            map_prob = slim.fully_connected(cls_prob,
                                            cfg.MEM.C,
                                            activation_fn=tf.nn.relu,
                                            trainable=is_training,
                                            weights_initializer=initializer_fc,
                                            biases_initializer=tf.constant_initializer(0.0),
                                            scope="map_prob")
            map_comp = tf.reshape(map_prob, [-1, 1, 1, cfg.MEM.C], name="map_comp")
            map_pool = tf.tile(map_comp, [1, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1], name="map_pool")
            self._score_summaries[iter].append(map_prob)
            self._act_summaries.append(map_prob)

        return map_pool

    def _toppred(self, cls_pred, is_training, name, iter):
        initializer_fc = tf.random_normal_initializer(mean=0.0,
                                                      stddev=cfg.MEM.STD * cfg.MEM.FP_R)
        with tf.variable_scope(name):
            cls_pred_hot = tf.one_hot(cls_pred, self._num_classes, name="encode")
            # just make the representation more dense
            map_pred = slim.fully_connected(cls_pred_hot,
                                            cfg.MEM.C,
                                            activation_fn=tf.nn.relu,
                                            trainable=is_training,
                                            weights_initializer=initializer_fc,
                                            biases_initializer=tf.constant_initializer(0.0),
                                            scope="map_pred")
            map_comp = tf.reshape(map_pred, [-1, 1, 1, cfg.MEM.C], name="map_comp")
            map_pool = tf.tile(map_comp, [1, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1], name="map_pool")
            self._score_summaries[iter].append(map_pred)
            self._act_summaries.append(map_pred)

        return map_pool

    def _input(self, net, is_training, name, iter):
        num_layers = cfg.MEM.IN_L
        in_conv = cfg.MEM.IN_CONV
        xavier = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope(name):
            # the first part is already done
            for i in range(2, num_layers + 1):
                net = slim.conv2d(net, cfg.MEM.C, [in_conv, in_conv],
                                  activation_fn=tf.nn.relu,
                                  trainable=is_training,
                                  weights_initializer=xavier,
                                  biases_initializer=tf.constant_initializer(0.0),
                                  scope="conv%02d" % i)
                self._score_summaries[iter].append(net)
                self._act_summaries.append(net)

        return net

    def _mem_update(self, pool5_mem, pool5_input, is_training, name, iter):
        feat_initializer = tf.random_normal_initializer(mean=0.0,
                                                        stddev=cfg.MEM.U_STD)
        mem_initializer = tf.random_normal_initializer(mean=0.0,
                                                       stddev=cfg.MEM.U_STD * cfg.MEM.FM_R)
        feat_gate_initializer = tf.random_normal_initializer(mean=0.0,
                                                             stddev=cfg.MEM.U_STD / cfg.MEM.VG_R)
        mem_gate_initializer = tf.random_normal_initializer(mean=0.0,
                                                            stddev=cfg.MEM.U_STD * cfg.MEM.FM_R / cfg.MEM.VG_R)
        mconv = cfg.MEM.CONV
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                biases_initializer=tf.constant_initializer(0.0),
                                trainable=is_training):
                # compute the gates and features
                p_input = slim.conv2d(pool5_input,
                                      cfg.MEM.C,
                                      [mconv, mconv],
                                      weights_initializer=feat_initializer,
                                      scope="input_p")
                p_reset = slim.conv2d(pool5_input,
                                      1,
                                      [mconv, mconv],
                                      weights_initializer=feat_gate_initializer,
                                      scope="reset_p")
                p_update = slim.conv2d(pool5_input,
                                       1,
                                       [mconv, mconv],
                                       weights_initializer=feat_gate_initializer,
                                       scope="update_p")
                # compute the gates and features from the hidden memory
                m_reset = slim.conv2d(pool5_mem,
                                      1,
                                      [mconv, mconv],
                                      weights_initializer=mem_gate_initializer,
                                      biases_initializer=None,
                                      scope="reset_m")
                m_update = slim.conv2d(pool5_mem,
                                       1,
                                       [mconv, mconv],
                                       weights_initializer=mem_gate_initializer,
                                       biases_initializer=None,
                                       scope="update_m")
                # get the reset gate, the portion that is kept from the previous step
                reset_gate = tf.sigmoid(p_reset + m_reset, name="reset_gate")
                reset_res = tf.multiply(pool5_mem, reset_gate, name="m_input_reset")
                m_input = slim.conv2d(reset_res,
                                      cfg.MEM.C,
                                      [mconv, mconv],
                                      weights_initializer=mem_initializer,
                                      biases_initializer=None,
                                      scope="input_m")
                # Non-linear activation
                pool5_new = tf.nn.relu(p_input + m_input, name="pool5_new")
                # get the update gate, the portion that is taken to update the new memory
                update_gate = tf.sigmoid(p_update + m_update, name="update_gate")
                # the update is done in a difference manner
                mem_diff = tf.multiply(update_gate, pool5_new - pool5_mem,
                                       name="mem_diff")

            self._score_summaries[iter].append(p_reset)
            self._score_summaries[iter].append(p_update)
            self._score_summaries[iter].append(m_reset)
            self._score_summaries[iter].append(m_update)
            self._score_summaries[iter].append(reset_gate)
            self._score_summaries[iter].append(update_gate)
            self._score_summaries[iter].append(mem_diff)

        return mem_diff

    def _input_module(self, pool5_nb,
                      cls_score_nb, cls_prob_nb, cls_pred_nb, bbox_pred_nb,
                      is_training, iter):
        pool5_comb = self._bottomtop(pool5_nb, cls_score_nb, bbox_pred_nb, is_training, "bottom_top", iter)
        pool5_input = self._input(pool5_comb, is_training, "pool5_input", iter)

        return pool5_input

    def _rel_input_module(self, rel_pool5_nb, rel_cls_score_nb, rel_cls_prob_nb, rel_cls_pred_nb, is_training,
                          iter):
        pool5_comb = self._rel_bottomtop(rel_pool5_nb, rel_cls_score_nb, is_training, "rel_bottom_top", iter)
        pool5_input = self._input(pool5_comb, is_training, "rel_pool5_input", iter)
        return pool5_input

    def _build_conv(self, is_training):
        # Get the head
        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope, self._scope):
            # get the region of interest
            rois, batch_ids, inv_rois, inv_batch_ids, bbox_targets, bbox_inside_weights = \
                self._target_memory_layer("target")
            rel_rois, rel_batch_ids, inv_rel_rois, inv_rel_batch_ids = self._target_rel_memory_layer("rel_target")
            isc_rois, isc_batch_ids, inv_isc_rois, inv_isc_batch_ids = self._target_isc_memory_layer("isc_target")
            # self.compute_target_map_layer("rel_map_target")
            # region of interest pooling
            pool5 = self._crop_rois(net_conv, rois, batch_ids, "pool5")
            pool5_nb = tf.stop_gradient(pool5, name="pool5_nb")
            # region of relation interest pooling
            rel_pool5 = self._crop_rois(net_conv, rel_rois, rel_batch_ids, "rel_pool5")
            isc_pool5 = self._crop_rois(net_conv, isc_rois, isc_batch_ids, "isc_pool5")
            all_rel_pool5 = tf.concat([rel_pool5, isc_pool5], axis=3)
            rel_pool5 = self._rel_integrate_conv(all_rel_pool5, is_training)
            rel_pool5_nb = tf.stop_gradient(rel_pool5, name="rel_pool5_nb")

        # initialize the normalization vector, note here it is the batch ids
        count_matrix_raw, self._count_crops = self._inv_crops(self._count_base, inv_rois, batch_ids, "count_matrix")
        self._count_matrix = tf.stop_gradient(count_matrix_raw, name='cm_nb')
        self._count_matrix_eps = tf.maximum(self._count_matrix, cfg.EPS, name='count_eps')
        self._score_summaries[0].append(self._count_matrix)

        rel_count_matrix_raw, self._rel_count_crops = self._inv_crops(self._count_base, inv_rel_rois,
                                                                      rel_batch_ids, "rel_count_matrix")
        self._rel_count_matrix = tf.stop_gradient(rel_count_matrix_raw, name='rel_cm_nb')
        self._rel_count_matrix_eps = tf.maximum(self._rel_count_matrix, cfg.EPS, name='rel_count_eps')
        self._score_summaries[0].append(self._rel_count_matrix)

        fc7 = self._head_to_tail(pool5, is_training)
        rel_fc7 = self._head_to_tail(rel_pool5, is_training, prefix='rel')

        with tf.variable_scope(self._scope, self._scope):
            vert_unary, edge_unary = self._fc_init(fc7, rel_fc7, is_training, "unary") # 512 dim
            vert_factor = self._vert_rnn_forward(vert_unary, "vert_rnn", reuse=False)
            edge_factor = self._edge_rnn_forward(edge_unary, "edge_rnn", reuse=False)
            edge_ctx = self._compute_edge_context(vert_factor, edge_factor, is_training, reuse=False)
            edge_factor = self._edge_rnn_forward(edge_ctx, "edge_rnn", reuse=True)
            vert_ctx = self._compute_vert_context(edge_factor, vert_factor, is_training, reuse=False)
            vert_factor = self._vert_rnn_forward(vert_ctx, "vert_rnn", reuse=True)

        # First iteration
        with tf.variable_scope(self._scope, self._scope):
            # region classification
            cls_score_conv, cls_prob_conv = self._cls_init(vert_factor, is_training)
            bbox_pred_conv = self._reg_init(vert_factor, is_training)

        with tf.variable_scope(self._scope, self._scope):
            rel_score_conv, rel_prob_conv = self._cls_init(edge_factor, is_training, prefix='rel')

        return cls_score_conv, bbox_pred_conv, pool5_nb, rel_score_conv, rel_pool5_nb, \
               rois, batch_ids, inv_rois, inv_batch_ids, rel_rois, rel_batch_ids, inv_rel_rois, inv_rel_batch_ids, \
               vert_factor, edge_factor

    def _build_pred_all(self, is_training, mem, cls_score_conv, bbox_pred_conv, rois, batch_ids,
                        rel_mem, rel_score_conv, rel_rois, rel_batch_ids, vert_factor, edge_factor, iter):
        if cfg.MEM.CT_L:
            mem_net = self._context(mem, is_training, "context", iter)
            rel_mem_net = self._context(rel_mem, is_training, "rel_context", iter)
        else:
            mem_net = mem
            rel_mem_net = rel_mem
        mem_ct_pool5 = self._crop_rois(mem_net, rois, batch_ids, "mem_ct_pool5", iter)  # N * 7 * 7 * 512
        mem_fc7 = self._local_fc_iter(mem_ct_pool5, is_training, "fc7", iter)
        rel_mem_ct_pool5 = self._crop_rois(rel_mem_net, rel_rois, rel_batch_ids, "rel_mem_ct_pool5", iter)
        rel_mem_fc7 = self._local_fc_iter(rel_mem_ct_pool5, is_training, "rel_fc7", iter)
        if iter == 0:
            cls_score_mem = None
            bbox_pred_mem = None
            rel_score_mem = None
            # create the new variable for futural use.
            u = tf.get_variable("weight_obj_factor", shape=[1], dtype=tf.float32, trainable=is_training)
            v = tf.get_variable("weight_obj_mem", shape=[1], dtype=tf.float32, trainable=is_training)
            p = tf.get_variable("weight_rel_factor", shape=[1], dtype=tf.float32, trainable=is_training)
            q = tf.get_variable("weight_rel_mem", shape=[1], dtype=tf.float32, trainable=is_training)
        else:
            u = tf.get_variable("weight_obj_factor")
            v = tf.get_variable("weight_obj_mem")
            p = tf.get_variable("weight_rel_factor")
            q = tf.get_variable("weight_rel_mem")
            mem_fc7_ld, rel_mem_fc7_ld = self._fc_init(mem_fc7, rel_mem_fc7, is_training, "unary") # reuse
            vert_factor_new = tf.add(u * vert_factor, v * mem_fc7_ld, name="vert_factor_new")
            edge_factor_new = tf.add(p * edge_factor, q * rel_mem_fc7_ld, name="edge_factor_new")
            edge_ctx = self._compute_edge_context(vert_factor_new, edge_factor_new, is_training, reuse=True)
            edge_factor = self._edge_rnn_forward(edge_ctx, "edge_rnn", reuse=True)
            vert_ctx = self._compute_vert_context(edge_factor_new, vert_factor_new, is_training, reuse=True)
            vert_factor = self._vert_rnn_forward(vert_ctx, "vert_rnn", reuse=True)
            cls_score_mem = self._cls_iter(vert_factor, is_training, "cls_iter", iter)
            bbox_pred_mem = self._bbox_iter(vert_factor, is_training, "bbox_iter", iter)
            rel_score_mem = self._rel_cls_iter(edge_factor, is_training, "rel_cls_iter", iter)
        cls_score, cls_prob, cls_pred, bbox_pred = self._comb_conv_mem(cls_score_conv, cls_score_mem,
                                                                       bbox_pred_conv, bbox_pred_mem,
                                                                       "comb_conv_mem", iter)
        rel_cls_score, rel_cls_prob, rel_cls_pred = self._rel_comb_conv_mem(rel_score_conv,
                                                                            rel_score_mem,
                                                                            "rel_comb_conv_mem", iter)
        return cls_score, cls_prob, cls_pred, bbox_pred, rel_cls_score, rel_cls_prob, rel_cls_pred, \
               vert_factor, edge_factor

    def _build_pred(self, is_training, mem, cls_score_conv, bbox_pred_conv, rois, batch_ids, inv_rois, iter):
        if cfg.MEM.CT_L:
            mem_net = self._context(mem, is_training, "context", iter)
        else:
            mem_net = mem
        mem_ct_pool5 = self._crop_rois(mem_net, rois, batch_ids, "mem_ct_pool5", iter)  # N * 7 * 7 * 512
        mem_avg_pool5 = tf.reduce_mean(mem_ct_pool5, axis=[1, 2])
        # mem_fc7 = self._fc_iter(mem_ct_pool5, is_training, "fc7", iter)
        cls_score_mem = self._cls_iter(mem_avg_pool5, is_training, "cls_iter", iter)
        if iter > 0 and is_training:
            with tf.variable_scope("cls_iter", reuse=True):
                with tf.variable_scope("cls_score_mem", reuse=True):
                    cls_weights = tf.get_variable("weights")
            mem_ct_pool5_reshape = tf.reshape(tf.transpose(tf.reshape(
                tf.transpose(mem_ct_pool5, perm=[0, 3, 1, 2]),
                [-1, cfg.MEM.C, cfg.MEM.CROP_SIZE * cfg.MEM.CROP_SIZE]),
                perm=[1, 0, 2]), [cfg.MEM.C, -1])  # N*7*7*512 -> N*512*7*7 -> N*512*49 -> 512*N*49 -> 512*(49N)
            weighted_map = tf.matmul(tf.transpose(cls_weights, perm=[1, 0]), mem_ct_pool5_reshape)  # c*(49N)
            num_classes = len(cfg.ind_to_class)
            weighted_map = tf.reshape(
                tf.transpose(tf.reshape(weighted_map, [num_classes, -1, cfg.MEM.CROP_SIZE * cfg.MEM.CROP_SIZE]),
                             perm=[1, 0, 2]), [-1, num_classes, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE])  # N*c*7*7
            labels = self._labels
            fg_inds = tf.where(labels > 0)[:, 0]
            fg_labels = tf.gather(labels, fg_inds)[:, 0]
            fg_inv_rois = tf.gather(inv_rois, fg_inds)
            fg_weighted_map = tf.gather(weighted_map, fg_inds)
            fg_class_weighted_map, fg_inv_batch_ids = tf.py_func(select_map_by_class, [fg_weighted_map, fg_labels],
                                                                 [tf.float32, tf.int32],
                                                                 name="fg_obj_pool_map")  # N_fg*7*7*1
            fg_class_weighted_map.set_shape([None, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
            self._predictions["fg_obj_pool_map"].append(fg_class_weighted_map)

            class_weighted_map, inv_batch_ids = tf.py_func(select_map_by_class, [weighted_map, labels[:, 0]],
                                                           [tf.float32, tf.int32], name="obj_pool_map")
            class_weighted_map.set_shape([None, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
            self._predictions["obj_pool_map"].append(class_weighted_map)

            fg_obj_global_map, _ = self._inv_crops(fg_class_weighted_map, fg_inv_rois, fg_inv_batch_ids,
                                                   "fg_obj_global_map")
            self._predictions["fg_obj_global_map"].append(fg_obj_global_map)

        bbox_pred_mem = self._bbox_iter(mem_avg_pool5, is_training, "bbox_iter", iter)
        cls_score, cls_prob, cls_pred, bbox_pred = self._comb_conv_mem(cls_score_conv, cls_score_mem,
                                                                       bbox_pred_conv, bbox_pred_mem,
                                                                       "comb_conv_mem", iter)

        return cls_score, cls_prob, cls_pred, bbox_pred

    def _build_pred_rel(self, is_training, rel_mem, rel_score_conv, rel_rois, rel_batch_ids, inv_rel_rois, iter):
        if cfg.MEM.CT_L:
            rel_mem_net = self._context(rel_mem, is_training, "rel_context", iter)
        else:
            rel_mem_net = rel_mem
        rel_mem_ct_pool5 = self._crop_rois(rel_mem_net, rel_rois, rel_batch_ids, "rel_mem_ct_pool5", iter)
        rel_mem_avg_pool5 = tf.reduce_mean(rel_mem_ct_pool5, axis=[1, 2])
        # rel_mem_fc7 = self._fc_iter(rel_mem_ct_pool5, is_training, "rel_fc7", iter)
        rel_score_mem = self._rel_cls_iter(rel_mem_avg_pool5, is_training, "rel_cls_iter", iter)
        if iter > 0 and is_training:
            with tf.variable_scope("rel_cls_iter", reuse=True):
                with tf.variable_scope("rel_cls_score", reuse=True):
                    cls_weights = tf.get_variable("weights")
            mem_ct_pool5_reshape = tf.reshape(tf.transpose(tf.reshape(
                tf.transpose(rel_mem_ct_pool5, perm=[0, 3, 1, 2]),
                [-1, cfg.MEM.C, cfg.MEM.CROP_SIZE * cfg.MEM.CROP_SIZE]),
                perm=[1, 0, 2]), [cfg.MEM.C, -1])  # N*7*7*512 -> N*512*7*7 -> N*512*49 -> 512*N*49 -> 512*(49N)
            weighted_map = tf.matmul(tf.transpose(cls_weights, perm=[1, 0]), mem_ct_pool5_reshape)  # c*(49N)
            num_predicates = len(cfg.ind_to_predicates)
            weighted_map = tf.reshape(
                tf.transpose(tf.reshape(weighted_map, [num_predicates, -1, cfg.MEM.CROP_SIZE * cfg.MEM.CROP_SIZE]),
                             perm=[1, 0, 2]), [-1, num_predicates, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE])  # N*c*7*7
            predicates = self._predicates
            fg_inds = tf.where(predicates > 0)[:, 0]
            fg_predicates = tf.gather(predicates, fg_inds)[:, 0]
            fg_inv_rel_rois = tf.gather(inv_rel_rois, fg_inds)
            fg_weighted_map = tf.gather(weighted_map, fg_inds)
            fg_rel_class_weighted_map, fg_inv_batch_ids = tf.py_func(select_map_by_class,
                                                                     [fg_weighted_map, fg_predicates],
                                                                     [tf.float32, tf.int32],
                                                                     name="fg_rel_pool_map")  # N_fg*7*7*1
            fg_rel_class_weighted_map.set_shape([None, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
            self._predictions["fg_rel_pool_map"].append(fg_rel_class_weighted_map)

            rel_class_weighted_map, inv_batch_ids = tf.py_func(select_map_by_class, [weighted_map, predicates[:, 0]],
                                                               [tf.float32, tf.int32], name="rel_pool_map")
            rel_class_weighted_map.set_shape([None, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
            self._predictions["rel_pool_map"].append(rel_class_weighted_map)

            fg_rel_global_map, _ = self._inv_crops(fg_rel_class_weighted_map, fg_inv_rel_rois, fg_inv_batch_ids,
                                                   "fg_rel_global_map")
            self._predictions["fg_rel_global_map"].append(fg_rel_global_map)

        rel_cls_score, rel_cls_prob, rel_cls_pred = self._rel_comb_conv_mem(rel_score_conv,
                                                                            rel_score_mem,
                                                                            "rel_comb_conv_mem", iter)
        return rel_cls_score, rel_cls_prob, rel_cls_pred

    def _build_gru(self, state, input, is_training, iter, name):
        mem_gate_initializer = tf.random_normal_initializer(mean=0.0,
                                                            stddev=cfg.MEM.U_STD * cfg.MEM.FM_R / cfg.MEM.VG_R)
        mconv = cfg.MEM.CONV
        with tf.variable_scope(name):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                biases_initializer=tf.constant_initializer(0.0),
                                trainable=is_training):
                reset_gate = tf.sigmoid(slim.conv2d(input, 1, [mconv, mconv], weights_initializer=mem_gate_initializer,
                                                    scope="input_reset") +
                                        slim.conv2d(state, 1, [mconv, mconv], weights_initializer=mem_gate_initializer,
                                                    scope="state_reset"),
                                        name="reset_gate")
                update_gate = tf.sigmoid(slim.conv2d(input, 1, [mconv, mconv], weights_initializer=mem_gate_initializer,
                                                     scope="input_update") +
                                         slim.conv2d(state, 1, [mconv, mconv], weights_initializer=mem_gate_initializer,
                                                     scope="state_update"),
                                         name="reset_gate")
                reset_state = tf.multiply(reset_gate, state)
                _input = slim.conv2d(input, cfg.MEM.C, [mconv, mconv], weights_initializer=mem_gate_initializer,
                                     scope="input_control") + \
                         slim.conv2d(reset_state, cfg.MEM.C, [mconv, mconv], weights_initializer=mem_gate_initializer,
                                     scope="state_control")
                output = update_gate * state + (1. - update_gate) * tf.tanh(_input)
        self._score_summaries[iter].append(reset_gate)
        self._score_summaries[iter].append(update_gate)
        self._score_summaries[iter].append(reset_state)
        self._score_summaries[iter].append(output)
        return output

    def _build_mem_communicate(self, is_training, mem, rel_mem, iter):
        new_mem = self._build_gru(mem, rel_mem, is_training, iter, "comm_mem")
        new_rel_mem = self._build_gru(rel_mem, mem, is_training, iter, "comm_rel_mem")
        mem = new_mem
        rel_mem = new_rel_mem
        return mem, rel_mem

    def _build_update(self, is_training, mem, pool5_nb, cls_score, cls_prob, cls_pred, bbox_pred,
                      rois, batch_ids, inv_rois, inv_batch_ids, iter):
        cls_score_nb = tf.stop_gradient(cls_score, name="cls_score_nb")
        cls_prob_nb = tf.stop_gradient(cls_prob, name="cls_prob_nb")
        cls_pred_nb = tf.stop_gradient(cls_pred, name="cls_pred_nb")
        bbox_pred_nb = tf.stop_gradient(bbox_pred, name="bbox_pred_nb")
        pool5_mem = self._crop_rois(mem, rois, batch_ids, "pool5_mem", iter)
        pool5_input = self._input_module(pool5_nb,
                                         cls_score_nb, cls_prob_nb,
                                         cls_pred_nb, bbox_pred_nb, is_training, iter)
        mem_update = self._mem_update(pool5_mem, pool5_input, is_training, "mem_update", iter)
        mem_diff, _ = self._inv_crops(mem_update, inv_rois, inv_batch_ids, "inv_crop")
        self._score_summaries[iter].append(mem_diff)
        # Update the memory
        mem_div = tf.div(mem_diff, self._count_matrix_eps, name="div")
        mem = tf.add(mem, mem_div, name="add")
        self._score_summaries[iter].append(mem)

        return mem

    def _build_update_rel(self, is_training, rel_mem, rel_pool5_nb, rel_cls_score,
                          rel_cls_prob, rel_cls_pred, rel_rois, rel_batch_ids,
                          inv_rel_rois, inv_rel_batch_ids, iter):
        rel_cls_score_nb = tf.stop_gradient(rel_cls_score, name="rel_cls_score_nb")
        rel_cls_prob_nb = tf.stop_gradient(rel_cls_prob, name="rel_cls_prob_nb")
        rel_cls_pred_nb = tf.stop_gradient(rel_cls_pred, name="rel_cls_pred_nb")
        rel_pool5_mem = self._crop_rois(rel_mem, rel_rois, rel_batch_ids, "rel_pool5_mem", iter)
        rel_pool5_input = self._rel_input_module(rel_pool5_nb, rel_cls_score_nb, rel_cls_prob_nb,
                                                 rel_cls_pred_nb, is_training, iter)
        rel_mem_update = self._mem_update(rel_pool5_mem, rel_pool5_input, is_training, "rel_mem_update", iter)
        rel_mem_diff, _ = self._inv_crops(rel_mem_update, inv_rel_rois, inv_rel_batch_ids, "rel_inv_crop")
        self._score_summaries[iter].append(rel_mem_diff)
        rel_mem_div = tf.div(rel_mem_diff, self._rel_count_matrix_eps, name="rel_div")
        rel_mem = tf.add(rel_mem, rel_mem_div, name="rel_add")
        self._score_summaries[iter].append(rel_mem)

        return rel_mem

    def _build_update_from_rel(self, is_training, rel_mem, mem, obj_mem_obj_pool5_nb, inv_rois, inv_batch_ids,
                               pred_rel_rois,
                               pred_rel_batch_ids, name, iter):
        rel_mem_nb = tf.stop_gradient(rel_mem, name="rel_mem_nb")
        rel_mem_pool5_upd = self._crop_rois(rel_mem_nb, pred_rel_rois, pred_rel_batch_ids, "rel_mem_pool5_upd", iter)
        rel_mem_pool5_upd_flatten = slim.flatten(rel_mem_pool5_upd, scope='rel_mem_pool5_upd_flatten')
        obj_mem_obj_pool5_flatten = slim.flatten(obj_mem_obj_pool5_nb, scope="obj_mem_obj_pool5_flatten")
        num_rel = rel_mem_pool5_upd_flatten.shape[0]
        num_obj = obj_mem_obj_pool5_flatten.shape[0]
        dim = rel_mem_pool5_upd_flatten.shape[-1]
        assert dim == obj_mem_obj_pool5_flatten.shape[-1]
        rel_mem_pool5_upd_flatten = tf.tile(tf.expand_dims(rel_mem_pool5_upd_flatten, axis=0),
                                            [int(num_obj), 1, 1])  # N_obj, N_rel, d
        obj_mem_obj_pool5_flatten = tf.tile(tf.expand_dims(obj_mem_obj_pool5_flatten, axis=1),
                                            [1, int(num_rel), 1])  # N_obj, N_rel, d

        concat_feat = tf.reshape(tf.concat([obj_mem_obj_pool5_flatten, rel_mem_pool5_upd_flatten], axis=2),
                                 [-1, dim * 2])

        xavier = tf.contrib.layers.variance_scaling_initializer()

        with tf.variable_scope(name):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=None,
                                trainable=is_training,
                                weights_initializer=xavier,
                                biases_initializer=tf.constant_initializer(0.0)):
                weight = slim.fully_connected(concat_feat,
                                              1,
                                              scope="weight")
            weight = tf.nn.softmax(tf.reshape(weight, [num_obj, num_rel, 1]), dim=1)  # N_obj, N_rel, 1
            weight = weight[:, :, :, None, None]  # N_obj, N_rel, 1, 1, 1
            rel_mem_pool5_upd_expd = rel_mem_pool5_upd[None]  # 1, N_rel, 7, 7, 512
            weighted_obj_input = tf.reduce_sum(tf.multiply(weight, rel_mem_pool5_upd_expd), axis=1,
                                               name="weighted_obj_input")  # N_obj, 7, 7, 512
            obj_mem_update_from_rel_mem = self._mem_update(obj_mem_obj_pool5_nb, weighted_obj_input, is_training,
                                                           "obj_mem_update_from_rel_mem", iter)
            obj_mem_update_from_rel_mem_diff, _ = self._inv_crops(obj_mem_update_from_rel_mem, inv_rois, inv_batch_ids,
                                                                  "obj_mem_update_from_rel_inv_crop")
            self._score_summaries[iter].append(obj_mem_update_from_rel_mem_diff)
            obj_mem_update_from_rel_div = tf.div(obj_mem_update_from_rel_mem_diff, self._count_matrix_eps,
                                                 name="obj_mem_update_from_rel_div")
            mem = tf.add(mem, obj_mem_update_from_rel_div, name="obj_mem_update_from_rel_add")
            self._score_summaries[iter].append(mem)
        return mem

    def _l2_loss(self, input, target):
        return tf.reduce_mean(tf.reduce_sum((input - target) ** 2 / 2, axis=[1, 2, 3]))

    def _build_memory(self, is_training, is_testing):
        # initialize memory
        mem = self._mem_init(is_training, "mem_init")
        rel_mem = self._rel_mem_init(is_training, "rel_mem_init")
        # convolution related stuff
        cls_score_conv, bbox_pred_conv, pool5_nb, rel_score_conv, rel_pool5_nb, \
        rois, batch_ids, inv_rois, inv_batch_ids, \
        rel_rois, rel_batch_ids, inv_rel_rois, inv_rel_batch_ids, vert_factor, edge_factor = self._build_conv(
            is_training)
        # Separate first prediction
        reuse = None
        # Memory iterations
        for iter in range(cfg.MEM.ITER):
            print('ITERATION: %02d' % iter)
            self._mems.append(mem)
            self._rel_mems.append(rel_mem)
            with tf.variable_scope(self._scope, reuse=reuse):
                # Use memory to predict the output
                '''
                cls_score, cls_prob, cls_pred, bbox_pred = self._build_pred(is_training,
                                                                            mem,
                                                                            cls_score_conv,
                                                                            bbox_pred_conv,
                                                                            rois, batch_ids, inv_rois, iter)

                rel_cls_score, rel_cls_prob, rel_cls_pred = self._build_pred_rel(is_training,
                                                                                 rel_mem,
                                                                                 rel_score_conv,
                                                                                 rel_rois,
                                                                                 rel_batch_ids, inv_rel_rois, iter)
                '''
                cls_score, cls_prob, cls_pred, bbox_pred, \
                rel_cls_score, rel_cls_prob, rel_cls_pred, vert_factor, edge_factor = \
                    self._build_pred_all(is_training, mem, cls_score_conv, bbox_pred_conv, rois, batch_ids,
                                         rel_mem, rel_score_conv, rel_rois, rel_batch_ids, vert_factor, edge_factor, iter)

                # if iter == cfg.MEM.ITER - 1:
                #    break

                # Update the memory with all the regions
                mem = self._build_update(is_training, mem,
                                         pool5_nb, cls_score, cls_prob, cls_pred, bbox_pred,
                                         rois, batch_ids, inv_rois, inv_batch_ids, iter)

                rel_mem = self._build_update_rel(is_training, rel_mem, rel_pool5_nb, rel_cls_score, rel_cls_prob,
                                                 rel_cls_pred, rel_rois, rel_batch_ids, inv_rel_rois,
                                                 inv_rel_batch_ids, iter)

                #mem, rel_mem = self._build_mem_communicate(is_training, mem, rel_mem, iter)

                '''
                # predict relations
                rel_cls_score, rel_cls_prob, rel_cls_pred, \
                pred_rels, pred_rel_rois, pred_rel_batch_ids, \
                pred_inv_rel_rois, pred_inv_rel_batch_ids, \
                obj_mem_obj_pool5_nb, obj_mem_rel_pool5_nb = self._pred_rel(
                    is_training, mem, rel_mem, rois, batch_ids, rel_rois, rel_batch_ids,
                    inv_rel_rois, inv_rel_batch_ids, iter)

                # update relation memory
                rel_mem = self._build_update_rel(is_training, rel_mem, obj_mem_rel_pool5_nb, rel_cls_score,
                                                 rel_cls_prob, rel_cls_pred, pred_rel_rois, pred_rel_batch_ids,
                                                 pred_inv_rel_rois, pred_inv_rel_batch_ids, iter)

                # transfer the information from relation memory to object memory
                # mem = self._build_update_from_rel(is_training, rel_mem, mem, obj_mem_obj_pool5_nb, inv_rois, inv_batch_ids,
                #                                  pred_rel_rois, pred_rel_batch_ids, "mem_update_from_rel",
                #                                  iter)
                '''
            if iter == 0:
                reuse = True

        return rois, cls_prob, bbox_pred, rel_cls_prob

    def _add_memory_losses(self, name):
        cross_entropy = []
        rel_cross_entropy = []
        bbox_loss = []
        assert len(self._predictions["cls_score"]) == cfg.MEM.ITER
        assert len(self._predictions["rel_cls_score"]) == cfg.MEM.ITER
        assert len(self._predictions["bbox_pred"]) == cfg.MEM.ITER
        with tf.variable_scope(name):
            # load the groundtruth
            label = tf.reshape(self._targets["labels"], [-1])
            predicate = tf.reshape(self._targets["predicates"], [-1])
            relation = self._targets["relations"]
            bbox_targets = self._targets["bbox_targets"]
            bbox_inside_weights = self._targets["bbox_inside_weights"]
            bbox_outside_weights = self._targets["bbox_outside_weights"]

            for iter in range(cfg.MEM.ITER):
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

            ce_rest = tf.stack(cross_entropy[1:], name="cross_entropy_rest")
            bbox_loss_rest = tf.stack(bbox_loss[1:], name="bbox_loss_rest")
            rel_ce_rest = tf.stack(rel_cross_entropy[1:], name="rel_cross_entropy_rest")

            self._losses['cross_entropy_image'] = cross_entropy[0]
            self._losses['cross_entropy_memory'] = tf.reduce_mean(ce_rest, name='cross_entropy')
            self._losses['cross_entropy'] = self._losses['cross_entropy_image'] \
                                            + cfg.MEM.WEIGHT * self._losses['cross_entropy_memory']
            self._losses['bbox_loss_image'] = bbox_loss[0]
            self._losses['bbox_loss_memory'] = tf.reduce_mean(bbox_loss_rest, name='bbox_loss')
            self._losses['bbox_loss'] = self._losses['bbox_loss_image'] \
                                        + cfg.MEM.WEIGHT * self._losses['bbox_loss_memory']

            self._losses['rel_cross_entropy_image'] = rel_cross_entropy[0]
            self._losses['rel_cross_entropy_memory'] = tf.reduce_mean(rel_ce_rest, name='rel_cross_entropy')
            self._losses['rel_cross_entropy'] = self._losses['rel_cross_entropy_image'] \
                                                + cfg.MEM.REL_WEIGHT * self._losses['rel_cross_entropy_memory']

            loss = self._losses['cross_entropy']
            # regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = self._losses['cross_entropy'] + self._losses['bbox_loss'] \
                                         + self._losses['rel_cross_entropy']  # \
            # + cfg.REL_MAP_WEIGHT * self._losses['obj_map_loss'] + \
            # cfg.OBJ_MAP_WEIGHT * self._losses['rel_map_loss']

            self._event_summaries.update(self._losses)

        return loss

    def _create_summary(self):
        """
        Note: The merge_all() function will merge all the tf.summary.xxx.
        So the summary appended after the val_summaries will also included in training summaries.
        """
        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self._add_gt_image_summary())
            for iter in range(cfg.MEM.ITER):
                # val_summaries.append(self._add_pred_memory_summary(iter))
                val_summaries.append(self._add_memory_summary(iter))
                val_summaries.append(self._add_rel_memory_summary(iter))
                for var in self._score_summaries[iter]:
                    self._add_score_iter_summary(iter, var)
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))

            for var in self._act_summaries:
                self._add_zero_summary(var)

        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(val_summaries)

    def create_architecture(self, mode, num_classes, num_predicates, tag=None):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._memory_size = tf.placeholder(tf.int32, shape=[2])
        self._rel_memory_size = tf.placeholder(tf.int32, shape=[2])
        self._rois = tf.placeholder(tf.float32, shape=[None, 5])  # including batch_id and coord
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 4])
        self._labels = tf.placeholder(tf.int32, shape=[None, 1])
        self._gt_labels = tf.placeholder(tf.int32, shape=[None, 1])
        self._rel_rois = tf.placeholder(tf.float32, shape=[None, 5])
        self._isc_rois = tf.placeholder(tf.float32, shape=[None, 5])
        self._relations = tf.placeholder(tf.int32, shape=[None, 2])
        self._predicates = tf.placeholder(tf.int32, shape=[None, 1])
        self._bbox_targets = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
        self._bbox_inside_weights = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
        self._bbox_outside_weights = tf.placeholder(tf.float32, shape=[None, 4 * num_classes])
        self._count_base = tf.ones([1, cfg.MEM.CROP_SIZE, cfg.MEM.CROP_SIZE, 1])
        self._num_roi = tf.placeholder(tf.int32, shape=[])
        self._num_rel = tf.placeholder(tf.int32, shape=[])
        self._edge_mask_inds = tf.placeholder(tf.int32, shape=[None, ])
        self._edge_segment_inds = tf.placeholder(tf.int32, shape=[None, ])
        self._edge_pair_mask_inds = tf.placeholder(tf.int32, shape=[None, 2])
        self._edge_pair_segment_inds = tf.placeholder(tf.int32, shape=[None, ])
        self._tag = tag

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
            rois, cls_prob, bbox_pred, rel_cls_prob = self._build_memory(training, testing)

        layers_to_output = {'rois': rois}

        if not testing:
            self._add_memory_losses("loss")
            layers_to_output.update(self._losses)
            self._create_summary()

        layers_to_output.update(self._predictions)

        return layers_to_output

    def _parse_dict(self, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._rois: blobs['rois'],
                     self._isc_rois: blobs['isc_rois'],
                     self._rel_rois: blobs['rel_rois'],
                     self._gt_boxes: blobs['gt_boxes'], self._gt_labels: blobs['gt_labels'],
                     self._memory_size: blobs['memory_size'], self._rel_memory_size: blobs['memory_size'],
                     self._labels: blobs['labels'], self._relations: blobs['relations'],
                     self._predicates: blobs['predicates'], self._bbox_targets: blobs['bbox_targets'],
                     self._bbox_inside_weights: blobs['bbox_inside_weights'],
                     self._bbox_outside_weights: blobs['bbox_outside_weights'],
                     self._num_roi: blobs['num_roi'], self._num_rel: blobs['num_rel'],
                     self._edge_mask_inds: blobs['rel_mask_inds'],
                     self._edge_segment_inds: blobs['rel_segment_inds'],
                     self._edge_pair_mask_inds: blobs['rel_pair_mask_inds'],
                     self._edge_pair_segment_inds: blobs['rel_pair_segment_inds']
                     }
        return feed_dict

    def _parse_test_dict(self, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._rois: blobs['rois'],
                     self._isc_rois: blobs['rel_rois'],
                     self._rel_rois: blobs['isc_rois'],
                     self._memory_size: blobs['memory_size'], self._rel_memory_size: blobs['memory_size'],
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
        return output

    def train_step(self, sess, blobs, train_op):
        out = sess.run([self._losses["cross_entropy"],
                        self._losses['bbox_loss'],
                        self._losses['rel_cross_entropy'],
                        # self._losses['obj_map_loss'],
                        # self._losses['rel_map_loss'],
                        self._losses['total_loss'],
                        train_op],
                       feed_dict=self._parse_dict(blobs))
        return {'loss_cls': out[0], 'loss_bbox': out[1], 'loss_rel': out[2], 'loss_total': out[3]}

    def train_step_with_summary(self, sess, blobs, train_op, summary_grads):
        out = sess.run(
            [self._losses["cross_entropy"],
             self._losses['bbox_loss'],
             self._losses['rel_cross_entropy'],
             # self._losses['obj_map_loss'],
             # self._losses['rel_map_loss'],
             self._losses['total_loss'],
             self._summary_op,
             summary_grads,
             train_op],
            feed_dict=self._parse_dict(blobs))
        return {'loss_cls': out[0], 'loss_bbox': out[1], 'loss_rel': out[2], 'loss_total': out[3], 'summary': out[4],
                'gsummary': out[5]}

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


class vgg16(Memory_ISGG_relmix, vgg16):
    def __init__(self):
        Memory_ISGG_relmix.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'


class resnetv1(Memory_ISGG_relmix, resnetv1):
    def __init__(self, num_layers=50):
        Memory_ISGG_relmix.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._num_layers = num_layers
        self._scope = 'resnet_v1_%d' % num_layers
        resnetv1._decide_blocks(self)


class mobilenetv1(Memory_ISGG_relmix, mobilenetv1):
    def __init__(self):
        Memory_ISGG_relmix.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
        self._scope = 'MobilenetV1'
