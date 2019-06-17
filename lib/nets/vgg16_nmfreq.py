# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from nets.network_freq import Network_freq
from model.config import cfg


class vgg16(Network_freq):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'

    def _image_to_head(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                              trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                              trainable=False, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=False, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=False, scope='conv5')

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def _head_to_tail(self, pool5, is_training, prefix=None, reuse=None, open_scope=True):
        prefix = '' if prefix is None else prefix+'_'
        if open_scope:
            with tf.variable_scope(self._scope, self._scope, reuse=reuse):
                pool5_flat = slim.flatten(pool5, scope=prefix+'flatten')
                fc6 = slim.fully_connected(pool5_flat, 4096, scope=prefix+'fc6')
                if is_training:
                    fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                       scope=prefix+'dropout6')
                fc7 = slim.fully_connected(fc6, 4096, scope=prefix+'fc7')
                if is_training:
                    fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                                       scope=prefix+'dropout7')
        else:
            pool5_flat = slim.flatten(pool5, scope=prefix + 'flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope=prefix + 'fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                   scope=prefix + 'dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope=prefix + 'fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                                   scope=prefix + 'dropout7')
        return fc7

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                            v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def restore_variables_from_npz(self, sess, pretrained_model):
        data_dict = np.load(pretrained_model, encoding="latin1").item()
        for key in data_dict:
            with tf.variable_scope(self._scope, reuse=True):
                with tf.variable_scope(key.split('_')[0], reuse=True):
                    if key.startswith("conv"):
                        with tf.variable_scope(key, reuse=True):
                            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                                var = tf.get_variable(subkey) # vgg_16/conv1/conv1_1/weights
                                sess.run(var.assign(data))
                    else:
                        #continue# for CBAM
                        for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                            var = tf.get_variable(subkey)  # vgg_16/fc6/weights
                            sess.run(var.assign(data))

    def get_variables_to_restore_nofix(self, variables, var_keep_dic):
        variables_to_restore = []
        for v in variables:
            # exclude learning rate
            if v.name =='Variable:0':
                continue
            if v.name.split(':')[0] in var_keep_dic:
                # exclude the prediction layer
                if 'cls_score' not in v.name and 'bbox_pred' not in v.name and \
                                list(v.shape) == var_keep_dic[v.name.split(':')[0]]:
                    print('Variables restored: %s' % v.name)
                    variables_to_restore.append(v)
                else:
                    print("Variables skipped: %s" % v.name)
        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                # fix the vgg16 issue from conv weights to fc weights
                # fix RGB to BGR
                fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
                fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv,
                                              self._scope + "/fc7/weights": fc7_conv,
                                              self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc6/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv,
                                                                                                      self._variables_to_fix[
                                                                                                          self._scope + '/fc7/weights:0'].get_shape())))
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
