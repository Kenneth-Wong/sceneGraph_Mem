from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from model.train_val import filter_roidb, SolverWrapper
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from roi_data_layer.data_runner import DataRunnerMP
from utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import os
import sys
import glob
import time

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class MemorySolverWrapper(SolverWrapper):
    """
      A wrapper class for the training process of spatial memory
    """

    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, self.imdb.num_predicates,
                                                  tag='default')
            # Define the loss
            loss = layers['total_loss']

            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            grad_summaries = []
            for grad, var in gvs:
                if 'SMN' not in var.name and 'GMN' not in var.name and 'ISGG' not in var.name and 'vgg_16' not in var.name:
                    continue
                grad_summaries.append(tf.summary.histogram('TRAIN/' + var.name, var))
                if grad is not None:
                    grad_summaries.append(tf.summary.histogram('GRAD/' + var.name, grad))

            # Double the gradient of the bias if set
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad, var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        if not np.allclose(scale, 1.0):
                            grad = tf.multiply(grad, scale)
                        final_gvs.append((grad, var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)
            self.summary_grads = tf.summary.merge(grad_summaries)

            # We will handle the snapshots ourselves
            self.saver = tf.train.Saver(max_to_keep=100000)
            # Write the train and validation information to tensorboard
            self.writer = tf.summary.FileWriter(self.tbdir, sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return lr, train_op

    def get_data_runner(self, sess, data_layer):

        def data_generator():
            while True:
                yield data_layer.forward()

        def task_generator():
            while True:
                yield data_layer._get_next_minibatch_inds()

        task_func = data_layer._get_next_minibatch
        data_runner = DataRunnerMP(task_func, task_generator, capacity=24)

        return data_runner

    def train_model(self, sess, max_iters):
        # Build data layers for both training and validation set
        self.data_layer = RoIDataLayer(self.imdb, self.roidb, self.bbox_means, self.bbox_stds)
        self.data_layer_val = RoIDataLayer(self.valimdb, self.valroidb, self.bbox_means, self.bbox_stds, random=True)

        data_runner = self.get_data_runner(sess, self.data_layer)
        val_data_runner = self.get_data_runner(sess, self.data_layer_val)

        # Construct the computation graph
        lr, train_op = self.construct_graph(sess)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize_nofix(sess) #self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess,
                                                                                   str(sfiles[-1]),
                                                                                   str(nfiles[-1]))
        data_runner.start_processes(sess, n_processes=3)
        val_data_runner.start_processes(sess, n_processes=3)

        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_iter = iter
        last_summary_time = time.time()
        # Make sure the lists are not empty
        stepsizes.append(max_iters)
        stepsizes.reverse()
        next_stepsize = stepsizes.pop()
        while iter < max_iters + 1:
            # Learning rate
            if iter == next_stepsize + 1:
                # Add snapshot here before reducing the learning rate
                self.snapshot(sess, iter)
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr, rate))
                next_stepsize = stepsizes.pop()

            timer.tic()
            # Get training data, one batch at a time
            #blobs = self.data_layer.forward()
            blobs = data_runner.get_feed_blobs()

            now = time.time()
            if iter == 1 or \
                    (now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL and  \
                                     iter - last_summary_iter > cfg.TRAIN.SUMMARY_ITERS):
                # Compute the graph with summary
                out = self.net.train_step_with_summary(sess, blobs, train_op, self.summary_grads)
                summary, gsummary = out['summary'], out['gsummary']
                #mem, rel_mem = self.net.debug_step(sess, blobs, [self.net._mems[0], self.net._rel_mems[0]])
                #out = self.net.train_step(sess, blobs, train_op)
                self.writer.add_summary(summary, float(iter))
                self.writer.add_summary(gsummary, float(iter + 1))
                # Also check the summary on the validation set
                #blobs_val = self.data_layer_val.forward()
                blobs_val = val_data_runner.get_feed_blobs()
                summary_val = self.net.get_summary(sess, blobs_val)
                self.valwriter.add_summary(summary_val, float(iter))
                last_summary_iter = iter
                last_summary_time = now
            else:
                # Compute the graph without summary
                out = self.net.train_step(sess, blobs, train_op)
            timer.toc()

            # Display training information
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                stats = 'iter: %d / %d, lr: %f' % (iter, max_iters, lr.eval())
                for key in sorted(out):
                    if key.startswith('loss'):
                        stats += ', %s: %.6f' %(key, out[key])
                stats += ', speed: {:.3f}s / iter'.format(timer.average_time)
                print(stats)

            # Snapshotting
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess, iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            iter += 1

        if last_snapshot_iter != iter - 1:
            self.snapshot(sess, iter - 1)

        self.writer.close()
        self.valwriter.close()


def train_net(network, imdb, roidb, valimdb, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
    """Train a Faster R-CNN network with memory."""
    #roidb = filter_roidb(roidb)
    #valroidb = filter_roidb(valroidb)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = MemorySolverWrapper(sess, network, imdb, roidb, valimdb, valroidb,
                                 output_dir, tb_dir,
                                 pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, max_iters)
        print('done solving')
