#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
#from models.run_net import PDetNet
from tensorflow.contrib.model_pruning.PlateDet.plate_detect.models.run_net import PDetNet

from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import os
import re
from tensorflow.contrib.model_pruning.python import pruning

def summaries_gradients_hist(grads):
    # Add histograms for gradients.
    summaries = set()
    for grad, var in grads: 
        if grad is not None:
            summaries.add(tf.summary.histogram(var.op.name + '/gradients', grad))
            summaries.add(tf.summary.histogram(var.op.name, var))
    return summaries

def summaries_gradients_norm(grads):
    summaries = set()
    #gradients norm
    g_norm = []
    for g, v in grads:
        if g is not None:
            print(g.name)
#            tmep_name = g.name.split("/")
#           name = tmep_name[3]+ "/" + tmep_name[4] + "/" + tmep_name[5] + "/" + tmep_name[6] + "/norm"
            gn = tf.norm(g, name=g.name.split(":")[0]+"/norm")
            g_norm.append(gn)
    for gn in g_norm:
        if "BiasAdd_grad" not in gn.name.split("/") and "batchnorm" not in gn.name.split("/"):
            summaries.add(tf.summary.scalar(gn.name, gn))
    return summaries

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
          List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    is_training = True
    # data pipeline
    imgs, true_boxes = gen_data_batch(re.sub(r'examples/', '', cfg.data_path), cfg.batch_size*cfg.train.num_gpus)
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    true_boxes_split = tf.split(true_boxes, cfg.train.num_gpus)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
    lr = tf.train.piecewise_constant(global_step, cfg.train.lr_steps, cfg.train.learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    summaries_buf = []
    summaries=set()
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(cfg.train.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cfg.train.tower, i)) as scope:
                    model = PDetNet(imgs_split[i], true_boxes_split[i], is_training)
                    loss = model.compute_loss()
                    tf.get_variable_scope().reuse_variables()
                    grads_and_vars = optimizer.compute_gradients(loss)
                    #
                    gradients_norm = summaries_gradients_norm(grads_and_vars)
                    gradients_hist = summaries_gradients_hist(grads_and_vars)
                    #summaries_buf.append(gradients_norm)
                    summaries_buf.append(gradients_hist)
                    ##sum_set = set()
                    ##sum_set.add(tf.summary.scalar("loss", loss))
                    ##summaries_buf.append(sum_set)
                    summaries_buf.append({tf.summary.scalar("loss", loss)})
                    #
                    tower_grads.append(grads_and_vars)
                    if i == 0:
                        current_loss = loss
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PDetNet")
    grads = average_gradients(tower_grads)
    with tf.control_dependencies(update_op):
        #train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op,*update_op)

    # GPU config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ##pruning add by lzlu
    # Parse pruning hyperparameters
    pruning_hparams = pruning.get_pruning_hparams().parse(cfg.prune.pruning_hparams)
    
    # Create a pruning object using the pruning hyperparameters
    pruning_obj = pruning.Pruning(pruning_hparams, global_step=global_step)
    
    # Use the pruning_obj to add ops to the training graph to update the masks
    # The conditional_mask_update_op will update the masks only when the
    # training step is in [begin_pruning_step, end_pruning_step] specified in
    # the pruning spec proto
    mask_update_op = pruning_obj.conditional_mask_update_op()
    
    # Use the pruning_obj to add summaries to the graph to track the sparsity
    # of each of the layers
    pruning_summaries = pruning_obj.add_pruning_summaries()
    
    summaries |= pruning_summaries
    for summ in summaries_buf:
        summaries |= summ
        
    summaries.add(tf.summary.scalar('lr', lr))    

    summary_op = tf.summary.merge(list(summaries), name='summary_op')
        
    if cfg.summary.summary_allowed:
        summary_writer = tf.summary.FileWriter(logdir=cfg.summary.logs_path, graph=sess.graph,
                                               flush_secs=cfg.summary.summary_secs)
    
    # Create a saver
    saver = tf.train.Saver()
    ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)

    if cfg.train.fine_tune == 0:
        # init
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, cfg.train.rstd_path)

    # running
    for i in range(0, cfg.train.max_batches):
        _, loss_, gstep, sval, _ = sess.run([train_op, current_loss, global_step, summary_op,mask_update_op])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if i % 1000 == 0 and i < 10000:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)
        if i % 10000 == 0:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)
        if cfg.summary.summary_allowed and gstep % cfg.summary.summ_steps == 0:
            summary_writer.add_summary(sval, global_step=gstep)


if __name__ == '__main__':
    train()
