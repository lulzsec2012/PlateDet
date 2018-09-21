#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
sys.path.append('../models')
from run_net import PDetNet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import os
import re



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

        # Keep in mind that the Variables are redundant becausethey are shared
        # across towers. So .. we will just return the first tower's pointer to the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    is_training = True
    # data pipeline
    imgs, true_boxes = gen_data_batch(cfg.data_path, cfg.batch_size*cfg.train.num_gpus) 
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    true_boxes_split = tf.split(true_boxes, cfg.train.num_gpus)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
    lr = tf.train.piecewise_constant(global_step, cfg.train.lr_steps, cfg.train.learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    losses = []
    summaries = set()
    summaries_buf = []
    
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(cfg.train.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cfg.train.tower, i)) as scope:
                    model = PDetNet(imgs_split[i], true_boxes_split[i], is_training)
                    loss, loss_summ = model.compute_loss()
                    tf.get_variable_scope().reuse_variables()
                    grads = optimizer.compute_gradients(loss)
                    gradients_summ = summaries_gradients_norm(grads)
                    gradients_hist = summaries_gradients_hist(grads)
                    
                    tower_grads.append(grads)
                    losses.append(loss)
                    summaries_buf.append(loss_summ)
                    summaries_buf.append(gradients_summ)
                    summaries_buf.append(gradients_hist)

                    if i == 0:
                        current_loss = loss
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PDetNet")
        
    grads = average_gradients(tower_grads)
    with tf.control_dependencies(update_op):
        # train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op,update_op)

    for summ in summaries_buf:
        summaries |= summ

    summaries.add(tf.summary.scalar('lr', lr))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

        
    # GPU config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create a saver
    saver = tf.train.Saver()
    ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # init
    if cfg.train.pretrained:
        restore_ckpt = cfg.train.restore_ckpt_path
        print('restore_ckpt=',restore_ckpt)
        saver.restore(sess, restore_ckpt)
    else:
        sess.run(tf.global_variables_initializer())

    if True:
        summary_writer = tf.summary.FileWriter(logdir=ckpt_dir, graph=sess.graph)
    # running
    for i in range(0, cfg.train.max_batches):
        _, loss_ , gstep, summary_var = sess.run([train_op, current_loss, global_step, summary_op])
        if(i % 100 == 0):
            print(i,': ', loss_)
            summary_writer.add_summary(summary_var, global_step=gstep)
        if i % 1000 == 0 and i < 10000:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)
        if i % 10000 == 0:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)


if __name__ == '__main__':
    train()
