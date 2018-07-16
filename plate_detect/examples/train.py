#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import PDetNet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

cfg.train.image_resized = 608
imgs, true_boxes = gen_data_batch(cfg.data_path, cfg.batch_size)

is_training = True
model = PDetNet(imgs, true_boxes, is_training)

loss = model.compute_loss()
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
lr = tf.train.piecewise_constant(global_step, cfg.train.lr_steps, cfg.train.learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PDetNet")

with tf.control_dependencies(update_op):
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
saver = tf.train.Saver()
ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)

gs = 0
batch_per_epoch = 2000
with tf.Session() as sess:
    configer = tf.ConfigProto()
    configer.gpu_options.per_process_gpu_memory_fraction = 0.999
    sess=tf.Session(config=configer)
    sess.run(tf.global_variables_initializer())
    for i in range(gs, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if i % 1000 == 0 and i < 10000:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)
        if i % 10000 == 0:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)

