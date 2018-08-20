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
        #记住,变量是冗余的.因为它们是共享的GPU,所以我们将返回第一个GPU的指针变量
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    is_training = True
    # data pipeline
    imgs, true_boxes = gen_data_batch(cfg.data_path, cfg.batch_size*cfg.train.num_gpus) 
    # print('imgs=',imgs) ###(n,608,608,3)
    # print('true_boxes=',true_boxes) ###(n,1,5)
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    true_boxes_split = tf.split(true_boxes, cfg.train.num_gpus)
    #如果trainabl=True还将变量添加到图表集合中 GraphKeys.TRAINABLE_VARIABLES（请参阅variables.Variable）
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
    #常数分片学习率衰减,当走到一定步长时更改学习率
    lr = tf.train.piecewise_constant(global_step, cfg.train.lr_steps, cfg.train.learning_rate)
    #创建Adam优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    #tf.variable_scope(): 通过 tf.get_variable()为变量名指定命名空间.
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(cfg.train.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cfg.train.tower, i)) as scope:
                    model = PDetNet(imgs_split[i], true_boxes_split[i], is_training)
                    loss = model.compute_loss()
                    #当前变量作用域可以用tf.get_variable_scope()进行检索并且reuse 标签可以通过调用tf.get_variable_scope().reuse_variables()设置为True.
                    tf.get_variable_scope().reuse_variables()
                    #compute_gradients(loss,var_list=None,gate_gradients=GATE_OP,aggregation_method=None,colocate_gradients_with_ops=False,grad_loss=None)
                    #作用：对于在变量列表（var_list）中的变量计算对于损失函数的梯度,这个函数返回一个（梯度,变量）对的列表,其中梯度就是相对应变量的梯度了.这是minimize()函数的第一个部分,
                    #参数： 
                    # loss: 待减小的值 
                    # var_list: 默认是在GraphKey.TRAINABLE_VARIABLES. 
                    # gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH. 
                    # aggregation_method: Specifies the method used to combine gradient terms. Valid values are defined in the class AggregationMethod. 
                    # colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op. 
                    # grad_loss: Optional. A Tensor holding the gradient computed for loss.
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    if i == 0:
                        current_loss = loss
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        # 标准库使用各种已知的名称来收集和检索与图形相关联的值 例如,如果没有指定,则 tf.Optimizer 子类默认优化收集的变量tf.GraphKeys.TRAINABLE_VARIABLES,但也可以传递显式的变量列表
                        #TRAINABLE_VARIABLES将由优化器训练的变量对象的子集
                        vars_det = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="PDetNet")
        
    grads = average_gradients(tower_grads)
    #返回一个控制依赖的上下文管理器，使用with关键字可以让在这个上下文环境中的操作都在control_inputs 执行
    with tf.control_dependencies(update_op):
        # train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars_det)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        #当这个op结点运行完成，所有作为input的ops都被运行完成
        train_op = tf.group(apply_gradient_op,*update_op)

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
    #tf.global_variables_initializer()能够将所有的变量一步到位的初始化,非常的方便,将会初始化所有在tf.GraphKeys.GLOBAL_VARIABLES 中的变量.
    sess.run(tf.global_variables_initializer())
    # running
    for i in range(0, cfg.train.max_batches):
        _, loss_ = sess.run([train_op, current_loss])
        if(i % 100 == 0):
            print(i,': ', loss_)
        if i % 1000 == 0 and i < 10000:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)
        if i % 10000 == 0:
            saver.save(sess, ckpt_dir+str(i)+'_plate.ckpt', global_step=global_step, write_meta_graph=False)


if __name__ == '__main__':
    train()
