#!/usr/bin/env python
# encoding: utf-8
# File: network_quant.py
# Author: shenhua Wu <shwu@ingenic.com>


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import sys
sys.path.append('..')
import numpy as np
from config import cfg
from dorefa_jz import get_dorefa_jz
from contextlib import contextmanager

fw32, fw8, fw4, fw2 = None, None, None, None
fa32, fa8, fa4, fa2 = None, None, None, None
fw, fa, fw = None, None, None

@contextmanager
def custom_getter_scope(custom_getter):
    scope = tf.get_variable_scope()
    if False:
        with tf.variable_scope(
                scope, custom_getter=custom_getter,
                auxiliary_name_scope=False):
            yield
    if True:
        ns = tf.get_default_graph().get_name_scope()
        with tf.variable_scope(
                scope, custom_getter=custom_getter):
            with tf.name_scope(ns + '/' if ns else ''):
                yield

def remap_variables(fn):
    tf.python_io.tf_record_iterator
    def custom_getter(getter, *args, **kwargs):
        v = getter(*args, **kwargs)
        return fn(v)
    return custom_getter_scope(custom_getter)

def network_arg_scope(
        is_training=True, weight_decay=cfg.train.weight_decay, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=False):
    batch_norm_params = {
        'is_training': is_training, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
        #'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
        'trainable': cfg.train.bn_training,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            # activation_fn=tf.nn.relu,
            activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class Network(object):
    def __init__(self):
        pass

    def inference(self, mode, inputs, scope='PDetNet'):
        is_training = mode
        print(inputs)
        if cfg.train.is_quantize:
            global fw, fa
            global fw32, fw8, fw4, fw2, fa32, fa8, fa4, fa2, fg
            bitwidth = cfg.quant.bitwidth
            bitwidth_ = bitwidth.split(",")
            if fw2 == None:
                fw32, fw8, fw4, fw2, fa32, fa8, fa4, fa2, fg = get_dorefa_jz()
                fw = quant_weight(int(bitwidth_[0]))
                fa = quant_activation(int(bitwidth_[1]))
        #
        def new_get_variable(v):
            name = v.op.name
            if not name.endswith('weights'):
                return v
            else:
                if cfg.train.is_quantize:
                    print(bitwidth,'_',name)
                    if cfg.quant.quant_layers_weight:
                        return quant_layers_weight(v)
                    else:
                        return fw(v)
                else:
                    return v

        assert inputs != None
        with slim.arg_scope(network_arg_scope(is_training=is_training)),\
        remap_variables(new_get_variable):
            with tf.variable_scope(scope, reuse=False):
                conv0 = conv2d(inputs, 32, 2, name='conv_0')
                pool1 = maxpool2x2(conv0, name='pool_1')
                conv2 = conv2d(pool1, 32, 1, name='conv_2')
                conv3 = conv2d(conv2, 32, 1, name='conv_3')
                route4 = route([pool1, conv2, conv3], name='route_4')
                conv5 = conv2d(route4, 32, 2, name='conv_5')
                conv6 = conv2d(conv5, 32, 1, name='conv_6')
                conv7 = conv2d(conv6, 32, 1, name='conv_7')
                route8 = route([conv7, conv5], name='route_8')
                conv9 = conv2d(route8, 32, 1, name='conv_9')
                conv10 = conv2d(conv9, 32, 1, name='conv_10')
                route11 = route([conv10, conv7], name='route_11')
                conv12 = conv2d(route11, 32, 1, name='conv_12')
                conv13 = conv2d(conv12, 32, 1, name='conv_13')
                route14 = route([conv13, conv10, conv5], name='route_14')
                conv15 = conv2d(route14, 32, 1, name='conv_15')
                conv16 = conv2d(conv15, 32, 1, name='conv_16')
                route17 = route([conv5, conv13, conv16], name='route_17')
                conv18 = conv2d(route17, 64, 2, name='conv_18')
                conv19 = conv2d(conv18, 64, 1, name='conv_19')
                conv20 = conv2d(conv19, 32, 1, name='conv_20')
                route21 = route([conv20, conv18], name='route_21')
                conv22 = conv2d(route21, 64, 1, name='conv_22')
                conv23 = conv2d(conv22, 32, 1, name='conv_23')
                route24 = route([conv23, conv20], name='route_24')
                conv25 = conv2d(route24, 64, 1, name='conv_25')
                conv26 = conv2d(conv25, 32, 1, name='conv_26')
                route27 = route([conv26, conv23, conv18], name='route_27')
                conv28 = conv2d(route27, 32, 1, name='conv_28')
                conv29 = conv2d(conv28, 32, 1, name='conv_29')
                route30 = route([conv18, conv26, conv29], name='route_30')
                conv31 = conv2d(route30, 128, 2, name='conv_31')
                conv32 = conv2d(conv31, 128, 1, name='conv_32')
                conv33 = conv2d(conv32, 64, 1, name='conv_33')
                route34 = route([conv33, conv31], name='route_34')
                conv35 = conv2d(route34, 128, 1, name='conv_35')
                conv36 = conv2d(conv35, 64, 1, name='conv_36')
                route37 = route([conv36, conv33], name='route_37')
                conv38 = conv2d(route37, 128, 1, name='conv_38')
                conv39 = conv2d(conv38, 64, 1, name='conv_39')
                route40 = route([conv39, conv36, conv31], name='route_40')
                conv41 = conv2d(route40, 128, 1, name='conv_41')
                conv42 = conv2d(conv41, 64, 1, name='conv_42')
                unpool43 = unpool2x2(conv42, name='unpool_43')
                unpool45 = unpool2x2(conv39, name='unpool_45')
                unpool47 = unpool2x2(conv31, name='unpool_47')
                route48 = route([conv18, conv26, conv29, unpool43, unpool45, unpool47], name='route_48')
                conv49 = conv2d(route48, 128, 1, name='conv_49')
                unpool50 = unpool2x2(conv49, name='unpool_50')
                route51 = route([conv5, conv13, conv16, unpool50], name='route_51')
                conv52 = conv2d(route51, 64, 1, name='conv_52')
                unpool53 = unpool2x2(conv52, name='unpool_53')
                route54 = route([conv2, conv3, unpool53], name='route_54')
                conv55 = conv2d(route54, (cfg.classes+5)*cfg.num_anchors, 1, name='conv_55')
                conv56 = slim.conv2d(conv55, num_outputs=(cfg.classes+5)*cfg.num_anchors, kernel_size=[3,3], stride=1, scope='conv_56', activation_fn=None, normalizer_fn=None)
                print('conv_56', conv56.get_shape())
                if is_training:
                    #l2_loss = tf.add_n(slim.losses.get_regularization_losses())
                    l2_loss = tf.add_n(tf.losses.get_regularization_losses())
                    return conv56, l2_loss
                else:
                    return conv56

def conv2d(inputs, c_outputs, s, name):
    num_filters = c_outputs
    kernel_size = [3,3]
    strides = s
    # weights_initializer = tf.constant_initializer(__weights_dict[name]['weights'])
    output = slim.convolution2d(inputs,
                            # weights_initializer = weights_initializer,
                            num_outputs = num_filters,
                            kernel_size=kernel_size,
                            stride=strides,
                            scope=name)
    if cfg.train.is_quantize:
        if cfg.quant.quant_layers_activation:
            output = quant_layers_activation(name,output)
        else:
            output = fa(output)
        output = fg(output)
    print('===============================')
    print(name, output.get_shape())
    return output

def route(input_list, name):
    with tf.name_scope(name):
        output = tf.concat(input_list, 3, name='concat')
    print(name, output.get_shape())
    return output

def maxpool2x2(input, name):
    output = slim.max_pool2d(input, kernel_size=[2, 2], stride=2, scope=name)
    print(name, output.get_shape())
    return output

def unpool2x2(input, name):
    with tf.name_scope(name):
        out = tf.concat([input, tf.zeros_like(input)], 3, name='concat_1')
        output = tf.concat([out, tf.zeros_like(out)], 2, name='concat_2')
        n, h, w, c = input.get_shape()[0], input.get_shape()[1], input.get_shape()[2], input.get_shape()[3]
        res = tf.reshape(output, (-1, h*2, w*2, c))
        print(name, res.get_shape())
    return res

def quant_layers_weight(v):
    quant_layers_weight_2bit = cfg.quant.quant_layers_weight_2bit.split(",")
    quant_layers_weight_4bit = cfg.quant.quant_layers_weight_4bit.split(",")
    quant_layers_weight_8bit = cfg.quant.quant_layers_weight_8bit.split(",")
    conv = v.op.name.split("/")[-2]
    if   conv in quant_layers_weight_2bit:
        return fw2(v)
    elif conv in quant_layers_weight_4bit:
        return fw4(v)
    elif conv in quant_layers_weight_8bit:
        return fw8(v)
    else:
        return fw(v)

def quant_layers_activation(name,output):
    quant_layers_activation_2bit = cfg.quant.quant_layers_activation_2bit.split(",")
    quant_layers_activation_4bit = cfg.quant.quant_layers_activation_4bit.split(",")
    quant_layers_activation_8bit = cfg.quant.quant_layers_activation_8bit.split(",")
    if   name in quant_layers_activation_2bit:
        return fa2(output)
    elif name in quant_layers_activation_4bit:
        return fa4(output)
    elif name in quant_layers_activation_8bit:
        return fa8(output)
    else:
        return fa(output)

def quant_weight(bitW):
    if   bitW == 8:
        fw = fw8
    elif bitW == 4:
        fw = fw4
    elif bitW == 2:
        fw = fw2
    else:
        fw = fw32
    return fw

def quant_activation(bitA):
    if   bitA == 8:
        fa = fa8
    elif bitA == 4:
        fa = fa4
    elif bitA == 2:
        fa = fa2
    else:
        fa = fa32
    return fa
