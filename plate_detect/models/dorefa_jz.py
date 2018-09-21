#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
# modify Author:shenhua Wu <shwu@ingenic.com>
import tensorflow as tf

def get_dorefa_jz():
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def clip_through(x,min,max):
        return tf.clip_by_value(x, min, max)

    def quantize_a(x, k):
        k = k -1
        n = float(2**k)
        with G.gradient_override_map({"Round": "Identity"}):
            x = tf.round(x * n)
            x = clip_through(x,-n,n-1) / n
            return x

    def quantize_google(x,k):
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)
        scale = (2**k - 1)/(x_max - x_min)
        with G.gradient_override_map({"Round": "Identity"}):
            x = tf.round( (x - x_min) * scale)
            x = x / scale + x_min
        return x

    def quantize_w(x, k):
        k = k -1
        n = float(2**k)
        with G.gradient_override_map({"Round": "Identity"}):
            x = tf.round(x * n)
            x = clip_through(x,-n,n-1) / n
        return x

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            x = tf.round(x*n)
            x = clip_through(x,0,n) / n
        return x

    def fw2(x):
        print ('quantized weight every channel ')
        if len(x.shape) == 4:
            re_c = tf.transpose(x, [3, 0, 1, 2])
            c = tf.abs(re_c)
            c = tf.reduce_max(c, 1)
            c = tf.reduce_max(c, 1)
            c = tf.reduce_max(c, 1,keepdims = True)
            c = tf.reshape(c, [x.shape[-1],1,1,1])
            c = tf.stop_gradient(c)
            res = re_c / c
            res = quantize_w(res, 2)
            res = c * res
            res = tf.transpose(res, [1,2,3,0])
        elif len(x.shape) == 2 and True:
            c = tf.abs(x)
            c = tf.reduce_max(c, 0,keepdims = True)
            c = tf.stop_gradient(c)
            res = x / c
            res = quantize_w(res, 2)
            res = c * res
        return res

    def fw4(x):
        print ('quantized weight every channel ')
        if len(x.shape) == 4:
            re_c = tf.transpose(x, [3, 0, 1, 2])
            c = tf.abs(re_c)
            c = tf.reduce_max(c, 1)
            c = tf.reduce_max(c, 1)
            c = tf.reduce_max(c, 1,keepdims = True)
            c = tf.reshape(c, [x.shape[-1],1,1,1])
            c = tf.stop_gradient(c)
            res = re_c / c
            res = quantize_w(res, 4)
            res = c * res
            res = tf.transpose(res, [1,2,3,0])
        elif len(x.shape) == 2 and True:
            c = tf.abs(x)
            c = tf.reduce_max(c, 0,keepdims = True)
            c = tf.stop_gradient(c)
            res = x / c
            res = quantize_w(res, 4)
            res = c * res
        return res

    def fw8(x):
        print ('quantized weight every channel ')
        if len(x.shape) == 4:
            re_c = tf.transpose(x, [3, 0, 1, 2])
            c = tf.abs(re_c)
            c = tf.reduce_max(c, 1)
            c = tf.reduce_max(c, 1)
            c = tf.reduce_max(c, 1,keepdims = True)
            c = tf.reshape(c, [x.shape[-1],1,1,1])
            c = tf.stop_gradient(c)
            res = re_c / c
            res = quantize_w(res, 8)
            res = c * res
            res = tf.transpose(res, [1,2,3,0])
        elif len(x.shape) == 2 and True:
            c = tf.abs(x)
            c = tf.reduce_max(c, 0,keepdims = True)
            c = tf.stop_gradient(c)
            res = x / c
            res = quantize_w(res, 8)
            res = c * res
        return res

    def fw32(x):
        return x
    
    def fa32(x):
        return x
    
    def fa8(x):
        x = tf.clip_by_value(x,0,6)
        x = x / 6
        x = quantize(x,8)
        return x

    def fa4(x):
        x = tf.clip_by_value(x,0,3)
        x = x / 3
        x = quantize(x,4)
        return x
        
    def fa2(x):
        x = tf.clip_by_value(x,0,1) 
        x = quantize(x,2)
        return x

    def fg(x):
        return x
    return fw32, fw8, fw4, fw2, fa32, fa8, fa4, fa2, fg
