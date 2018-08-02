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
import cv2
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dim_w = 2048
dim_h = 1024
scale = False
size = 608
g_step = 50000

need_resize = True if dim_w%512==0 and dim_h%512==0 else False
w_equal_h = True if dim_w==dim_h else False
is_training = False
cfg.batch_size = 1
t = 0.95

if scale == True:
    need_resize = False
    w_equal_h = False

imgs_holder = tf.placeholder(tf.float32, shape=[1, dim_h, dim_w, 3])
model = PDetNet(imgs_holder, None, is_training)
img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
boxes, scores, classes = model.predict(img_hw, iou_threshold=0.5, score_threshold=t)

saver = tf.train.Saver()
if size == 608:
    ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)
else:
    ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_416)

image_path = '/mllib/dataset/PLATE_DET/data/test_data/'

with tf.Session() as sess:
    configer = tf.ConfigProto()
    # configer.gpu_options.per_process_gpu_memory_fraction = 0.999
    configer.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess=tf.Session(config=configer)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    # print(ckpt.model_checkpoint_path)
    #saver.restore(sess, ckpt.model_checkpoint_path)
    print(ckpt_dir+str(g_step)+'_plate.ckpt-'+str(g_step+1))
    saver.restore(sess, ckpt_dir+str(g_step)+'_plate.ckpt-'+str(g_step+1))
    
    imgs = os.listdir(image_path)
    for i in imgs:
        if 'jpg' not in i:
            if 'png' not in i:
                continue
        image_1 = cv2.imread(os.path.join(image_path, i))
        image = cv2.imread(os.path.join(image_path, i))
        image = cv2.resize(image, (1920, 1080))
        image[0:image_1.shape[0], 0:image_1.shape[1], :] = image_1

        if w_equal_h:
            image = cv2.resize(image, (dim_w, dim_h))
        elif need_resize:
            image_back = cv2.imread(os.path.join(image_path, i))
            image_back = cv2.resize(image_back, (2048, 1024))
            image_back[0:1024, 0:1920, :] = image[0:1024, 0:1920, :]
            image = cv2.resize(image_back, (dim_w, dim_h))
            print(image.shape)
        else:
            image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            image_back = cv2.imread(os.path.join(image_path, i))
            image_back = cv2.resize(image_back, (2048, 1024))
            image_back[0:image.shape[0], 0:image.shape[1], :] = image[0:1024, 0:image.shape[1], :]
            image = cv2.resize(image_back, (dim_w, dim_h))
        h, w, c = image.shape
        image_data = np.array(image, dtype='float32') / 255.0

        boxes_, scores_, classes_ = sess.run([boxes, scores, classes], feed_dict={img_hw:[h ,w], imgs_holder: np.reshape(image_data, [1, dim_h, dim_w, 3])})
        print(boxes_)
        print(scores_)
        print(classes_)

        img = np.floor(image_data * 255 + 0.5).astype('uint8')
        for i in range(boxes_.shape[0]):
            box = boxes_[i]
            y_top, x_left, y_bottom, x_right = box
            cv2.rectangle(img, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (0,255,255), 1)

        cv2.imshow('res', img)
        cv2.waitKey()
