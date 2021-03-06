#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from config import cfg

def parser(example):
    feats = tf.parse_single_example(example, features={'label_and_class' : tf.FixedLenFeature([5], tf.float32),
                                                       'feature': tf.FixedLenFeature([], tf.string)})
    coord = feats['label_and_class']
    coord = tf.reshape(coord, [1, 5])

    img = tf.decode_raw(feats['feature'], tf.uint8)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [608, 608, 3])
    #img = tf.image.resize_images(img, [cfg.train.image_resized, cfg.train.image_resized])
    #temp = np.random.randint(320, 608)
    #img = tf.image.resize_images(img, [temp, temp])
    #print('image_resized', cfg.train.image_resized)

    img = tf.image.random_hue(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3) # 0.8~1.2
    img = tf.image.random_brightness(img, max_delta=0.16) # 0.1
    img = tf.image.random_saturation(img, lower=0.7, upper=1.3) # 0.8~1.2
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)
    return img, coord

def gen_data_batch(tf_records_filename, batch_size):
    dt = tf.data.TFRecordDataset(tf_records_filename)
    dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.prefetch(batch_size)
    dt = dt.shuffle(buffer_size=5*batch_size)
    dt = dt.repeat()
    dt = dt.batch(batch_size)
    iterator = dt.make_one_shot_iterator()
    imgs, true_boxes = iterator.get_next()

    return imgs, true_boxes

def parser_test_data(tf_records_filename):
    '''
    load image and label from tf records
    '''
    input_queue = tf.train.string_input_producer([tf_records_filename], num_epochs=1, shuffle=False)
    reader = tf.TFRecordReader()
    key, value = reader.read(input_queue)

    feats = tf.parse_single_example(value, features={'label' : tf.FixedLenFeature([4], tf.float32),
                                                     'feature': tf.FixedLenFeature([], tf.string)})
    coord = feats['label']
    coord = tf.reshape(coord, [1, 4])

    img = tf.decode_raw(feats['feature'], tf.uint8)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [1, 1024, 2048, 3])

    return img, coord

if __name__ == '__main__':
    tf_records_filename = cfg.data_path

    imgs, true_boxes = gen_data_batch(tf_records_filename, cfg.batch_size)
    imgs_split = tf.split(imgs, cfg.train.num_gpus)
    true_boxes_split = tf.split(true_boxes, cfg.train.num_gpus)
    print(imgs, true_boxes)
    sess = tf.Session()
    for i in range(2):
        for j in range(4):
            imgs_, true_boxes_ = sess.run([imgs_split[j], true_boxes_split[j]])
            print(imgs_.shape)
            print(true_boxes_.shape)

    cfg.train.image_resized=384
    for i in range(10):
        for j in range(4):
            imgs_, true_boxes_ = sess.run([imgs_split[j], true_boxes_split[j]])
            print(imgs_.shape)
            print(true_boxes_.shape)
