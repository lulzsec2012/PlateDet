#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
#from models.run_net import PDetNet
from tensorflow.contrib.model_pruning.PlateDet.plate_detect.models.run_net import PDetNet
from tensorflow.contrib.model_pruning.python import pruning
from prepare_data.gen_data_batch import parser_test_data
from config import cfg
import cv2
import os
import re
from tqdm import tqdm
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

def accuracy(test_file):
    dim_w = 2048
    dim_h = 1024
    is_training = False
    g_step = 50000
    cfg.batch_size = 1
    size = 608
    t = 0.9
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX:cfg.ckpt_path_608:",cfg.ckpt_path_608)
    if size == 608:
        ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_608)
    else:
        ckpt_dir = re.sub(r'examples/', '', cfg.ckpt_path_416)

    img, true_boxes = parser_test_data(test_file)
    with tf.Session() as sess:
        imgs_holder = tf.placeholder(tf.float32, shape=[1, dim_h, dim_w, 3])
        model = PDetNet(imgs_holder, None, is_training)
        img_hw = tf.placeholder(dtype=tf.float32, shape=[2])
        boxes, scores, classes = model.predict(img_hw, iou_threshold=0.5, score_threshold=t)

        configer = tf.ConfigProto()
        configer.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess=tf.Session(config=configer)
        
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, ckpt_dir+str(g_step)+'_plate.ckpt-'+str(g_step+1))
        saver.restore(sess, cfg.train.eval_ckpt)
        sess.run(tf.local_variables_initializer())
        print(ckpt_dir+str(g_step)+'_plate.ckpt-'+str(g_step+1))

        correct = 0
        wrong = 0
        all_images = 0
        with tf.Graph().as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop():
                    image_, gt_box = sess.run([img, true_boxes])
                    all_images += 1
                    boxes_, scores_, classes_ = sess.run([boxes, scores, classes], feed_dict={img_hw:[dim_h ,dim_w], imgs_holder:image_})
                    boxes_[:, [0, 1, 2, 3]] = boxes_[:, [1, 0, 3, 2]]
                    if all_images % 100 == 0:
                        print("done images: {}".format(all_images))
                    for i in range(boxes_.shape[0]):
                        Iou = IoU(boxes_[i], gt_box)
                        if np.max(Iou) > 0.7:
                            correct += 1
                        else:
                            wrong += 1
            except tf.errors.OutOfRangeError:
                print('done')
                accuracy = float(correct) / float(correct + wrong)
                recall = float(correct) / float(all_images)
                print("All images:\n {}".format(all_images))
                print("Accuracy: {:.4f}%".format(accuracy))
                print("Recall: {:.4f}%".format(recall))
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    test_file = re.sub(r'examples', '', os.getcwd()) + '/data/plate_detect_test.records'
    accuracy(test_file)
