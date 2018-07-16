#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import PDetNet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import cv2
import os
import re
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

def gen_gt_box(coord_str):
    '''
    convert coord_string_list to coord_float_list
    return coord.shape: (n, 4) --> [xmin, ymin, xmax, ymax]
    '''
    if len(coord_str) == 8:
        coord_x = []
        coord_y = []
        for i in range(4):
            coord_x.append(float(coord_str[i*2]))
            coord_y.append(float(coord_str[i*2+1]))
            coord_x.sort()
            coord_y.sort()
        coord = [coord_x[0], coord_y[0], coord_x[3], coord_y[3]]
    else:
        coord = [float(i) for i in coord_str]
    coord = np.array([coord])
    return coord

def extract_image(image_path, coord):
    '''
    convert image.shape(h, w, 3) to image.shape(1080, 1920, 3)
    if ground truth box is beyond the boundary, regard the image as invalid_image
    '''
    image_1 = cv2.imread(image_path)
    h, w, _ = image_1.shape
    if h == 1080 and w == 1920:
        image = image_1
    elif h < 1080 and w < 1920:
        image = cv2.resize(image_1, (1920, 1080))
        image[0:h, 0:w, :] = image_1
    elif h < 1080 and w > 1920:
        image = cv2.resize(image_1, (1920, 1080))
        if coord[0][0] < (w-1920)/2 or coord[0][2] > (w-1920)/2+192:
            return False
        image[0:h, 0:1920, :] = image_1[0:h, int((w-1920)/2):int((w-1920)/2)+1920, :]
        coord[0][0], coord[0][2] = coord[0][0] - (w - 1920)/2, coord[0][2] - (w - 1920)/2
    elif h > 1080 and w < 1920:
        if coord[0][1] < (h-1080)/2 or coord[0][1] > (h-1080)/2+1080:
            return False
        image = cv2.resize(image_1, (1920, 1080))
        image[0:1080, 0:w, :] = image_1[int((h-1080)/2):int((h-1080)/2)+1080, 0:w, :]
        coord[0][1], coord[0][3] = coord[0][1] - (h-1080)/2, coord[0][3] - (h-1080)/2
    else:
        if coord[0][1] < (h-1080)/2 or coord[0][3] > (h-1080)/2+1080 \
          or coord[0][0] < (w-1920)/2 or coord[0][2] > (w-1920)/2+1920:
            return False
        image = cv2.resize(image_1, (1920, 1080))
        image[0:1080, 0:1920, :] = image_1[int((h-1080)/2):int((h-1080)/2)+1080, int((w-1920)/2):int((w-1920)/2)+1920, :]
        coord[0][0], coord[0][2] = coord[0][0] - (w - 1920)/2, coord[0][2] - (w - 1920)/2
        coord[0][1], coord[0][3] = coord[0][1] - (h-1080)/2, coord[0][3] - (h-1080)/2
    return [image, coord]


def resize_image(image, coord, w_equal_h, need_resize):
    '''
    convert image.shape(1080, 1920, 3) to image.shape(1024, 2048, 3)
    '''
    if w_equal_h:
        image_back = cv2.resize(image, (dim_w, dim_h))
    elif need_resize:
        image_back = cv2.resize(image, (2048, 1024))
        image_back[0:1024, 0:1920, :] = image[1080-1024:1080, 0:1920, :]
        coord[0][1], coord[0][3] = coord[0][1]-(1080-1024), coord[0][3]-(1080-1024)
    else:
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
        image_back = cv2.resize(image, (2048, 1024))
        image_back[0:image.shape[0], 0:image.shape[1], :] = image[0:1024, 0:image.shape[1], :]
    return image_back, coord


def accuracy(test_file):
    dim_w = 2048
    dim_h = 1024
    scale = False
    size = 608
    g_step = 40000

    need_resize = True if dim_w%512==0 and dim_h%512==0 else False
    w_equal_h = True if dim_w==dim_h else False
    is_training = False
    cfg.batch_size = 1
    t = 0.5

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

    images_path = open(test_file, 'r')

    with tf.Session() as sess:
        configer = tf.ConfigProto()
        configer.gpu_options.per_process_gpu_memory_fraction = 0.999
        sess=tf.Session(config=configer)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        print(ckpt.model_checkpoint_path)
        #saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt_dir+str(g_step)+'_plate.ckpt-'+str(g_step+1))

        imgs = images_path.readlines()

        all_images = len(imgs)
        invalid_images = 0
        correct = 0
        wrong = 0
        for i in tqdm(imgs):
            image_path = re.split(' ', i.strip())[0]
            coord_str = re.split(' ', i.strip())[1:]
            coord = gen_gt_box(coord_str)

            image_and_coord = extract_image(image_path, coord)
            if image_and_coord is not False:
                pass
            else:
                invalid_images += 1
                continue

            image, gt_box = resize_image(image_and_coord[0], image_and_coord[1], w_equal_h, need_resize)
            h, w, c = image.shape
            image_data = np.array(image, dtype='float32') / 255.0

            boxes_, scores_, classes_ = sess.run([boxes, scores, classes], feed_dict={img_hw:[h ,w], imgs_holder: np.reshape(image_data, [1, dim_h, dim_w, 3])})

            img = np.floor(image_data * 255 + 0.5).astype('uint8')
            boxes_[:, [0, 1, 2, 3]] = boxes_[:, [1, 0, 3, 2]]
            for i in range(boxes_.shape[0]):
                Iou = IoU(boxes_[i], gt_box)
                if np.max(Iou) > 0.7:
                    correct += 1
                else:
                    wrong += 1

        accuracy = correct / (correct + wrong)
        recall = correct / (all_images - invalid_images)
        print("All images:\n {}".format(all_images))
        print("Invalid images:\n {}".format(invalid_images))
        print("Accuracy: {:.2f}%".format(accuracy))
        print("Recall: {:.2f}%".format(recall))


if __name__ == '__main__':
    test_file = re.sub(r'examples', '', os.getcwd()) + 'data/test_list/test.txt'
    print(test_file)
    accuracy(test_file)
