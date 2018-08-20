#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import sys
sys.path.append('..')
from config import cfg

def parser(example):
    # 解析读入的一个records文件
    #解析tfrecord文件的每条记录,使用tf.parse_single_example来解析：
    #调用接口解析一行样本
    feats = tf.parse_single_example(example, features={'label_and_class' : tf.FixedLenFeature([5], tf.float32),
                                                       'feature': tf.FixedLenFeature([], tf.string)})
    coord = feats['label_and_class']
    coord = tf.reshape(coord, [1, 5])
    # 将字符串解析成图像对应的像素组
    img = tf.decode_raw(feats['feature'], tf.uint8)
    # 将img转化成float32
    #这里对图像数据做归一化,是关键,没有这句话,精度不收敛,为0.1左右，
    # 有了这里的归一化处理,精度与原始数据一致
    img = tf.cast(img, tf.float32) / 255.0
    #这里将图片还原成原来的维度
    img = tf.reshape(img, [608, 608, 3])
    #img = tf.image.resize_images(img, [cfg.train.image_resized, cfg.train.image_resized])
    #temp = np.random.randint(320, 608)
    #img = tf.image.resize_images(img, [temp, temp])
    #print('image_resized', cfg.train.image_resized)
    #在[-max_delta, max_delta]的范围随机调整图片的色相,max_delta的取值在[0, 0.5]之间
    img = tf.image.random_hue(img, max_delta=0.1)
    #在[lower, upper]的范围随机调整图的对比度
    img = tf.image.random_contrast(img, lower=0.7, upper=1.3) # 0.8~1.2
    #在[-max_delta, max_delta)的范围随机调整图片的亮度
    img = tf.image.random_brightness(img, max_delta=0.16) # 0.1
    #在[lower, upper]的范围随机调整图的饱和度
    img = tf.image.random_saturation(img, lower=0.7, upper=1.3) # 0.8~1.2
    img = tf.minimum(img, 1.0)
    img = tf.maximum(img, 0.0)
    return img, coord

def gen_data_batch(tf_records_filename, batch_size):
    #从tfrecord文件创建TFRecordDataset：
    dt = tf.data.TFRecordDataset(tf_records_filename)
    #map方法可以接受任意函数以对dataset中的数据进行处理;另外,可使用repeat、shuffle、batch方法对dataset进行重复、混洗、分批；用repeat复制dataset以进行多个epoch
    #num_parallel_calls参数加速
    dt = dt.map(parser, num_parallel_calls=4)
    dt = dt.prefetch(batch_size)
    #shuffle的功能为打乱dataset中的元素,它有一个参数buffersize,表示打乱时使用的buffer的大小
    dt = dt.shuffle(buffer_size=5*batch_size)
    #repeat的功能就是将整个序列重复多次,主要用来处理机器学习中的epoch,假设原先的数据是一个epoch,使用repeat(2)就可以将之变成2个epoch
    #注意,如果直接调用repeat()的话,生成的序列就会无限重复下去,没有结束,因此也不会抛出tf.errors.OutOfRangeError异常
    dt = dt.repeat()
    #batch就是将多个元素组合成batch,按照输入元素第一个维度
    dt = dt.batch(batch_size)
    #从dataset中实例化了一个Iterator,这个Iterator是一个“one shot iterator”，即只能从头到尾读取一次
    iterator = dt.make_one_shot_iterator()
    #表示从iterator里取出一个batch元素,one_batch_element只是一个Tensor,并不是一个实际的值
    #调用sess.run(one_batch_element)后,才能真正地取出一个值
    imgs, true_boxes = iterator.get_next()

    return imgs, true_boxes

def parser_test_data(tf_records_filename):
    '''
    load image and label from tf records
    '''
    #根据文件名生成一个队列
    input_queue = tf.train.string_input_producer([tf_records_filename], num_epochs=1, shuffle=False)
    #create a reader from file queue
    reader = tf.TFRecordReader()
    ##返回文件名和文件
    key, value = reader.read(input_queue)
    #get feats from serialized example
    feats = tf.parse_single_example(value, features={'label' : tf.FixedLenFeature([4], tf.float32),
                                                     'feature': tf.FixedLenFeature([], tf.string)})
    coord = feats['label']
    coord = tf.reshape(coord, [1, 4])
    # 将字符串解析成图像对应的像素组
    img = tf.decode_raw(feats['feature'], tf.uint8)
    # print('img=',img) ###(1, 1024, 2048, 3)
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
