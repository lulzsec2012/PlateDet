from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import os

cfg = edict()

cfg.classes = 1
cfg.num_anchors = 9
cfg.num_anchors_per_layer = 9
cfg.num = 9
cfg.anchors = np.array([[50,20], [75,35], [100,50], [120,55], [150,60],
                        [170,65], [200,75], [250, 100], [350, 128]])
cfg.names = ['plate']
cfg.batch_size = 32

cnt_path = os.getcwd()
cfg.data_path = '/mllib/dataset/PLATE_DET/data/train_data/plate_detect_train.records'
cfg.ckpt_path = cnt_path + '/ckpt'
cfg.ckpt_path_416 = cfg.ckpt_path + '/ckpt_416/'
cfg.ckpt_path_608 = cfg.ckpt_path + '/ckpt_608/'

# training options
cfg.train = edict()

cfg.train.ignore_thresh = .5
cfg.train.momentum = 0.9
cfg.train.bn_training = True
cfg.train.weight_decay = 0.0005
cfg.train.learning_rate = [1e-3, 1e-4, 1e-5]
cfg.train.max_batches = 60010
cfg.train.lr_steps = [40000., 50000.]
cfg.train.lr_scales = [.1, .1]
cfg.train.max_truth = 1
cfg.train.mask = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
cfg.train.image_resized = 608
cfg.train.num_gpus = 4
cfg.train.tower = 'tower'

