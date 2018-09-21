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

cfg.train.pretrained = True
cfg.train.restore_ckpt_path = '{CKPT_PATH}/60000_plate.ckpt-230004'
cfg.train.is_quantize = True 

cfg.quant = edict()

cfg.quant.bitwidth = '4,2,32'

cfg.quant.quant_layers_weight = True
cfg.quant.quant_layers_weight_2bit = ''
cfg.quant.quant_layers_weight_4bit = 'conv_0,conv_2,conv_3'
cfg.quant.quant_layers_weight_8bit = ''

cfg.quant.quant_layers_activation = True
cfg.quant.quant_layers_activation_2bit = 'conv_38,conv_39,conv_41'
cfg.quant.quant_layers_activation_4bit = ''
cfg.quant.quant_layers_activation_8bit = ''

# quant_layers_weight_optional = 'conv_0,conv_2,conv_3,conv_5,conv_6,conv_7,conv_9,conv_10,conv_12,conv_13,conv_15,conv_16,conv_18,conv_19,conv_20,conv_22,conv_23,conv_25,conv_26,conv_28,conv_29,conv_31,conv_32,conv_33,conv_35,conv_36,conv_38,conv_39,conv_41,conv_42,conv_49,conv_52,conv_55,conv_56'

# quant_layers_activation_optional = 'conv_0,conv_2,conv_3,conv_5,conv_6,conv_7,conv_9,conv_10,conv_12,conv_13,conv_15,conv_16,conv_18,conv_19,conv_20,conv_22,conv_23,conv_25,conv_26,conv_28,conv_29,conv_31,conv_32,conv_33,conv_35,conv_36,conv_38,conv_39,conv_41,conv_42,conv_49,conv_52,conv_55'
