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
cfg.data_path = cnt_path + '/data/plate_detect_train.records'
cfg.ckpt_path = cnt_path + '/ckpt'
cfg.ckpt_path_416 = cfg.ckpt_path + '/ckpt_416/'
cfg.ckpt_path_608 = cfg.ckpt_path + '/ckpt_608_full/'
#cfg.ckpt_path_608 = cfg.ckpt_path + '/ckpt_608_pruning_0.5/'

# training options
cfg.train = edict()
#cfg.train.eval_ckpt=cfg.ckpt_path_608+'30000_plate.ckpt-90002'
cfg.train.eval_ckpt=cfg.ckpt_path_608+'60000_plate.ckpt-60001'
cfg.train.ignore_thresh = .5
cfg.train.momentum = 0.9
cfg.train.bn_training = True
cfg.train.weight_decay = 0.0005
#cfg.train.max_batches = 60010
#cfg.train.learning_rate = [1e-3, 1e-4, 1e-5]
#cfg.train.lr_steps = [40000., 50000.]
cfg.train.fine_tune = 1
cfg.train.rstd_path = cfg.ckpt_path + '/ckpt_608_full/60000_plate.ckpt-60001'
cfg.train.max_batches = 40010
cfg.train.learning_rate = [1e-4,1e-5,1e-5]
cfg.train.lr_steps = [80000., 90000.]

cfg.train.lr_scales = [.1, .1]
cfg.train.max_truth = 1
cfg.train.mask = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
cfg.train.image_resized = 608
cfg.train.num_gpus = 4
cfg.train.tower = 'tower'


# quant options
cfg.quant = edict()
cfg.quant.bitwidth = "8,8,32"
cfg.quant.is_quantize = 0

# summary options
cfg.summary = edict()
cfg.summary.summary_allowed = True
cfg.summary.summ_steps = 500
cfg.summary.summary_secs = 600
cfg.summary.logs_path=cfg.ckpt_path_608

# prune options
cfg.prune = edict()
cfg.prune.pruning_hparams='name=plateRec_pruning,begin_pruning_step=60000,end_pruning_step=100000,target_sparsity=0.5,sparsity_function_begin_step=60000,sparsity_function_end_step=100000'
