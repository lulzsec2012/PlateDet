import numpy as np
import numpy.random as npr
import os

data_dir = 'train_list'

with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
    lines = f.readlines()


with open(os.path.join(data_dir, "train.txt"), "w") as f:
    nums = len(lines)
    base_num = nums
    lines_keep = npr.choice(len(lines), size=int(base_num),replace=False) #int(base_num*12)

    for i in lines_keep:
        f.write(lines[i])
