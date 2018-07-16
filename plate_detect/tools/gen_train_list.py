import os
import re

img_txt = 'train.txt.backup'

target_txt = 'train.txt'
tf_file = open(target_txt, 'w')
with open(img_txt, 'r') as f:
    while 1:
        line = f.readline()
        if not line:
            break
        plate_name = re.findall(r'\/.*\/(.*)\.jpg', line.strip())[0]
        print(plate_name)
        label_txt = line.strip()[:-4] + '.txt'
        label_file = open(label_txt, 'r')
        label = label_file.readline().strip()
        plate_and_label = line.strip() + ' ' + label
        label_file.close()
        tf_file.write(plate_and_label + '\n')

tf_file.close()
