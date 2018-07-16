import os
import re
import cv2

source_path = '/home/xjyu/kgduan/Object_Detect/plate_detect/data/plate_list/all_plate_crop_box.txt'

with open(source_path, 'r') as f:
    while 1:
        line = f.readline()
        if not line:
            break
        img_path = re.split(' ', line)[0]
        coord = [int(float(i)) for i in re.split(' ', line.strip())[1:]]
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        cv2.rectangle(img, (coord[0], coord[1]), (coord[2], coord[3]), (0,255,255), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
