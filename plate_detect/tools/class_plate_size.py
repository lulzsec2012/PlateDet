#-*- coding:utf-8 -*-

import os
import re
import numpy as np
import cv2

plate_300 = open('./plate_300.txt', 'w')
plate_300_600 = open('./plate_300_600.txt', 'w')
plate_600 = open('./plate_600.txt', 'w')

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

count = 0

with open('plate_list.txt') as f:
    while 1:
        line = f.readline()
        if not line:
            break
        image_path = re.split(' ', line)[0]
        coord_str = re.split(' ', line)[1:]
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

        w = coord[2] - coord[0]
        if w <= 300:
            plate_300.write(line)
        if w > 300 and w < 600:
            plate_300_600.write(line)
        if w >= 600:
            plate_600.write(line)

plate_300.close()
plate_300_600.close()
plate_600.close()
