import os
import cv2

images_file = '/plate_crop_box.txt'
with open(images_file) as f:
    images = f.readlines()

print(len(images))

for i in images:
    img = cv2.imread(i.split(' ')[0])
    height, width, channel = img.shape
    print(i)
    cv2.imshow('img', img)
    cv2.waitKey(0)

