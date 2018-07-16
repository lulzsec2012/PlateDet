# encoding : utf-8

import os
import re
import numpy as np
import cv2

def convert(size, box):
    '''
    convert (xmin, ymin, xmax, ymax) to (cx/w, cy/h, bw/w, bw/h)
    param:
        size: tuple (im_width, im_height)
        box: list [xmin, ymin, xmax, ymax]
    return:
        tuple (cx/w, cy/h, bw/w, bw/h)
    '''
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

def crop_img(filename, coord_target_file, box_target_file, target_path):
    '''
    crop original plate img
    img.shape -> (608, 608, 3)
    '''
    count = 0
    coord_tf = open(coord_target_file, 'w')
    box_tf = open(box_target_file, 'w')
    with open(filename, 'r') as f:
        while 1:
            line = f.readline()
            print(line)
            if not line:
                break
            image_path = re.split(' ', line)[0]
            # generate coord: [xmin, ymin, xmax, ymax]
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

            img = cv2.imread(image_path)
            imh, imw, imc = img.shape
            if imh <= 608 or imw <= 608:
                continue

            plate_height, plate_width = coord[3] - coord[1], coord[2] - coord[0]
            if plate_width > 1000:
                continue
            if plate_width*2 > imw or plate_height*2 > imh:
                continue
            crop_imh, crop_imw = 608, 608
            if plate_width < 600 and plate_height < 600:
                bad = True
                while bad:
                    crop_x_left_offset = np.random.randint(0, int(crop_imw - plate_width))
                    crop_y_top_offset = np.random.randint(0, int(crop_imh - plate_height))

                    crop_im_x_left, crop_im_y_top = coord[0] - crop_x_left_offset, coord[1] - crop_y_top_offset
                    crop_im_x_right, crop_im_y_bottom = crop_im_x_left + crop_imw, crop_im_y_top + crop_imh
                    if crop_im_x_left < 0 or crop_im_y_top < 0 or crop_im_x_right > imw or crop_im_y_bottom > imh:
                        pass
                    else:
                        bad = False
                plate_x_left, plate_x_right = crop_x_left_offset, crop_x_left_offset + plate_width
                plate_y_top, plate_y_bottom = crop_y_top_offset, crop_y_top_offset + plate_height
                print(int(crop_im_y_top), int(crop_im_y_bottom), int(crop_im_x_left), int(crop_im_x_right))
                crop_img = img[int(crop_im_y_top):int(crop_im_y_bottom), int(crop_im_x_left):int(crop_im_x_right), :]
                height, width, channel = crop_img.shape
                #cv2.circle(crop_img, (int(plate_x_left), int(plate_y_top)), 3, (0,0,255))
                #cv2.circle(crop_img, (int(plate_x_right), int(plate_y_bottom)), 3, (0,0,255))
                #cv2.imshow('res', crop_img)
                #cv2.waitKey(0)
                b = [plate_x_left, plate_x_right, plate_y_top, plate_y_bottom]
                bb = convert((width, height), b)
                #txt_name = 'plate_608' + str(count) + '_0705.txt'
                img_name = 'plate_less_than_600_' + str(count).zfill(7) + '_0705.jpg'
                target_img_path = os.path.join(target_path, img_name)
                cv2.imwrite(target_img_path, crop_img)
                coord_tf.write(target_img_path + ' ' + str(0) + ' ' + ' '.join([str(a)[:10] for a in bb]) + '\n')
                box_tf.write(target_img_path + ' ' + ' '.join([str(a) for a in b]) + '\n')
                count += 1
            else:
                crop_imw, crop_imh = int(plate_width*1.5), int(plate_width*1.5)
                bad = True
                while bad:
                    crop_x_left_offset = np.random.randint(0, int(crop_imw - plate_width))
                    crop_y_top_offset = np.random.randint(0, int(crop_imh - plate_height))

                    crop_im_x_left, crop_im_y_top = coord[0] - crop_x_left_offset, coord[1] - crop_y_top_offset
                    crop_im_x_right, crop_im_y_bottom = crop_im_x_left + crop_imw, crop_im_y_top + crop_imh
                    if crop_im_x_left < 0 or crop_im_y_top < 0 or crop_im_x_right > imw or crop_im_y_bottom > imh:
                        pass
                    else:
                        bad = False
                plate_x_left, plate_x_right = crop_x_left_offset, crop_x_left_offset + plate_width
                plate_y_top, plate_y_bottom = crop_y_top_offset, crop_y_top_offset + plate_height
                print(int(crop_im_y_top), int(crop_im_y_bottom), int(crop_im_x_left), int(crop_im_x_right))
                crop_img = img[int(crop_im_y_top):int(crop_im_y_bottom), int(crop_im_x_left):int(crop_im_x_right), :]
                height, width, channel = crop_img.shape
                #cv2.circle(crop_img, (int(plate_x_left), int(plate_y_top)), 3, (0,0,255))
                #cv2.circle(crop_img, (int(plate_x_right), int(plate_y_bottom)), 3, (0,0,255))
                #cv2.imshow('res', crop_img)
                #cv2.waitKey(0)
                b = [plate_x_left, plate_x_right, plate_y_top, plate_y_bottom]
                bb = convert((width, height), b)
                b = [b[0]*608/width, b[1]*608/height, b[2]*608/width, b[3]*608/height]
                #txt_name = 'plate_608' + str(count) + '_0705.txt'
                img_name = 'plate_greater_than_600_' + str(count).zfill(7) + '_0705.jpg'
                crop_img = cv2.resize(crop_img, (608, 608), interpolation=cv2.INTER_LINEAR)
                target_img_path = os.path.join(target_path, img_name)
                cv2.imwrite(target_img_path, crop_img)
                coord_tf.write(target_img_path + ' ' + str(0) + ' ' + ' '.join([str(a)[:10] for a in bb]) + '\n')
                box_tf.write(target_img_path + ' ' + ' '.join([str(a) for a in b]) + '\n')
                count += 1

def crop_resize_img(filename, coord_target_file, box_target_file, target_path):
    count = 0
    coord_tf = open(coord_target_file, 'w')
    box_tf = open(box_target_file, 'w')
    with open(filename, 'r') as f:
        while 1:
            line = f.readline()
            print(line)
            if not line:
                break
            image_path = re.split(' ', line)[0]
            # generate coord: [xmin, ymin, xmax, ymax]
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

            img = cv2.imread(image_path)
            imh, imw, imc = img.shape
            if imh <= 608 or imw <= 608:
                continue

            plate_height, plate_width = coord[3] - coord[1], coord[2] - coord[0]
            if plate_width > 1000:
                continue
            if plate_width*3 > imw or plate_height*3 > imh:
                continue

            scale = float(str(np.random.uniform(0.599, 0.801))[:4])
            crop_imh, crop_imw = int(608/scale), int(608/scale)
            if crop_imh > imh or crop_imw > imw:
                continue
            if plate_width < 700 and plate_height < 700:
                bad = True
                while bad:
                    crop_x_left_offset = np.random.randint(0, int(crop_imw - plate_width))
                    crop_y_top_offset = np.random.randint(0, int(crop_imh - plate_height))

                    crop_im_x_left, crop_im_y_top = coord[0] - crop_x_left_offset, coord[1] - crop_y_top_offset
                    crop_im_x_right, crop_im_y_bottom = crop_im_x_left + crop_imw, crop_im_y_top + crop_imh
                    if crop_im_x_left < 0 or crop_im_y_top < 0 or crop_im_x_right > imw or crop_im_y_bottom > imh:
                        pass
                    else:
                        bad = False
                plate_x_left, plate_x_right = crop_x_left_offset, crop_x_left_offset + plate_width
                plate_y_top, plate_y_bottom = crop_y_top_offset, crop_y_top_offset + plate_height
                print(int(crop_im_y_top), int(crop_im_y_bottom), int(crop_im_x_left), int(crop_im_x_right))
                crop_img = img[int(crop_im_y_top):int(crop_im_y_bottom), int(crop_im_x_left):int(crop_im_x_right), :]
                height, width, channel = crop_img.shape
                #cv2.circle(crop_img, (int(plate_x_left), int(plate_y_top)), 3, (0,0,255))
                #cv2.circle(crop_img, (int(plate_x_right), int(plate_y_bottom)), 3, (0,0,255))
                #cv2.imshow('res', crop_img)
                #cv2.waitKey(0)
                b = [plate_x_left, plate_x_right, plate_y_top, plate_y_bottom]
                bb = convert((width, height), b)
                #txt_name = 'plate_608' + str(count) + '_0705.txt'
                img_name = 'resize_plate_less_than_600_' + str(count).zfill(7) + '_0705.jpg'
                target_img_path = os.path.join(target_path, img_name)
                crop_img = cv2.resize(crop_img, (608, 608), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(target_img_path, crop_img)
                coord_tf.write(target_img_path + ' ' + str(0) + ' ' + ' '.join([str(a)[:10] for a in bb]) + '\n')
                box_tf.write(target_img_path + ' ' + ' '.join([str(a) for a in b]) + '\n')
                count += 1
            else:
                crop_imw, crop_imh = int(plate_width*1.5), int(plate_width*1.5)
                if crop_imh > imh or crop_imw > imw:
                    continue
                bad = True
                while bad:
                    crop_x_left_offset = np.random.randint(0, int(crop_imw - plate_width))
                    crop_y_top_offset = np.random.randint(0, int(crop_imh - plate_height))

                    crop_im_x_left, crop_im_y_top = coord[0] - crop_x_left_offset, coord[1] - crop_y_top_offset
                    crop_im_x_right, crop_im_y_bottom = crop_im_x_left + crop_imw, crop_im_y_top + crop_imh
                    if crop_im_x_left < 0 or crop_im_y_top < 0 or crop_im_x_right > imw or crop_im_y_bottom > imh:
                        pass
                    else:
                        bad = False
                plate_x_left, plate_x_right = crop_x_left_offset, crop_x_left_offset + plate_width
                plate_y_top, plate_y_bottom = crop_y_top_offset, crop_y_top_offset + plate_height
                print(int(crop_im_y_top), int(crop_im_y_bottom), int(crop_im_x_left), int(crop_im_x_right))
                crop_img = img[int(crop_im_y_top):int(crop_im_y_bottom), int(crop_im_x_left):int(crop_im_x_right), :]
                height, width, channel = crop_img.shape
                #cv2.circle(crop_img, (int(plate_x_left), int(plate_y_top)), 3, (0,0,255))
                #cv2.circle(crop_img, (int(plate_x_right), int(plate_y_bottom)), 3, (0,0,255))
                #cv2.imshow('res', crop_img)
                #cv2.waitKey(0)
                b = [plate_x_left, plate_x_right, plate_y_top, plate_y_bottom]
                bb = convert((width, height), b)
                b = [b[0]*608/width, b[1]*608/height, b[2]*608/width, b[3]*608/height]
                #txt_name = 'plate_608' + str(count) + '_0705.txt'
                img_name = 'plate_greater_than_600_' + str(count).zfill(7) + '_0705.jpg'
                crop_img = cv2.resize(crop_img, (608, 608), interpolation=cv2.INTER_LINEAR)
                target_img_path = os.path.join(target_path, img_name)
                cv2.imwrite(target_img_path, crop_img)
                coord_tf.write(target_img_path + ' ' + str(0) + ' ' + ' '.join([str(a)[:10] for a in bb]) + '\n')
                box_tf.write(target_img_path + ' ' + ' '.join([str(a) for a in b]) + '\n')
                count += 1

if __name__ == '__main__':
    filename = 'plate_list.txt'
    coord_target_file = 'resize_plate_crop_coord.txt'
    box_target_file = 'resize_plate_crop_box.txt'
    target_path = '/resize_plate_crop'
    crop_resize_img(filename, coord_target_file, box_target_file, target_path)
