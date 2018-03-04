#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
To do data augmentation based on rotation. 
@author:cuirundong 
@file: prepocessing.py 
@time: 2017/06/02

"""
from __future__ import division, print_function, absolute_import

import os
import os.path as osp
import cv2
import random

def rotate(img, angle):
    height = img.shape[0]
    width = img.shape[1]

    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rot_img_shape = (width, height)
    if angle % 180 == 0:
        rot_img_shape = (width, height)
    elif angle % 90 == 0:
        rot_img_shape = (height, width)
    else:
        print('Angle is not supported!')
        exit(1)
    rotateImg = cv2.warpAffine(img, rotateMat, rot_img_shape)
    return rotateImg

if __name__ == '__main__':
    ori_dir = '/home/xiaoran/qxr/dataset/Cervical-Cancer-Screening/train'
    rotate_dir = '/home/xiaoran/qxr/dataset/Cervical-Cancer-Screening/train/rotate_seg_val'

    num_t1 = 0
    num_t2 = 0
    num_t3 = 0
    fid = open('/home/xiaoran/qxr/dataset/Cervical-Cancer-Screening/train/seg_val.txt', 'r')
    for line in fid:
        imgpath = line.strip().split()[0]
        set = imgpath.split('/')[1]
        type = imgpath.split('/')[2]
        imgname = imgpath.split('/')[3]
        print(set, type, imgname)
        rotate_set_dir = osp.join(rotate_dir, set)
        if not osp.exists(rotate_set_dir):
            os.mkdir(rotate_set_dir)
        rotate_sub_dir = osp.join(rotate_set_dir, type)
        if not osp.exists(rotate_sub_dir):
            os.mkdir(rotate_sub_dir)

        imgpath = osp.join(ori_dir, imgpath)
        img = cv2.imread(imgpath)
        if img is None:
            print('The image is None!')
            continue
        if type == 'Type_1':
            num_t1 += 1
            img_r90 = rotate(img, 90)
            wripath = osp.join(rotate_sub_dir, imgname.split('.')[0] + '_r90.jpg')
            cv2.imwrite(wripath, img_r90)
            #img_r180 = rotate(img, 180)
            #wripath = osp.join(rotate_sub_dir, imgname.split('.')[0] + '_r180.jpg')
            #cv2.imwrite(wripath, img_r180)
        elif type == 'Type_2':
            num_t2 += 1
            img_r90 = rotate(img, 90)
            wripath = osp.join(rotate_sub_dir, imgname.split('.')[0] + '_r90.jpg')
            cv2.imwrite(wripath, img_r90)
        elif type == 'Type_3':
            num_t3 += 1
            img_r90 = rotate(img, 90)
            wripath = osp.join(rotate_sub_dir, imgname.split('.')[0] + '_r90.jpg')
            cv2.imwrite(wripath, img_r90)

    print(num_t1, num_t2, num_t3)


