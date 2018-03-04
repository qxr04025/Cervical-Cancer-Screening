#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
To divide the train and validation datasets.
@author:cuirundong 
@file: div_train_val_resize.py 
@time: 2017/05/30

"""
from __future__ import division, print_function, absolute_import

import os
import os.path as osp
import cv2

def div_train_val_resize(ori_path, resize_path):
    #fid_train = open('./cancer_train_a+1.txt', 'a+')
    #fid_val = open('./cancer_val_a+1.txt', 'a+')

    #num_train = 0
    #num_val = 0
    dirlists = os.listdir(ori_path)
    dirlists.sort()
    pid = 0
    for subdir in dirlists:
        orisub_path = osp.join(ori_path, subdir)
        resizesub_path = osp.join(resize_path, subdir)
        if not osp.exists(resizesub_path):
            os.mkdir(resizesub_path)
        imglists = os.listdir(orisub_path)
        imglists.sort()
        #num = 0
        for imgname in imglists:
            imgpath = osp.join(orisub_path, imgname)
            rew_imgpath = osp.join(resizesub_path, imgname)
            img = cv2.imread(imgpath)
            if img is None:
                print(imgpath + ' The image is None!')
                continue
            #num += 1
            imw = img.shape[1]
            imh = img.shape[0]
            #rate = 256*1.0/min(imw, imh)
            #img_re = cv2.resize(img, (int(imw*rate), int(imh*rate)))
            img_re = cv2.resize(img, (448, 448))
            print(img_re.shape)
            cv2.imwrite(rew_imgpath, img_re)

            # spl = rew_imgpath.split('/')
            # if num%5 != 0:
            #     fid_train.write(spl[-3]+'/'+spl[-2]+'/'+spl[-1]+' '+str(pid)+'\n')
            #     num_train = num_train + 1
            # else:
            #     fid_val.write(spl[-3]+'/'+spl[-2]+'/'+spl[-1]+' '+str(pid)+'\n')
            #     num_val = num_val + 1
        pid = pid + 1

    # fid_train.close()
    # fid_val.close()

    print('pid:', pid-1)
    #print('total set:', num_train+num_val, 'train set:', num_train, 'val set:', num_val)


def div_train_val(ori_path, write_path):
    fid_train = open(osp.join(write_path, 'seg_train.txt'), 'a+')
    fid_val = open(osp.join(write_path, 'seg_val.txt'), 'a+')

    num_train = 0
    num_val = 0
    dirlists = os.listdir(ori_path)
    dirlists.sort()
    pid = 0
    for subdir in dirlists:
        orisub_path = osp.join(ori_path, subdir)
        imglists = os.listdir(orisub_path)
        imglists.sort()
        num = 0
        for imgname in imglists:
            imgpath = osp.join(orisub_path, imgname)
            print(imgpath)
            num += 1
            spl = imgpath.split('/')
            if num%15 != 0:
                fid_train.write(spl[-3]+'/'+spl[-2]+'/'+spl[-1]+' '+str(pid)+'\n')
                num_train = num_train + 1
            else:
                fid_val.write(spl[-3]+'/'+spl[-2]+'/'+spl[-1]+' '+str(pid)+'\n')
                num_val = num_val + 1
        pid = pid + 1
    fid_train.close()
    fid_val.close()
    print('pid:', pid-1)
    print('total set:', num_train + num_val, 'train set:', num_train, 'val set:', num_val)


def aug_train(ori_path, write_path):
    fid = open(osp.join(write_path, 'aug_train.txt'), 'a+')
    dirlists = os.listdir(ori_path)
    dirlists.sort()
    pid = 0
    for subdir in dirlists:
        orisub_path = osp.join(ori_path, subdir)
        imglists = os.listdir(orisub_path)
        imglists.sort()
        for imgname in imglists:
            imgpath = osp.join(orisub_path, imgname)
            print(imgpath)
            spl = imgpath.split('/')
            fid.write(spl[-4] + '/' + spl[-3] + '/' + spl[-2] + '/' + spl[-1] + ' ' + str(pid) + '\n')
        pid = pid + 1
    fid.close()
    print('pid:', pid - 1)

def rotate_txt(ori_path, write_path):
    fid = open(osp.join(write_path, 'rotate_seg_val.txt'), 'a+')
    dirlists = os.listdir(ori_path)
    dirlists.sort()
    pid = 0
    for subdir in dirlists:
        orisub_path = osp.join(ori_path, subdir)
        imglists = os.listdir(orisub_path)
        imglists.sort()
        for imgname in imglists:
            imgpath = osp.join(orisub_path, imgname)
            print(imgpath)
            spl = imgpath.split('/')
            fid.write(spl[-4] + '/' + spl[-3] + '/' + spl[-2] + '/' + spl[-1] + ' ' + str(pid) + '\n')
        pid = pid + 1
    fid.close()
    print('pid:', pid - 1)

        

if __name__ == '__main__':
    ori_path = '/home/xiaoran/qxr/dataset/Cervical-Cancer-Screening/train/rotate_seg_val/additional'
    write_path = '/home/xiaoran/qxr/dataset/Cervical-Cancer-Screening/train'
    #resize_path = '/home/xiaoran/qxr/dataset/Cervical-Cancer-Screening/train/seg448/additional'
    # if not osp.exists(resize_path):
    #     os.mkdir(resize_path)
    # resize_path = osp.join(resize_path, 'additional')
    # if not osp.exists(resize_path):
    #     os.mkdir(resize_path)
    rotate_txt(ori_path, write_path)
