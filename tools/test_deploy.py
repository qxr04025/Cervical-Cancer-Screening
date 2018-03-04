#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
To inference the test dataset.
Author:qinxiaoran

"""
from __future__ import division, print_function, absolute_import

import _init_paths
import caffe
import argparse
import sys
import os
import os.path as osp
import cv2
import numpy as np
from IPython import embed

def get_imdb(data_dir):
    file_name = 'testlist.txt'
    file_name = os.path.join(data_dir, file_name)
    fid = open(file_name, 'r')
    image_names = []
    for im_name in fid:
        image_names.append(im_name.strip())

    imdb = image_names

    return imdb

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='vgg classification')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')

    args = parser.parse_args()

    return args

def im_list_to_blob(ims):
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

if __name__ == '__main__':
    args = parse_args()
    
    rootdir = '/home/qinxiaoran/project/cervical_cancer'
    prototxt = osp.join(rootdir, 'models/vgg_ori+seg+crop400_224/deploy.prototxt')
    #caffemodel = osp.join(rootdir, 'snapshot/binary-class/binary-class_iter_1000.caffemodel')
    caffemodel= osp.join(rootdir, 'snapshot/vgg_ori+seg+crop400_224/_iter_50000.caffemodel')
    #mean_file = osp.join(rootdir, 'data/train/ori+seg_aug_mean.npy')

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('Loaded network {:s}'.format(caffemodel))

    data_dir = osp.join(rootdir, 'data')
    save_dir = osp.join(rootdir, 'res')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    imdb = get_imdb(data_dir)
    #imdb = ['val_14019']

    save_file = osp.join(save_dir, 'res_vgg_ori+seg+crop400_224_iter5w.txt')
    fid = open(save_file, 'w')
    #fid.write('image_name,Type_1,Type_2,Type_3\n')
    for img_name in imdb:
        #print(img_name, img_label)
        im = cv2.imread(osp.join(data_dir, 'test', img_name))
        #if im is None:
        #   fid.write(img_name + ' ' + str(0) + '\n')
        #    continue
        im = im.astype(np.float32, copy=False)
        im = cv2.resize(im, (224, 224))
        #meanvalue = np.load(mean_file).mean(1).mean(1).reshape(1, 1, 3)
        #im = im - meanvalue 
        im = im / 256

        blob = im_list_to_blob([im])
        forward_kwargs = {'data': blob.astype(np.float32, copy=False)}
        res = net.forward(**forward_kwargs)
        res = res['prob'][0]
        res_index = np.argsort(res)[-1]
        fid.write(img_name + ' ' + str(res_index) + '\n')
        print(img_name, res_index)      
    
    fid.close()



   








