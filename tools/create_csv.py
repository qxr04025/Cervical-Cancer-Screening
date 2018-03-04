#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
To create .csv result files.
Author:qinxiaoran

"""
from __future__ import division, print_function, absolute_import

if __name__ == '__main__':
    c1 = 0
    c2 = 0
    c3 = 0
    with open('res/res_vgg_ori+seg+crop400_224_iter5w.csv','w') as fout:
        fout.write("image_name,Type_1,Type_2,Type_3\n")
        for line in open('res/res_vgg_ori+seg+crop400_224_iter5w.txt'):
            filename = line.strip().split()[0]
            calssname = line.strip().split()[1]
            if calssname == '0':
                fout.write(filename+","+str(0.76)+","+str(0.12)+","+str(0.12)+"\n")
                c1=c1+1
            if calssname == '1':
                fout.write(filename+","+str(0.12)+","+str(0.76)+","+str(0.12)+"\n")
                c2=c2+1
            if calssname == '2':
                fout.write(filename+","+str(0.12)+","+str(0.12)+","+str(0.76)+"\n")
                c3=c3+1
    print(c1,c2,c3)
