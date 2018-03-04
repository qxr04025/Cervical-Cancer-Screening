#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 

Author:qinxiaoran

"""
from __future__ import division, print_function, absolute_import
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '../..', 'python')
add_path(caffe_path)