#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
To evaluate the test accuracy.
Author:qinxiaoran

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0,1,0.001)
N = 512
acc = 0.71
loss = -1*N*acc*np.log(x)-1*N*(1-acc)*np.log((1-x)/2)
loss = loss/N
print("The min of loss is " , np.min(loss))
print("The prob to reach min is ",0.001*np.argmin(loss))
plt.xlabel('loss')
plt.ylabel('accuracy')
plt.title('Acc vs Loss')
plt.plot(x,loss)
#plt.show()
#acc = np.arange(0,1,0.001)

loss2 = -1*N*acc*np.log(0.71)-1*N*(1-acc)*np.log((1-0.71)/2)
loss2 = loss2/N
#plt.plot(acc,loss2)
print(loss2)
