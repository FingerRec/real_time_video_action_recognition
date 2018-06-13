#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2018/6/9 15:34
     # @Author  : Awiny
     # @Site    :
     # @Project : C3D-tensorflow
     # @File    : real_time_input_data.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning



import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


def clip_images_to_tensor(video_imgs, num_frames_per_clip=16, crop_size=112):
    data = []
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    tmp_data = video_imgs
    img_datas = []
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img_datas.append(img)
     # data.append(img_datas)

    # pad (duplicate) data/label if less than batch_size
    data.append(img_datas)

    np_arr_data = np.array(data).astype(np.float32)

    return np_arr_data
