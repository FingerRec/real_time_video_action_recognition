#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2018/6/8 21:11
     # @Author  : Awiny
     # @Site    :
     # @Project : C3D-tensorflow
     # @File    : real_time_c3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import c3d_model
from real_time_input_data import *
import numpy as np
import cv2
import heapq

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS

images_placeholder = tf.placeholder(tf.float32, shape=(1, 16, 112, 112, 3))
labels_placeholder = tf.placeholder(tf.int64, shape=1)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(1, 16,112,112,3))
  labels_placeholder = tf.placeholder(tf.int64, shape=1)
  return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
    #with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def run_one_sample(norm_score, sess, video_imgs):
    """
    run_one_sample and get the classification result
    :param norm_score:
    :param sess:
    :param video_imgs:
    :return:
    """
   # images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
#    start_time = time.time()
#    video_imgs = np.random.rand(1, 16, 112, 112, 3).astype(np.float32)
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: video_imgs}
            )
    top1_predicted_label = np.argmax(predict_score)
    predict_score = np.reshape(predict_score,101)
    #print(predict_score)
    top5_predicted_value = heapq.nlargest(5, predict_score)
    top5_predicted_label = predict_score.argsort()[-5:][::-1]
    return top1_predicted_label, top5_predicted_label, top5_predicted_value


def build_c3d_model():
    """
    build c3d model
    :return:
    norm_score:
    sess:
    """
    #model_name = "pretrained_model/c3d_ucf101_finetune_whole_iter_20000_TF.model.mdlp"
    #model_name = "pretrained_model/conv3d_deepnetA_sport1m_iter_1900000_TF.model"
    model_name = "pretrained_model/sports1m_finetuning_ucf101.model"
    # Get the sets of images and labels for training, validation, and
    with tf.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
        }
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            logit = c3d_model.inference_c3d(
                images_placeholder[0 * FLAGS.batch_size:(0 + 1) * FLAGS.batch_size,:,:,:,:], 0.6,
                FLAGS.batch_size, weights, biases)
            logits.append(logit)
    logits = tf.concat(logits, 0)
    norm_score = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)
    return norm_score, sess


def real_time_recognition(video_path):
    """
    real time video classification
    :param video_path:the origin video_path
    :return:
    """
    norm_score, sess = build_c3d_model()
    video_src = video_path
    cap = cv2.VideoCapture(video_src)
    count = 0
    video_imgs = []
    predicted_label_top5 = []
    top5_predicted_value = []
    predicted_label = 0
    classes = {}
    flag = False
    with open('./list/classInd.txt', 'r') as f:
        for line in f:
            content = line.strip('\r\n').split(' ')
            classes[content[0]] = content[1]
   # print(classes)
    while True:
        ret, img = cap.read()
        if type(img) == type(None):
            break
        float_img = img.astype(np.float32)
        video_imgs.append(float_img)
        count += 1
        if count == 16:
            video_imgs_tensor = clip_images_to_tensor(video_imgs, 16, 112)
            predicted_label, predicted_label_top5, top5_predicted_value = run_one_sample(norm_score, sess, video_imgs_tensor)
            count = 0
            video_imgs = []
            flag = True
          # channel_1, channel_2, channel_3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        if flag:
            for i in range(5):
                cv2.putText(img, str(top5_predicted_value[i])+ ' : ' + classes[str(predicted_label_top5[i] + 1)], (10, 15*(i+1)),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                            1, False)

        cv2.imshow('video', img)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()


def main(_):
    video_path = input("please input the video path to be classification:")
    real_time_recognition(video_path)

if __name__ == '__main__':
    tf.app.run()