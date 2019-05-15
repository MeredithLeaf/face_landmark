import tensorflow as tf
import numpy as np
import cv2
import random

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x.W):
    return tf.nn.conv2d(x,W,strides)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#图片大小是178*178
#关键点数是10
KEYPOINT_INDEX = 10
IMGSIZE = 178

x = tf.placeholder("float", shape=[None, IMGSIZE, IMGSIZE, 3])
y_ = tf.placeholder("float", shape=[None, KEYPOINT_INDEX])
keep_prob = tf.placeholder("float")

def model():
    W_conv1 = weight_variable([3,3,3,32])
    b_conv1 = bias_variable([32])
    
    #[178,178] -- [176,176] -- [88,88]
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    
    #[88,88] -- [86,86] --[43,43]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_conv3 = weight_variable([2,2,64,128])
    b_conv3 = bias_variable([128])
    
    #[43,43] -- [42,42] -- [21,21]
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    #全链接层
    W_fc1 = weight_variable([21*21*128, 500])
    b_fc1 = bias_variable([500])
    h_pool3_flag = tf.reshape(h_pool3, [-1, 21*21*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    W_fc2 = weight_variable([500, 500])
    b_fc2 = bias_variable([500])
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    
    W_fc3 = weight_variable([500, KEYPOINT_INDEX])
    b_fc3 = bias_variable([KEYPOINT_INDEX])
    
2
