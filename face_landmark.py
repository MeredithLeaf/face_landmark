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
