import numpy as np
import tensorflow as tf
import json
import glob
import os
import random
import cv2
from cfg import cfg
def l1_loss(src, dst):
    return tf.reduce_mean(tf.abs(src - dst))

def l2_loss(src, dst):
    return tf.reduce_mean(tf.nn.l2_normalize((src-dst)**2,dim=0))

def softmax_cross_entropy_loss(prediction, gt):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt))

def cosine_loss(a, b):
    transpose_a = tf.transpose(a)
    transpose_b = tf.transpose(b)
    inner_a = tf.sqrt(tf.reduce_sum(transpose_a * transpose_a, 1))
    inner_b = tf.sqrt(tf.reduce_sum(transpose_b * transpose_b, 1))
    inner_pro = tf.reduce_sum(transpose_a * transpose_b, 1)
    score = tf.div(inner_pro, inner_a * inner_b + 1e-8)
    output= tf.reduce_sum(0.5-0.5*score)
    return output