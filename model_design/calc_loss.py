import tensorflow as tf
import numpy as np
import os
import cv2

import sys

sys.path.append('../')

import config




def build_loss(logits, labels):
    print('logits: ', logits.get_shape().as_list())
    print('labels: ', labels.get_shape().as_list())

    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    )
    print('ce_loss: ', cross_entropy_loss.get_shape().as_list())








