import tensorflow as tf
import os
import cv2
import numpy as np
import sys

from model_design import model_design_nn
import config
from data_prepare import data_loader



class Inference():
    def __init__(self, model_save_dir=config.model_save_dir):
        self.load_model(model_save_dir)

    def load_model(self, model_save_dir):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.images = tf.placeholder(dtype=tf.float32, shape=[None, config.img_h, config.img_w, config.img_ch])
            net_out = model_design_nn.cls_net(self.images, is_training=False)
            self.scores = tf.keras.layers.Softmax()(net_out)
            saver = tf.train.Saver(max_to_keep=3)

        with self.sess.as_default():
            ckpt_path = tf.train.latest_checkpoint(model_save_dir)
            print('latest_checkpoint_path: ', ckpt_path)
            if ckpt_path is not None:
                saver.restore(self.sess, ckpt_path)
            else:
                print('ckpt not exists, task over!')
                exit(0)

    def do_predict(self, img_cv2):
        img_cv2 = cv2.resize(img_cv2, dsize=(config.img_w, config.img_h), interpolation=cv2.INTER_LINEAR_EXACT)
        img_data = data_loader.image_preprocess_by_normality(img_cv2)
        _images = np.expand_dims(img_data, axis=0)
        _scores = self.sess.run(self.scores, feed_dict={self.images: _images})
        bg_prob, seat_belt_prob = _scores[0]
        return seat_belt_prob > bg_prob, seat_belt_prob



if __name__=='__main__':

    LC_Inference = Inference()

    img_dir = './data/cls_crop_result_on_real'
    img_fn_list = os.listdir(img_dir)
    correct_cnt = 0
    for idx, img_fn in enumerate(img_fn_list):
        print('----------------- img_fn: %s'%img_fn)
        img_fpath = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        result = LC_Inference.do_predict(img_cv2)
        print(result)























