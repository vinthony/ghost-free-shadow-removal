from __future__ import division
from networks import *
from utils import *
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import scipy.stats as st
import sys
import tensorflow as tf
import time

EPS = 1e-12

class Deshadower(object):
    def __init__(self, model_path, vgg_19_path, use_gpu, hyper):
        self.vgg_19_path = vgg_19_path
        self.model = model_path
        self.hyper = hyper 
        self.channel = 64
        if use_gpu<0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=str(use_gpu)
        self.setup_model()
    
    def setup_model(self):
        # set up the model and define the graph
        with tf.variable_scope(tf.get_variable_scope()):
            self.input=tf.placeholder(tf.float32,shape=[None,None,None,3])
            target=tf.placeholder(tf.float32,shape=[None,None,None,3])
            gtmask = tf.placeholder(tf.float32,shape=[None,None,None,1])

            # build the model
            self.shadow_free_image,predicted_mask=build_aggasatt_joint(self.input, self.channel, vgg_19_path=self.vgg_19_path)

            loss_mask = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gtmask,tf.nn.sigmoid(predicted_mask)))
    
            # Perceptual Loss
            loss_percep = compute_percep_loss(self.shadow_free_image, target,vgg_19_path=self.vgg_19_path) 
            # Adversarial Loss
            with tf.variable_scope("discriminator"):
                predict_real,pred_real_dict = build_discriminator(self.input,target)
            with tf.variable_scope("discriminator", reuse=True):
                predict_fake,pred_fake_dict = build_discriminator(self.input, self.shadow_free_image)

            d_loss=(tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))) * 0.5
            g_loss=tf.reduce_mean(-tf.log(predict_fake + EPS))

            loss = loss_percep*0.2 + loss_mask
            
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        ckpt=tf.train.get_checkpoint_state(self.model)
            
        print("[i] contain checkpoint: ", ckpt)
        saver_restore=tf.train.Saver([var for var in tf.trainable_variables() if 'discriminator' not in var.name])
        print('loaded '+ckpt.model_checkpoint_path)
        saver_restore.restore(self.sess,ckpt.model_checkpoint_path)

        sys.stdout.flush()

        
    def run(self, img):
        iminput = expand(img)
        st=time.time()
        imoutput = self.sess.run([self.shadow_free_image],feed_dict={self.input:iminput})
        print("Test time  = %.3f " % (time.time()-st ))
        imoutput=decode_image(imoutput)
        return imoutput 
