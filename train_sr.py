from __future__ import division
import os,time,cv2,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from networks import build_discriminator
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from networks import *
from utils import *
import scipy.stats as st
import argparse,sys

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="pre-trained", help="path to folder containing the model")
parser.add_argument("--data_dir", default="./Dataset/ISTD_Dataset/", help="path to real dataset")
parser.add_argument("--save_model_freq", default=1, type=int, help="frequency to save model")
parser.add_argument("--use_gpu", default=0, type=int, help="which gpu to use")
parser.add_argument("--use_da", default=0.5, type=float, help="[0~1], the precentage of synthesized dataset")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training", default=1, help="training or testing")
parser.add_argument("--continue_training", action="store_true", help="search for checkpoint in the subfolder specified by `task` argument")
ARGS = parser.parse_args()

task='logs/'+ARGS.task
is_training=ARGS.is_training==1
continue_training=ARGS.continue_training
hyper=ARGS.is_hyper==1
current_best = 65535
maxepoch=151
EPS = 1e-12
channel = 64 # number of feature channels to build the model, set to 64
vgg_19_path = './Models/imagenet-vgg-verydeep-19.mat'

test_w,test_h = 640,480

if ARGS.use_gpu<0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(ARGS.use_gpu)

train_real_root=[ARGS.data_dir]

# set up the model and define the graph
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,None,None,3])
    target=tf.placeholder(tf.float32,shape=[None,None,None,3])
    gtmask = tf.placeholder(tf.float32,shape=[None,None,None,1])

    # build the model
    shadow_free_image,predicted_mask=build_aggasatt_joint(input,channel,vgg_19_path=vgg_19_path)

    loss_mask = tf.reduce_mean(tf.keras.losses.binary_crossentropy(gtmask,tf.nn.sigmoid(predicted_mask)))
    
    # Perceptual Loss
    loss_percep = compute_percep_loss(shadow_free_image, target,vgg_19_path=vgg_19_path) 
    # Adversarial Loss
    with tf.variable_scope("discriminator"):
        predict_real,pred_real_dict = build_discriminator(input,target)
    with tf.variable_scope("discriminator", reuse=True):
        predict_fake,pred_fake_dict = build_discriminator(input,shadow_free_image)

    d_loss=(tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))) * 0.5
    g_loss=tf.reduce_mean(-tf.log(predict_fake + EPS))

    loss = loss_percep*0.2 + loss_mask

train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminator' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]
g_opt=tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss*100+g_loss, var_list=g_vars) # optimizer for the generator
d_opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss,var_list=d_vars) # optimizer for the discriminator

for var in tf.trainable_variables():
    print("Listing trainable variables ... ")
    print(var)

saver=tf.train.Saver(max_to_keep=None)

if not os.path.isdir(task):
    os.makedirs(task)

######### Session #########
sess=tf.Session()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(task)
print("[i] contain checkpoint: ", ckpt)

if ckpt and continue_training:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables()])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)
# test doesn't need to load discriminator
elif not is_training:
    saver_restore=tf.train.Saver([var for var in tf.trainable_variables() if 'discriminator' not in var.name])
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)

sys.stdout.flush()

if is_training:
    # please follow the dataset directory setup in README
    input_images_path=prepare_data(train_real_root,stage=['train_A']) # no reflection ground truth for real images
    syn_images=prepare_data(train_real_root,stage=['synC'])

    print("[i] Total %d training images, first path of real image is %s." % (len(input_images_path), input_images_path[0]))
    
    num_train=len(input_images_path)+len(syn_images)
    all_l=np.zeros(num_train, dtype=float)
    all_percep=np.zeros(num_train, dtype=float)
    all_grad=np.zeros(num_train, dtype=float)
    all_g=np.zeros(num_train, dtype=float)
    for epoch in range(1,maxepoch):
        input_images_ids,target_images_ids=[None]*num_train,[None]*num_train
        epoch_st = time.time()

        if os.path.isdir("%s/%04d"%(task,epoch)):
            continue
        cnt=0
        for id in np.random.permutation(num_train):
            st=time.time()
            if input_images_ids[id] is None:
                _id=id%len(input_images_path)
                running_idx = (epoch-1)*num_train+cnt
                magic = np.random.rand()
                current_img_id = ''

                inputimg = cv2.imread(input_images_path[_id],-1)
                neww=np.random.randint(256, 480) # w is the longer width[]
                newh=round((neww/inputimg.shape[1])*inputimg.shape[0])

                if magic < ARGS.use_da: #choose from fake images
                    is_syn = True
                    current_img_id = random.sample(syn_images,1)[0]
                    iminput,imtarget,maskgt = parpare_image_syn(current_img_id,(neww,newh),da=True,stage='synC')
                else:
                    is_syn = False
                    current_img_id = input_images_path[_id]
                    iminput,imtarget,maskgt = parpare_image(current_img_id,(neww,newh),da=True,stage=['_M','_C','_B'])
                
               # alternate training, update discriminator every two iterations
                if cnt%2==0:
                    fetch_list=[d_opt]
                    # update D
                    _=sess.run(fetch_list,feed_dict={input:iminput,target:imtarget,gtmask:maskgt})

                # update G                
                fetch_list=[g_opt,shadow_free_image,d_loss,g_loss,loss,loss_percep]
                _,imoutput,current_d,current_g,current,current_percep=\
                    sess.run(fetch_list,feed_dict={input:iminput,target:imtarget,gtmask:maskgt})

                all_l[id]=current
                all_percep[id]=current_percep
                all_g[id]=current_g
                g_mean=np.mean(all_g[np.where(all_g)])

                if running_idx% 500==0:
                    print("iter: %d %d || D: %.2f || G: %.2f %.2f || mean all: %.2f || percp: %.2f %.2f || time: %.2f"%
                        (epoch,cnt,current_d,current_g,g_mean,
                            np.mean(all_l[np.where(all_l)]),
                            current_percep, np.mean(all_percep[np.where(all_percep)]),
                            time.time()-st))

                    fileid = os.path.splitext(os.path.basename(input_images_path[_id]))[0]
                    imoutput=decode_image(imoutput)
                    iminput=decode_image(iminput)
                    imtarget=decode_image(imtarget)
                    cv2.imwrite("%s/%s_%s.png"%(task, running_idx, fileid),np.concatenate((iminput,imoutput,imtarget),axis=1))

                cnt+=1
                input_images_ids[id]=1.
                target_images_ids[id]=1.

        print('epoch %s use %s'%(epoch,time.time()-epoch_st))

        # save model and images every epoch
        if epoch % ARGS.save_model_freq == 0:
            saver.save(sess,"%s/lasted_model.ckpt"%task)
            sys.stdout.flush()
else:
    subtask=task.replace('/','_') # if you want to save different testset separately
    stage='test_A'
    for val_path in prepare_data([ARGS.data_dir],stage=[stage]):
        if not os.path.isfile(val_path):
            continue
        iminput,imtarget,maskgt = parpare_image(val_path,(test_w,test_h))
        st=time.time()
        imoutput=sess.run([shadow_free_image],feed_dict={input:iminput})
        print("Test time %.3f for image %s"%(time.time()-st, val_path))

        imtarget=decode_image(imtarget)
        imoutput=decode_image(imoutput)

        if not os.path.isdir("./results/%s"%(subtask+stage)):
            os.makedirs("./results/%s"%(subtask+stage))
        cv2.imwrite("./results/%s/%s.png"%(subtask+stage,os.path.splitext(os.path.basename(val_path))[0]),imoutput) # output transmission layer
