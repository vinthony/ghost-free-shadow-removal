import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os,time,cv2,scipy.io,random
from PIL import Image
from PIL import ImageEnhance,ImageFilter
from networks import build_vgg19


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def prepare_data(train_path, stage=['train_A']):
    input_names=[]
    image1=[]
    for dirname in train_path:
        for subfolder in stage:
            train_b = dirname + "/"+ subfolder+"/"
            for root, _, fnames in sorted(os.walk(train_b)):
                for fname in fnames:
                    if is_image_file(fname):
                        input_names.append(os.path.join(train_b, fname))
    return input_names
    
def decode_image(img,resize=False,sz=(640,480)):
    imw,imh = sz
    img = np.squeeze(np.minimum(np.maximum(img,0.0),1.0))
    if resize:
        img = resize_to_test(img,sz=(imw,imh))
    img = np.uint8(img*255.0)
    if len(img.shape) ==2:
        return np.repeat(np.expand_dims(img,axis=2),3,axis=2)
    else:
        return img

def expand(im):
  if len(im.shape) == 2:
    im = np.expand_dims(im,axis=2)
  im = np.expand_dims(im,axis=0)
  return im


def resize_to_test(img,sz=(640,480)):
  imw,imh = sz
  return cv2.resize(np.float32(img),(imw,imh),cv2.INTER_CUBIC)


def encode_image(img_path,sz=(256,256),resize=True):
  imw,imh = sz
  input_image = cv2.imread(img_path,-1)
  
  if resize:
    input_image=cv2.resize(np.float32(input_image),(imw,imh),cv2.INTER_CUBIC)

  return input_image/255.0


# dataload for images
def parpare_image(val_path,sz=(640,480),da=False,stage=['_M','_T','_B']):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))
  imtarget = encode_image(val_path.replace('_A',stage[1]),(imw,imh))
  gtmask = encode_image(val_path.replace('_A',stage[2]),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,imtarget,gtmask] ]

  iminput,imtarget,gtmask = [expand(x) for x in (iminput,imtarget,gtmask) ]

  return iminput,imtarget,gtmask

# dataload for synthesized images
def parpare_image_syn(val_path,sz=(640,480),da=False,stage='train_shadow_free'):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))
  val_mask_name = val_path.split('/')[-1].split('_')[-1]
  gtmask = encode_image(val_path.replace(stage,'train_B').replace(val_path.split('/')[-1],val_mask_name),(imw,imh))

  val_im_name = '_'.join(val_path.split('/')[-1].split('_')[0:-1])+'.jpg'
  imtarget = encode_image(val_path.replace(stage,'shadow_free').replace(val_path.split('/')[-1],val_im_name),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,imtarget,gtmask] ]

  iminput,imtarget,gtmask = [expand(x) for x in (iminput,imtarget,gtmask) ]

  return iminput,imtarget,gtmask

#### LOSSES
def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))

def compute_percep_loss(input, output, reuse=False, vgg_19_path='None'):
    vgg_real=build_vgg19(output*255.0,vgg_path=vgg_19_path,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,vgg_path=vgg_19_path,reuse=True)
    p0=compute_l1_loss(vgg_real['input'],vgg_fake['input'])
    p1=compute_l1_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_l1_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    p3=compute_l1_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    p4=compute_l1_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    p5=compute_l1_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    return p0+p1+p2+p3+p4+p5

def parpare_image_fake_generator(val_path,im_mask_path,sz=(640,480)):

  imw,imh = sz
  immask  = encode_image(im_mask_path,(imw,imh))
  imshadowfree = encode_image(val_path,(imw,imh))

  imshadowfree,immask = [expand(x) for x in (imshadowfree,immask) ]
  
  return imshadowfree,immask

