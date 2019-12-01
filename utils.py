import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os,time,cv2,scipy.io,random
from PIL import Image
from PIL import ImageEnhance,ImageFilter
import pydensecrf.densecrf as dcrf


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
    
def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')
    
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

def rmse_lab(imtarget,imoutput,immask):
    
    imtarget = np.float32(cv2.cvtColor(imtarget,cv2.COLOR_BGR2Lab))
    imoutput = np.float32(cv2.cvtColor(imoutput,cv2.COLOR_BGR2Lab))

    imtarget[:,:,0] = imtarget[:,:,0]*100/255.
    imtarget[:,:,1] = imtarget[:,:,1]-128
    imtarget[:,:,2] =imtarget[:,:,2]-128
    
    imoutput[:,:,0] = imoutput[:,:,0]*100/255.
    imoutput[:,:,1] = imoutput[:,:,1]-128
    imoutput[:,:,2] = imoutput[:,:,2]-128
    
    if len(immask.shape) == 2:
      immask = immask[:,:,np.newaxis]

    mask_binary = immask/255.0
    
    err_masked = np.sum(abs(imtarget*mask_binary-imoutput*mask_binary))
    num_of_mask = np.sum(mask_binary)

    return err_masked,num_of_mask
    
def encode_image_gauss(img_path,sz=(640,480)):
    imw,imh = sz
    im = Image.open(img_path)
    imx = im.resize((imw//10,imh//10))
    imx = imx.resize((imw,imh))
    imx = imx.filter(ImageFilter.GaussianBlur(radius=32))
    imx = np.asarray(imx)/255.
    return imx

def expand(im):
  if len(im.shape) == 2:
    im = np.expand_dims(im,axis=2)
  im = np.expand_dims(im,axis=0)
  return im

def cv2pil(img,color=True):
  rgb = np.uint8(img*255.0)
  if color:
    rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)

  im = Image.fromarray(rgb)
  return im

def pil2cv(img):
  imnp = np.array(np.uint8(img))
  return cv2.cvtColor(imnp,cv2.COLOR_RGB2BGR)

def resize_to_test(img,sz=(640,480)):
  imw,imh = sz
  return cv2.resize(np.float32(img),(imw,imh),cv2.INTER_CUBIC)


def encode_image(img_path,sz=(256,256),resize=True):
  imw,imh = sz
  input_image = cv2.imread(img_path,-1)
  
  if resize:
    input_image=cv2.resize(np.float32(input_image),(imw,imh),cv2.INTER_CUBIC)

  return input_image/255.0


def parpare_image_fake_generator(val_path,im_mask_path,sz=(640,480)):

  imw,imh = sz
  immask  = encode_image(im_mask_path,(imw,imh))
  imshadowfree = encode_image(val_path.replace('_A','_C'),(imw,imh))

  syn_pil = cv2pil(imshadowfree)
  bri = ImageEnhance.Brightness(syn_pil)

  syn_shadow = bri.enhance(1-0.5)

  offset_x = 0
  offset_y = 0

  pilMask = cv2pil(immask,color=False)
  pilMask = pilMask.filter(ImageFilter.GaussianBlur(radius=3))

  syn_pil.paste(syn_shadow,(offset_x,offset_y),pilMask)

  iminput = np.float32(pil2cv(syn_pil))/255.0

  iminput,immask = [expand(x) for x in (iminput,immask) ]

  return iminput,immask



def parpare_image_fake(val_path,sz=(640,480),da=True,stage=['_M','_T','_B']):
  imw,imh = sz

  # pwd_mask = '/home/liuxuebo/Datasets/ISTD_dataset/train/train_B/'
  # image_names = [ os.path.join(pwd_mask,x) for x in os.listdir(pwd_mask) if '.png' in x]
  # im_mask_path = random.choice(image_names)

  imshadowfree = encode_image(val_path.replace('_A','_T'),(imw,imh))  
  imshadow = encode_image(val_path,(imw,imh))
  immask = encode_image(val_path.replace('_A',stage[2]),(imw,imh)) #mask

  # read the non-shadow-image
  # imshadow = encode_image(val_path,(imw,imh))
  # read the random mask.
  # immask  = encode_image(im_mask_path,(imw,imh))

  syn_pil = cv2pil(imshadowfree)
  bri = ImageEnhance.Brightness(syn_pil)

  syn_shadow = bri.enhance(1-0.5)
  # image_bright = bri.enhance(1+factor/2)

  offset_x = 0
  offset_y = 0

  pilMask = cv2pil(immask,color=False)
  pilMask = pilMask.filter(ImageFilter.GaussianBlur(radius=3))

  syn_pil.paste(syn_shadow,(offset_x,offset_y),pilMask)

  iminput = np.float32(pil2cv(syn_pil))/255.0

  if da:
    if np.random.random_sample() > 0.5:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imshadow,immask = [cv2.flip(x,_c) for x in [iminput,immask,imshadow,immask] ]
  
  iminput,immask,imshadow,immask = [expand(x) for x in (iminput,immask,imshadow,immask) ]

  return iminput,immask,imshadow,immask


def parpare_image(val_path,sz=(640,480),da=False,stage=['_M','_T','_B']):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))

  if stage[0]=='_M':
    immask = encode_image(val_path.replace('_A',stage[0]).replace('.png','.jpg'),(imw,imh))
  elif stage[0]=='_D':
    immask = encode_image_gauss(val_path.replace('_A',stage[0]),(imw,imh))
  elif stage[0]=='_G':
    immask = encode_image_gauss(val_path.replace('_A','_M').replace('.png','.jpg'),(imw,imh))
  else:
    immask = encode_image(val_path.replace('_A',stage[0]),(imw,imh))

  imtarget = encode_image(val_path.replace('_A',stage[1]),(imw,imh))
  gtmask = encode_image(val_path.replace('_A',stage[2]),(imw,imh))


  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,immask,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,immask,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,immask,imtarget,gtmask] ]

  
  iminput,immask,imtarget,gtmask = [expand(x) for x in (iminput,immask,imtarget,gtmask) ]

  return iminput,immask,imtarget,gtmask


def parpare_image_motif(val_path,sz=(640,480),da=False):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))
  immask = encode_image(val_path.replace('synthesized','real_mask'),(imw,imh))
  imtarget = encode_image(val_path.replace('synthesized','real_image'),(imw,imh))


  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imtarget = [cv2.flip(x,_c) for x in [iminput,immask,imtarget] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,immask,imtarget = [np.rot90(x,_c) for x in [iminput,immask,imtarget] ]
  
  iminput,immask,imtarget = [expand(x) for x in (iminput,immask,imtarget) ]

  return iminput,immask,imtarget


def parpare_image_SRD(val_path,sz=(640,480),da=False):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))
  imtarget = encode_image(val_path.replace('/shadow','/shadow_free'),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget = [cv2.flip(x,_c) for x in [iminput,imtarget] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget = [np.rot90(x,_c) for x in [iminput,imtarget] ]
  
  iminput,imtarget = [expand(x) for x in (iminput,imtarget) ]

  return iminput,imtarget


def selected_image(val_path,sz=(640,480),da=False):
  imw,imh = sz

  val_path = random.sample(val_path,1)[0]
  iminput = encode_image(val_path,(imw,imh))

  val_mask_name = val_path.split('/')[-1].split('_')[-1]
  immask = encode_image(val_path.replace('syn_shadow','train_B').replace(val_path.split('/')[-1],val_mask_name),(imw,imh))

  val_im_name = '_'.join(val_path.split('/')[-1].split('_')[0:-1])+'.jpg'

  imtarget = encode_image(val_path.replace('syn_shadow','shadow_free').replace(val_path.split('/')[-1],val_im_name),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imtarget = [cv2.flip(x,_c) for x in [iminput,immask,imtarget] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,immask,imtarget = [np.rot90(x,_c) for x in [iminput,immask,imtarget] ]

  
  iminput,immask,imtarget = [expand(x) for x in (iminput,immask,imtarget) ]

  return iminput,immask,imtarget

def parpare_image_syn_de(val_path,sz=(640,480),da=False):
  imw,imh = sz

  ids = val_path.split('/')[-1]
  base_path = os.path.dirname(val_path).replace('_SYN','_C')

  img_id, mask_id = ids.split('_')[0],ids.split('_')[1]

  shadow_free_path = os.path.join(base_path,img_id+'.png')
  mask_path = os.path.join(base_path.replace('_C','_B'),mask_id)
  iminput = encode_image(val_path,(imw,imh))


  immask = encode_image(mask_path.replace('_B','_M').replace('.png','.jpg'),(imw,imh))

  imtarget = encode_image(shadow_free_path,(imw,imh))
  gtmask = encode_image(mask_path,(imw,imh))

  if da:
    if np.random.random_sample() > 0.5:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,immask,imtarget,gtmask] ]
  
  iminput,immask,imtarget,gtmask = [expand(x) for x in (iminput,immask,imtarget,gtmask) ]

  return iminput,immask,imtarget,gtmask

def parpare_image_syn(val_path,sz=(640,480),da=False,stage='train_shadow_free'):
  imw,imh = sz

  iminput = encode_image(val_path,(imw,imh))

  val_mask_name = val_path.split('/')[-1].split('_')[-1]
  gtmask = encode_image(val_path.replace(stage,'train_B').replace(val_path.split('/')[-1],val_mask_name),(imw,imh))
  immask = encode_image(val_path.replace(stage,'train_B').replace(val_path.split('/')[-1],val_mask_name),(imw,imh))

  val_im_name = '_'.join(val_path.split('/')[-1].split('_')[0:-1])+'.jpg'
  imtarget = encode_image(val_path.replace(stage,'shadow_free').replace(val_path.split('/')[-1],val_im_name),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,immask,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,immask,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,immask,imtarget,gtmask] ]

  iminput,immask,imtarget,gtmask = [expand(x) for x in (iminput,immask,imtarget,gtmask) ]

  return iminput,immask,imtarget,gtmask

def parpare_image_deshadow(val_path,sz=(640,480),val=False,da=False,stage=['masks','shadow_free_new','_B']):

  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh))
  immask = encode_image(val_path.replace('/shadow','/'+stage[0]),(imw,imh))
  imtarget = encode_image(val_path.replace('/shadow','/'+stage[1]),(imw,imh))
  gtmask = immask

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,immask,imtarget,gtmask = [cv2.flip(x,_c) for x in [iminput,immask,imtarget,gtmask] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,immask,imtarget,gtmask = [np.rot90(x,_c) for x in [iminput,immask,imtarget,gtmask] ]

  
  iminput,immask,imtarget,gtmask = [expand(x) for x in (iminput,immask,imtarget,gtmask) ]

  return iminput,immask,imtarget,gtmask


def parpare_image_desnow(val_path,sz=(640,480),da=False):
  imw,imh = sz
  iminput = encode_image(val_path,(imw,imh)) 
  imtarget = encode_image(val_path.replace('synthetic','gt'),(imw,imh))

  if da:
    if np.random.random_sample() > 0.75:
      _c = random.choice([-1,0,1])
      # data augumentation
      iminput,imtarget = [cv2.flip(x,_c) for x in [iminput,imtarget] ]

    if imw == imh:
      # rotate
      _c = random.choice([0,1,2,3])
      # data augumentation
      iminput,imtarget = [np.rot90(x,_c) for x in [iminput,imtarget] ]
  
  iminput,imtarget = [expand(x) for x in (iminput,imtarget) ]

  return iminput,imtarget
