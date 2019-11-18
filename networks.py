import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os,time,cv2,scipy.io


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def lrelu(x):
    return tf.maximum(x*0.2,x)

def relu(x):
    return tf.maximum(0.0,x)

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


def se_block(input_feature, name, ratio=8):
    
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)   
        assert squeeze.get_shape()[1:] == (1,1,channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='bottleneck_fc')   
        assert excitation.get_shape()[1:] == (1,1,channel//ratio)
        excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc')    
        assert excitation.get_shape()[1:] == (1,1,channel)
        scale = input_feature * excitation    
    return scale

def build_vgg19(input,vgg_19_path,reuse=False):
    vgg_path=scipy.io.loadmat(vgg_19_path)
    print("[i] Loaded pre-trained vgg19 parameters")
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_path['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
        net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32), name='vgg_conv5_3')
        net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34), name='vgg_conv5_4')
        net['pool5'] = build_net('pool', net['conv5_4'])
        return net


def spp(net,channel=64,scope='g_pool'):

    # here we build the pooling stack
    net_2 = tf.layers.average_pooling2d(net,pool_size=4,strides=4,padding='same')
    net_2 = slim.conv2d(net_2,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'2')

    net_8 = tf.layers.average_pooling2d(net,pool_size=8,strides=8,padding='same')
    net_8 = slim.conv2d(net_8,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'8')

    net_16 = tf.layers.average_pooling2d(net,pool_size=16,strides=16,padding='same')
    net_16 = slim.conv2d(net_16,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'16')

    net_32 = tf.layers.average_pooling2d(net,pool_size=32,strides=32,padding='same')
    net_32 = slim.conv2d(net_32,channel,[1,1],activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'32')

    net = tf.concat([
      tf.image.resize_bilinear(net_2,(tf.shape(net)[1],tf.shape(net)[2])),
      tf.image.resize_bilinear(net_8,(tf.shape(net)[1],tf.shape(net)[2])),
      tf.image.resize_bilinear(net_16,(tf.shape(net)[1],tf.shape(net)[2])),
      tf.image.resize_bilinear(net_32,(tf.shape(net)[1],tf.shape(net)[2])),
      net],axis=3)

    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'sf')

    return net

def agg(net,channel=64,scope='g_agg'):
    net = se_block(net,scope+'att2')
    net = slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope=scope+'agg2')
    return net


def build_aggasatt_joint(input,channel=64,vgg_19_path='None'):
    print("[i] Hypercolumn ON, building hypercolumn features ... ")
    vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0,vgg_19_path)
    for layer_id in range(1,6):
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)

    sf=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_sf')

    net0=slim.conv2d(sf,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net1=slim.conv2d(net0,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    
    net = tf.concat([net0,net1],axis=3)
    netaggi_0 = agg(net,scope='g_aggi_0')
    netaggm_0 = agg(net,scope='g_aggm_0')

    net1 = netaggi_0 * tf.nn.sigmoid(netaggm_0)

    net2=slim.conv2d(net1,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net3=slim.conv2d(net2,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')

    #agg
    netaggi_1 = agg(tf.concat([netaggi_0,net3,net2],axis=3),scope='g_aggi_1')
    netaggm_1 = agg(tf.concat([netaggm_0,net3,net2],axis=3),scope='g_aggm_1')

    net3 = netaggi_1 * tf.nn.sigmoid(netaggm_1)

    net4=slim.conv2d(net3,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net5=slim.conv2d(net4,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    
    #agg
    netaggi_2 = agg(tf.concat([netaggi_1,net5,net4],axis=3),scope='g_aggi_2')
    netaggm_2 = agg(tf.concat([netaggm_1,net5,net4],axis=3),scope='g_aggm_2')

    net6 = netaggi_2 * tf.nn.sigmoid(netaggm_2)


    net6=slim.conv2d(net6,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net7=slim.conv2d(net6,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')

    #agg
    netimg = agg(tf.concat([netaggi_1,netaggi_2,net6,net7],axis=3),scope='g_aggi_3')
    netmask = agg(tf.concat([netaggm_1,netaggm_2,net6,net7],axis=3),scope='g_aggm_3')

    netimg = spp(netimg,scope='g_imgpool')
    netmask = spp(netmask,scope='g_maskpool')

    netimg = netimg * tf.nn.sigmoid(netmask)

    netimg=slim.conv2d(netimg,3,[1,1],rate=1,activation_fn=None,scope='g_conv_img')
    netmask=slim.conv2d(netmask,1,[1,1],rate=1,activation_fn=None,scope='g_conv_mask')

    return netimg,netmask


