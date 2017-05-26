# -*- coding: UTF-8 -*-
import caffe
from caffe import layers as L, params as P

def input_data_layer(name, crop_size, source=None, backend=None, batch_size=None, mean_file=None, deploy=False):
  
  if deploy == False: 
  # train step
    if not isinstance(source,list):
      raise('a source list include train data and val data is needed at Train Step')
    if not isinstance(batch_size,list):
      raise('a batch_size list include train data and val data is needed at Train Step')
    if not backend is None:
      raise('data backend is neeed at Train Step')
    if mean_file is None:
      transform_param = dict(mirror=True, crop_size=crop_size, mean_file=mean_file)
    else:
      transform_param = dict(mirror=True, crop_size=crop_size, mean_value=[104, 117, 123])
    
    if len(batch_size) == 1:
      train_batch_size = batch_size[0]
      val_batch_size = batch_size[0]
    else：
      train_batch_size = batch_size[0]
      val_batch_size = batch_size[1]
    
    if len(source) == 1:
      train_data = source[0]
      val_data = source[0]
    else：
      train_data = source[0]
      val_data = source[1]

    data, label = L.Data(name=name,
                             include=dict(phase=caffe.TRAIN),
                             transform_param=transform_param
                             batch_size=train_batch_size,
                             backend=backend,
                             source=train_data,
                             ntop=2)
 
    transform_param[mirror]=False
    test_data = L.Data(name=name,
                             include=dict(phase=caffe.TEST),
                             transform_param=transform_param,
                             batch_size=val_batch_size,
                             backend=backend,
                             source=val_data,
                             top=['data','label'],
                             ntop=0)

    return data, label, test_data
  else:
  # test step
    data  = L.Input(name=name,input_param=dict(shape=dict(dim=[1,3,crop_size,crop_size])))  
    return data,None,None

def conv_relu(bottom, ks, nout, stride=1, pad=0, **kwargs):
  conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, **kwargs)
  return conv, L.ReLU(conv, in_place=True)

def conv_relu_norm(bottom, ks, nout, stride=1, pad=0, local_size, alpha, beta, **kwargs):
  conv, relu = conv_relu(bottom, ks, nout, stride, pad, **kwargs)
  return L.LRN(n.relu1, local_size=local_size, alpha=alpha, beta=beta)
  
def fc_relu(bottom, nout, , **kwargs):
  fc = L.InnerProduct(bottom, num_output=nout, **kwargs)
  return fc, L.ReLU(fc, in_place=True)
  
def fc_relu_drop(bottom, fc_param, dropout_ratio=0.5):
    fc = L.InnerProduct(bottom, num_output=fc_param['num_output'],
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type=fc_param['weight_type'], std=fc_param['weight_std']),
                        bias_filler=dict(type='constant', value=fc_param['bias_value']))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop

def max_pool(bottom, ks, stride=1):
  return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv_bn_scale(bottom, nout, bias_term=False, **kwargs):
  '''Helper to build a conv -> BN -> relu block.
  '''
  conv = L.Convolution(bottom, num_output=nout, bias_term=bias_term,
                         **kwargs)
  bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)
  scale = L.Scale(bn, bias_term=True, in_place=True)
  return conv, bn, scale
  
def factorization_conv_bn_scale_relu(net, bottom, name, num_output=64, kernel_size=3, stride=1, pad=0, bn=True, relu=True):
    conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, stride=stride, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='xavier', std=1),
                         bias_filler=dict(type='constant', value=0.2))
    net.__dict__[name] = conv
    if bn == True:
      net.__dict__[name+'_bn'] = L.BatchNorm(conv, use_global_stats=net.deploy, in_place=True)
      net.__dict__[name+'_scale'] = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    if relu == True:
      net.__dict__[name+'_relu'] = L.ReLU(conv, in_place=True)

    return conv

def factorization_conv_mxn(net, bottom, name, num_output=64, kernel_h=1, kernel_w=7, stride=1, pad_h=3, pad_w=0):
    conv_mxn = L.Convolution(bottom, num_output=num_output, kernel_h=kernel_h, kernel_w=kernel_w, stride=stride,
                             pad_h=pad_h, pad_w=pad_w,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier', std=0.01),
                             bias_filler=dict(type='constant', value=0.2))
    net.__dict__[name] = conv_mxn
    net.__dict__[name+'_bn'] = L.BatchNorm(conv_mxn, use_global_stats=net.deploy, in_place=True)
    net.__dict__[name+'_scale'] = L.Scale(conv_mxn, scale_param=dict(bias_term=True), in_place=True)
    net.__dict__[name+'_relu'] = L.ReLU(conv_mxn, in_place=True)

    return conv_mxn 
   
def inception(net, bottom, stage, conv_output):
    conv_1x1 = factorization_conv_bn_scale_relu(net, bottom, 'conv_1x1', kernel_size=1, num_output=conv_output['conv_1x1'], bn=True)
    conv_3x3_reduce = factorization_conv_bn_scale_relu(net, bottom, 'conv_3x3_reduce', kernel_size=1, num_output=conv_output['conv_3x3_reduce'], bn=True)
    conv_3x3 = factorization_conv_bn_scale_relu(net, conv_3x3_reduce, 'conv_3x3_reduce', num_output=conv_output['conv_3x3'], bn=True)
    conv_5x5_reduce = factorization_conv_bn_scale_relu(net, bottom, 'conv_5x5_reduce', kernel_size=1, num_output=conv_output['conv_5x5_reduce'], bn=True)
    conv_5x5 = factorization_conv_bn_scale_relu(net, conv_5x5_reduce, 'conv_5x5', kernel_size=5, num_output=conv_output['conv_5x5'], bn=True)
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
    net.__dict__['inception_%s_pool'%stage] = pool
    pool_proj = factorization_conv_bn_scale_relu(net, conv_5x5_reduce, 'conv_5x5', kernel_size=1, num_output=conv_output['pool_proj'], bn=True)
    
    concat = L.Concat(conv_1x1, conv_3x3, conv_5x5, pool_proj)
    return concat

def inception_bn(net, bottom, stage, conv_output):
    conv_1x1 = factorization_conv_bn_scale_relu(net, bottom, 'inception_%s_1x1'%stage, num_output=conv_output['conv_1x1'], kernel_size=1)
    conv_3x3_reduce = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_3x3_reduce'%stage, num_output=conv_output['conv_3x3_reduce'], kernel_size=1)
    conv_3x3 = factorization_conv_bn_scale_relu(net,conv_3x3_reduce, 'inception_%s_3x3'%stage, num_output=conv_output['conv_3x3'], kernel_size=3, pad=1)
    conv_5x5_reduce = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_5x5_reduce'%stage, num_output=conv_output['conv_5x5_reduce'], kernel_size=1)
    conv_5x5 = factorization_conv_bn_scale_relu(net,conv_5x5_reduce, 'inception_%s_5x5'%stage, num_output=conv_output['conv_5x5'], kernel_size=5, pad=2)
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
    net.__dict__['inception_%s_pool'%stage] = pool
    pool_proj = factorization_conv_bn_scale_relu(net, pool, 'inception_%s_pool_proj'%stage, num_output=conv_output['pool_proj'], kernel_size=1)
    concat = L.Concat(conv_1x1, conv_3x3, conv_5x5, pool_proj)
    
    return concat

def stem_v3_299x299(net, bottom):
    """
    input:3x299x299
    output:192x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv1_3x3_s2 = factorization_conv_bn_scale_relu(net, bottom, 'conv1_3x3_s2', num_output=32, kernel_size=3, stride=2)  # 32x149x149
    conv2_3x3_s1 = factorization_conv_bn_scale_relu(net,conv1_3x3_s2, 'conv2_3x3_s1', num_output=32, kernel_size=3)  # 32x147x147
    conv3_3x3_s1 = factorization_conv_bn_scale_relu(net,conv2_3x3_s1, 'conv3_3x3_s1', num_output=64, kernel_size=3, pad=1)  # 64x147x147
    net.pool1_3x3_s2 = L.Pooling(conv3_3x3_s1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x73x73
    conv4_3x3_reduce = factorization_conv_bn_scale_relu(net,net.pool1_3x3_s2, 'conv4_3x3_reduce', num_output=80, kernel_size=1)  # 80x73x73
    conv4_3x3 = factorization_conv_bn_scale_relu(net,conv4_3x3_reduce, 'conv4_3x3', num_output=192, kernel_size=3)  # 192x71x71
    return L.Pooling(conv4_3x3, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 192x35x35


def inception_v3_a(net, bottom, stage, pool_proj_num_output=32):
    """
    input:192or256or288x35x35
    output:256or288x35x35
    :param pool_proj_num_output: num_output of pool_proj
    :param bottom: bottom layer
    :return: layers
    """
    conv_1x1 = factorization_conv_bn_scale_relu(net, bottom, 'inception_%s_1x1'%stage, num_output=64, kernel_size=1)  # 64x35x35
    conv_5x5_reduce = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_5x5_reduce'%stage, num_output=48, kernel_size=1)  # 48x35x35
    conv_5x5 = factorization_conv_bn_scale_relu(net,conv_5x5_reduce, 'inception_%s_5x5'%stage, num_output=64, kernel_size=5, pad=2)  # 64x35x35
    conv_3x3_reduce = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_3x3_reduce'%stage, kernel_size=1, num_output=64)  # 64x35x35
    conv_3x3 = factorization_conv_bn_scale_relu(net,conv_3x3_reduce, 'inception_%s_3x3'%stage, kernel_size=3, num_output=96, pad=1)  # 96x35x35
    conv_3x3_2 = factorization_conv_bn_scale_relu(net,conv_3x3, 'inception_%s_3x3_2'%stage, kernel_size=3, num_output=96, pad=1)  # 96x35x35
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 192x35x35
    net.__dict__['inception_%s_pool'%stage] = pool
    pool_proj = factorization_conv_bn_scale_relu(net, pool, 'inception_%s_pool_proj'%stage, kernel_size=1, num_output=pool_proj_num_output)  # 32x35x35
    concat = L.Concat(conv_1x1, conv_5x5, conv_3x3_2, pool_proj)  # 256or288(64+64+96+32or64)x35x35
    
    return concat


def reduction_v3_a(net, bottom, stage):
    """
    input:288x35x35
    output:768x17x17
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 384x17x17
    net.__dict__['reduction_%s_pool'%stage] = pool

    conv_3x3 = factorization_conv_bn_scale_relu(net, bottom, kernel_size=3,'reduction_%s_3x3'%stage, num_output=384, stride=2)  # 384x17x17
    conv_3x3_2_reduce = factorization_conv_bn_scale_relu(net,bottom,'reduction_%s_3x3_2_reduce'%stage, num_output=64, kernel_size=1)  # 64x35x35
    conv_3x3_2 = factorization_conv_bn_scale_relu(net,conv_3x3_2_reduce, 'reduction_%s_3x3_2'%stage, num_output=96, kernel_size=3, pad=1)  # 96x35x35
    conv_3x3_3 = factorization_conv_bn_scale_relu(net,conv_3x3_2,'reduction_%s_3x3_3'%stage, num_output=96, kernel_size=3, stride=2)  # 96x17x17

    concat = L.Concat(pool, conv_3x3, conv_3x3_3)  # 768(288+384+96)x17x17
    
    return concat


def inception_v3_b(net, bottom, stage, outs=128):
    """
    input:768x17x17
    output:768x17x17
    :param outs: num_outputs
    :param bottom: bottom layer
    :return: layers
    """
    pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 768x17x17
    net.__dict__['inception_%s_pool_ave'%stage] = pool_ave
    conv_1x1 = factorization_conv_bn_scale_relu(net, pool_ave,'inception_%s_1x1'%stage, num_output=192, kernel_size=1)  # 192x17x17
    conv_1x1_2 = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x1_2'%stage, num_output=192, kernel_size=1)  # 192x17x17
    conv_1x7_reduce = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x7_reduce'%stage, num_output=outs, kernel_size=1)  # outsx17x17
    conv_1x7 = factorization_conv_mxn(net,conv_1x7_reduce,'inception_%s_1x7'%stage, num_output=outs, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # outsx17x17
    conv_7x1 = factorization_conv_mxn(net,conv_1x7,'inception_%s_7x1'%stage, num_output=192, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 192x17x17
    conv_7x1_reduce = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_7x1_reduce'%stage, num_output=outs, kernel_size=1)  # outsx17x17
    conv_7x1_2 = factorization_conv_mxn(net,conv_7x1_reduce,'inception_%s_7x1_2'%stage, num_output=outs, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # outsx17x17
    conv_1x7_2 = factorization_conv_mxn(net,conv_7x1_2,'inception_%s_1x7_2'%stage, num_output=outs, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # outsx17x17
    conv_7x1_3 = factorization_conv_mxn(net,conv_1x7_2,'inception_%s_7x1_3'%stage, num_output=outs, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # outsx17x17
    conv_1x7_3 = factorization_conv_mxn(net,conv_7x1_3,'inception_%s_1x7_3'%stage, num_output=192, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 192x17x17

    concat = L.Concat(conv_1x1_2, conv_7x1, conv_1x7_3, conv_1x1)  # 768(192+192+192+192)x17x17
    return concat


def reduction_v3_b(net, bottom, stage):
    """
    input:768x17x17
    output:1280x8x8
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 768x8x8
    net.__dict__['reduction_%s_pool'%stage] = pool

    conv_3x3_reduce = factorization_conv_bn_scale_relu(net, bottom,         'reduction_%s_3x3_reduce'%stage, num_output=192, kernel_size=1)  # 192x17x17
    conv_3x3 = factorization_conv_bn_scale_relu(net, conv_3x3_reduce,'reduction_%s_3x3'%stage,        num_output=320, kernel_size=3, stride=2)  # 320x8x8
    conv_1x7_reduce = factorization_conv_bn_scale_relu(net, bottom,         'reduction_%s_1x7_reduce'%stage,num_output=192, kernel_size=1)  # 192x17x17
    conv_1x7 = factorization_conv_mxn(net, conv_1x7_reduce,'reduction_%s_1x7'%stage, num_output=192, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)# 192x17x17
    conv_7x1 = factorization_conv_mxn(net, conv_1x7,'reduction_%s_7x1'%stage, num_output=192, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 192x17x17
    conv_3x3_2 = factorization_conv_bn_scale_relu(net, conv_7x1,'reduction_%s_3x3_2'%stage, num_output=192, kernel_size=3, stride=2)  # 192x8x8

    concat = L.Concat(pool, conv_3x3, conv_3x3_2)  # 1280(768+320+192)x8x8
    return concat


def inception_v3_c(bottom, pool=P.Pooling.AVE):
    """
    input:1280or2048x8x8
    output:2048x8x8
    :param pool: pool_type
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=pool)  # 1280or2048x8x8
    net.__dict__['inception_%s_pool'%stage] = pool
    conv_1x1 = factorization_conv_bn_scale_relu(net, pool, 'inception_%s_1x1'%stage, num_output=192, kernel_size=1)  # 192x8x8
    conv_1x1_2 = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_1x1_2'%stage, num_output=320, kernel_size=1)  # 320x8x8
    conv_1x3_reduce = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_1x3_reduce'%stage, num_output=384, kernel_size=1)  # 384x8x8
    conv_1x3 = factorization_conv_mxn(net,conv_1x3_reduce, 'inception_%s_1x3'%stage, num_output=384, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 384x8x8
    conv_3x1 = factorization_conv_mxn(net,conv_1x3_reduce, 'inception_%s_3x1'%stage, num_output=384, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 384x8x8
    conv_3x3_reduce = factorization_conv_bn_scale_relu(net,bottom, 'inception_%s_3x3_reduce'%stage, num_output=448, kernel_size=1)  # 448x8x8
    conv_3x3 = factorization_conv_bn_scale_relu(net,conv_3x3_reduce, 'inception_%s_3x3'%stage, num_output=384, kernel_size=3, pad=1)  # 384x8x8
    conv_1x3_2 = factorization_conv_mxn(net,conv_3x3, 'inception_%s_1x3_2'%stage, num_output=384, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 384x8x8
    conv_3x1_2 = factorization_conv_mxn(net,conv_3x3, 'inception_%s_3x1_2'%stage, num_output=384, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 384x8x8

    concat = L.Concat(conv_1x1_2, conv_1x3, conv_3x1, conv_1x3_2, conv_3x1_2, conv_1x1)  # 2048(192+320+384+384+384+384)x8x8
    return concat

 def stem_v4_299x299(net, bottom):
    """
    input:3x299x299
    output:384x35x35
    :param bottom: bottom layer
    :return: layers
    """
    conv1_3x3_s2 = factorization_conv_bn_scale_relu(net, bottom, 'conv1_3x3_s2', num_output=32, kernel_size=3, stride=2)  # 32x149x149
    conv2_3x3_s1 = factorization_conv_bn_scale_relu(net, conv1_3x3_s2, 'conv2_3x3_s1',  num_output=32, kernel_size=3, stride=1)  # 32x147x147
    conv3_3x3_s1 = factorization_conv_bn_scale_relu(net, conv2_3x3_s1, 'conv3_3x3_s1', num_output=64, kernel_size=3, stride=1, pad=1)  # 64x147x147

    net.inception_stem1_pool = L.Pooling(conv3_3x3_s1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x73x73
    inception_stem1_3x3_s2 = factorization_conv_bn_scale_relu(net, conv3_3x3_s1, 'inception_stem1_3x3_s2', num_output=96, kernel_size=3, stride=2)  # 96x73x73
    net.inception_stem1 = L.Concat(net.inception_stem1_pool, inception_stem1_3x3_s2)  # 160x73x73

    inception_stem2_3x3_reduce = factorization_conv_bn_scale_relu(net, net.inception_stem1, 'inception_stem2_3x3_reduce', num_output=64, kernel_size=1)  # 64x73x73
    inception_stem2_3x3 = factorization_conv_bn_scale_relu(net, inception_stem2_3x3_reduce, 'inception_stem2_3x3', num_output=96, kernel_size=3)  # 96x71x71
    inception_stem2_7x1_reduce = factorization_conv_bn_scale_relu(net, inception_stem1, 'inception_stem2_7x1_reduce',  num_output=64, kernel_size=1)  # 64x73x73
    inception_stem2_7x1 = factorization_conv_mxn(net, inception_stem2_7x1_reduce, 'inception_stem2_7x1', num_output=64, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 64x73x73
    inception_stem2_1x7 = factorization_conv_mxn(net, inception_stem2_7x1, 'inception_stem2_1x7', num_output=64, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 64x73x73
    inception_stem2_3x3_2 = factorization_conv_bn_scale_relu(net, inception_stem2_1x7,'inception_stem2_3x3_2', num_output=96, kernel_size=3)  # 96x71x71
    
    net.inception_stem2 = L.Concat(inception_stem2_3x3, inception_stem2_3x3_2)  # 192x71x71

    inception_stem3_3x3_s2 = factorization_conv_bn_scale_relu(net, net.inception_stem2, 'inception_stem3_3x3_s2', num_output=192, stride=2)  # 192x35x35
    net.inception_stem3_pool = L.Pooling(net.inception_stem2, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 192x35x35
    inception_stem3 = L.Concat(inception_stem3_3x3_s2, net.inception_stem3_pool)  # 384x35x35

    return inception_stem3


def inception_v4_a(net, bottom, stage):
    """
    input:384x35x35
    output:384x35x35
    :param bottom: bottom layer
    :return: layers
    """
    pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 384x35x35
    net.__dict__['inception_%s_pool_ave'%stage] = pool_ave
    conv_1x1 = factorization_conv_bn_scale_relu(net, pool_ave, 'inception_%s_1x1'%stage, num_output=96, kernel_size=1)  # 96x35x35
    conv_1x1_2 = factorization_conv_bn_scale_relu(net, bottom, 'inception_%s_1x1_2'%stage,num_output=96, kernel_size=1)  # 96x35x35
    conv_3x3_reduce = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_3x3_reduce'%stage, num_output=64, kernel_size=1)  # 64x35x35
    conv_3x3 = factorization_conv_bn_scale_relu(net, conv_3x3_reduce,'inception_%s_3x3'%stage, num_output=96, kernel_size=3, pad=1)  # 96x35x35
    conv_3x3_2_reduce = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_3x3_2_reduce'%stage, num_output=64, kernel_size=1)  # 64x35x35
    conv_3x3_2 = factorization_conv_bn_scale_relu(net, conv_3x3_2_reduce,'inception_%s_3x3_2'%stage, num_output=96, kernel_size=3, pad=1)  # 96x35x35
    conv_3x3_3 = factorization_conv_bn_scale_relu(net, conv_3x3_2,'inception_%s_3x3_3'%stage, num_output=96, kernel_size=3, pad=1)  # 96x35x35

    concat = L.Concat(conv_1x1, conv_1x1_2, conv_3x3, conv_3x3_3)  # 384(96+96+96+96)x35x35

    return concat


def reduction_v4_a(net, bottom, stage):
    """
    input:384x35x35
    output:1024x17x17
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 384x17x17
    net.__dict__['reduction_%s_pool'%stage] = pool
    conv_3x3 = factorization_conv_bn_scale_relu(net, bottom, 'reduction_%s_3x3', num_output=384, kernel_size=3, stride=2)  # 384x17x17

    conv_3x3_2_reduce = factorization_conv_bn_scale_relu(net, bottom, num_output=192, 'reduction_%s_3x3_2_reduce', kernel_size=1)  # 192x35x35
    conv_3x3_2 =  factorization_conv_bn_scale_relu(net, conv_3x3_2_reduce, 'reduction_%s_3x3_2',num_output=224, kernel_size=3, stride=1, pad=1)  # 224x35x35
    conv_3x3_3 =  factorization_conv_bn_scale_relu(net, conv_3x3_2, 'reduction_%s_3x3_3',num_output=256, kernel_size=3, stride=2)  # 256x17x17

    concat = L.Concat(pool, conv_3x3, conv_3x3_3)  # 1024(384+384+256)x17x17

    return concat


def inception_v4_b(net, bottom, stage):
    """
    input:1024x17x17
    output:1024x17x17
    :param bottom: bottom layer
    :return: layers
    """
    pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 1024x17x17
    net.__dict__['inception_%s_pool_ave'%stage] = pool_ave
    conv_1x1 = factorization_conv_bn_scale_relu(net, pool_ave,'inception_%s_1x1', num_output=128, kernel_size=1)  # 128x17x17

    conv_1x1_2 = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x1_2', num_output=384, kernel_size=1)  # 384x17x17

    conv_1x7_reduce = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x7_reduce', num_output=192, kernel_size=1)  # 192x17x17
    conv_1x7 = factorization_conv_mxn(net, conv_1x7_reduce,'inception_%s_1x7', num_output=224, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 224x17x17
    conv_7x1 = factorization_conv_mxn(net, conv_1x7,'inception_%s_7x1', num_output=256, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 256x17x17

    conv_1x7_2_reduce = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x7_2_reduce', num_output=192, kernel_size=1)  # 192x17x17
    conv_1x7_2 = factorization_conv_mxn(net, conv_1x7_2_reduce,'inception_%s_1x7_2', num_output=192, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 192x17x17
    conv_7x1_2 = factorization_conv_mxn(net, conv_1x7_2,'inception_%s_7x1_2', num_output=224, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 224x17x17
    conv_1x7_3 = factorization_conv_mxn(net, conv_7x1_2,'inception_%s_1x7_3', num_output=224, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 224x17x17
    conv_7x1_3 = factorization_conv_mxn(net, conv_1x7_3,'inception_%s_7x1_3', num_output=256, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 256x17x17

    concat = L.Concat(conv_1x1, conv_1x1_2, conv_7x1, conv_7x1_3)  # 1024(128+384+256+256)x17x17

    return concat


def reduction_v4_b(net, bottom, stage):
    """
    input:1024x17x17
    output:1536x8x8
    :param bottom: bottom layer
    :return: layers
    """
    pool = L.Pooling(bottom, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 1024x8x8
    net.__dict__['inception_%s_pool'%stage] = pool
    conv_3x3_reduce = factorization_conv_bn_scale_relu(net, bottom,'reduction_%s_3x3_reduce', num_output=192, kernel_size=1)  # 192x17x17
    conv_3x3 = factorization_conv_bn_scale_relu(net, conv_3x3_reduce,'reduction_%s_3x3', num_output=192, kernel_size=3, stride=2)  # 192x8x8

    conv_1x7_reduce = factorization_conv_bn_scale_relu(net, bottom,'reduction_%s_1x7_reduce', num_output=256, kernel_size=1)  # 256x17x17
    conv_1x7 = factorization_conv_mxn(net, conv_1x7_reduce,'reduction_%s_1x7', num_output=256, kernel_h=1, kernel_w=7, pad_h=0, pad_w=3)  # 256x17x17
    conv_7x1 = factorization_conv_mxn(net, conv_1x7,,'reduction_%s_7x1' num_output=320, kernel_h=7, kernel_w=1, pad_h=3, pad_w=0)  # 320x17x17
    conv_3x3_2 = factorization_conv_bn_scale_relu(net, conv_7x1,'reduction_%s_3x3_2', num_output=320, kernel_size=3, stride=2)  # 320x8x8

    concat = L.Concat(pool, conv_3x3, conv_3x3_2)  # 1536(1024+192+320)x8x8

    return concat


def inception_v4_c(net, bottom, stage):
    """
    input:1536x8x8
    output:1536x8x8
    :param bottom: bottom layer
    :return: layers
    """
    pool_ave = L.Pooling(bottom, kernel_size=3, stride=1, pad=1, pool=P.Pooling.AVE)  # 1536x8x8
    net.__dict__['inception_%s_pool_ave'%stage] = pool_ave
    conv_1x1 = factorization_conv_bn_scale_relu(net, pool_ave,'inception_%s_1x1', num_output=256, kernel_size=1)  # 256x8x8
    conv_1x1_2 = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x1_2', num_output=256, kernel_size=1)  # 256x8x8
    conv_1x1_3 = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x1_3', num_output=384, kernel_size=1)  # 384x8x8
    conv_1x3 = factorization_conv_mxn(net, conv_1x1_3,'inception_%s_1x3', num_output=256, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 256x8x8
    conv_3x1 = factorization_conv_mxn(net, conv_1x1_3,'inception_%s_3x1', num_output=256, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 256x8x8
    conv_1x1_4 = factorization_conv_bn_scale_relu(net, bottom,'inception_%s_1x1_4', num_output=384, kernel_size=1)  # 384x8x8
    conv_1x3_2 = factorization_conv_mxn(net, conv_1x1_4,'inception_%s_1x3_2', num_output=448, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 448x8x8
    conv_3x1_2 = factorization_conv_mxn(net, conv_1x3_2,'inception_%s_3x1_2', num_output=512, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 512x8x8
    conv_1x3_3 = factorization_conv_mxn(net, conv_3x1_2,'inception_%s_1x3_3', num_output=256, kernel_h=1, kernel_w=3, pad_h=0, pad_w=1)  # 256x8x8
    conv_3x1_3 = factorization_conv_mxn(net, conv_3x1_2,'inception_%s_3x1_3', num_output=256, kernel_h=3, kernel_w=1, pad_h=1, pad_w=0)  # 256x8x8

    concat = L.Concat(conv_1x1, conv_1x1_2, conv_1x3, conv_3x1, conv_1x3_3, conv_3x1_3)  # 1536(256+256+256+256+256+256)x17x17

    return concat

def resnext_block(net, bottom, stage, base_output=64, card=32, is_match=False):
    """
    input:4*base_output x n x n
    output:4*base_output x n x n
    :param base_output: base num_output of branch2
    :param bottom: bottom layer
    :return: layers

    Args:
        card:
    """
    conv1 = factorization_conv_bn_scale_relu(net, bottom, 'resx%s_conv_1'%stage, num_output=base_output * (card / 16), kernel_size=1) 
    conv2 = factorization_conv_bn_scale_relu(net, conv1, 'resx%s_conv_2'%stage, num_output=base_output * (card / 16), kernel_size=3) 
    conv3 = factorization_conv_bn_scale_relu(net, conv2, 'resx%s_conv_3'%stage, num_output=base_output * 4, kernel_size=1, relu=False) 

    if is_match: 
      match = factorization_conv_bn_scale_relu(net, bottom, 'resx%s_match_conv'%stage, num_output=base_output * 4, kernel_size=1, relu=False) 
    else:
      match = bottom
    net.eltwise = L.Eltwise(match, conv3, eltwise_param=dict(operation=1))
    net.eltwise_relu = L.ReLU(eltwise, in_place=True)

    return net.eltwise
