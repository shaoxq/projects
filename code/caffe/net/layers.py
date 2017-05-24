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