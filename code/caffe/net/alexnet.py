# -*- coding: UTF-8 -*-
import caffe
from caffe import layers as L, params as P
from layers import *

def alexnet(train_data=None, val_data=None, mean_file=None, train_batch_size=None, test_batch_size=None, backend=None, deploy=False):
  """
  Generate the caffe's network specification train_val.prototxt file for Alexnet model,
  described in the [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) publication.
  """
  if deploy == False:
      if train_data is None or val_data is None or train_batch_size is None or test_batch_size is None or backend:
        raise Exception("Invalide input!")

  n = caffe.NetSpec()
  
  n.data,n.label,n.test_data = input_data_layer('data',
                                    crop_size=227,
                                    source=[train_data,val_data],
                                    backend=backend,
                                    batch_size=[train_batch_size,test_batch_size],
                                    mean_file=mean_file, 
                                    deploy=deploy)
    
  n.conv1,n.relu1,n.norm1 = conv_relu_norm(n.data, num_output=96,  kernel_size=11, stride=4,
                            local_size=5, alpha=0.0001, beta=0.75
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0),
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.pool1 = L.Pooling(n.norm1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
  
  n.conv2,n.relu2,n.norm2 = conv_relu(n.pool1, num_output=256, pad=2, kernel_size=5,
                            local_size=5, alpha=0.0001, beta=0.75
                            group=2,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0.1),
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.pool2 = max_pool(n.norm2, kernel_size=3, stride=2)
 
  n.conv3,n.relu3 = conv_relu(n.pool2, num_output=384, pad=1, kernel_size=3,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0),
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
 
  n.conv4,n.relu4 = conv_relu(n.relu3, num_output=384, pad=1, kernel_size=3,
                            group=2,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0.1),
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

  n.conv5,n.relu5 = conv_relu(n.relu4, num_output=256, pad=1, kernel_size=3,
                            group=2,
                            weight_filler=dict(type='gaussian', std=0.01),
                            bias_filler=dict(type='constant', value=0.1),
                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.pool5 = max_pool(n.relu5, kernel_size=3, stride=2)
 
  n.fc6,n.relu6 = fc_relu(n.pool5, num_output=4096,
                           weight_filler=dict(type='gaussian', std=0.005),
                           bias_filler=dict(type='constant', value=0.1),
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.drop6 = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5)
 
  n.fc7,n.relu7 = fc_relu(n.drop6, name='fc7', num_output=4096,
                           weight_filler=dict(type='gaussian', std=0.005),
                           bias_filler=dict(type='constant', value=0.1),
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  n.drop7 = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5)
  
  n.fc8 = L.InnerProduct(n.drop7, name='fc8', num_output=1000,
                           weight_filler=dict(type='gaussian', std=0.01),
                           bias_filler=dict(type='constant', value=0),
                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
  if deploy == False:
    n.accuracy = L.Accuracy(n.fc8, n.label, include=dict(phase=caffe.TEST))
    n.loss = L.SoftmaxWithLoss(n.fc8, n.label)
  else:
    n.prob  = L.Softmax(n.fc8)
    
  return  'name: "AlexNet"\n' + str(n.to_proto())
 
if __name__ == '__main__':  
  print alexnet('./examples/imagenet/ilsvrc12_train_lmdb', './examples/imagenet/ilsvrc12_val_lmdb','data/ilsvrc12/imagenet_mean.binaryproto', 256, 50, P.Data.LMDB)
  print alexnet(deploy=True)
  
  