# -*- coding: UTF-8 -*-
import caffe
from caffe import layers as L, params as P
from layers import *

def inception_v1(train_data=None, val_data=None, mean_file=None, train_batch_size=None, test_batch_size=None, backend=None, deploy=False):
  
  if deploy == False:
      if train_data is None or val_data is None or train_batch_size is None or test_batch_size is None or backend:
        raise Exception("Invalide input!")

  n = caffe.NetSpec()
  
  n.deploy = False
  
  n.data,n.label,n.test_data = input_data_layer('data',
                                    crop_size=227,
                                    source=[train_data,val_data],
                                    backend=backend,
                                    batch_size=[train_batch_size,test_batch_size],
                                    deploy=deploy)
  n.conv1_7x7_s2 = L.Convolution(n.data, num_output=64, kernel_size=7, stride=2, pad=3,
                                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                       weight_filler=dict(type='xavier', weight_std=1),
                                       bias_filler=dict(type='constant', value=0.2))
  n.conv1_relu_7x7 = L.ReLU(n.conv1_7x7_s2, in_place=True)
  n.pool1_3x3_s2 = L.Pooling(n.conv1_7x7_s2, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)
  n.pool1_norm1 = L.LRN(n.pool1_3x3_s2, local_size=5, alpha=1e-4, beta=0.75)

  n.conv2_3x3_reduce = L.Convolution(n.pool1_norm1, kernel_size=1, num_output=64, stride=1,
                                           param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                           weight_filler=dict(type='xavier', weight_std=1),
                                           bias_filler=dict(type='constant', value=0.2))
  n.conv2_relu_3x3_reduce = L.ReLU(n.conv2_3x3_reduce, in_place=True)

  n.conv2_3x3 = L.Convolution(n.conv2_3x3_reduce, num_output=192, kernel_size=3, stride=1, pad=1,
                                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                    weight_filler=dict(type='xavier', weight_std=1),
                                    bias_filler=dict(type='constant', value=0.2))
  n.conv2_relu_3x3 = L.ReLU(n.conv2_3x3, in_place=True)
  n.conv2_norm2 = L.LRN(n.conv2_3x3, local_size=5, alpha=1e-4, beta=0.75)
  n.pool2_3x3_s2 = L.Pooling(n.conv2_norm2, kernel_size=3, stride=1, pad=1, pool=P.Pooling.MAX)

  n.inception_3a_output = inception(n, n.pool2_3x3_s2, '3a', dict(conv_1x1=64, conv_3x3_reduce=96, conv_3x3=128, conv_5x5_reduce=16,
                                           conv_5x5=32, pool_proj=32))
  n.inception_3b_output = inception(n, n.inception_3a_output, '3b', dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=192, conv_5x5_reduce=32,
                                                  conv_5x5=96, pool_proj=64))
  n.pool3_3x3_s2 = L.Pooling(n.inception_3b_output, kernel_size=3, stride=2, pool=P.Pooling.MAX)
  n.inception_4a_output = inception(n, n.pool3_3x3_s2, '4a', dict(conv_1x1=192, conv_3x3_reduce=96, conv_3x3=208, conv_5x5_reduce=16,
                                           conv_5x5=48, pool_proj=64))
  if deploy == False:
    # loss 1
    n.loss1_ave_pool = L.Pooling(n.inception_4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
    n.loss1_conv = L.Convolution(n.loss1_ave_pool, num_output=128, kernel_size=1, stride=1,
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                     weight_filler=dict(type='xavier', weight_std=1),
                                     bias_filler=dict(type='constant', value=0.2))
    n.loss1_relu_conv = L.ReLU(n.loss1_conv, in_place=True)
    n.loss1_fc, n.loss1_relu_fc, n.loss1_drop_fc = \
            fc_relu_drop(n.loss1_conv, dict(num_output=1024, weight_type='xavier', weight_std=1, bias_type='constant',
                                            bias_value=0.2), dropout_ratio=0.7)
    n.loss1_classifier = L.InnerProduct(n.loss1_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
    n.loss1_loss = L.SoftmaxWithLoss(n.loss1_classifier, n.label, loss_weight=0.3)
   
    n.loss1_accuracy_top1 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1))
    n.loss1_accuracy_top5 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

  n.inception_4b_output = inception(n, n.inception_4a_output,'4b', dict(conv_1x1=160, conv_3x3_reduce=112, conv_3x3=224, conv_5x5_reduce=24,
                                                  conv_5x5=64, pool_proj=64))
  n.inception_4c_output = inception(n, n.inception_4b_output, '4c', dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=256, conv_5x5_reduce=24,
                                                  conv_5x5=64, pool_proj=64))
  n.inception_4d_output = inception(n, n.inception_4c_output, '4d', dict(conv_1x1=112, conv_3x3_reduce=144, conv_3x3=288, conv_5x5_reduce=32,
                                                  conv_5x5=64, pool_proj=64))
  if deploy == False:
    # loss 2
    n.loss2_ave_pool = L.Pooling(n.inception_4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
    n.loss2_conv = L.Convolution(n.loss2_ave_pool, num_output=128, kernel_size=1, stride=1,
                                     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                     weight_filler=dict(type='xavier', weight_std=1),
                                     bias_filler=dict(type='constant', value=0.2))
    n.loss2_relu_conv = L.ReLU(n.loss2_conv, in_place=True)
    n.loss2_fc, n.loss2_relu_fc, n.loss2_drop_fc = \
            fc_relu_drop(n.loss2_conv, dict(num_output=1024, weight_type='xavier', weight_std=1, bias_type='constant',
                                            bias_value=0.2), dropout_ratio=0.7)
    n.loss2_classifier = L.InnerProduct(n.loss2_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
    n.loss2_loss = L.SoftmaxWithLoss(n.loss2_classifier, n.label, loss_weight=0.3)
    n.loss2_accuracy_top1 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1))
    n.loss2_accuracy_top5 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

  n.inception_4e_output = inception(n, n.inception_4d_output,, '4e', dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320, conv_5x5_reduce=32,
                                                  conv_5x5=128, pool_proj=128))
  n.inception_5a_output = inception(n, n.pool4_3x3_s2, '5a', dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320, conv_5x5_reduce=32,
                                           conv_5x5=128, pool_proj=128))
  n.inception_5b_output = inception(n, n.inception_5a_output, '5b',  dict(conv_1x1=384, conv_3x3_reduce=192, conv_3x3=384, conv_5x5_reduce=48,
                                                  conv_5x5=128, pool_proj=128))
  n.pool5_7x7_s1 = L.Pooling(n.inception_5b_output, kernel_size=7, stride=1, pool=P.Pooling.AVE)
  n.pool5_drop_7x7_s1 = L.Dropout(n.pool5_7x7_s1, in_place=True,
                                        dropout_param=dict(dropout_ratio=0.4))
  n.loss3_classifier = L.InnerProduct(n.pool5_7x7_s1, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
  
  if deploy == False:
    n.loss3_loss = L.SoftmaxWithLoss(n.loss3_classifier, n.label, loss_weight=1)
    n.loss3_accuracy_top1 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1))
    n.loss3_accuracy_top5 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1),
                                            accuracy_param=dict(top_k=5))
  else:
     n.prob  = L.Softmax(n.loss3_classifier)
    
  return  'name: "GooLeNet"\n' + str(n.to_proto())

def inception_v2(train_data=None, val_data=None, mean_file=None, train_batch_size=None, test_batch_size=None, backend=None, deploy=False):
  
  if deploy == False:
      if train_data is None or val_data is None or train_batch_size is None or test_batch_size is None or backend:
        raise Exception("Invalide input!")

  n = caffe.NetSpec()
  n.deploy = deploy
  
  n.data,n.label,n.test_data = input_data_layer('data',
                                    crop_size=227,
                                    source=[train_data,val_data],
                                    backend=backend,
                                    batch_size=[train_batch_size,test_batch_size],
                                    deploy=deploy)

  n.conv1_7x7_s2, n.conv1_7x7_s2_bn, n.conv1_7x7_s2_scale, n.conv1_7x7_relu = \
            factorization_conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, stride=2, pad=3)
  n.pool1_3x3_s2 = L.Pooling(n.conv1_7x7_s2, kernel_size=3, stride=2, pool=P.Pooling.MAX)

  n.conv2_3x3_reduce, n.conv2_3x3_reduce_bn, n.conv2_3x3_reduce_scale, n.conv2_3x3_reduce_relu = \
            factorization_conv_bn_scale_relu(n.pool1_3x3_s2, num_output=64, kernel_size=1)

  n.conv2_3x3, n.conv2_3x3_bn, n.conv2_3x3_scale, n.conv2_3x3_relu = \
            factorization_conv_bn_scale_relu(n.conv2_3x3_reduce, num_output=192, kernel_size=3, pad=1)
  n.pool2_3x3_s2 = L.Pooling(n.conv2_3x3, kernel_size=3, stride=2, pool=P.Pooling.MAX)

  n.inception_3a_output = inception_bn(n, n.pool2_3x3_s2, '3a', dict(conv_1x1=64, conv_3x3_reduce=96, conv_3x3=128, conv_5x5_reduce=16,
  n.inception_3b_output = inception_bn(n, n.inception_3a_output, '3b', dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=192,
  n.inception_4a_output = inception_bn(n, n.pool3_3x3_s2, '4a', dict(conv_1x1=192, conv_3x3_reduce=96, conv_3x3=208, conv_5x5_reduce=16,
                                              conv_5x5=48, pool_proj=64))
  if deploy == False:
    # loss 1
    n.loss1_ave_pool = L.Pooling(n.inception_4a_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
    n.loss1_conv, n.loss1_conv_bn, n.loss1_conv_scale, n.loss1_relu_conv = \
            factorization_conv_bn_scale_relu(n.loss1_ave_pool, num_output=128, kernel_size=1)
    n.loss1_fc, n.loss1_relu_fc, n.loss1_drop_fc = \
            fc_relu_drop(n.loss1_conv, dict(num_output=1024, weight_type='xavier', weight_std=1,
                                            bias_type='constant', bias_value=0.2), dropout_ratio=0.7)
    n.loss1_classifier = L.InnerProduct(n.loss1_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
    n.loss1_loss = L.SoftmaxWithLoss(n.loss1_classifier, n.label, loss_weight=0.3)
    n.loss1_accuracy_top1 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1))
    n.loss1_accuracy_top5 = L.Accuracy(n.loss1_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

  n.inception_4b_output = inception_bn(n, n.inception_4a_output, '4b', dict(conv_1x1=160, conv_3x3_reduce=112, conv_3x3=224,
  n.inception_4c_output = inception_bn(n, n.inception_4b_output, '4c', dict(conv_1x1=128, conv_3x3_reduce=128, conv_3x3=256,
  n.inception_4d_output = inception_bn(n, n.inception_4c_output, '4d', dict(conv_1x1=112, conv_3x3_reduce=144, conv_3x3=288,
                                                     conv_5x5_reduce=32, conv_5x5=64, pool_proj=64))
  if deploy == False:
    # loss 2
    n.loss2_ave_pool = L.Pooling(n.inception_4d_output, kernel_size=5, stride=3, pool=P.Pooling.AVE)
    n.loss2_conv, n.loss2_conv_bn, n.loss2_conv_scale, n.loss2_relu_conv = \
            factorization_conv_bn_scale_relu(n.loss2_ave_pool, num_output=128, kernel_size=1)
    n.loss2_fc, n.loss2_relu_fc, n.loss2_drop_fc = \
            fc_relu_drop(n.loss2_conv, dict(num_output=1024, weight_type='xavier', weight_std=1,
                                               bias_type='constant', bias_value=0.2), dropout_ratio=0.7)
    n.loss2_classifier = L.InnerProduct(n.loss2_fc, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))
    n.loss2_loss = L.SoftmaxWithLoss(n.loss2_classifier, n.label, loss_weight=0.3)
    n.loss2_accuracy_top1 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1))
    n.loss2_accuracy_top5 = L.Accuracy(n.loss2_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))

  n.inception_4e_output = inception_bn(n, n.inception_4d_output, '4e', dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320,
  n.inception_5a_output = inception_bn(n, n.pool4_3x3_s2, '5a', dict(conv_1x1=256, conv_3x3_reduce=160, conv_3x3=320,
  n.inception_5b_output = inception_bn(n, n.inception_5a_output, '5b', dict(conv_1x1=384, conv_3x3_reduce=192, conv_3x3=384,
                                                     conv_5x5_reduce=48, conv_5x5=128, pool_proj=128))
  n.pool5_7x7_s1 = L.Pooling(n.inception_5b_output, kernel_size=7, stride=1, pool=P.Pooling.AVE)
  n.pool5_drop_7x7_s1 = L.Dropout(n.pool5_7x7_s1, in_place=True,
                                        dropout_param=dict(dropout_ratio=0.4))
  n.loss3_classifier = L.InnerProduct(n.pool5_7x7_s1, num_output=self.classifier_num,
                                            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                            weight_filler=dict(type='xavier'),
                                            bias_filler=dict(type='constant', value=0))

  if deploy == False:
    n.loss3_loss = L.SoftmaxWithLoss(n.loss3_classifier, n.label, loss_weight=1)
    n.loss3_accuracy_top1 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1))
    n.loss3_accuracy_top5 = L.Accuracy(n.loss3_classifier, n.label, include=dict(phase=1),
                                               accuracy_param=dict(top_k=5))
  else:
    n.prob  = L.Softmax(n.loss3_classifier)
       
  return n.to_proto()

def inception_v3(train_data=None, val_data=None, mean_file=None, train_batch_size=None, test_batch_size=None, backend=None, deploy=False):
  
  if deploy == False:
      if train_data is None or val_data is None or train_batch_size is None or test_batch_size is None or backend:
        raise Exception("Invalide input!")

  n = caffe.NetSpec()
  n.deploy = deploy
  n.data,n.label,n.test_data = input_data_layer('data',
                                    crop_size=299,
                                    source=[train_data,val_data],
                                    backend=backend,
                                    batch_size=[train_batch_size,test_batch_size],
                                    deploy=deploy)
  # stem 
  n.pool2_3x3_s2 = stem_v3_299x299(n, n.data)  # 192x35x35
  # inception_v3_a
  n.inception_a1_output = inception_v3_a(n, n.pool2_3x3_s2, 'a1')  # 256x35x35
  n.inception_a2_output = inception_v3_a(n, n.inception_a1_output 'a2', pool_proj_num_output=64)  # 288x35x35
  n.inception_a3_output = inception_v3_a(n, n.inception_a2_output 'a3', pool_proj_num_output=64)  # 288x35x35

  # reduction_v3_a
  n.reduction_a_concat = reduction_v3_a(n.inception_a3_output)  # 768x17x17

  # 4 x inception_v3_b
  n.inception_b1_concat = inception_v3_b(n, n.reduction_a_concat, 'b1', outs=128)  # 768x17x17
  n.inception_b2_concat = inception_v3_b(n, n.inception_b1_concat, 'b2', outs=160)  # 768x17x17
  n.inception_b3_concat = inception_v3_b(n, n.inception_b2_concat, 'b3', outs=160)  # 768x17x17
  n.inception_b4_concat = inception_v3_b(n, n.inception_b3_concat, 'b4', outs=192)  # 768x17x17

  if deploy == False:
    # loss 1
    n.auxiliary_loss_ave_pool = L.Pooling(n.inception_b4_concat, kernel_size=5, stride=3,
                                              pool=P.Pooling.AVE)  # 768x5x5
    n.auxiliary_loss_conv, n.auxiliary_loss_conv_bn, n.auxiliary_loss_conv_scale, n.auxiliary_loss_relu_conv = \
            factorization_conv_bn_scale_relu(n.auxiliary_loss_ave_pool, num_output=128, kernel_size=1)  # 128x1x1
    n.auxiliary_loss_fc = L.InnerProduct(n.auxiliary_loss_conv, num_output=768,
                                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                             weight_filler=dict(type='xavier', std=0.01),
                                             bias_filler=dict(type='constant', value=0))
    n.auxiliary_loss_fc_relu = L.ReLU(n.auxiliary_loss_fc, in_place=True)
    n.auxiliary_loss_classifier = L.InnerProduct(n.auxiliary_loss_fc, num_output=self.classifier_num,
                                                     param=[dict(lr_mult=1, decay_mult=1),
                                                            dict(lr_mult=2, decay_mult=0)],
                                                     weight_filler=dict(type='xavier'),
                                                     bias_filler=dict(type='constant', value=0))
    n.auxiliary_loss = L.SoftmaxWithLoss(n.auxiliary_loss_classifier, n.label, loss_weight=0.4)
    
  # reduction_v3_b
  n.reduction_b_concat = reduction_v3_b(n, n.inception_b4_concat, 'b')  # 1280x8x8

  #  2 x inception_v3_c
  n.inception_c1_concat = inception_v3_c(n, n.reduction_b_concat, 'c1')  # 2048x8x8
  n.inception_c2_concat = inception_v3_c(n, n.inception_c1_concat, 'c2', pool=P.Pooling.MAX)  # 2048x8x8

  n.pool_8x8_s1 = L.Pooling(n.inception_c2_concat, kernel_size=8, pool=P.Pooling.AVE)
  n.pool_8x8_s1_drop = L.Dropout(n.pool_8x8_s1, dropout_param=dict(dropout_ratio=0.2))
  n.classifier = L.InnerProduct(n.pool_8x8_s1_drop, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
    
  if deploy == False:
    # loss 2
    n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
    n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
    n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),
                                         accuracy_param=dict(top_k=5))
  else: 
    n.prob  = L.Softmax(n.classifier)   
  
  return n.to_proto()
  
def inception_v4(train_data=None, val_data=None, mean_file=None, train_batch_size=None, test_batch_size=None, backend=None, deploy=False):  
  if deploy == False:
      if train_data is None or val_data is None or train_batch_size is None or test_batch_size is None or backend:
        raise Exception("Invalide input!")

  n = caffe.NetSpec()
  n.deploy = deploy
  n.data,n.label,n.test_data = input_data_layer('data',
                                    crop_size=299,
                                    source=[train_data,val_data],
                                    backend=backend,
                                    batch_size=[train_batch_size,test_batch_size],
                                    deploy=deploy)
  # stem
   n.inception_stem3 = stem_v4_299x299(n, n.data)  # 384x35x35
  # 4 x inception_a
  n.inception_a1_concat = inception_v4_a(n, bottom, 'a1') # 384x35x35
  n.inception_a2_concat = inception_v4_a(n, n.inception_a0_concat, 'a2') # 384x35x35
  n.inception_a3_concat = inception_v4_a(n, n.inception_a0_concat, 'a3') # 384x35x35
  n.inception_a4_concat = inception_v4_a(n, n.inception_a0_concat, 'a4') # 384x35x35
  # reduction_v4_a
  n.reduction_a_concat = reduction_v4_a(n, n.inception_a4_concat, 'a')  # 1024x17x17

  # 7 x inception_b
  n.inception_b1_concat = inception_v4_b(n, bottom, 'b1')# 1024x17x17
  n.inception_b2_concat = inception_v4_b(n, bottom, 'b2')# 1024x17x17
  n.inception_b3_concat = inception_v4_b(n, bottom, 'b3')# 1024x17x17
  n.inception_b4_concat = inception_v4_b(n, bottom, 'b4')# 1024x17x17
  n.inception_b5_concat = inception_v4_b(n, bottom, 'b5')# 1024x17x17
  n.inception_b6_concat = inception_v4_b(n, bottom, 'b6')# 1024x17x17
  n.inception_b7_concat = inception_v4_b(n, bottom, 'b7')# 1024x17x17

  # reduction_v4_b
  n.reduction_b_concat =  reduction_v4_b(n, n.inception_b7_concat, 'b')  # 1536x8x8

  # 3 x inception_c
  n.inception_c1_concat = inception_v4_c(n, bottom, 'c1') #1536x8x8
  n.inception_c2_concat = inception_v4_c(n, bottom, 'c2') #1536x8x8
  n.inception_c3_concat = inception_v4_c(n, bottom, 'c3') #1536x8x8

  n.pool_8x8_s1 = L.Pooling(n.inception_c3_concat, pool=P.Pooling.AVE, global_pooling=True)  # 1536x1x1
  n.pool_8x8_s1_drop = L.Dropout(n.pool_8x8_s1, dropout_param=dict(dropout_ratio=0.2))
  n.classifier = L.InnerProduct(n.pool_8x8_s1_drop, num_output=self.classifier_num,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))
  if deploy == False:
    n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
    n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
    n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),
                                         accuracy_param=dict(top_k=5))
  else:
    n.prob = L.Softmax(n.classifier)

  return n.to_proto() 
  
if __name__ == '__main__':  

  print inception_v1('./examples/imagenet/ilsvrc12_train_lmdb', './examples/imagenet/ilsvrc12_val_lmdb','data/ilsvrc12/imagenet_mean.binaryproto', 256, 50, P.Data.LMDB)
  print inception_v1(deploy=True)
  
  print inception_v2('./examples/imagenet/ilsvrc12_train_lmdb', './examples/imagenet/ilsvrc12_val_lmdb','data/ilsvrc12/imagenet_mean.binaryproto', 256, 50, P.Data.LMDB)
  print inception_v2(deploy=True)
  
  