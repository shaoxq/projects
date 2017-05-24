# coding:utf-8
import caffe
from caffe import layers as L, params as P
from layers import * 
 
def _resnet_block(name, n, bottom, nout, branch1=False, initial_stride=2):
    '''Basic ResNet block.
    '''
    if branch1:
        res_b1 = 'res{}_branch1'.format(name)
        bn_b1 = 'bn{}_branch1'.format(name)
        scale_b1 = 'scale{}_branch1'.format(name)
        n[res_b1], n[bn_b1], n[scale_b1] = _conv_bn_scale(
            bottom, 4*nout, kernel_size=1, stride=initial_stride, pad=0)
    else:
        initial_stride = 1
 
    res = 'res{}_branch2a'.format(name)
    bn = 'bn{}_branch2a'.format(name)
    scale = 'scale{}_branch2a'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        bottom, nout, kernel_size=1, stride=initial_stride, pad=0)
    relu2a = 'res{}_branch2a_relu'.format(name)
    n[relu2a] = L.ReLU(n[scale], in_place=True)
    res = 'res{}_branch2b'.format(name)
    bn = 'bn{}_branch2b'.format(name)
    scale = 'scale{}_branch2b'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2a], nout, kernel_size=3, stride=1, pad=1)
    relu2b = 'res{}_branch2b_relu'.format(name)
    n[relu2b] = L.ReLU(n[scale], in_place=True)
    res = 'res{}_branch2c'.format(name)
    bn = 'bn{}_branch2c'.format(name)
    scale = 'scale{}_branch2c'.format(name)
    n[res], n[bn], n[scale] = _conv_bn_scale(
        n[relu2b], 4*nout, kernel_size=1, stride=1, pad=0)
    res = 'res{}'.format(name)
    if branch1:
        n[res] = L.Eltwise(n[scale_b1], n[scale])
    else:
        n[res] = L.Eltwise(bottom, n[scale])
    relu = 'res{}_relu'.format(name)
    n[relu] = L.ReLU(n[res], in_place=True)

def resnet50(train_data, val_data, mean_file, train_batch_size, test_batch_size, backend, deploy=False):
    '''ResNet 50 layers architecture.
    '''  
    n = caffe.NetSpec()
    n.data,n.label,n.test_data = input_data_layer('data',
                                    crop_size=224,
                                    source=[train_data,val_data],
                                    backend=backend,
                                    batch_size=[train_batch_size,test_batch_size],
                                    deploy=deploy)
    n.conv1, n.bn_conv1, n.scale_conv1 = _conv_bn_scale(
        n.data, 64, bias_term=True, kernel_size=7, pad=3, stride=2)
    n.conv1_relu = L.ReLU(n.scale_conv1)
    n.pool1 = L.Pooling(
        n.conv1_relu, kernel_size=3, stride=2, pool=P.Pooling.MAX)
 
    _resnet_block('2a', n, n.pool1, 64, branch1=True, initial_stride=1)
    _resnet_block('2b', n, n.res2a_relu, 64)
    _resnet_block('2c', n, n.res2b_relu, 64)
 
    _resnet_block('3a', n, n.res2c_relu, 128, branch1=True)
    _resnet_block('3b', n, n.res3a_relu, 128)
    _resnet_block('3c', n, n.res3b_relu, 128)
    _resnet_block('3d', n, n.res3c_relu, 128)
 
    _resnet_block('4a', n, n.res3d_relu, 256, branch1=True)
    _resnet_block('4b', n, n.res4a_relu, 256)
    _resnet_block('4c', n, n.res4b_relu, 256)
    _resnet_block('4d', n, n.res4c_relu, 256)
    _resnet_block('4e', n, n.res4d_relu, 256)
    _resnet_block('4f', n, n.res4e_relu, 256)
 
    _resnet_block('5a', n, n.res4f_relu, 512, branch1=True)
    _resnet_block('5b', n, n.res5a_relu, 512)
    _resnet_block('5c', n, n.res5b_relu, 512)
 
    n.pool5 = L.Pooling(
        n.res5c_relu, kernel_size=7, stride=1, pool=P.Pooling.AVE)

    n.fc1000 = L.InnerProduct(n.pool5, num_output=1000)

    if deploy == False:
      n.accuracy = L.Accuracy(n.fc1000, n.label, include=dict(phase=caffe.TEST))
      n.loss = L.SoftmaxWithLoss(n.fc1000, n.label)
    else:
      n.prob  = L.Softmax(n.fc1000)
    
    return  'name: "ResNet50"\n' + str(n.to_proto())

if __name__ == '__main__':

    print resnet50('./examples/imagenet/ilsvrc12_train_lmdb', './examples/imagenet/ilsvrc12_val_lmdb','data/ilsvrc12/imagenet_mean.binaryproto', 256, 50, P.Data.LMDB)
    print resnet50('', '','', '', '', '')