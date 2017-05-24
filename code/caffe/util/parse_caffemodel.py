# coding:utf-8
# description: 从caffemodel文件解析出网络训练信息，以类似train.prototxt的形式输出到屏幕

import _init_paths
import caffe.proto.caffe_pb2 as caffe_pb2

def convert_caffemodel_to_prototxt(caffemodel_filename):

  model = caffe_pb2.NetParameter()
  # load model from binary file
  f=open(caffemodel_filename, 'rb')
  model.ParseFromString(f.read())
  f.close()
  
  layers = model.layer
  #print 'name: "%s"'%model.name
  proto_string = 'name: "%s"\n'%model.name
  for layer in layers:
    #print 'layer {'
    proto_string += 'layer {\n'
    #print '  name: "%s"'%layer.name
    proto_string += '  name: "%s"\n'%layer.name
    #print '  type: "%s"'%layer.type
    proto_string +='  type: "%s"\n'%layer.type
    
    tops = layer.top
    for top in tops:
        #print '  top: "%s"'%top
        proto_string += '  top: "%s"\n'%top
    
    bottoms = layer.bottom
    for bottom in bottoms:
        #print '  bottom: "%s"'%bottom
        proto_string += '  bottom: "%s"\n'%bottom
    
    if len(layer.include)>0:
        #print '  include {'
        proto_string += '  include {\n'
        includes = layer.include
        phase_mapper={
            '0': 'TRAIN',
            '1': 'TEST'
        }
        
        for include in includes:
            if include.phase is not None:
                #print '    phase: %s\n'%phase_mapper[str(include.phase)]
                proto_string +=  '    phase: %s\n'%phase_mapper[str(include.phase)]
        #print '  }'
        proto_string += '  }\n'
    
    if layer.transform_param is not None and layer.transform_param.scale is not None and layer.transform_param.scale!=1:
        #print '  transform_param {'
        proto_string += '  transform_param {\n'
        #print '    scale: %s'%layer.transform_param.scale
        proto_string += '    scale: %s\n'%layer.transform_param.scale
        #print '  }'
        proto_string += '  }\n'

    if layer.data_param is not None and (layer.data_param.source!="" or layer.data_param.batch_size!=0 or layer.data_param.backend!=0):
        #print '  data_param: {'
        proto_string += '  data_param: {\n'
        if layer.data_param.source is not None:
            proto_string += '    source: "%s"\n'%layer.data_param.source
        if layer.data_param.batch_size is not None:
            #print '    batch_size: %d'%layer.data_param.batch_size
            proto_string += '    batch_size: %d\n'%layer.data_param.batch_size
        if layer.data_param.backend is not None:
            #print '    backend: %s'%layer.data_param.backend
            proto_string += '    backend: %s\n'%layer.data_param.backend
        #print '  }'
        proto_string += '  }\n'
        
    if layer.param is not None:
        params = layer.param
        for param in params:
            #print '  param {'
            proto_string += '  param {\n'
            if param.lr_mult is not None:
                #print '    lr_mult: %s'% param.lr_mult
                proto_string += '    lr_mult: %s\n'% param.lr_mult
            #print '  }'
            proto_string += '  }\n'
    
    if layer.convolution_param is not None:
        #print '  convolution_param {'
        proto_string += '  convolution_param {\n'
        conv_param = layer.convolution_param
        if conv_param.num_output is not None:
            #print '    num_output: %d'%conv_param.num_output
            proto_string += '    num_output: %d\n'%conv_param.num_output
        if len(conv_param.kernel_size) > 0:
            for kernel_size in conv_param.kernel_size:
                #print '    kernel_size: ',kernel_size
                proto_string += '    kernel_size: %s\n'%kernel_size
        if len(conv_param.stride) > 0:
            for stride in conv_param.stride:
                #print '    stride: ', stride
                proto_string += '    stride: %s\n'% stride
        if conv_param.weight_filler is not None:
            #print '    weight_filler {'
            proto_string += '    weight_filler {\n'
            #print '      type: "%s"'%conv_param.weight_filler.type
            proto_string += '      type: "%s"\n'%conv_param.weight_filler.type
            #print '    }'
            proto_string += '    }\n'
        if conv_param.bias_filler is not None:
            #print '    bias_filler {'
            proto_string += '    bias_filler {\n'
            #print '      type: "%s"'%conv_param.bias_filler.type
            proto_string += '      type: "%s"\n'%conv_param.bias_filler.type
            #print '    }'
            proto_string += '    }\n'
        #print '  }'
        proto_string += '  }\n'
    
    #print '}'
    proto_string += '}\n'
  return proto_string