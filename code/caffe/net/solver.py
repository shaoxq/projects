# -*- coding: UTF-8 -*-
import caffe
from google.protobuf import text_format

def prob_solver(net_prototxt, train_sample_num, test_sample_num, snapshot_prefix):
 
  train_epoch = 10
  test_time_per_epoch = 10
  saving_per_epoch = 3

  net = caffe.proto.caffe_pb2.NetParameter()
  text_format.Merge(open(net_prototxt).read(), net)
  layers = net.layer
  train_batch_size=None
  val_batch_size=None
  for layer in layers:
    if layer.type == 'Data':
#      print 'name: "%s"'%layer.name
      if layer.include[0].phase == caffe.TRAIN:
        train_batch_size = layer.data_param.batch_size
      else:
        val_batch_size = layer.data_param.batch_size
  if train_batch_size is None or val_batch_size is None:
    raise exception("can't find batch_size from Input Data Layer.")
   
  epoch = train_sample_num / train_batch_size
  solver = caffe.proto.caffe_pb2.SolverParameter()
  
  solver.net = net_prototxt
  solver.test_iter.append(test_sample_num / val_batch_size)
  solver.test_interval= epoch / test_time_per_epoch
  solver.lr_policy = 'step'
  solver.base_lr = 0.005
  solver.gamma = 0.5
  solver.stepsize = epoch
  solver.display = epoch / test_time_per_epoch
  solver.max_iter= train_epoch * epoch
  solver.momentum = 0.95
  solver.weight_decay = 0.0005
  solver.snapshot = epoch / saving_per_epoch
  solver.snapshot_prefix = snapshot_prefix  
  solver.solver_mode = caffe.proto.caffe_pb2.SolverParameter.GPU
  solver.type = 'SGD'
  
  return str(solver)
