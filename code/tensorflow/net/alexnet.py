# -*- coding: UTF-8 -*-
import tensorflow as tf
from .network import Network

class alexnet(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.label = tf.placeholder(tf.float32, shape=[None, 1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'label':self.label})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, name='conv1')
             .lrn(5, 0.0001, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .lrn(5, 0.0001, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, name='pool2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name='drop6')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='drop7')
             .fc(1000, relu=False, name='fc8')
             .softmax(name='prob'))

    def build_loss():
        predict = self.get_output('prob')
        label = self.get_output('label')
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=label))
        