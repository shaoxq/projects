# -*- coding: UTF-8 -*-
def caffe_write_net(proto_string, prototxt_file):
    #写入prototxt文件
    with open(prototxt_file, 'w') as f:
        f.write(str(proto_string)