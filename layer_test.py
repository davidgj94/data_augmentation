import sys; sys.path.insert(0, "/home/davidgj/projects/deeplab-v2/python")
import caffe

caffe.set_mode_gpu()
solver = caffe.SGDSolver('tests_layer/solver.prototxt')
net = solver.net
net.forward()