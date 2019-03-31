import sys; sys.path.insert(0, "/home/davidgj/projects_v2/caffe-segnet-cudnn5/python")
import caffe

num_times = int(sys.argv[1])
caffe.set_mode_gpu()
solver = caffe.SGDSolver('tests_layer/solver.prototxt')
net = solver.net
for _ in range(num_times):
	net.forward()
