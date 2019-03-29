WORK_DIR=$(pwd)
cd ..
ROOT_DIR=$(pwd)
cd $WORK_DIR

CAFFE_DIR="${ROOT_DIR}"/deeplab-v2
PYCAFFE_DIR="${CAFFE_DIR}"/python
CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe.bin

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

CMD="${CAFFE_BIN} train --solver=tests_layer/solver.prototxt --gpu=0"
echo Running ${CMD} && ${CMD}

