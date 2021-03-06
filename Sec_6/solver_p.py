from __future__ import division
import sys

caffe_root = '/home/yuz4/caffe/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('/scratch/16824/models/bvlc_reference_caffenet.caffemodel')

niter = 60000
train_loss = np.zeros(niter)

f = open('log.txt', 'w')

for it in range(niter): 
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
    f.write('{0: f}\n'.format(train_loss[it]))

f.close()

# solver.step(80000)
