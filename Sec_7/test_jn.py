import numpy as np
import os, sys
from PIL import Image

# Make sure that caffe is on the python path:
caffe_root = '/home/yuz4/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/scratch/16824/data/testlist_both.txt'
result_file = 'joint_results.txt'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/home/yuz4/Sec_7/test.prototxt',
                '/home/yuz4/Sec_7/models/model_iter_40000.caffemodel',
                caffe.TEST)

test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
accuracy = 0

f = open(result_file, 'w')
print(batch_count)
for i in range(batch_count):

	out = net.forward()
	print(out)
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break

		line = test_list[id].split(' ')
		lbl = []

		fname =	line[0]
		lbl_cls = int(line[1])
		for ii in range(2,6):
			lbl.append(int(line[ii]))
		
		with Image.open(os.path.join('/scratch/16824/data/crop_imgs',fname)) as im:
			width, height = im.size
		
		prop = net.blobs['fc8_reg'].data[j]
		prop_cls = out['cls_loss'][j].argmax()

		f.write(fname)
		f.write('{0: d}{1: f}{2: f}{3: f}{4: f}'.format(prop_cls, prop[0]*width, prop[1]*height, prop[2]*width, prop[3]*height))
		f.write('\n')
f.close()
