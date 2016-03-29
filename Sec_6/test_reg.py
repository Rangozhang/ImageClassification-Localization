import numpy as np
import os, sys
from PIL import Image

# Make sure that caffe is on the python path:
caffe_root = '/home/yuz4/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/scratch/16824/data/testlist_bbox.txt'
result_file = 'bbox_results.txt'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('/home/yuz4/Sec_6/test.prototxt',
                '/home/yuz4/Sec_6/models/model_iter_60000.caffemodel',
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
		for ii in range(1,5):
			lbl.append(int(line[ii]))
		fname = line[0]
		
		with Image.open(os.path.join('/scratch/16824/data/crop_imgs',fname)) as im:
			width, height = im.size
		
		prop = net.blobs['fc8_reg'].data[j]
		#blob_prop = net.blobs['fc8_reg']
		#prop = np.array( caffe.io.blobproto_to_array(blob_prop) )
		print(prop)


		f.write(fname)
		f.write('{0: f}{1: f}{2: f}{3: f}'.format(prop[0]*width, prop[1]*height, prop[2]*width, prop[3]*height))
		f.write('\n')
f.close()
