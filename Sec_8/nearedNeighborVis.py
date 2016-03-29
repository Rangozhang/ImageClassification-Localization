import numpy as np
import os, sys
from PIL import Image

caffe_root = '/home/yuz4/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def featureDistance(f1, f2):
	return np.linalg.norm(f1-f2)

selectedImages = []
selectedImages.append('/scratch/16824/data/crop_imgs/12/COCO_train2014_000000317534_1.jpg') #[2]
selectedImages.append('/scratch/16824/data/crop_imgs/1/COCO_train2014_000000004131_2.jpg') #[0]
selectedImages.append('/scratch/16824/data/crop_imgs/24/COCO_train2014_000000161762_1.jpg') #[1]

featureLayer = 'fc7'

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net('loc_test.prototxt',
                'models/localization.caffemodel',
                caffe.TEST)

test_listfile = '/scratch/16824/data/testlist_both.txt'
test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
test_list_filename = []
test_list_feature = []

data_counts = len(test_list)
batch_size = net.blobs['data'].data.shape[0]
batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
test_list_distance = np.zeros(data_counts)

print("Extracting image features from test list...")
for i in range(batch_count):
	out = net.forward()
	#print(net.blobs[featureLayer].data.shape)
	for j in range(batch_size):
		id = i * batch_size + j
		if id >= data_counts:
			break

		line = test_list[id].split(' ')
		test_list_filename.append(line[0])		

		# is the feature in data or the weight? TODO: need to be clarified
		prop = net.blobs[featureLayer].data[j]
		test_list_feature.append(np.copy(prop))

print("Computing distance between selected and test images...")
#TODO: transformer a image and feed in the network
net.blobs['data'].reshape(3, 3, 227, 227)

transformer = caffe.io.Transformer({'data': (3, 3, 227, 227)})#net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 104, 104]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

image_batch = np.zeros((3, 3, 227, 227))

for i in [2,1,0]:
	im = caffe.io.load_image(selectedImages[i])
	im = caffe.io.resize_image(im, (256, 256))

	#center croping
	center = np.array((256, 256)) / 2.0
	croped = np.tile(center, np.array([1, 2]))[0] + np.concatenate([
		-np.array([227, 227])/ 2.0, 
		np.array([227, 227])/ 2.0])
	im = im[croped[0]:croped[2], croped[1]:croped[3], :]
	image_batch[i] = np.copy(transformer.preprocess('data', im))

net.blobs['data'].data[...] = image_batch
out = net.forward(start='conv1')
		
for i in range(0, 3):
	test_list_distance.fill(0)

	feature = net.blobs[featureLayer].data[i]
	print "feature"
	print(feature.sum())
	#TODO: compute distnace
	for j in range(0, data_counts):
		#print("-"*10)
		#print(feature.sum())
		#print(test_list_feature[j].sum())
		test_list_distance[j] = featureDistance(feature, test_list_feature[j])
		#print(test_list_distance[j])

	#TODO: argmanx first 10
	first10 = np.array(test_list_distance).argsort()[:10] #[-10:][::-1]
	for ind in first10:
		#print(ind)
		print(test_list_filename[ind])
		#print(test_list_distance[ind])
		#print(test_list_feature[ind].sum())
		#print(feature.sum())
		#print(test_list_feature[ind])
		#print(feature)
		#print("-"*20)
	print('\n')
