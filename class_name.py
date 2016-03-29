import numpy as np

x = np.loadtxt('COCO_class_name_id.txt',str, comments=None,delimiter=' ')
f = open('COCO_id_name.txt', 'w')
for i in range(x.size):
	f.write(str(i))
	f.write(' ')
	f.write(x[i])
	f.write('\n')
