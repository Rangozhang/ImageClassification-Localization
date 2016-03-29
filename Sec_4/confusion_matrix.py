import os, sys
import numpy as np

def get_confusion_matrix(gt_file, ts_file, num_class):
	conf_mat = np.zeros((num_class, num_class), dtype=float)
   	gt_list = np.loadtxt(gt_file, str, comments=None, delimiter='\n')
	ts_list = np.loadtxt(ts_file, str, comments=None, delimiter='\n')
	lens = len(gt_list)
	for i in xrange(lens):
		ts = int(ts_list[i].split(' ')[1])
		gt = int(gt_list[i].split(' ')[1])
		conf_mat[gt][ts] += 1
        conf_mat = conf_mat/conf_mat.sum(1)[np.newaxis].T
	return conf_mat

if __name__ == '__main__':
	conf_mat = get_confusion_matrix('/scratch/16824/data/testlist_class.txt', 'cls_results.txt', 30)
	for row in conf_mat:
		for col in row:
			print(str(np.around(col,decimals=3)) + ' '),
		print('\n')
	i = 0
	for row in conf_mat:
		sorted = row.argsort()[-4:][::-1]
		print(str(i) + "&"),
		print(sorted),
		print('&'),
		print(np.around(row[sorted], decimals=2)),
		print("\\\\")
		i = i+1
