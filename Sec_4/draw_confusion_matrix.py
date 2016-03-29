import os, sys
#sys.path.insert(0, '../')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from confusion_matrix import get_confusion_matrix

conf_mat = get_confusion_matrix('/scratch/16824/data/testlist_class.txt', 'cls_results.txt', 30)

sum_per_row = np.sum(conf_mat, axis=1)
conf_mat = (0.0 + conf_mat) / sum_per_row[:, np.newaxis]
#print " ".join(str(ele) for ele in np.sum(conf_mat, axis=1))

plt.matshow(conf_mat)
#plt.show()
plt.xticks(np.arange(0,30,5))
plt.yticks(np.arange(0,30,5))

plt.savefig("test.png")

