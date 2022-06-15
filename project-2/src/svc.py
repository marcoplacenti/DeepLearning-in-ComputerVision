import os
import numpy as np
from sklearn import svm
from numpy import savez_compressed

mustache = os.listdir('./data/samples/projections/projections/mustache/')
no_mustache = os.listdir('./data/samples/projections/projections/no_mustache/')

agg_codes = []
agg_labels = []
for dir in mustache:
    codes = np.load('./data/samples/projections/projections/mustache/'+dir+'/projected_w.npz')
    codes = codes['w']
    agg_codes.append(codes.flatten())
    agg_labels.append(0)

for dir in no_mustache:
    codes = np.load('./data/samples/projections/projections/no_mustache/'+dir+'/projected_w.npz')
    codes = codes['w']
    agg_codes.append(codes.flatten())
    agg_labels.append(1)

agg_codes = np.array(agg_codes)

clf = svm.SVC(kernel='linear')
clf.fit(agg_codes, agg_labels)

weights = clf.coef_.reshape(1, 18, 512)

savez_compressed('./data/samples/projections/mustache_directions.npz', w=weights)
