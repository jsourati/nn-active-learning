import sys
import pickle
import pdb
import numpy as np

import NNAL
import NNAL_tools
import pdb
from NNAL_tools import test_training_part

exp_id = sys.argv[1]
run_id = sys.argv[2]
data_path = sys.argv[3]
target_classes_path = sys.argv[4]
# string '_init_t.ckpt' will be added to model_save_path
# where `t` is the index of experiment
model_save_path = sys.argv[5]
results_save_path = '%s/%s/results.dat'% (exp_id, run_id)

# directory of the folder where indices are to be saved
index_save_path = '%s/%s'% (exp_id, run_id)

print("Loading data..")
# ------------------------------------
if target_classes_path=='NA':
    print("Using full data set..")
    dat = pickle.load(open(data_path,'rb'))
    feats = dat[0]
    labels = dat[1]
else:
    categories_path = \
        "/common/projects2/jamshid/Caltech-101/101_ObjectCategories/"
    feats,labels = NNAL_tools.filter_classes(
        categories_path, data_path, target_classes_path)

# convert the labels into one-hot vectors
c = len(np.unique(np.array(labels)))
hot_labels = np.zeros((len(labels), c))
for i in range(len(labels)):
    hot_labels[i,labels[i]] = 1.


train_inds, test_inds = test_training_part(hot_labels.T, .3)
# saving train and test indices
np.savetxt('%s/train_inds.txt'% index_save_path, train_inds, fmt='%d')
np.savetxt('%s/test_inds.txt'% index_save_path, test_inds, fmt='%d')

pool_CalDat = feats[train_inds,:,:,:]
pool_CalLabels = hot_labels[train_inds,:]
test_CalDat = feats[test_inds,:,:,:]
test_CalLabels = hot_labels[test_inds,:]

print("Starting the querying experiments..")
# -------------------------------------------
# learning parameters
epochs = 10
learning_rate = 1e-3
dropout_rate = 0.5
train_batch_size = 50
eval_batch_size = 100

# querying parameters
k = int(sys.argv[6])
B = 100
max_queries = int(sys.argv[7])

methods = ['random', 'entropy', 'rep-entropy', 'fi']
#methods = ['fi']

# querying
# --------
# get some initial data set
npool = pool_CalDat.shape[0]
init_size = int(sys.argv[8])
rand_inds = np.random.permutation(npool)[:init_size]
init_train_dat = [pool_CalDat[rand_inds,:,:], 
                  pool_CalLabels[rand_inds,:]]
# the pool
pool_inds = list(set(np.arange(npool)) - set(rand_inds))
pool_CalDat = pool_CalDat[pool_inds,:,:,:]
pool_CalLabels = pool_CalLabels[pool_inds,:]

# these indices are based on rows of the "train_inds.txt"
np.savetxt('%s/init_train_inds.txt'% index_save_path, rand_inds, fmt='%d')
np.savetxt('%s/pool_inds.txt'% index_save_path, pool_inds, fmt='%d')

accs_t, fi_queries_t = NNAL.run_AlexNet_AL(
    pool_CalDat, pool_CalLabels, test_CalDat, test_CalLabels, 
    learning_rate, dropout_rate, epochs, 
    k, B, methods, max_queries, 
    train_batch_size, 
    model_save_path, 
    results_save_path,
    index_save_path,
    eval_batch_size, 
    init_train_dat
    )


