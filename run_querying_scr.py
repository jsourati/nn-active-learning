import sys
import pickle
import numpy as np

import NNAL
from NNAL_tools import test_training_part

#data_path = sys.argv[1]
# string '_init_t.ckpt' will be added to model_save_path
# where `t` is the index of experiment
#model_save_path = sys.argv[2]
#results_save_path = sys.argv[3]

print("Loading data..")
# ------------------------------------
dat = pickle.load(open(data_path,'rb'))
caltech_10_dat = dat[0]
# convert the labels into one-hot vectors
c = len(np.unique(np.array(dat[1])))
caltech_10_labels = np.zeros((len(dat[1]), c))
for i in range(len(dat[1])):
    caltech_10_labels[i,dat[1][i]] = 1.
    

train_inds, test_inds = test_training_part(caltech_10_labels.T, .3)

train_CalDat = caltech_10_dat[train_inds,:,:,:]
train_CalLabels = caltech_10_labels[train_inds,:]
test_CalDat = caltech_10_dat[test_inds,:,:,:]
test_CalLabels = caltech_10_labels[test_inds,:]

print("Starting the querying experiments..")
# -------------------------------------------
# learning parameters
epochs = 10
learning_rate = 1e-3
dropout_rate = 0.5
train_batch_size = 50
eval_batch_size = 100

# querying parameters
k = 10
B = 150
runs = 10
max_queries = 100


methods = ['random', 'entropy', 'egl', 'rep-entropy', 'fi']
accs = {}
fi_queries = []
for i in range(len(methods)):
    if methods[i]=='fi':
        accs.update({methods[i]: []})
    else:
        accs.update({methods[i]: 
                     np.zeros((runs, int(max_queries/k)+1))})

# querying
for t in range(runs):
    print(20*'-' + '  Run number %d  '% t +20*"-")
    
    accs_t, fi_queries_t = NNAL.run_AlexNet_AL(
        train_CalDat, train_CalLabels, test_CalDat, test_CalLabels, 
        learning_rate, dropout_rate, epochs, k, B, methods, 
        max_queries, train_batch_size, 
        '%s_init_%d.ckpt'% (model_save_path, t),
        '%s_%d.dat'% (results_save_path, t),
        eval_batch_size)

    # learning parameters
epochs = 10
learning_rate = 1e-3
dropout_rate = 0.5
train_batch_size = 50
eval_batch_size = 100

# querying parameters
k = 10
B = 150
runs = 1
max_queries = 100

methods = ['random', 'entropy', 'egl', 'rep-entropy', 'fi']
accs = {}
fi_queries = []
for i in range(len(methods)):
    if methods[i]=='fi':
        accs.update({methods[i]: []})
    else:
        accs.update({methods[i]: 
                     np.zeros((runs, int(max_queries/k)+1))})

accs,fi_queries = pickle.load(open('%s_accs_%d.dat'% (results_savt_path, t), 'rb'))
    
# querying
# -----------------
for t in range(runs):
    print(20*'-' + '  Run number %d  '% t +20*"-")
    
    accs_t, fi_queries_t = NNAL.run_AlexNet_AL(
        train_CalDat, train_CalLabels, test_CalDat, test_CalLabels, 
        learning_rate, dropout_rate, epochs, k, B, methods, 
        max_queries, train_batch_size, 
        '%s_init_%d.ckpt'% (model_save_path, t),
        '%s_%d.dat'% (results_save_path, t),
        eval_batch_size)
    
    for M in methods:
        if M=='fi':
            accs[M] += [accs_t[M]]
            fi_queries += [fi_queries_t]
        else:
            accs[M][t,:] = accs_t[M]
            
    pickle.dump([accs, fi_queries], 
                open('%s_accs_%d.dat'% (results_savt_path, t), 'wb'))
