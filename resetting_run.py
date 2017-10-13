from IPython import get_ipython
ipython=get_ipython()
ipython.magic("%reset -f")

# starting the counter and save it
import numpy as np
import time
t=0
runs = 10
np.savetxt('counter.txt', [t, runs], fmt='%d')

while t<runs:
    print(20*'-' + '  Run number %d  '% t +20*"-")
    
    # importing packages
    import sys
    import time
    import pickle

    import NNAL
    import pdb
    from NNAL_tools import test_training_part

    T1 = time.time()
    
    print("Loading data..")
    # ------------------------------------
    data_path = "/common/projects2/jamshid/Caltech-101/resized_data_10class.p"
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
    B = 100
    max_queries = 200

    methods = ['random', 'entropy', 'rep-entropy', 'fi']
    accs = {}
    fi_queries = []
    for i in range(len(methods)):
        if methods[i]=='fi':
            accs.update({methods[i]: []})
        else:
            accs.update({methods[i]: 
                         np.zeros((runs, int(max_queries/k)+1))})

    # get some initial data set
    init_size = 10
    rand_inds = np.random.permutation(train_CalDat.shape[0])[:init_size]
    init_train_dat = [train_CalDat[rand_inds,:,:], train_CalLabels[rand_inds,:]]
    
    # querying
    accs_t, fi_queries_t = NNAL.run_AlexNet_AL(
        train_CalDat, train_CalLabels, test_CalDat, test_CalLabels, 
        learning_rate, dropout_rate, epochs, k, B, methods, 
        max_queries, train_batch_size, 
        'results/saved_models/Wombat_init_%d.ckpt'% t,
        'Wombat_accs_%d.dat'% t,
        eval_batch_size, init_train_dat)

    for M in methods:
        if M=='fi':
            accs[M] += [accs_t[M]]
            fi_queries += [fi_queries_t]
        else:
            accs[M][t,:] = accs_t[M]

    pickle.dump([accs, fi_queries], 
                open('results/Wombat_accs_exp_%d.dat'% t, 'wb'))
    
    # computing the duration
    T2 = time.time()
    dT = (T2 - T1)/60
    with open('durations.txt', 'a') as file:
        file.write('%f.4 \n'% dT)

    # updating the counter and storing it
    t += 1
    np.savetxt('counter.txt', [t, runs], fmt='%d')
    
    # resetting everyting
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic("%reset -f")
    
    # re-loading the counter
    import numpy as np
    t, runs = np.loadtxt('counter.txt', dtype=int)
