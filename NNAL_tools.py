import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import pdb
import sys
import copy
import cv2
import os
#import NN

read_file_path = "/home/ch194765/repos/atlas-active-learning/"
sys.path.insert(0, read_file_path)
import prep_dat

def uncertainty_filtering(posteriors, B):
    """Filtering data by keeping only the most `B` uncertain
    samples of the data set
    
    The posteriors are assumed to be in form of [n_samples, n_classes]
    """
    
    # take care of zero posteriors (to be fed to logarithms)
    posteriors[posteriors==0] += 1e-8
    
    # uncertainties
    entropies = -np.sum(posteriors * np.log(posteriors), axis=0)
    selected_unlabeled = np.argsort(-entropies)[:B]
    
    return selected_unlabeled

def enlist_gradients(TF_vars, B, par_list):
    """Take a TensorFlow variable, which contains several cost function
    (with possibly variable size), and unstack them and create a list of
    their gradients so that they can be calculated in a single call of
    TensorFlow
    
    The variable `TF_vars` could be, e.g., log-posteriors of the model,
    whose number of columns is equal to number of classes, but the 
    number of its rows is variables (or None) and so it should be given.
    The input `B` represents the number of rows that we expect to have
    for `TF_vars` in the time of running. 
    
    The function should get the list of parameters with respect to which
    the gradients are to be taken too.
    """
    
    # extract number of classes
    c = TF_vars.shape[1].value
    
    # first, unstacking the TF variable into an array
    vars_array = np.empty((B,c), dtype=object)
    # along the class-axis
    unstacked_vars = tf.unstack(TF_vars, axis=1)
    # .. and then along the sample-axis
    for j in range(c):
        vars_array[:,j] = np.array(tf.unstack(unstacked_vars[j], B))
        
    # forming the list of gradients
    funcs = np.reshape(vars_array, c*B)
    grads = [tf.gradients(func, par_list) for func in funcs]
    
    return grads

def compute_entropy(PMFs):
    """Computing entropy of a given finite-length PMF
    
    The PMFs matrix should have dimension [n_classes, n_samples].
    That is, columnwise.
    """
    
    # getting rid of the zero probabilities so that logarithm
    # function can be applied
    PMFs[PMFs==0] += 10e-8
    
    # compuing Shannon entropy
    entropies = -np.sum(PMFs * np.log(PMFs), axis=0)
    
    return entropies


def divide_training(train_dat, init_size, batch_size):
    """Partitioning a given training data into an initial
    labeled data set (to initialize the model) and a pool of 
    unlabeled samples, and then batching the initial labeled
    set
    
    It is assumed that the data is column-wise
    """
    
    train_features = train_dat[0]
    train_labels = train_dat[1]
    
    # random assignment of indices to the labeled data
    # set or to the unlabeled pool
    train_size = train_features.shape[1]
    rand_inds = np.random.permutation(train_size)
    init_inds = rand_inds[:init_size]
    unlabeled_inds = rand_inds[init_size:]
    
    init_features = train_features[:, init_inds]
    init_labels = train_labels[:, init_inds]
    pool_features = train_features[:, unlabeled_inds]
    pool_labels = train_labels[:, unlabeled_inds]

    # manually creating batches for initial training
    batch_inds = prep_dat.gen_batch_inds(init_size, batch_size)
    batch_of_data = prep_dat.gen_batch_matrices(init_features, batch_inds)
    batch_of_labels = prep_dat.gen_batch_matrices(init_labels, batch_inds)
    
    return batch_of_data, batch_of_labels, pool_features, pool_labels

def init_MNIST(init_size, batch_size, classes=None):
    """Prepare the MNIST data to use in our active learning
    framework, by partitioning it into three sets of (1) initial
    labeled data, (2) unlabeled data (to query from) and (3)
    test data
    """

    # loading the data from the TensorFlow library
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    # normalizing everything
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    if classes:
        indics = np.sum(mnist.train.labels[:,classes], axis=1)>0
        train_images = mnist.train.images[indics,:]
        train_labels = mnist.train.labels[indics,:]
        train_labels = train_labels[:,classes]
        #
        indics = np.sum(mnist.test.labels[:,classes], axis=1)>0
        test_images = mnist.test.images[indics,:]
        test_labels = mnist.test.labels[indics,:]
        test_labels = test_labels[:,classes]
        
    # randomly selecting that many samples from the training data set
    train_size = train_images.shape[0]
    rand_inds = np.random.permutation(train_size)
    init_inds = rand_inds[:init_size]
    unlabeled_inds = rand_inds[init_size:]
    #
    init_train_images = train_images[init_inds, :]
    init_train_labels = train_labels[init_inds, :]
    pool_images = train_images[unlabeled_inds, :]
    pool_labels = train_labels[unlabeled_inds, :]

    # manually creating batches for initial training
    batch_inds = prep_dat.gen_batch_inds(
        init_size, batch_size)
    batch_of_data = prep_dat.gen_batch_matrices(
        init_train_images.T, batch_inds)
    batch_of_labels = prep_dat.gen_batch_matrices(
        init_train_labels.T, batch_inds)
    
    return batch_of_data, batch_of_labels, pool_images.T, pool_labels.T, \
        test_images.T, test_labels.T

def init_restricted_classes(X_train, Y_train, classes,
                            per_class_size):
    """Preparing a  data set to be used as the initial
    labeled data set in an active learning framework
    """

    class_inds = np.where(Y_train.T==1.)[1] == classes[0]
    class_inds = np.where(class_inds)[0]
    n_class = len(class_inds)
    # 
    rand_inds = np.random.permutation(n_class)
    selected_inds = class_inds[rand_inds[:per_class_size]]
    init_X_train = X_train[selected_inds,:,:,:]
    init_Y_train = Y_train[:,selected_inds]
    # updating the pool
    X_pool = np.delete(X_train, selected_inds, 0)
    Y_pool = np.delete(Y_train, selected_inds, 1)
    for i in range(1, len(classes)):
        class_inds = np.where(Y_pool.T==1.)[1] == classes[i]
        class_inds = np.where(class_inds)[0]
        n_class = len(class_inds)
        # 
        rand_inds = np.random.permutation(n_class)
        selected_inds = class_inds[rand_inds[:per_class_size]]
        init_X_train = np.concatenate(
            (init_X_train, X_pool[selected_inds,:,:,:]), axis=0)
        init_Y_train = np.concatenate(
            (init_Y_train, Y_pool[:,selected_inds]), axis=1)
        # updating the pool
        X_pool = np.delete(X_pool, selected_inds, 0)
        Y_pool = np.delete(Y_pool, selected_inds, 1)


    return init_X_train, init_Y_train, X_pool, Y_pool

def modify_MNIST(batch_of_data, batch_of_labels, pool_images, 
                 pool_labels, new_images, new_labels):
    """Modifying a current batch of training data, and unlabeled 
    pool, by a newly generated labeled data
    
    The batch of data is in form [n_samples, n_features]
    """
    
    batch_size = batch_of_data[0].shape[0]
    # number of initial labeled samles
    n_batches = len(batch_of_data)
    n = (n_batches-1)*batch_size + batch_of_data[-1].shape[0]
    d = batch_of_data[0].shape[1]
    c = batch_of_labels.shape[1]

    # merge all the batches, to randomly parition again after adding
    # the new labeled samples
    init_data = np.zeros((n, d))
    init_data[:batch_size,:] = batch_of_data[0]
    init_labels[:batch_size,:] = batch_of_data[0]
    for i in range(1, n_batches):
        if i==n_batches-1:
            init_data[i*batch_size:n,:] = batch_of_data[i]
        else:
            init_data[i*batch_size:(i+1)*batch_size,:] = batch_of_data[i]
            
        
    
def update_batches(batch_of_data, batch_of_labels, new_data, 
                   new_labels, method='regular'):
    """Updating a set of existing batches of training data with a newly
    labeled sampels to finetune the network after expanding the
    training data

    There are two possible ways to extend the training batches:

    * 'Regular method': the newly labeled data will be aded 
    to the whole (unbatched) training data and then the re-batch
    them in a regular way
    
    * 'Emphasized method': the newly added samples will be added to 
    all the previous batches, hence size of the batches will increase
    
    """
    
    batch_size = batch_of_data[0].shape[1]

    if method=='regular':
        # un-batching the data
        training_data = np.concatenate(batch_of_data, axis=1)
        training_labels = np.concatenate(batch_of_labels, axis=1)
        
        # append the newly labeled samples
        training_data = np.concatenate((training_data, new_data), axis=1)
        training_labels = np.concatenate((training_labels, new_labels), axis=1)
        
        # batch again
        batch_inds = prep_dat.gen_batch_inds(training_data.shape[1], batch_size)
        batch_of_data = prep_dat.gen_batch_matrices(training_data, batch_inds)
        batch_of_labels = prep_dat.gen_batch_matrices(training_labels, batch_inds)
        
    elif method=='emphasized':
        # append the newly labeled samples to all the batches
        for i in range(len(batch_of_data)):
            batch_of_data[i] = np.concatenate((batch_of_data[i], new_data), axis=1)
            batch_of_labels[i] = np.concatenate((batch_of_labels[i], new_labels), axis=1)
    else:
        raise ValueError("Specified method does not exist.")
        
    return batch_of_data, batch_of_labels

def prepare_finetuning_data(X_train, Y_train, Q, Y_Q, 
                            old_data_to_keep, batch_size):
    """Preparing the data after receiving new labels to update/finetune
    the model accordingly

    This function takes the set of previously labeled data set, together
    with a newly labeled queries to use them to update/fine-tune the
    model based on the newly added labels. Here, we use part of the old
    labeled data set too, to prevent the network from being overfitted 
    to the new labels.
    """

    n_old = X_train.shape[0]
    if old_data_to_keep > n_old:
        old_X_train = X_train
        old_Y_train = Y_train
    else:
        # randomly selecting some of the old labeled samples
        rand_inds = np.random.permutation(n_old)
        old_X_train = X_train[rand_inds[:old_data_to_keep],:,:,:]
        old_Y_train = Y_train[:, rand_inds[:old_data_to_keep]]

    # mixing the new and old labels
    new_X_train = np.concatenate((old_X_train, Q), axis=0)
    new_Y_train = np.concatenate((old_Y_train, Y_Q), axis=1)
    
    return new_X_train, new_Y_train


def Alex_features_MNIST(bulk_size):
    """Pre-processing MNIST data to make them consistent with AlexNet, and
    then extract the features as the output of the 7'th layer
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    # we cannot give the whole training data at once because of OOM
    # forming the division indices
    train_size = mnist.train.images.shape[0]
    divisions = np.append(np.arange(0, train_size, bulk_size), train_size)
        
    train_features = np.zeros((4096, train_size))
    for t in range(len(divisions)-1):
        inds = np.arange(divisions[t], divisions[t+1])
        Alexified_images = np.zeros((len(inds), 227, 227, 3))
        for i in range(len(inds)):
            img = np.reshape(
                mnist.train.images[inds[i],:], (28,28))
            img = cv2.resize(img.astype(np.float32), (227,227))
            img = img.reshape((227,227,1))
            img = np.repeat(img, 3, axis=2)
            img -= imagenet_mean
            Alexified_images[i,:,:,:] = img.reshape((1,227,227,3))
        
        train_features[:, inds] = NN.AlexNet_features(Alexified_images).T
        print("%d / %d" % (t, len(divisions)-1))

    print("Extracting test features...")
    test_size = mnist.test.images.shape[0]
    divisions = np.append(np.arange(0, test_size, bulk_size), test_size)
        
    test_features = np.zeros((4096, test_size))
    for t in range(len(divisions)-1):
        inds = np.arange(divisions[t], divisions[t+1])
        Alexified_images = np.zeros((len(inds), 227, 227, 3))
        for i in range(len(inds)):
            img = np.reshape(
                mnist.test.images[inds[i],:], (28,28))
            img = cv2.resize(img.astype(np.float32), (227,227))
            img = img.reshape((227,227,1))
            img = np.repeat(img, 3, axis=2)
            img -= imagenet_mean
            Alexified_images[i,:,:,:] = img.reshape((1,227,227,3))
        
        test_features[:, inds] = NN.AlexNet_features(Alexified_images).T
        print("%d / %d" % (t, len(divisions)-1))
    
    return train_features, test_features


def batch_posteriors(model, X, batch_size, session):
    """Computing posterior probability of a large set of samples
    after dividing them into batches so that computations can be
    done with a limited amount of memory
    
    This function is especially useful when GPU's are being used
    with limited memory. Here, `model` is a tensorflow model with
    `posteriors` as the variable that returns the posterior 
    probabilities of given inputs. Also, note that input `X`
    is assumed to be a tensor of format [batch, width, height,
    channels].
    """
    
    n = X.shape[0]
    c = model.output.get_shape()[0].value
    posteriors = np.zeros((c, n))
    
    # batch-wise computations
    quot, rem = np.divmod(n, batch_size)
    for i in range(quot):
        if i<quot-1:
            inds = np.arange(i*batch_size, (i+1)*batch_size)
        else:
            inds = slice(i*batch_size, n)
            
        iter_X = X[inds,:,:,:]
        posteriors[:,inds] = session.run(
            model.posteriors, feed_dict={model.x:iter_X})
        
    return posteriors
    
def SDP_query_distribution(A, k):
    """Solving SDP problem in FIR-based active learning
    to obtain the query distribution
    """
    
    n = len(A)
    d = A[0].shape[0]
    
    """Preparing the variables"""
    # vector c (in the objective)
    cvec = matrix(
            np.concatenate((np.zeros(n), 
                            np.ones(d)))
            )
    # matrix inequality constraints
    G, h = inequality_cvx_matrix(A)
    # equality constraint (for having probabilities)
    A_eq = matrix(
            np.concatenate((np.ones(n), 
                            np.zeros(d)))).trans()
    b_eq = matrix(1.)
    
    """Solving SDP"""
    soln = solvers.sdp(cvec, Gs=G, hs=h, A=A_eq, b=b_eq)
    
    return soln

def inequality_cvx_matrix(A, k=None):
    """Preparing inequality vectorized matrices needed
    to form the SDP of FIR-based active learning
    
    CAUTIOUN: `matrix` function in cvxopt, creates a
    matrix in a different way than the numpy `array`.
    Specifically, if you give a numpy array to this
    function, it transposes the array and assign it
    to the matrix. So we should alway transpose the
    array once we want to covnert them into cvxopt
    matrix.

    If `k` is given, it means that an extra constraint
    should be applied to prevent the query PMF from 
    becoming too peaky (degenerate distributin)
    """
    
    n = len(A)
    d = A[0].shape[0]
    
    # first form matrices that include A:
    # forming the rows in constraint matrix
    # corresponding to q_i's, since these remain
    # unchagned through all the first d constraints
    # can can be formed only once
    unchanged_arr = np.zeros((n, (d+1)**2))
    # also compute the part related to positivity
    # constraint in this same loop
    positivity_const = np.zeros((n+d, n**2))
    for i in range(n):
        app_A = append_zero(A[i])
        unchanged_arr[i,:] = np.ravel(app_A.T)
        positivity_const[i, i*n+i] = 1.
        
    # we should also construct the right-hand-side
    # of the inequality constraints (h's)
    h = []
    G = []
    for j in range(d):
        ineq_arr = np.zeros((n+d, (d+1)**2))
        ineq_arr[:n,:] = unchanged_arr
        # matrix for t_j
        ineq_arr[n+j,-1] = 1.
        G += [matrix(-ineq_arr.T)]
        # the corresponding h-term
        h_mat = np.zeros((d+1, d+1))
        h_mat[j, -1] = -1.
        h_mat[-1, j] = -1.
        h += [matrix(-h_mat)]
        
    # Also, include the positivity constraints
    G += [matrix(-positivity_const.T)]
    h += [matrix(np.zeros((n, n)))]
    
    # add q_i <= 1/k if k is given
    if k:
        G += [matrix(positivity_const.T)]
        h += [matrix(np.eye(n)/np.float(k))]
    
    return G, h
    
def shrink_gradient(grad, method, args=None):
    """Shrinking gradient vectors by summing-up 
    the sub-components, or choosing a subset of 
    the derivatives
    """
    
    if method=='sum':
        
        layer_num = int(len(grad) / 2)
        shrunk_grad = np.zeros(layer_num)

        for t in range(layer_num):
            grW = grad[2*t]
            grb = grad[2*t+1]
            # summing up derivatives related to
            # the parameters of each layer
            grad_size = np.prod(grW.shape)+len(grb)
            shrunk_grad[t] = (np.sum(
                grW) + np.sum(grb))/grad_size
                
    elif method=='rand':
        # layers to sample from
        layer_inds = args['layer_inds']
        
        # number of parameters randomly sampled
        # per layer
        nppl = args['nppl']
        
        shrunk_grad = np.zeros(nppl, len(layer_inds))
        rand_inds = np.zeros(nppl, len(layer_inds))
        for t in range(len(layer_inds)):
            # layer grad in a single vector
            grW = np.ravel(grad[2*t])
            grb = grad[2*t+1]
            gr = np.concatenate((grW,  grb))
            
            shrunk_grad[:, t] = gr[args['inds'][:,t]]
                
    return np.ravel(shrunk_grad)

def append_zero(A):
    """Function for appending zeros as the
    last row and column of a given square matrix
    """
    
    d = A.shape[0]
    A = np.insert(A, d, 0, axis=1)
    A = np.insert(A, d, 0, axis=0)
    
    return A
    
def sample_query_dstr(q_dstr, k, replacement=True):
    """Drawing a batch of samles from the query
    distribution.
    """

    if q_dstr.min()<-.01:
            warnings.warn('Optimal q has significant'+
                          ' negative values..')    
    q_dstr[q_dstr<0] = 0.
    
    if replacement:
        # drawing samples without replacement
        Q_inds = q_dstr.cumsum(
            ).searchsorted(np.random.sample(k))
        Q_inds = np.unique(Q_inds)

        # if we need to make sure exactly k samples 
        # will be drawn
        k_sample = False
        if k_sample:
            # keep sampling until k samples is obtained
            while len(Q_inds) < k:
                rand_ind = q_dstr.cumsum(
                    ).searchsorted(np.random.sample(1))
                if not((Q_inds==rand_ind).any()):
                    Q_inds = np.append(Q_inds, rand_ind)

        # in case of numerical issue, fix it
        if (Q_inds==len(q_dstr)).any():
            Q_inds[Q_inds==len(q_dstr)] = len(q_dstr)-1
    else:
        # draw samples with replacement 
        # this way we can always make sure of having
        # exactly k samles.
        rem_inds = np.arange(len(q_dstr))
        Q_inds = []
        while len(Q_inds)<k:
            single_ind = [q_dstr.cumsum(
                    ).searchsorted(
                    np.random.sample(1))]
            Q_inds += [rem_inds[single_ind][0]]
            # remove the last drawn sample from PMF
            rem_inds = np.delete(rem_inds, single_ind)
            q_dstr = np.delete(q_dstr, single_ind)
            # re-normalization
            # (make sure not all masses are zero)
            if all(q_dstr==0):
                q_dstr[:] = 1.
            q_dstr = q_dstr / np.sum(q_dstr)
        
        Q_inds = np.array(Q_inds)
        
    return Q_inds
            
            
def prepare_data_4Alex(path, folders=None):
    """Preparing a given data set to be used for
    testing or fine-tuning AlexNet model
    """
    all_images = []
    all_labels = []
    if not(folders):
        # if no specific folders are specified 
        # read all the folders assuming
        # that there are only sub-directories
        folders = os.listdir(path)
        
    for i in range(len(folders)):
        image_files = os.listdir(
            os.path.join(path, folders[i]))
        
        for j in range(len(image_files)):
            img_path = os.path.join(
                path, folders[i], image_files[j])
            
            all_images += [cv2.imread(img_path)]
            all_labels += [i]
        
            
    return all_images, all_labels
    
