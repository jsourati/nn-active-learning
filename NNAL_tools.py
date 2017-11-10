import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import pdb
import sys
import copy
import h5py
import cv2
import os
import NN
import pickle

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

def test_training_part(labels, test_ratio):
    """Paritioning a given labeled data set into test and 
    training partitions, with a given test-to-total ratio
    
    The labels matrix is assumed to be a hot-one column-wise
    matrix.
    """
    
    (c,n) = labels.shape
    
    test_inds = []
    train_inds = np.arange(n)
    # randomly selecting indices from each class
    for j in range(c):
        class_inds = np.where(np.where(labels)[0]==j)[0]
        test_size = round(len(class_inds)*test_ratio)
        rand_inds = np.random.permutation(
            len(class_inds))[:test_size]
        test_class_inds = class_inds[rand_inds]
        test_inds += list(test_class_inds)
        
    test_inds = np.array(test_inds)
    train_inds = np.delete(train_inds, test_inds)

    return train_inds, test_inds
        

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


def idxBatch_posteriors(model, inds,
                        img_path_list,
                        batch_size,
                        session, col,
                        extra_feed_dict={}):
    """A function similar to `batch_posteriors()`
    but working with indices and image-paths 
    rather than a 4D array that is already loaded
    by the data
    
    Similar to `batch_posteriors()` this 
    function also gives a column-wise array 
    of posteriors no matter if the given network
    is column-wise or row-wise. Columns in the
    output array are ordered such that the `i`-th
    column includes the posteriors of the image
    with global index `inds[i]` (i.e., that image
    can be access by its path 
    `image_path_list[ inds[i] ]`
    """
    
    n = len(inds)
    if col:
        c = model.output.get_shape()[0].value
    else:
        c = model.output.get_shape()[1].value
    posteriors = np.zeros((c, n))
        
    # preparing batches
    # each batch has indices (batch_of_inds)
    # that are in terms of the input variable
    # "inds"
    if not(batch_size): 
        batch_of_inds = np.arange(
            len(inds)).tolist()
    else:
        batch_of_inds = prep_dat.gen_batch_inds(
            n, batch_size)
    
    # computing the posteriors
    for inner_inds in batch_of_inds:
        # load the data
        X = NN.load_winds(inds[inner_inds],
                          img_path_list)
        # preparing the feed-dictionary for 
        # the network
        feed_dict = {model.x: X}
        feed_dict.update(extra_feed_dict)
        
        # load the array noticing if the network
        # gives column-wise or row-wise 
        # array of posteriors
        if col:
            posteriors[:,inner_inds] = session.run(
                model.posteriors, 
                feed_dict=feed_dict)
        else:
            posteriors[:, inner_inds] = session.run(
                model.posteriors, 
                feed_dict=feed_dict).T
        
    return posteriors


def batch_posteriors(model, X, batch_size, session, 
                     col, extra_feed_dict={}):
    """Computing posterior probability of a large set of samples
    after dividing them into batches so that computations can be
    done with a limited amount of memory
    
    This function is especially useful when GPU's are being used
    with limited memory. Here, `model` is a tensorflow model with
    `posteriors` as the variable that returns the posterior 
    probabilities of given inputs. Also, note that input `X`
    is assumed to be a tensor of format [batch, width, height,
    channels].
    
    Another point is that the model might output the posteriors
    in a column-wise (with shape `[n_features, n_samples]`) or
    row-wise (with shape `[n_samples, n_features]`). But we 
    always want the function to return in the former format.
    Hence, we get a flag `col` to identify if the model outputs
    the posteriors columnwise (`col=True`) or not (`col=False`).
    In case of the latter, we transpose the output posteriors.
    
    :Parameters:
    
      **model** : CNN or AlexNet_CNN class object
        the (convolutional) neural network that has a property
        called `posteriors` which outputs class posterior
        probabilities for a given set of samples
        
      **X** : array
        data array with shape [batch, height, width, n_channels]
        
      **batch_size** : positive integer
        size of batches for batch-computation of the
        posteriors
        
      **session** : tf.Session()
        the tensorflow session operating on the model
        
      **col** : logical flag (True or False)
        a flag identifying if the `model` outputs the 
        posteriors column-wise (`True`) or row-wise 
        (`False`)
        
      **extra_feed_dict** : dictionary
        any extra dictionary needed to be fed to the mdel
        other than the input data, e.g. the dropout
        rate, if needed
    """
    
    n = X.shape[0]
    
    if batch_size: 
        if col:
            c = model.output.get_shape()[0].value
        else:
            c = model.output.get_shape()[1].value

        posteriors = np.zeros((c, n))

        # batch-wise computations
        quot, rem = np.divmod(n, batch_size)
        for i in range(quot):
            if i<quot-1:
                inds = np.arange(i*batch_size, (i+1)*batch_size)
            else:
                inds = np.arange(i*batch_size, n)

            iter_X = X[inds,:,:,:]

            # add any extra dictionary to the main feed_dict,
            # which is dictionary related to the inputs. 
            # E.g., dictionaries including dropout probabilities
            feed_dict = {model.x: iter_X}
            feed_dict.update(extra_feed_dict)
            if col:
                posteriors[:,inds] = session.run(
                    model.posteriors, 
                    feed_dict=feed_dict)
            else:
                posteriors[:, inds] = session.run(
                    model.posteriors, 
                    feed_dict=feed_dict).T
    else:
        feed_dict = {model.x: X}
        feed_dict.update(extra_feed_dict)
        posteriors = session.run(
            model.posteriors, feed_dict=feed_dict)
        if not(col):
            posteriors = posteriors.T
        
    return posteriors

def batch_accuracy(model, X, Y, batch_size, session, col=True):
    """Similar to `batch_posteriors()` this function computes
    accuracies according to a given test samples, using
    batches
    """
    
    n = X.shape[0]
    acc = 0
    if col:
        c = model.output.get_shape()[0].value
    else:
        c = model.output.get_shape()[1].value
    # batch-wise computations
    quot, rem = np.divmod(n, batch_size)
    for i in range(quot):
        if i<quot-1:
            inds = np.arange(i*batch_size, (i+1)*batch_size)
        else:
            inds = np.arange(i*batch_size, n)
            
        iter_X = X[inds,:,:,:]
        if col:
            iter_Y = Y[:,inds]
        else:
            iter_Y = Y[inds,:]
            
        iter_acc = session.run(
            model.accuracy, 
            feed_dict={model.x: iter_X,
                       model.y_: iter_Y, 
                       model.KEEP_PROB: 1.}
            )
        acc += iter_acc * len(inds)
        
    return acc/n
    
def SDP_query_distribution(A, X_pool, lambda_, k):
    """Solving SDP problem in FIR-based active learning
    to obtain the query distribution
    """
    
    n = len(A)
    d = X_pool.shape[0]
    tau = A[0].shape[0]
    
    pool_norms = np.sum(X_pool**2, axis=0)
    
    """Preparing the variables"""
    # vector c (in the objective)
    cvec = matrix(
            np.concatenate((-lambda_*pool_norms, 
                            np.ones(tau)))
            )
    # matrix inequality constraints
    G, h = inequality_cvx_matrix(A)
    # equality constraint (for having probabilities)
    left_A = np.concatenate((X_pool, np.ones((1, n))), axis=0)
    A_eq = matrix(
        np.concatenate((left_A, np.zeros((d+1, tau))), axis=1))
    
    b_eq = np.zeros(d+1)
    b_eq[-1] = 1.
    b_eq = matrix(b_eq)
    
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
            
    elif method=='max':

        layer_num = int(len(grad) / 2)
        shrunk_grad = np.zeros(layer_num)

        for t in range(layer_num):
            grW = grad[2*t]
            grb = grad[2*t+1]
            # Taking the gradient with maximum
            # magnitude
            grW_max = max(grW.flatten(), key=abs)
            grb_max = max(grb, key=abs)
            shrunk_grad[t] = max(grW_max, grb_max)
                
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
    
def resize_images(imgs, target_shape):
    """Resizing a set of 3D images with different sizes
    to a single target size
    """
    
    n = len(imgs)
    resized_imgs = np.zeros(((n,)+target_shape)+(3,))
    for i in range(n):
        resized = cv2.resize(
            imgs[i].astype(np.float32), target_shape)
        resized_imgs[i,:,:,:] = resized
        
    return resized_imgs

def filter_classes(categories_path, full_data_path, target_classes_path):
    """Filering a data set to a specific subset of classes
    
    The main assumtion is that the `categories_folder` contains only
    folders corresponding to different classes.
    """
    
    full_data = pickle.load(open(full_data_path,'rb'))
    full_labels = full_data[1]
    
    # get the list of all classes indices that correspond 
    # to the same indexing within the loaded full data
    #
    # E.g., if the first folder is "airplanes", the index
    # "0" in the `full_labels` should also correspond to
    # "airplanes" class too.
    #
    # Note that in `prepare_data_4Alex()`, when reading data
    # from all classes, the categories are read in an 
    # order consistent with `os.listdir(categories_path)`,
    # hence here we use the same function to list the
    # categories.
    class_names = os.listdir(categories_path)
 
    # get the target classes
    with open(target_classes_path, 'rb') as fname:
        rows = fname.readlines()
        
    target_classes = [
        row.strip().decode("utf-8") for row in rows]
    
    # filter these classes
    new_images = np.empty((0,)+full_data[0][0,:,:,:].shape)
    new_labels = []
    for i, C in enumerate(target_classes):
        class_idx = np.where(
            np.array(class_names)==C)[0][0]
        dat_idx = np.where(
            np.array(full_labels)==class_idx)[0]
        # adding data and labels
        new_images = np.concatenate(
            (new_images, full_data[0][dat_idx,:,:,:]), 
            axis=0)
        new_labels += len(dat_idx)*[i]
        
    return new_images, new_labels

def read_pretrained_VGG19(weights_path):
    """Reading weights of a pre-trained VGG-19
    """
    
    W = h5py.File(weights_path)

    # layers that inlcude trainable parameters in 
    # the Keras model for which the weights are saved
    related_layers = [1,3,6,8,11,13,15,17,20,22,24,26,
                      29,31,33,35,38,40,42]

    # extract the weights from the HDF5 file
    pretrained_pars = []
    for i in range(len(related_layers)):
        pretrained_pars += [
            [np.array(W['layer_'+str(related_layers[i])]['param_0']),
             np.array(W['layer_'+str(related_layers[i])]['param_1'])]
        ]
        
    return pretrained_pars

def load_weights_VGG19(model, weights_path, session):
    """Loading pre-trained weights to a given VGG model
    """
    
    # reading the weights
    pretrained_pars = read_pretrained_VGG19(weights_path)
    
    # network TF variable names
    TF_var_names = list(model.var_dict.keys())
    n_conv = 16
    
    # convolution layers
    for i in range(n_conv):
        # filters
        # --------
        # shape of the saved filters is of format:
        # [out_channels, in_channels ,height, width]
        # but our filters are in created in form of
        # [height, width, in_channles, out_channels]
        filter_tensor = pretrained_pars[i][0]
        # swapping axes of the saved tensor
        # in_channels <-> height
        swapped_filter = np.swapaxes(filter_tensor, 1, 2)
        # out_channles <-> height
        swapped_filter = np.swapaxes(swapped_filter, 0, 1)
        # out_channels <-> width
        swapped_filter = np.swapaxes(swapped_filter, 1, 3)
        
        var = model.var_dict[TF_var_names[i]][0]
        session.run(var.assign(swapped_filter))
            
        # biases
        # -------
        # these are easy to assign, since they are 1D 
        # arrays and there is no flipping mismatches
        var = model.var_dict[TF_var_names[i]][1]
        session.run(var.assign(pretrained_pars[i][1]))

    # FC layers
    for i in range(n_conv, len(pretrained_pars)-1):
        # matrix weight
        # ------------
        # again the matrices are transpose of each other
        var = model.var_dict[TF_var_names[i]][0]
        session.run(var.assign(pretrained_pars[i][0].T))
        
        # biases
        # ------
        var = model.var_dict[TF_var_names[i]][1]
        session.run(var.assign(np.expand_dims(
                    pretrained_pars[i][1], axis=1)))
