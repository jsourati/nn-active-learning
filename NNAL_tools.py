import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pdb
import sys

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

def init_MNIST(init_size, batch_size):
    """Prepare the MNIST data to use in our active learning
    framework, by partitioning it into three sets of (1) initial
    labeled data, (2) unlabeled data (to query from) and (3)
    test data
    """
    
    # loading the data from the TensorFlow library
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    
    # randomly selecting that many samples from the training data set
    train_size = mnist.train.images.shape[0]
    rand_inds = np.random.permutation(train_size)
    init_inds = rand_inds[:init_size]
    unlabeled_inds = rand_inds[init_size:]
    #
    init_train_images = mnist.train.images[init_inds, :]
    init_train_labels = mnist.train.labels[init_inds, :]
    pool_images = mnist.train.images[unlabeled_inds, :]
    pool_labels = mnist.train.labels[unlabeled_inds, :]

    # manually creating batches for initial training
    batch_inds = prep_dat.gen_batch_inds(init_size, batch_size)
    batch_of_data = prep_dat.gen_batch_matrices(init_train_images.T, batch_inds)
    batch_of_labels = prep_dat.gen_batch_matrices(init_train_labels.T, batch_inds)
    
    return batch_of_data, batch_of_labels, pool_images.T, pool_labels.T, \
        test_images.T, test_labels.T

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
