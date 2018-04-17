from matplotlib import pyplot as plt
import numpy as np
import linecache
import shutil
import pickle
import scipy
import nrrd
import yaml
import copy
import pdb
import os

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import PW_analyze_results
import NNAL_tools
import Influence
import NN


def prep_dat(init_size=100,
             digits=None):
    mnist = input_data.read_data_sets(
        'MNIST_data', one_hot=True)

    X_train = mnist.train.images
    Y_train = mnist.train.labels
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    # if specified, take only certain digits
    if digits is not None:
        digits = np.array(digits)
        # training
        inds = np.zeros(ntrain, dtype=bool)
        labels = np.argmax(Y_train, axis=1)
        for digit in digits:
            inds[labels==digit] = 1
        X_train = X_train[inds,:]
        Y_train = Y_train[inds,:]
        Y_train = Y_train[:,digits]
        ntrain = np.sum(inds)
        # test
        inds = np.zeros(ntest, dtype=bool)
        labels = np.argmax(Y_test, axis=1)
        for digit in digits:
            inds[labels==digit] = 1
        X_test = X_test[inds,:]
        Y_test = Y_test[inds,:]
        Y_test = Y_test[:,digits]
        ntest = np.sum(inds)

    # random selection of initial labels
    init_inds = np.random.permutation(
        ntrain)[:init_size]
    init_X_train = X_train[init_inds,:].T
    init_Y_train = Y_train[init_inds,:].T

    # remaining of the trainin data 
    pool_inds = set(np.arange(ntrain)) - \
                set(init_inds)
    pool_inds = list(pool_inds)
    X_pool = X_train[pool_inds,:].T
    Y_pool = Y_train[pool_inds,:].T

    return (init_X_train, init_Y_train), \
        (X_pool, Y_pool), (X_test.T, Y_test.T)


def train_LogisticReg(save_dir,
                      init_size=100,
                      digits=None,
                      batch_size=50,
                      learning_rate=1e-4,
                      epochs=60):
    """Create and train a simple logistic regression
    for classifying all or a subset of MNIST
    digits
    """

    """ preparing the data """
    init_dat, pool_dat, test_dat = prep_dat(
        init_size, digits)
    init_X_train, init_Y_train = init_dat
    X_pool, Y_pool = pool_dat
    X_test, Y_test = test_dat

    """ preparing the model """
    dropout = [[0], 0.5]
    c = [10 if digits is None else len(digits)][0]
    x = tf.placeholder(tf.float32, [784,None])
    pw_dict = {'fc1': [100, 'fc'],
               'fc2': [c, 'fc']}
    model = NN.CNN(x, pw_dict, 'LogisticReg',
                   dropout=dropout)
    model.get_optimizer(learning_rate)
    model.feature_layer = model.x

    """ initial training """
    n = init_X_train.shape[1]
    b = batch_size
    with tf.Session() as sess:
        model.initialize_graph(sess)

        for i in range(epochs):
            batches = NN.gen_batch_inds(n,b)

            for batch_inds in batches:
                batch_X = init_X_train[:,batch_inds]
                batch_Y = init_Y_train[:,batch_inds]

                sess.run(model.train_step,feed_dict={
                    model.x:batch_X,
                    model.y_:batch_Y,
                    model.keep_prob:model.dropout_rate})
            print(i,end=',')

        model.save_weights(save_dir)

    return init_dat, pool_dat, test_dat


def eval_test_model(model, sess, test_dat):
    
    b = 5000
    X_test, Y_test = test_dat
    n = X_test.shape[1]
    
    batches = NN.gen_batch_inds(n,b)
    tP, tTP, tFP = 0,0,0
    for batch_inds in batches:
        batch_X = X_test[:,batch_inds]
        batch_Y = Y_test[:,batch_inds]
        
        preds = sess.run(model.prediction,feed_dict={
            model.x:batch_X,
            model.keep_prob:1.})

        batch_Y = np.argmax(batch_Y, axis=0)
        (P,N,TP,
         FP,TN,FN) = PW_analyze_results.get_preds_stats(
             preds, batch_Y)

        tP += P
        tTP += TP
        tFP += FP

    # compute total Pr/Rc and F1
    Pr = tTP / (tTP + tFP)
    Rc = tTP / tP
    F1 = 2. / (1/Pr + 1/Rc)

    return F1


def stoch_approx_Influence_pool(model, sess, 
                                X_train, X_pool,
                                pool_inds):

    n = X_train.shape[1]
    max_iter = 500

    # preparing feed_dict
    feed_dict = {model.x: X_pool[:,pool_inds],
                 model.keep_prob: 1.}

    # loss gradients of the pool samples
    grads, labels = NN.LLFC_grads(model,sess,feed_dict)

    # start the stochastic estimatin
    V_t = grads
    for t in range(max_iter):
        # Hessian of a random labeled data
        rand_ind = [np.random.randint(n)]
        feed_dict = {model.x:X_train[:,rand_ind],
                     model.keep_prob:1.}
        H = -NN.LLFC_hess(model,sess,feed_dict)

        # iteration's step
        V_t = grads + V_t - H@V_t/10

    return V_t, labels

def stoch_approx_Influence_train(model, sess, 
                                 X_train, Y_train,
                                 train_inds):

    n = X_train.shape[1]
    max_iter = 500

    # preparing feed_dict
    feed_dict = {model.x: X_train[:,train_inds],
                 model.keep_prob: 1.}

    # loss gradients of the pool samples
    grads = NN.LLFC_grads(model,sess,feed_dict, Y_train)

    # start the stochastic estimatin
    V_t = grads
    for t in range(max_iter):
        # Hessian of a random labeled data
        rand_ind = [np.random.randint(n)]
        feed_dict = {model.x:X_train[:,rand_ind],
                     model.keep_prob:1.}
        H = -NN.LLFC_hess(model,sess,feed_dict)

        # iteration's step
        V_t = grads + V_t - H@V_t/10

    return V_t

        
    
