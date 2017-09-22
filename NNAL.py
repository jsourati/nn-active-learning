import numpy as np
import tensorflow as tf
import pdb
import sys
import pickle
import warnings
import NN
import NNAL_tools
from cvxopt import matrix, solvers


def test_MNIST(iters, B, k, init_size, batch_size, epochs, 
               train_dat=None, test_dat=None):
    """Evaluate active learning based on Fisher information,
    or equivalently expected change of the model, over MNIST
    data set
    """
    
    # preparing MNIST data set
    if not(train_dat):
        batch_of_data, batch_of_labels, pool_images, pool_labels, \
            test_images, test_labels = NNAL_tools.init_MNIST(init_size, batch_size)
    else:
        test_images = test_dat[0]
        test_labels = test_dat[1]
        batch_of_data, batch_of_labels, pool_images, pool_labels = \
            NNAL_tools.divide_training(train_dat, init_size, batch_size)
    
    # FI-based querying
    print("Doing FI-based querying")
    fi_accs, fi_data, fi_labels = \
        querying_iterations_MNIST(batch_of_data, batch_of_labels, 
                                  pool_images, pool_labels, 
                                  test_images, test_labels,
                                  iters, k, epochs, method="FI")

    print("Doing random querying")
    rand_accs, rand_data, rand_labels = \
        querying_iterations_MNIST(batch_of_data, batch_of_labels, 
                                  pool_images, pool_labels, 
                                  test_images, test_labels,
                                  iters, k, epochs, method="random")

    print("Doing uncertainty sampling")
    ent_accs, ent_data, ent_labels = \
        querying_iterations_MNIST(batch_of_data, batch_of_labels, 
                                  pool_images, pool_labels, 
                                  test_images, test_labels,
                                  iters, k, epochs, method="entropy")
            
    return fi_accs, rand_accs, ent_accs


def querying_iterations_MNIST(batch_of_data, batch_of_labels, 
                             pool_images, pool_labels, 
                             test_images, test_labels,
                             iters, k, epochs, method):
    
    c = pool_labels.shape[0]
    d = pool_images.shape[0]
    accs = np.zeros((c+1,iters+1))
    
    # initial training
    with tf.Session() as sess:
        
        print("Initializing the model...")
        
        # input and output placeholders
        x = tf.placeholder(tf.float32, shape=[d, None])
        y_ = tf.placeholder(tf.float32, shape=[10, None])

        # parameters
        W = tf.Variable(tf.zeros([10, d]))
        b = tf.Variable(tf.zeros([10,1]))
        
        # initializing
        sess.run(tf.global_variables_initializer())

        # outputs of the network
        y = tf.matmul(W,x) + b
        posteriors = tf.nn.softmax(tf.transpose(y))
        #log_posteriors = tf.log(posteriors)
        
        # cross entropy as the training objective
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y_), 
                                                    logits=tf.transpose(y)))

        # optimization iteration
        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

        # initial training
        for _ in range(epochs):
            for i in range(len(batch_of_data)):
                train_step.run(feed_dict={x: batch_of_data[i], 
                                          y_: batch_of_labels[i]})
        
        # initial accuracy
        correct_prediction = tf.equal(tf.argmax(y,0), tf.argmax(y_,0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accs[0,0] = accuracy.eval(feed_dict={x: test_images, 
                                             y_: test_labels})
        # accuracies in each class
        for j in range(1,c+1):
            indics = test_labels[j-1,:]==1
            accs[j,0] = accuracy.eval(feed_dict={x: test_images[:,indics], 
                                                 y_: test_labels[:,indics]})

        # start the querying iterations
        print("Starting the querying iterations..")
        added_labels = []
        #added_images = np.zeros((iters, d))
        for t in range(1, iters+1):
            
            if method=="FI":
                """FI-based querying"""
                # compute all the posterior probabilities
                pool_posteriors = sess.run(posteriors, feed_dict=
                                           {x: pool_images, y_: pool_labels})
                
                # using the normalized pool-samples
                pool_norms = np.sum(pool_images**2, axis=0)
                pool_norms /= pool_norms.max()
                
                # norm of posteriors
                pool_posteriors_norms = np.sum(pool_posteriors**2, axis=1)
                # scores
                scores = (pool_norms+1)*(1-pool_posteriors_norms)

                # take the best k scores
                #bests = np.argsort(-scores)[:100]
                #Q = np.array([bests[np.random.randint(100)]])
                Q = np.argsort(-scores)[:k]
                
            elif method=="random":
                """randomd querying"""
                Q = np.random.randint(0, pool_images.shape[1], k)
                
            elif method=="entropy":
                # compute all the posterior probabilities
                pool_posteriors = sess.run(posteriors, feed_dict=
                                           {x: pool_images, y_: pool_labels})
                entropies = NNAL_tools.compute_entropy(pool_posteriors.T)
                Q = np.argsort(-entropies)[:k]
            
            new_train_data = pool_images[:,Q]
            new_train_labels = pool_labels[:,Q]
            
            #added_images[t-1,:] = np.squeeze(new_train_data)
            added_labels += [np.where(new_train_labels)[0][0]]
            
            batch_of_data, batch_of_labels = \
                NNAL_tools.update_batches(batch_of_data, 
                                          batch_of_labels,
                                          new_train_data,
                                          new_train_labels,
                                          'regular')
            
            # fine-tuning
            sess.run(tf.global_variables_initializer())
            for _ in range(epochs):
                for i in range(len(batch_of_data)):
                    train_step.run(feed_dict={x: batch_of_data[i], 
                                              y_: batch_of_labels[i]})

            accs[0,t] = accuracy.eval(feed_dict={x: test_images, 
                                               y_: test_labels})
            # accuracies in each class
            for j in range(1,c+1):
                indics = test_labels[j-1,:]==1
                accs[j,t] = accuracy.eval(feed_dict={x: test_images[:,indics], 
                                                     y_: test_labels[:,indics]})
            # update the pool
            np.delete(pool_images, Q, 1)
            np.delete(pool_labels, Q, 1)
            
            nL = np.concatenate(batch_of_data, axis=1).shape[1]
            print("Iteration %d is done. Number of labels: %d" % (t, nL))
    
    return accs, batch_of_data, batch_of_labels

def CNN_query(model, k, B, pool_X, method, session, 
              batch_size=None, shrink_method=None):
    """Querying a number of unlabeled samples from a given pool
    
    :Parameters:
    
      **model** : CNN model object
        any CNN class object which has methods, `output` as the 
        output of the network, and `posteriors` as the estimated
        posterior probability of the classes
        
      **k** : positive integer
        number of queries to be selected
        
      **B** : positive integer
        number of samples to keep in uncertainty filterins
        (only will be used in `egl` and `fi-` methods)
        
      **pool_X** : 4D tensors
        pool of unlabeled samples that is stored in format
        `[batch, rows, columns, n_channels]`
        
      **method** : string
        the querying method
        
      **session** : tf.Session()
        the tensorflow session operating on the model
        
      **batch_size** : integers (default is None)
        size of the batches for batch-wise computation of
        posteriors and gradients; if not provided, full data
        will be used at once in those computations, which is
        prone to out-of-memory error especially when GPU's
        are being used
    """

    if method=='egl':
        # uncertainty filtering
        print("Uncertainty filtering...")
        if batch_size:
            posteriors = NNAL_tools.batch_posteriors(
                model, pool_X, batch_size, session)
        else:
            posteriors = session.run(
                model.posteriors, feed_dict={model.x:pool_X})
            
        sel_inds = NNAL_tools.uncertainty_filtering(posteriors, B)
        sel_posteriors = posteriors[:, sel_inds]

        # EGL scoring
        print("Computing the scores..")
        c = model.output.get_shape()[0].value
        scores = np.zeros(B)
        for i in range(B):
            # gradients of samples one-by-one
            grads = session.run(
                model.grad_log_posts, 
                feed_dict={model.x:np.expand_dims(
                        pool_X[sel_inds[i],:,:,:], axis=0)})

            T = len(grads['0'])
            for j in range(c):
                class_score = 0.
                for t in range(T):
                    class_score += np.sum(grads[str(j)][t]**2)
                scores[i] += class_score*sel_posteriors[j,i]

        # select the highest k scores
        Q_inds = sel_inds[np.argsort(-scores)[:k]]

    elif method=='random':
        n = pool_X.shape[0]
        Q_inds = np.random.permutation(n)[:k]
        
    elif method=='entropy':
        # computing the posteriors
        if batch_size:
            posteriors = NNAL_tools.batch_posteriors(
                model, pool_X, batch_size, session)
        else:
            posteriors = session.run(
                model.posteriors, feed_dict={model.x:pool_X})
            
        entropies = NNAL_tools.compute_entropy(posteriors)
        Q_inds = np.argsort(-entropies)[:k]
        
    elif method=='fi':
        # uncertainty filtering
        print("Uncertainty filtering...")
        if batch_size:
            posteriors = NNAL_tools.batch_posteriors(
                model, pool_X, batch_size, session)
        else:
            posteriors = session.run(
                model.posteriors, feed_dict={model.x:pool_X})
            
        sel_inds = NNAL_tools.uncertainty_filtering(posteriors, B)
        sel_posteriors = posteriors[:, sel_inds]
        
        # forming A-matrices
        layer_num = len(model.layer_type)
        c = model.output.get_shape()[0].value
        A = []
        for i in range(B):
            # gradients of samples one-by-one
            grads = session.run(
                model.grad_log_posts, 
                feed_dict={model.x:np.expand_dims(
                        pool_X[sel_inds[i],:,:,:], axis=0)})

            Ai = np.zeros((layer_num, layer_num))
            for j in range(c):
                shrunk_grad = NNAL_tools.shrink_gradient(
                    grads[str(j)], 'sum')
                Ai += sel_posteriors[j,i]*np.outer(
                    shrunk_grad,shrunk_grad) + np.eye(
                    layer_num)*1e-5
            A += [Ai]
        # SDP
        print('Solving SDP..')
        soln = NNAL_tools.SDP_query_distribution(A, k)
        print('status: %s'% (soln['status']))
        
        q_opt = np.array(soln['x'][:B])
        if q_opt.min()<-.01:
            warnings.warn('Optimal q has significant'+
                          ' negative values..')    
        q_opt[q_opt<0] = 0.

        # draw k samples from the obtained query distribution
        Q_inds = q_opt.cumsum(
            ).searchsorted(np.random.sample(k))
        Q_inds = np.unique(Q_inds)
        
        # for now make sure we get exactly k samples
        k_sample = False
        if k_sample:
            # keep sampling until k samples is obtained
            while len(Q_inds) < k:
                rand_ind = q_opt.cumsum(
                    ).searchsorted(np.random.sample(1))
                if not((Q_inds==rand_ind).any()):
                    Q_inds = np.append(Q_inds, rand_ind)
                
        # in case of numerical issue, fix it
        if (Q_inds==B).any():
            Q_inds[Q_inds==B] = B-1
        Q_inds = sel_inds[Q_inds]
        
    elif method=='rep-entropy':
        # uncertainty filtering
        print("Uncertainty filtering...")
        if batch_size:
            posteriors = NNAL_tools.batch_posteriors(
                model, pool_X, batch_size, session)
        else:
            posteriors = session.run(
                model.posteriors, feed_dict={model.x:pool_X})
            
        sel_inds = NNAL_tools.uncertainty_filtering(posteriors, B)
        sel_posteriors = posteriors[:, sel_inds]
        n = pool_X.shape[0]
        nsel_inds = list(set(np.arange(n)) - set(sel_inds))
        
        print("Finding Similarities..")
        # extract the features
        F = model.extract_features(pool_X, session, batch_size)
        F_uncertain = F[:, sel_inds]
        norms_uncertain = np.sqrt(np.sum(F_uncertain**2, axis=0))
        F_rest_pool = F[:, nsel_inds]
        norms_rest = np.sqrt(np.sum(F_rest_pool**2, axis=0))
        
        # compute cos-similarities between filtered images
        # and the rest of the unlabeled samples
        dots = np.dot(F_rest_pool.T, F_uncertain)
        norms_outer = np.outer(norms_rest, norms_uncertain)
        sims = dots / norms_outer
            
        print("Greedy optimization..")
        # start from empty set
        Q_inds = []
        rem_inds = np.arange(B)
        # add most representative samples one by one
        for i in range(k):
            rep_scores = np.zeros(B-i)
            for j in range(B-i):
                cand_Q = Q_inds + [rem_inds[j]]
                rep_scores[j] = np.sum(
                    np.max(sims[:, cand_Q], axis=1))
            iter_sel = rem_inds[np.argmax(rep_scores)]
            # update the iterating sets
            Q_inds += [iter_sel]
            rem_inds = np.delete(
                rem_inds, np.argmax(rep_scores))
            
        Q_inds = sel_inds[Q_inds]

    return Q_inds

def run_CNNAL(A, init_X_train, init_Y_train,
              X_pool, Y_pool, X_test, Y_test, epochs, 
              k, B, method, max_queries, train_batch=50, 
              eval_batch=None):
    """Starting with a CNN model that is trained with an initial
    labeled data set, and then perform certain number of querying 
    iterations using a specified active learning method
    """
    
    test_acc = []
    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, A.save_path)
        test_acc += [A.accuracy.eval(feed_dict={
                    A.x: X_test, A.y_:Y_test})]
        print()
        print('Test accuracy: %g' %test_acc[0])

        # start querying
        new_X_train, new_Y_train = init_X_train, init_Y_train
        new_X_pool, new_Y_pool = X_pool, Y_pool
        A.get_gradients()
        # number of selected in each iteration is useful
        # when samling from a distribution and repeated
        # queries might be present
        query_num = []
        print(20*'-' + '  Querying  ' +20*"-")
        t = 0
        while sum(query_num) < max_queries:
            print("Iteration %d: "% t)
            Q_inds = CNN_query(A, k, B, new_X_pool, 
                               method, session, eval_batch)
            query_num += [len(Q_inds)]
            print('Query index: '+' '.join(str(q) for q in Q_inds))
            # prepare data for another training
            Q = new_X_pool[Q_inds,:,:,:]
            #pickle.dump(Q, open('results/%s/%d.p'% (method,t),'wb'))
            Y_Q = new_Y_pool[:,Q_inds]
            # remove the selected queries from the pool
            new_X_pool = np.delete(new_X_pool, Q_inds, axis=0)
            new_Y_pool = np.delete(new_Y_pool, Q_inds, axis=1)
            # update the model
            print("Updating the model: ", end='')
            new_X_train, new_Y_train = NNAL_tools.prepare_finetuning_data(
                new_X_train, new_Y_train, Q, Y_Q, 200+t, 50)
            for i in range(epochs):    
                A.train_graph_one_epoch(new_X_train, new_Y_train, 
                                        train_batch, session)
                print(i, end=', ')

            test_acc += [A.accuracy.eval(
                    feed_dict={A.x: X_test, A.y_:Y_test})]
            print()
            print('Test accuracy: %g' %test_acc[t+1])
            t += 1
            
    return np.array(test_acc), np.append(0, np.array(query_num))
            
        
