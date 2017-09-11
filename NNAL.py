import numpy as np
import tensorflow as tf
import pdb
import sys
import NNAL_tools


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

def CNN_query(model, k, B, pool_X, method, session, batch_size=None):
    """Querying a number of unlabeled samples from a given pool
    """

    if method=='expected_change':
        # uncertainty filtering
        print("Computing the posteriors...")
        if batch_size:
            posteriors = NNAL_tools.batch_posteriors(
                model, pool_X, batch_size, session)
        else:
            posteriors = session.run(
                model.posteriors, feed_dict={model.x:pool_X})
            
        sel_inds = NNAL_tools.uncertainty_filtering(posteriors, B)
        sel_posteriors = posteriors[:, sel_inds]

        # FI scoring
        print("Computing the scores..")
        c = model.output.get_shape()[0].value
        scores = np.zeros(B)
        for i in range(B):
            # gradients of samples one-by-one
            #pdb.set_trace()
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
        Q_inds = np.argsort(-scores)[:k]

    return Q_inds
