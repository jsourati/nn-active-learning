import numpy as np
import tensorflow as tf
import pdb
import sys
import NNAL_tools


def test_MNIST(iters, B, k, init_size, batch_size, epochs):
    """Evaluate active learning based on Fisher information,
    or equivalently expected change of the model, over MNIST
    data set
    """
    c = 10
    accs = np.zeros(iters+1)
    
    # preparing MNIST data set
    batch_of_data, batch_of_labels, pool_images, pool_labels, \
        test_images, test_labels = NNAL_tools.init_MNIST(init_size, batch_size)
    
    # initial training
    with tf.Session() as sess:
        
        print("Initializing the model...")
        
        # input and output placeholders
        x = tf.placeholder(tf.float32, shape=[784, None])
        y_ = tf.placeholder(tf.float32, shape=[10, None])

        # parameters
        W = tf.Variable(tf.zeros([10, 784]))
        b = tf.Variable(tf.zeros([10,1]))
        
        # initializing
        sess.run(tf.global_variables_initializer())

        # outputs of the network
        y = tf.matmul(W,x) + b
        posteriors = tf.nn.softmax(tf.transpose(y))
        log_posteriors = tf.log(posteriors)
        
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
        accs[0] = accuracy.eval(feed_dict={x: test_images, 
                                           y_: test_labels})
        
        # forming the gradients list
        print("Listing the gradients..")
        grads = NNAL_tools.enlist_gradients(log_posteriors, B, [W,b])
        
        # start the querying iterations
        finetune_epochs = 1
        # optimization step for fine-tuning
        finetune_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        print("Starting the querying iterations..")
        new_train_data = []
        new_train_labels = []
        for t in range(1, iters+1):
            # compute all the posterior probabilities
            pool_posteriors = sess.run(posteriors, feed_dict={x: pool_images, y_: pool_labels})
            
            # uncertain filtering
            filtered_pool_inds = NNAL_tools.uncertainty_filtering(pool_posteriors.T, B)
            
            # compute all the listed gradients in one single call
            print("Computing the gradients..")
            all_grads = sess.run(grads, feed_dict={x: pool_images[:, filtered_pool_inds], 
                                                   y_: pool_labels[:, filtered_pool_inds]})
            selected_posteriors = pool_posteriors[filtered_pool_inds, :]
            scores = np.zeros(B)
            for i in range(B):
                for j in range(c):
                    # summation of derivative-squared
                    class_grad_squared = np.sum(all_grads[i*c+j][0]**2)
                    class_grad_squared += np.sum(all_grads[i*c+j][1]**2)
                    scores[i] += class_grad_squared*pool_posteriors[i, j]
                    
            # take the best k scores
            Q = np.argsort(-scores)[:k]
            Q = filtered_pool_inds[Q]
            #Q = np.random.randint(0, pool_images.shape[0], k)
            
            new_train_data = pool_images[:,Q]
            new_train_labels = pool_labels[:,Q]
                
            if False:
                batch_of_data, batch_of_labels = \
                    NNAL_tools.update_batches(batch_of_data, 
                                              batch_of_labels,
                                              new_train_data,
                                              new_train_labels,
                                              'emphasized')
            else:
                batch_of_data, batch_of_labels = \
                    NNAL_tools.update_batches(batch_of_data, 
                                              batch_of_labels,
                                              new_train_data,
                                              new_train_labels,
                                              'regular')
            
            # fine-tuning
            sess.run(tf.global_variables_initializer())
            for _ in range(50):
                for i in range(len(batch_of_data)):
                    train_step.run(feed_dict={x: batch_of_data[i], 
                                              y_: batch_of_labels[i]})

            accs[t] = accuracy.eval(feed_dict={x: test_images, 
                                               y_: test_labels})
            # update the pool
            np.delete(pool_images, Q, 1)
            np.delete(pool_labels, Q, 1)
            
            nL = np.concatenate(batch_of_data, axis=1).shape[1]
            print("Iteration %d is done. Number of labels: %d" % (t, nL))
            
    return accs
