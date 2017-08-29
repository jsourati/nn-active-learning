import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import copy
import pdb
import sys

read_file_path = "/home/ch194765/repos/atlas-active-learning/"
sys.path.insert(0, read_file_path)
import prep_dat

def train_CNN_MNIST(epochs, batch_size):
    """Trianing a classification network for MNIST data set which includes
    two layers of convolution (CNN) followed by two fully connected  layers.
    """
    
    # data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    x = tf.placeholder(tf.float32, [784, None])
    x_image = tf.reshape(tf.transpose(x), [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, [10, None])
    
    # creating the CNN layers
    W_dict, b_dict = CNN_variables([5,5], [1, 32, 64])
    CNN_output = CNN_layers(W_dict, b_dict, x_image)
    
    # creating the first fully-connected layer
    CNN_output = tf.reshape(tf.transpose(CNN_output), [7*7*64, -1])
    W_fc1 = weight_variable([1024, 7*7*64])
    b_fc1 = bias_variable([1024,1])
    fc1_output = tf.nn.relu(tf.matmul(W_fc1, CNN_output) + b_fc1)
    
    # applying drop-out
    keep_prob = tf.placeholder(tf.float32)
    fc1_output_drop = tf.nn.dropout(fc1_output, keep_prob)
    
    # creating the second (last) fully-connected layer
    W_fc2 = weight_variable([10, 1024])
    b_fc2 = bias_variable([10,1])
    y = tf.matmul(W_fc2, fc1_output_drop) + b_fc2
    
    # training this network
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(y_), 
                                                logits=tf.transpose(y)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 0), tf.argmax(y_, 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    train_size = mnist.train.images.shape[0]
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for i in range(epochs):
            # preparing batches
            batch_inds = prep_dat.gen_batch_inds(train_size, batch_size)
            batch_of_data = prep_dat.gen_batch_matrices(mnist.train.images.T, batch_inds)
            batch_of_labels = prep_dat.gen_batch_matrices(mnist.train.labels.T, batch_inds)

            #batch = mnist.train.next_batch(50)
            for j in range(len(batch_of_data)):
                if j % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                            x: batch_of_data[j], y_: batch_of_labels[j], keep_prob: 1.0})
                    print('epoch %d-iteratin %d, training accuracy %g' % (i, j, train_accuracy))
            
                train_step.run(feed_dict={x: batch_of_data[j], y_: batch_of_labels[j], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
                    x: mnist.test.images.T, y_: mnist.test.labels.T, keep_prob: 1.0}))
        
        # extracting features, in a loop
        print("Extracting the features..")
        d_extracted = 1024
        train_features = np.zeros((d_extracted, mnist.train.images.shape[0]))
        for j in range(len(batch_of_data)):
            train_features[:,batch_inds[j]] = sess.run(fc1_output, feed_dict={
                    x: batch_of_data[j], y_: batch_of_labels[j], keep_prob: 1.0})
        test_features = sess.run(fc1_output, feed_dict={
                x: mnist.test.images.T, y_: mnist.test.labels.T, keep_prob: 1.0})
        
    return train_features, test_features

def CNN_layers(W_dict, b_dict, x):
    """Creating the output of CNN layers and return them as TF variables
    
    Each layer consists of a convolution, following by a max-pooling and
    a ReLu activation.
    The number of channels of the input, should match the number of
    input channels to the first layer based on the parameter dictionary.
    """
    
    L = len(W_dict)
    
    output = x
    for i in range(L):
        output = tf.nn.conv2d(output, W_dict[str(i)], 
                              strides=[1, 1, 1, 1], 
                              padding='SAME') + b_dict[str(i)]
        output = tf.nn.relu(output)
        output = max_pool(output, 2, 2)
        
    return output
    

def CNN_variables(kernel_dims, layer_list):
    """Creating the CNN variables
    
    We should have `depth_lists[0] = in_channels`.
    In the i-th layer, dimensionality of of the kernel `W` would be
    `(kernel_dims[i],kernel_dims[i])`, and the number of them (that is,
    the number of filters) would be `layer_list[i+1]`. Moreover, the number
    of its input channels is `layer_list[i]`.
    """
    
    if not(len(layer_list)==len(kernel_dims)+1):
        raise ValueError("List of  layers should have one more"+
                         "element than the list of kernel dimensions.")
    
    W_dict = {}
    b_dict = {}
    
    layer_num = len(layer_list)
    # size of W should be [filter_height, filter_width, in_channels, out_channels]
    # here filter_height = filter_width = kernel_dim
    for i in range(layer_num-1):
        W_dict.update({str(i):weight_variable([kernel_dims[i], kernel_dims[i], 
                                               layer_list[i], layer_list[i+1]])})
        b_dict.update({str(i): bias_variable([layer_list[i+1]])})
        
    return W_dict, b_dict


def weight_variable(shape):
    """Creating a kernel tensor with specified shape
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Creating a bias term with specified shape
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

    
def max_pool(x, w_size, stride):
    return tf.nn.max_pool(x, ksize=[1, w_size, w_size, 1],
                          strides=[1, stride, stride, 1], padding='SAME')
    

    
