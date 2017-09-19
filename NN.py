import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import copy
import pdb
import sys
import cv2
import NNAL_tools

read_file_path = "/home/ch194765/repos/atlas-active-learning/"
sys.path.insert(0, read_file_path)
import prep_dat

read_file_path = "/home/ch194765/repos/atlas-active-learning/AlexNet"
sys.path.insert(0, read_file_path)
import alexnet


def AlexNet_features(img_arr):
    """Extracting features from the pretrained alexnet 
    """
    
    tf.reset_default_graph()

    # creating the network
    # placeholder for input and dropout rate
    x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    
    # create model with default config 
    # ( == no skip_layer and 1000 units in the last layer)
    model = alexnet.AlexNet(
        x, keep_prob, 1000, [], 
        weights_path='/home/ch194765/repos/atlas-active-learning/AlexNet/bvlc_alexnet.npy')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Load the pretrained weights into the model
        model.load_initial_weights(sess)
        
        # extract the features
        features = sess.run(model.feature_layer, 
                            feed_dict={x: img_arr, keep_prob: 1})
        
    return features


class CNN(object):
    """Class of CNN models
    """
    
    def __init__(self, x, layer_dict, name):
        """Constructor takes the input placehoder, a dictionary
        whose keys are names of the layers and the items assigned to each
        key is a 2-element list inlcuding  depth of this layer and its type,
        and a name which will be assigned to the scope name of the variabels
        
        Type of each layer is either 'fc' (for fully connected) and 'conv' 
        for convolutional layers. Moreover, depth of each layer will be
        equal to the number of nodes (in fully-connected layers) and 
        number of filters (in convolutional layers). This list for 
        convolutional layers should have three elements, where the last
        element specifies the kernel size of that layer.
        
        The assumption is that at least the first layer is a CNN, hence
        depth of the input layer is the number of channels of the input.
        It is further assumed that the last layer of the network is
        not a CNN
        """
        
        self.x = x
        self.layer_type = []
        self.name = name
        
        # creating the network's variables
        self.var_dict = {}
        layer_names = list(layer_dict.keys())

        with tf.name_scope(name):
            for i, layer_name in enumerate(layer_dict):
                # extract previous depth
                if i==0:
                    prev_depth = x.shape[-1].value
                    output = x
                    
                if layer_dict[layer_name][1]=='conv':
                    kernel_dim = layer_dict[layer_name][2]
                    self.var_dict.update({
                            layer_name: [
                                weight_variable([kernel_dim[0], kernel_dim[1], 
                                                 prev_depth, layer_dict[layer_name][0]], 
                                                name=layer_name+'_weight'),
                                bias_variable([layer_dict[layer_name][0]],
                                              name=layer_name+'_bias')]
                            })

                    # output of the layer
                    output = tf.nn.conv2d(
                        output, self.var_dict[layer_name][0], strides=[1,1,1,1],
                        padding='SAME') + self.var_dict[layer_name][1]
                    output = tf.nn.relu(output)
                    output = max_pool(output, 2, 2)
                    
                    self.layer_type += ['conv']
                    # storing depth of the current layer for the next one
                    # if the next layer is fully-connected, the depth of this layer
                    # would be total number of neurons (and not just the channls)
                    if layer_dict[layer_names[i+1]][1]=='fc':
                        prev_depth = np.prod(output.get_shape()[1:]).value
                        output = tf.reshape(tf.transpose(output), [prev_depth, -1])
                    else:
                        prev_depth = output.get_shape()[-1].value
                    
                elif layer_dict[layer_name][1]=='fc': 
                    self.var_dict.update({
                            layer_name:[
                                weight_variable([layer_dict[layer_name][0], prev_depth], 
                                                name=layer_name+'_weight'),
                                bias_variable([layer_dict[layer_name][0], 1],
                                              name=layer_name+'_bias')]
                            })
                    self.layer_type += ['fc']
                    
                    # output of the layer
                    output = tf.matmul(
                        self.var_dict[layer_name][0], output) + self.var_dict[layer_name][1]
                    # apply relu activation only if we are NOT at the last layer 
                    if i < len(layer_dict)-1:
                        output = tf.nn.relu(output)
                        # set the output of the layer one before last as 
                        # the features that the network will extract
                        if i==len(layer_dict)-2:
                            self.features = output
                    prev_depth = layer_dict[layer_name][0]
                    
                else:
                    raise ValueError("Layer's type should be either 'fc'" + 
                                     "or 'conv'.")

            self.output = output
            # posterior
            posteriors = tf.nn.softmax(tf.transpose(output))
            self.posteriors = tf.transpose(posteriors)
                
    def initialize_graph(self, init_X_train, init_Y_train, 
                         train_batch, epochs,  addr=None):
        """Initializing a graph given an initial training data set 
        and saving the results if necessary
        """
        init = tf.global_variables_initializer()        
        with tf.Session() as sess:
            sess.run(init)
            
            print(20*'-' + '  Initialization  ' +20*"-")
            print("Epochs: ", end='')
            for i in range(epochs):    
                self.train_graph_one_epoch(
                    init_X_train, 
                    init_Y_train, 
                    train_batch, sess)

                print(i, end=', ')
            if addr:
                self.save_model(addr, sess)
        
    def save_model(self, addr, session):
        """Saving the current model
        """
        saver = tf.train.Saver()
        saver.save(session, addr)
        self.save_path = addr
        
    def extract_features(self, X, session, batch_size=None):
        """Extracting features
        """
        
        n = X.shape[0]
        if batch_size:
            d = self.features.get_shape()[0].value
            features = np.zeros((d, n))
            quot, rem = np.divmod(n, batch_size)
            for i in range(quot):
                if i<quot-1:
                    inds = np.arange(i*batch_size, (i+1)*batch_size)
                else:
                    inds = slice(i*batch_size, n)
                    
                iter_X = X[inds,:,:,:]
                features[:,inds] = session.run(
                    self.features, feed_dict={self.x:iter_X})
                
        else:
            features = session.run(
                self.features, feed_dict={self.x:X})
            
        return features
                    
        
    def get_optimizer(self, learning_rate):
        """Form the loss function and optimizer of the CNN graph
        """
        
        # number of classes
        c = self.output.get_shape()[0].value
        self.y_ = tf.placeholder(tf.float32, [c, None])
        
        # loss function
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.transpose(self.y_), 
                logits=tf.transpose(self.output)))
        
        # optimizer
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
        # define the accuracy
        correct_prediction = tf.equal(tf.argmax(self.output, 0), 
                                      tf.argmax(self.y_, 0))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def get_gradients(self):
        """Forming gradients of the log-posteriors
        """
        
        # collect all the trainable variabels
        Theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                  scope=self.name)
        
        self.grad_log_posts = {}
        c = self.output.get_shape()[0].value
        for j in range(c):
            self.grad_log_posts.update(
                {str(j): tf.gradients(tf.log(self.posteriors)[j, 0], Theta)})

        
    def train_graph_one_epoch(self, X_train, Y_train, batch_size, session):
        """Randomly partition the data into batches and complete one
        epoch of training
        
        Input feature vectors, `X_train` and labels, `Y_train` are columnwise
        """
        
        # random partitioning into batches
        train_size = X_train.shape[0]
        batch_inds = prep_dat.gen_batch_inds(train_size, batch_size)
        batch_of_data = prep_dat.gen_batch_tensors(X_train, batch_inds)
        batch_of_labels = prep_dat.gen_batch_matrices(Y_train, batch_inds)
        
        # completing an epoch
        for j in range(len(batch_of_data)):
            #if False:
            #    acc = self.accuracy.eval(feed_dict={
            #            self.x: batch_of_data[j], 
            #            self.y_: batch_of_labels[j]})
            
            session.run(self.train_step, feed_dict={self.x: batch_of_data[j], 
                                                    self.y_: batch_of_labels[j]})
    

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
        c = 10
        train_features = np.zeros((d_extracted, mnist.train.images.shape[0]))
        train_labels = np.zeros((c, mnist.train.images.shape[0]))
        for j in range(len(batch_of_data)):
            train_features[:,batch_inds[j]] = sess.run(fc1_output, feed_dict={
                    x: batch_of_data[j], y_: batch_of_labels[j], keep_prob: 1.0})
            train_labels[:, batch_inds[j]] = mnist.train.labels[batch_inds[j], :].T
        test_features = sess.run(fc1_output, feed_dict={
                x: mnist.test.images.T, y_: mnist.test.labels.T, keep_prob: 1.0})
        
    return (train_features, train_labels), (test_features, mnist.test.labels.T)   
    

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


def weight_variable(shape, name=None):
    """Creating a kernel tensor with specified shape
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """Creating a bias term with specified shape
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

    
def max_pool(x, w_size, stride):
    return tf.nn.max_pool(x, ksize=[1, w_size, w_size, 1],
                          strides=[1, stride, stride, 1], padding='SAME')
    

    
