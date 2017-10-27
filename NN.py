import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import copy
import pdb
import sys
#import cv2
import NNAL_tools

read_file_path = "/home/ch194765/repos/atlas-active-learning/"
sys.path.insert(0, read_file_path)
import prep_dat

read_file_path = "/home/ch194765/repos/atlas-active-learning/AlexNet"
sys.path.insert(0, read_file_path)
import alexnet
from alexnet import AlexNet


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
    
    def __init__(self, x, layer_dict, name, feature_layer=None):
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
        not a CNN.
        
        Also, there is the option of specifying a layer whose output
        could be used extracted feature vectors of the input samples.
        """
        
        self.x = x
        self.layer_type = []
        self.name = name
        
        # creating the network's variables
        self.var_dict = {}
        layer_names = list(layer_dict.keys())

        with tf.name_scope(name):
            for i in range(len(layer_dict)-1):
                # extract previous depth
                if i==0:
                    #prev_depth = x.shape[-1].value
                    self.output = x
                
                self.add_layer(
                    layer_dict[layer_names[i]], 
                    layer_names[i],
                    layer_dict[layer_names[i+1]][1])
                
                # set the output of the layer one before last as 
                # the features that the network will extract
                if i==feature_layer:
                    self.features = self.output
                
                self.layer_type += [
                    layer_dict[layer_names[i]][1]]
            
            self.add_layer(
                layer_dict[layer_names[-1]],
                layer_names[-1],
                last_layer=True)
            
            self.layer_type += [
                layer_dict[layer_names[-1]][1]]
            
            # posterior
            posteriors = tf.nn.softmax(tf.transpose(self.output))
            self.posteriors = tf.transpose(posteriors)
            
    def add_layer(self, layer_specs, name, 
                  next_layer_type=None, 
                  last_layer=True):
        """Adding a layer to the graph
        
        Type of the next layer should also be given so that
        the appropriate output can be prepared
        """
        
        if layer_specs[1]=='conv':
            # if the next layer is fully-connected we need
            # to output a flatten tensor
            if next_layer_type=='fc':
                pdb.set_trace()
                self.add_conv(layer_specs, name, flatten=True)
            else:
                self.add_conv(layer_specs, name, flatten=False)

        elif layer_specs[1]=='fc': 
            # apply relu activation only if we are NOT 
            # at the last layer 
            if last_layer:
                self.add_fc(layer_specs, name, activation=False)
            else:
                self.add_fc(layer_specs, name, activation=True)

        elif layer_specs[1] == 'pool':
            # if the next layer is fully-connected we need
            # to output a flatten tensor
            if next_layer_type=='fc':
                self.add_pool(layer_specs, flatten=True)
            else:
                self.add_pool(layer_specs, flatten=False)
        else:
            raise ValueError("Layer's type should be either 'fc'" + 
                             ", 'conv' or 'pool'.")
                
                
    def add_conv(self, layer_specs, name, flatten=True):
        """Adding a convolutional layer to the graph given 
        the specifications
        """
        kernel_dim = layer_specs[2]
        prev_depth = self.output.get_shape()[-1].value
        
        self.var_dict.update(
            {name: [
                    weight_variable([kernel_dim[0], kernel_dim[1], 
                                     prev_depth, layer_specs[0]], 
                                    name=name+'_weight'),
                    bias_variable([layer_specs[0]],
                                  name=name+'_bias')]
             }
            )
        # output of the layer
        output = tf.nn.conv2d(
            self.output, 
            self.var_dict[name][0], 
            strides=[1,1,1,1],
            padding='SAME') + self.var_dict[name][1]
        self.output = tf.nn.relu(output)
        
        # if the flatten flag is True, flatten the output tensor
        # into a 2D array, where each column has a vectorized
        # tensor in it
        if flatten:
            out_size = np.prod(self.output.get_shape()[1:]).value
            self.output = tf.reshape(
                tf.transpose(self.output), [out_size, -1])
    
    def add_fc(self, layer_specs, name, activation=True):
        """Adding a fully-connected layer with a given 
        specification to the graph
        """
        prev_depth = self.output.get_shape()[0].value
        self.var_dict.update(
            {name:[
                    weight_variable(
                        [layer_specs[0], prev_depth], 
                        name=name+'_weight'),
                    bias_variable(
                        [layer_specs[0], 1],
                        name=name+'_bias')]
                }
            )
        # output of the layer
        self.output = tf.matmul(self.var_dict[name][0], 
                           self.output) + self.var_dict[name][1]
        # apply activation function if necessary
        if activation:
            self.output = tf.nn.relu(self.output)
            
    def add_pool(self, layer_specs, flatten=False):
        """Adding a (max-)pooling layer with given specifications
        """
        pool_size = layer_specs[0]
        self.output = max_pool(self.output, 
                               pool_size[0], 
                               pool_size[1])
        # flatten the output if necessary
        if flatten:
            out_size = np.prod(self.output.get_shape()[1:]).value
            self.output = tf.reshape(
                tf.transpose(self.output), [out_size, -1])
            
    def add_unpool(self, layer_specs, flatten=False):
        """Adding an unpooling layer as the opposite layer of a
        pooling one, to increase size of the output
        
        For now, we are using NN interpolation.
        """
        pool_size = layer_specs[0]
            
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
        self.train_step = tf.train.AdamOptimizer(
            learning_rate).minimize(loss)
        
        # define the accuracy
        correct_prediction = tf.equal(tf.argmax(self.output, 0), 
                                      tf.argmax(self.y_, 0))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
                                               tf.float32))
        
    def get_gradients(self, start_layer=0):
        """Forming gradients of the log-posteriors
        """
        
        # collect all the trainable variabels
        gpars = tf.trainable_variables()[start_layer*2:]
        
        self.grad_log_posts = {}
        c = self.output.get_shape()[0].value
        for j in range(c):
            self.grad_log_posts.update(
                {str(j): tf.gradients(tf.log(self.posteriors)[j, 0], 
                                      gpars)})

        
    def train_graph_one_epoch(self, X_train, Y_train, batch_size, session):
        """Randomly partition the data into batches and complete one
        epoch of training
        
        Input feature vectors, `X_train` and labels, `Y_train` are columnwise
        """
        
        # random partitioning into batches
        train_size = X_train.shape[0]
        if train_size > batch_size:
            batch_inds = prep_dat.gen_batch_inds(
                train_size, batch_size)
            batch_of_data = prep_dat.gen_batch_tensors(
                X_train, batch_inds)
            batch_of_labels = prep_dat.gen_batch_matrices(
                Y_train, batch_inds, col=False)
        else:
            batch_of_data = [X_train]
            batch_of_labels = [Y_train]
        
        # completing an epoch
        for j in range(len(batch_of_data)):
            session.run(self.train_step, 
                        feed_dict={self.x: batch_of_data[j], 
                                   self.y_: batch_of_labels[j]})
    
class AlexNet_CNN(AlexNet):
    """
    """
    
    def __init__(self, x, dropout_rate, c, skip_layer, weights_path):
        self.x = x
        self.dropout_rate = dropout_rate
        keep_prob = tf.placeholder(tf.float32)
        AlexNet.__init__(self, self.x, keep_prob, c, 
                         skip_layer, weights_path)
        self.output = self.fc8
        self.posteriors = tf.nn.softmax(self.output)
        self.weights_path = weights_path
        
        
    def initialize_graph(self, session, addr=None):
        session.run(tf.global_variables_initializer())
        self.load_initial_weights(session)
        if addr:
            saver = tf.train.Saver()
            saver.save(session, addr)

    
    def extract_features(self, X, session, batch_size=None):
        """Extracting features
        """
        
        n = X.shape[0]

        # do not drop-out any nodes when extracting features
        if batch_size:
            d = self.feature_layer.shape[1].value
            features = np.zeros((d, n))
            quot, rem = np.divmod(n, batch_size)
            for i in range(quot):
                if i<quot-1:
                    inds = np.arange(i*batch_size, (i+1)*batch_size)
                else:
                    inds = slice(i*batch_size, n)
                    
                iter_X = X[inds,:,:,:]
                features[:,inds] = session.run(
                    self.feature_layer, 
                    feed_dict={self.x:iter_X, 
                               self.KEEP_PROB:1.}
                    ).T
                
        else:
            features = session.run(
                self.feature_layer, 
                feed_dict={self.x:X,
                           self.KEEP_PROB: 1.}
                ).T
            
        return features
        
        
    def get_optimizer(self, learning_rate):
        """Making the optimizer operation for the graph
        """
        # note that for AlexNet the output is row-wise
        c = self.output.get_shape()[1].value
        self.y_ = tf.placeholder(tf.float32, [None, c])
        
        # loss function
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits = self.output, labels = self.y_))
        # training operation
        self.pars = tf.trainable_variables()
        gradients = tf.gradients(loss, self.pars)
        gradients = list(zip(gradients, self.pars))

        # Create optimizer and apply gradient descent 
        # to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate)
        self.train_step = optimizer.apply_gradients(
            grads_and_vars=gradients)
        
        # also define the accuracy operation
        correct_pred = tf.equal(
            tf.argmax(self.output, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32))
        
    def get_gradients(self, start_layer=0):
        """Forming gradients of the log-posteriors
        """
        
        self.grad_log_posts = {}
        c = self.output.get_shape()[1].value
        
        # ys
        gpars = self.pars[start_layer*2:]

        for j in range(c):
            self.grad_log_posts.update(
                {str(j): tf.gradients(
                        ys=tf.log(self.posteriors)[0, j], 
                        xs=gpars,
                        grad_ys=1.)
                 }
                )

        
    def train_graph_one_epoch(self, X_train, Y_train, batch_size, session):
        """Randomly partition the data into batches and complete one
        epoch of training
        
        Input feature vectors, `X_train` and labels, 
        `Y_train` are columnwise
        """
        
        # random partitioning into batches
        train_size = X_train.shape[0]
        if train_size > batch_size:
            batch_inds = prep_dat.gen_batch_inds(
                train_size, batch_size)
            batch_of_data = prep_dat.gen_batch_tensors(
                X_train, batch_inds)
            batch_of_labels = prep_dat.gen_batch_matrices(
                Y_train, batch_inds, col=False)
        else:
            batch_of_data = [X_train]
            batch_of_labels = [Y_train]
        
        # completing an epoch
        for j in range(len(batch_of_data)):
            
            session.run(
                self.train_step, 
                feed_dict={self.x: batch_of_data[j], 
                           self.y_: batch_of_labels[j],
                           self.KEEP_PROB: self.dropout_rate}
                )


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
    

    
