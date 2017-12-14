import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import linecache
import copy
import h5py
import pdb
import sys
import cv2
import os

import NNAL_tools
import AL

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
    
    def __init__(self, 
                 x, 
                 layer_dict, 
                 name,
                 feature_layer=None,
                 dropout=None):
        """Constructor takes the input placehoder, a dictionary
        whose keys are names of the layers and the items assigned to each
        key is a 2-element list inlcuding  depth of this layer and its type,
        and a name which will be assigned to the scope name of the variabels
        
        The constructor goes through all the layers one-by-one in the same
        order as the items of `layer_dict` dictionary, and add each layer
        over the previous one. Each time the layers is added the output of
        the model (stored in `self.output`) will be updated to be the output
        of the last layer. Hence, when we added a layer with an index equal
        to the given `feature_layer`, the marker `self.features` will be 
        make equal to the output of this layer. Moreover, if the dropout
        is supposed to be applied on this layer, the output of this layer
        will be dropped-out with the given probability, at the time of 
        training.
        
        The assumption is that at least the first layer is a CNN, hence
        depth of the input layer is the number of channels of the input.
        It is further assumed that the last layer of the network is
        not a CNN.
        
        Also, there is the option of specifying a layer whose output
        could be used extracted feature vectors of the input samples.
        
        :Parameters:
        
            **x** : Tensorflow placeholder in format [n_batch, (H, W), n_channel]
                Input to the network 
        
            **layer_dict** : dictionary
                Information about all layers of the network in format 
            
                    {layer_name: layer_characteristic}

                Layer's characteristic, in turn, contains two or three 
                items dependending on the type of the layer. At this time
                this class supports only three types of layers:
               
                - Convolutional:   [# output channel, 'conv', kernel size]
                - Fully-connected: [# output channel, 'fc']
                - max-pooling:     [pool size, 'pool']
                
                 Note that "kernel size" and "pool size" are a list with 
                 two elements.
        
            **name**: string
                Name of Tensorflow scope of all the variables defined in
                this class.
        
            **feature_layer** : int (default: None)
                If given, is the index of the layer whose output will be 
                marked as the features extracted from the network; this 
                index should be given in terms of the order of layers in
                `layer_dict`
        
            **dropout** : list of two elements (default: None)
                If any layer should be dropped out during the training,
                this list contains the layers that need to be dropped out
                (first item) and the drop-out rate (second item).
        """
        
        self.x = x
        self.layer_type = []
        self.name = name
        
        self.keep_prob = tf.placeholder(
            tf.float32, name='keep_prob')
        if dropout:
            self.dropout_layers = dropout[0]
            self.dropout_rate = dropout[1]
        else:
            self.dropout_layers = []
            self.dropout_rate = 1.
        
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
                
                # dropping out the output layers if the layer
                # is in the list of dropped-out layers
                if i in self.dropout_layers:
                    self.output = tf.nn.dropout(
                        self.output, self.keep_prob)
                
                # set the output of the layer one before last as 
                # the features that the network will extract
                if i==feature_layer:
                    self.feature_layer = self.output
                
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
            self.posteriors = tf.transpose(posteriors, 
                                           name='posteriors')
            
    def add_layer(self, 
                  layer_specs, 
                  name, 
                  next_layer_type=None, 
                  last_layer=True):
        """Adding a layer to the graph
        
        Type of the next layer should also be given so that
        the appropriate output can be prepared
        
        :Parameters:
        
            **layer_specs** : list of three elements
                 specification list of the layer with
                 a format explaned in `__init__` as the
                 items of `layer_dict`
        
            **name** : string
        
            **next_layer_type** : string
                determining type of the next layer so that
                the output will be provided accordingly
        
            **last_layer** : binary flag
                determining whether the layer is the 
                last one; if it is `True` there won't be
                any activation at the output
        """
        
        if layer_specs[1]=='conv':
            # if the next layer is fully-connected,
            # output a flattened tensor
            if next_layer_type=='fc':
                self.add_conv(layer_specs, 
                              name, 
                              flatten=True)
            else:
                self.add_conv(layer_specs, 
                              name, 
                              flatten=False)

        elif layer_specs[1]=='fc': 
            # apply relu activation only if we are NOT 
            # at the last layer 
            if last_layer:
                self.add_fc(layer_specs, 
                            name, 
                            activation=False)
            else:
                self.add_fc(layer_specs, 
                            name, 
                            activation=True)

        elif layer_specs[1] == 'pool':
            # if the next layer is fully-connected we need
            # to output a flatten tensor
            if next_layer_type=='fc':
                self.add_pool(layer_specs, 
                              flatten=True)
            else:
                self.add_pool(layer_specs, 
                              flatten=False)
        else:
            raise ValueError(
                "Layer's type should be either 'fc'" + 
                ", 'conv' or 'pool'.")
                
                
    def add_conv(self, layer_specs, name, flatten=True):
        """Adding a convolutional layer to the graph given 
        the specifications
        """
        kernel_dim = layer_specs[2]
        prev_depth = self.output.get_shape()[-1].value
        
        self.var_dict.update(
            {name: [
                weight_variable(
                    [kernel_dim[0], 
                     kernel_dim[1], 
                     prev_depth, 
                     layer_specs[0]], 
                    name=name+'_weight'),
                bias_variable(
                    [layer_specs[0]],
                    name=name+'_bias')
            ]
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
            out_size = np.prod(
                self.output.get_shape()[1:]).value
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
        self.output = tf.matmul(
            self.var_dict[name][0], 
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
            
    def initialize_graph(self, 
                         session,
                         path=None):
        """Initializing the graph, and if given loading
        a set of pre-trained weights into the model
        
        :Parameters:
        
            **session** : Tensorflow session
                The active session in which the model is
                running
        
            **pretr_name** : string (default: None)
                Name of the model that has pre-trained
                weights. It will be `None` if no pre-
                trained weights are given
        
            **path** : string (default: None)
                Path to the pre-trained weights, if the
                the given model has one
        """
        session.run(tf.global_variables_initializer())
        
        if path:
            if self.name=="VGG19":
                NNAL_tools.load_weights_VGG19(
                    self, path, session)
            

    def save_weights(self, file_path):
        """Saving only the parameter values of the 
        current model into a .h5 file
        
        The file will have as many groups as the number
        of layers in the model (which is equal to the
        number of keys in `self.var_dict`. Each group has
        two datasets, one for the weight W, and one for
        the bias b.
        """
        
        f = h5py.File(file_path, 'w')
        for layer_name, pars in self.var_dict.items():
            L = f.create_group(layer_name)
            L.create_dataset('Weight', data=pars[0].eval())
            L.create_dataset('Bias', data=pars[1].eval())
            
        f.close()
        
    def load_weights(self, file_path, session):
        """Loading parameter values saved in a .h5 file
        into the tensorflow variables of the class object
        
        The groups in the .h5 file should match the layers
        in the model. Specifically, name of each group 
        needs to be the same as the name of the layers
        in `self.var_dict` (this is autmatically satisfied
        if the .h5 file is generated using self.save_model().
        """
        
        f = h5py.File(file_path)
        for layer_name, pars in self.var_dict.items():
            # weight
            value_to_load = np.array(
                f[layer_name]['Weight'])
            session.run(pars[0].assign(value_to_load))
            
            # bias
            value_to_load = np.array(
                f[layer_name]['Bias'])
            session.run(pars[1].assign(value_to_load))
            
    def add_assign_ops(self, file_path):
        """Adding operations for assigning values to
        the nodes of the class's graph. This method
        is for creating repeatedly assigning values to
        the nodes after finalizing the graph. It should
        be called before `sess.graph.finalize()` to 
        create the operation nodes before finalizing.
        Then, the created operation nodes can be 
        performed without any need to create new nodes.
        
        Note that such repeated value assignment to the
        nodes are necessary for, say, querying iterations
        where after selecting each set of queries the model
        should be trained from scratch (or from the point
        that we saved the weights beforehand).
        
        This function, together with `self.perform_assign_ops`
        will be used instead of `self.load_weights` when 
        value assignment needs to be done repeatedly after
        finalizing the graph.
        """

        f = h5py.File(file_path)
        self.assign_dict = {}
        for layer_name, pars in self.var_dict.items():
            # weight
            weight_to_load = np.array(
                f[layer_name]['Weight'])
            bias_to_load = np.array(
                f[layer_name]['Bias'])
            
            self.assign_dict.update({
                layer_name: [
                    tf.assign(pars[0], weight_to_load),
                    tf.assign(pars[1], bias_to_load)]
            })
            
    def perform_assign_ops(self,sess):
        """Performing assignment operations that have
        been created by `self.add_assign_ops`.

        This function, together with `self.add_assign_ops`
        will be used instead of `self.load_weights` when 
        value assignment needs to be done repeatedly after
        finalizing the graph.
        """
        
        for _, assign_ops in self.assign_dict.items():
            sess.run(assign_ops)

        
    def extract_features(self, inds, 
                         expr, 
                         session):
        """Extracting features
        """
        
        n = len(inds)
        d = self.feature_layer.get_shape()[0].value
        features = np.zeros((d, n))
        batch_size = expr.pars['batch_size']
        # preparing batch_of_inds, whose
        # indices are in terms of "inds"
        if batch_size > n: 
            batch_of_inds = [np.arange(
                n).tolist()]
        else:
            batch_of_inds = gen_batch_inds(
                n, batch_size)

        # extracting the features
        for inner_inds in batch_of_inds:
            # loading the data for this patch
            X,_ = load_winds(inds[inner_inds],
                           expr.imgs_path_file,
                           expr.pars['target_shape'],
                           expr.pars['mean'])

            features[:,inner_inds] = session.run(
                self.feature_layer, 
                feed_dict={self.x: X,
                           self.keep_prob:1.})
            
        return features
                    
        
    def get_optimizer(self, learning_rate, 
                      train_layers=[]):
        """Form the loss function and optimizer of the CNN graph
        
        :Parameters;
        
            **learning_rate** : positive float
                learning rate of the optimization, which is 
                proportional to the step length of the descent

            **layer_list** : list of strings
                list of names of those layers that are to be
                modified in the training step; if empty all
                the layers will be included. This list should
                be a subset of `self.var_dict.keys()`.
        """
        
        # number of classes
        c = self.output.get_shape()[0].value
        self.y_ = tf.placeholder(tf.float32, 
                                 [c, None],
                                 name='labels')
        
        # loss function
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.transpose(self.y_), 
                logits=tf.transpose(self.output)),
            name='loss')
        
        tf.summary.scalar('Loss', loss)
        
        # optimizer
        if len(train_layers)==0:
            self.train_step = tf.train.AdamOptimizer(
                learning_rate).minimize(
                    loss, name='train_step')
        else:
            self.train_layers = train_layers
            # if some layers are specified, only
            # modify these layers in the training
            var_list = []
            for layer in train_layers:
                var_list += self.var_dict[layer]
            
            self.train_step = tf.train.MomentumOptimizer(
                learning_rate,0.5).minimize(
                    loss, var_list=var_list)
        
        # define the accuracy
        self.prediction = tf.argmax(
            self.posteriors, 0, name='prediction')
        
    def get_gradients(self, grad_layers=[]):
        """Forming gradients of the log-posteriors
        """
        
        # collect all the trainable variabels
        self.grad_layers = grad_layers
        if len(grad_layers)==0:
            gpars = tf.trainable_variables()
        else:
            gpars = []
            for layer in grad_layers:
                gpars += self.var_dict[layer]
        
        self.grad_posts = {}
        c = self.output.get_shape()[0].value
        for j in range(c):
            self.grad_posts.update(
                {str(j): tf.gradients(
                    self.posteriors[j, 0], 
                    gpars, name='score_class_%d'% j)
             }
            )

        
    def train_graph_one_epoch(self, expr,
                              train_inds,
                              session,
                              TB_opt={}):
        """Randomly partition the data into 
        batches and complete one epoch of training
        
        
        :Parameters:
        
            **expr** : AL.Experiment object
        
            **train_inds** : list or array of integers
                indices of the samples in terms of the
                `img_path_list` of the give experiment
                based on which the model is to be modified

            **batch_size** : integer
                size of the batch for training or
                evaluating the accuracy
        
            **session** : Tensorflow Session 

            **TB_opt** : dictionary (default={})
                If the results are to be saved for 
                Tensorboard's use, this dictionary
                includes all the necessary information
                for saving the TB files:

                * `summs`: the merged summaries that are
                           are created outisde the function
                * `writer`: file writer f the TB for these
                            this epoch (it contains the path
                            in which the TB files will be saved)
                * `epoch_id`: index of the current epoch
                * `tag`: a tag for the current training epoch
        """
        
        # random partitioning into batches
        batch_size = expr.pars['batch_size']
        train_size = len(train_inds)
        if train_size > batch_size:
            batch_of_inds = gen_batch_inds(
                train_size, batch_size)
        else:
            batch_of_inds = [np.arange(
                train_size).tolist()]
        
        # completing an epoch
        for j in range(len(batch_of_inds)):
            # create the 4D array of batch images
            iter_inds = train_inds[batch_of_inds[j]]
            batch_of_imgs, batch_of_labels = load_winds(
                iter_inds, 
                expr.imgs_path_file, 
                expr.pars['target_shape'],
                expr.pars['mean'],
                expr.labels_file)
            
            batch_of_labels = AL.make_onehot(
                batch_of_labels, expr.nclass)

            if TB_opt:
                summary, _ = session.run(
                    [TB_opt['summs'], self.train_step], 
                    feed_dict={self.x: batch_of_imgs, 
                               self.y_: batch_of_labels,
                               self.keep_prob: self.dropout_rate})
            else:
                session.run(
                    self.train_step, 
                    feed_dict={self.x: batch_of_imgs, 
                               self.y_: batch_of_labels,
                               self.keep_prob: self.dropout_rate})

            # writing tensorboard files if necessary
            # (every 50 iterations)
            if TB_opt:
                if j%50 == 0:
                    # compute the accuracy
                    train_preds = self.predict(
                        expr, train_inds, session)
                    iter_acc = AL.get_accuracy(
                        train_preds, expr.labels[:,train_inds])
                    # adding accuracy to a summary
                    acc_summary = tf.Summary()
                    acc_summary.value.add(
                        tag='Accuracy',
                        simple_value=iter_acc)

                    TB_opt['writer'].add_summary(
                        summary, 
                        TB_opt['epoch_id']*len(batch_of_inds)+j)
                    TB_opt['writer'].add_summary(
                        acc_summary, 
                        TB_opt['epoch_id']*len(batch_of_inds)+j)
                    
    def validated_train(self,
                        expr,
                        sess,
                        train_inds,
                        valid_ratio,
                        const_inds=None,
                        print_flag=False):
        """Validated training of a CNN model

        :Parameters:

            **model** : CNN object
                the CNN model which has the method
                `train_graph_one_epoch()`

            **expr** : active learning experiment
                the experiment's object which contains
                the path to data directories

            **sess** : Tensorflow session


            **train_inds** : array of positive integers
                indices of samples inside the training
                data set

            **valid_ratio** : positive float (<1)
                ratio of the validatio data set and the
                one that is used for training

            **cosnt_inds** : array of positive integerses
                If given, it represents a set of samples
                that are constrained to be inside the 
                partition that is used for fine-tuning
                (and not the validation data set)
        
        """

        # separate the training indices to validation
        # and the ones to be used for fine-tuning
        labels = np.loadtxt(expr.labels_file)
        tuning_inds, valid_inds = NNAL_tools.test_training_part(
            labels[train_inds], valid_ratio)

        if const_inds:
            tuning_inds = np.append(tuning_inds, const_inds)


        # best accuracy is the initial accuracy 
        # in the beginning (like sorting)
        predicts = self.predict(expr,valid_inds,sess)
        best_acc = AL.get_accuracy(predicts,
                                   expr.labels_file,
                                   valid_inds)
        # save the initial weights in the current directory
        # as the temporary "best" weights so far
        self.save_weights('tmp_weights.h5')
        print('init. acc.: %f'% best_acc)

        for i in range(expr.pars['epochs']):
            self.train_graph_one_epoch(expr,tuning_inds,sess)

            # validating the model after each epoch
            predicts = self.predict(expr,valid_inds,sess)
            acc = AL.get_accuracy(predicts,
                                  expr.labels_file,
                                  valid_inds)
            if acc > best_acc + 1e-6:
                best_acc = acc
                self.save_weights("tmp_weights.h5")
            
            if print_flag:
                print('%d- (%f,%f)'% (i,acc,best_acc))

        # after fix number of iterations load the best
        # weights that is stored
        self.load_weights("tmp_weights.h5", sess)

        # delete the temporary file
        os.remove("tmp_weights.h5")

        
    def predict(self, expr, 
                inds, 
                session):
        """Generate a set of predictions for a set of
        data points
        
        The predictions will be in form of class labels
        for test samples whose indices are saved in
        text.txt of the given experiment.
        """
        
        n = len(inds)
        batch_size = expr.pars['batch_size']
        if n > batch_size:
            batch_inds = gen_batch_inds(
                n, batch_size)
        else:
            batch_inds = [np.arange(len(n)).tolist()]
            
        predicts = np.zeros(n)
        for j in range(len(batch_inds)):
            # create the 4D array of the current batch
            iter_inds = inds[batch_inds[j]]
            batch_of_imgs, _ = load_winds(
                iter_inds, 
                expr.imgs_path_file, 
                expr.pars['target_shape'],
                expr.pars['mean'])
            
            predicts[batch_inds[j]] = session.run(
                self.prediction, 
                feed_dict={self.x: batch_of_imgs, 
                           self.keep_prob: 1.})

        return predicts

    
class AlexNet_CNN(AlexNet):
    """
    """
    
    def __init__(self, x, dropout_rate, 
                 c, skip_layer, gpu_id):
        self.x = x
        self.dropout_rate = dropout_rate
        keep_prob = tf.placeholder(tf.float32)
        AlexNet.__init__(self, self.x, keep_prob, c, 
                         skip_layer)
        self.output = self.fc8
        self.posteriors = tf.nn.softmax(self.output)
        self.gpu_id = gpu_id
        
        
    def initialize_graph(self, session,
                         weights_path):
        session.run(tf.global_variables_initializer())
        self.WEIGHTS_PATH=weights_path
        self.load_initial_weights(session)
    
    def extract_features(self, inds, 
                         img_path_list,
                         session, batch_size):
        """Extracting features
        """
        
        n = len(inds)
        d = self.feature_layer.shape[1].value
        features = np.zeros((d,n))
        # preparing batch_of_inds, whose
        # indices are in terms of "inds"
        if not(batch_size): 
            batch_of_inds = [np.arange(
                n).tolist()]
        else:
            batch_of_inds = gen_batch_inds(
                n, batch_size)

        # extracting the features
        for inner_inds in batch_of_inds:
            # loading the data for this patch
            X = load_winds(inds[inner_inds],
                           img_path_list)
            features[:,inner_inds] = session.run(
                self.feature_layer, 
                feed_dict={self.x: X, 
                           self.KEEP_PROB:1.}).T
                
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
        self.prediction = tf.argmax(self.posteriors, 1)
        correct_pred = tf.equal(
            self.prediction, tf.argmax(self.y_, 1))
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
        
    def train_graph_one_epoch(self, expr, train_inds, 
                              batch_size, session):
        """Randomly partition the data into batches 
        and complete one epoch of training
        
        :Parameters:
        
            **expr** : object of class AL.Experiment
                An active learning experiment class that
                has access to the the data path so that
                the input `train_inds` can be translated
                into real images

            **train_inds** : 1D array of integer indices
                List of indices tha can be used to access
                the training images through the `expr`.
                The indices are stored with respect to
                the path list `expr.im_path_list`, and
                equivalently their labels `expr.labels`.

            **batch_size** : int
                Size of the training mini-batches
        
            **session** : Tensorflow session
                An active tensorflow session within 
                which the model is running
        """
        
        # random partitioning into batches
        train_size = len(train_inds)
        if train_size > batch_size:
            batch_of_inds = gen_batch_inds(
                train_size, batch_size)
        else:
            batch_of_inds = [np.arange(
                train_size).tolist()]
        
        # completing an epoch
        for j in range(len(batch_of_inds)):
            # create the 4D array of batch images
            iter_inds = train_inds[batch_of_inds[j]]
            batch_of_imgs, batch_of_labels = load_winds(
                iter_inds, expr, True)
            session.run(
                self.train_step, 
                feed_dict={self.x: batch_of_imgs, 
                           self.y_: batch_of_labels.T,
                           self.KEEP_PROB: self.dropout_rate}
                )
            
    def predict(self, expr, test_inds, 
                batch_size, session):
        """Generate a set of predictions for a set of
        data points
        
        The predictions will be in form of class labels
        for test samples whose indices are saved in
        text.txt of the given experiment.
        """
        
        test_size = len(test_inds)
        if test_size > batch_size:
            batch_inds = gen_batch_inds(
                test_size, batch_size)
        else:
            batch_inds = [np.arange(len(
                test_inds)).tolist()]
            
        n = len(test_inds)
        predicts = np.zeros(n)
        for j in range(len(batch_inds)):
            # create the 4D array of the current batch
            iter_inds = test_inds[batch_inds[j]]
            batch_of_imgs = load_winds(
                iter_inds, expr.img_path_list)
            
            predicts[batch_inds[j]] = session.run(
                self.prediction, 
                feed_dict={self.x: batch_of_imgs, 
                           self.KEEP_PROB: 1.})

        return predicts

def create_model(model_name,
                 dropout_rate, 
                 n_class,
                 learning_rate, 
                 grad_layers=[],
                 train_layers=[]):
    
    if model_name=='Alex':
        model = create_Alex(dropout_rate, 
                            n_class,
                            learning_rate, 
                            starting_layer)
    elif model_name=='VGG19':
        model = create_VGG19(dropout_rate, 
                             learning_rate,
                             n_class, 
                             grad_layers,
                             train_layers)
        
    return model

def create_Alex(dropout_rate,
                n_class,
                learning_rate,
                starting_layer):
    """Creating an AlexNet model 
    using `AlexNet_CNN` class
    """

    x = tf.placeholder(tf.float32, 
                       [None, 227, 227, 3])
    skip_layer = ['fc8']
    model = AlexNet_CNN(
        x, dropout_rate, n_class, skip_layer)
    
    model.get_optimizer(learning_rate)
    
    # getting the gradient operations
    model.get_gradients(starting_layer)
    
    return model

def create_VGG19(dropout_rate, learning_rate,
                 n_class, grad_layers,
                 train_layers):
    """Creating a VGG19 model using CNN class
    """
    
    # architechture dictionary
    vgg_dict = {'conv1':[64, 'conv', [3,3]],
                'conv2':[64, 'conv', [3,3]],
                'max1': [[2,2], 'pool'],
                'conv3':[128, 'conv', [3,3]],
                'conv4':[128, 'conv', [3,3]],
                'max2' :[[2,2], 'pool'],
                'conv5':[256, 'conv', [3,3]],
                'conv6':[256, 'conv', [3,3]],
                'conv7':[256, 'conv', [3,3]],
                'conv8':[256, 'conv', [3,3]],
                'max3': [[2,2], 'pool'],
                'conv9': [512, 'conv', [3,3]],
                'conv10':[512, 'conv', [3,3]],
                'conv11':[512, 'conv', [3,3]],
                'conv12':[512, 'conv', [3,3]],
                'max4': [[2,2], 'pool'],
                'conv13':[512, 'conv', [3,3]],
                'conv14':[512, 'conv', [3,3]],
                'conv15':[512, 'conv', [3,3]],
                'conv16':[512, 'conv', [3,3]],
                'max5':[[2,2], 'pool'],
                'fc1':[4096,'fc'],
                'fc2':[4096,'fc'],
                'fc3':[n_class,'fc']}


    dropout = [[21,22], dropout_rate]
    x = tf.placeholder(tf.float32,
                       [None, 224, 224, 3],
                       name='input')
    feature_layer = len(vgg_dict) - 2
    
    # creating the architecture
    model = CNN(x, vgg_dict, 'VGG19', 
                feature_layer, dropout)

    # forming optimizer and gradient operator
    print('Optimizers..')
    model.get_optimizer(learning_rate, train_layers)
    print('Gradients..')
    model.get_gradients(grad_layers)

    return model
    
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
    """Creating a kernel tensor 
    with specified shape
    
    Here, as for the initialization we use
    the strategy that He et al. (2015), 
    "Delving deep into rectifiers: Surpassing 
    human level..."such that the outputs have 
    unit (reasonably large)
    
    It consists of Gaussian initialization
    with zero-mean and a specific variance.
    """
    
    # using Eq (10) of He et al., assuming
    # ReLu activation, independence of 
    # elements of the weight tensors, 
    # and independence between weights and
    # input tensors
    if len(shape)>2:
        # conv. layer
        # shape[0] : kernel dim_1 
        # shape[1] : kernel dim_2
        # shape[2] : input channels
        n = shape[0]*shape[1]*shape[2]
        std = np.sqrt(2/n)
    else:
        # fc layer
        n = shape[1]
        std = np.sqrt(2/n)
    
    initial = tf.random_normal(
        shape, mean=0., stddev=std)
    
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    """Creating a bias term with specified shape
    """
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name=name)

    
def max_pool(x, w_size, stride):
    return tf.nn.max_pool(
        x, ksize=[1, w_size, w_size, 1],
        strides=[1, stride, stride, 1], 
        padding='SAME')
    
def load_winds(inds, 
               imgs_path_file, 
               target_shape,
               mean=None,
               labels_file=None):
    """Creating a 4D array that contains a
    number of 3D images 
    """

    ntrain = len(inds)
    # read the first image to get the number of 
    path = linecache.getline(
        imgs_path_file, 
        inds[0]+1).splitlines()[0]
    img = np.float64(cv2.imread(path))
    img = cv2.resize(img,target_shape)
    if mean:
        img -= mean

    nchannels = img.shape[-1]
    batch_of_data = np.zeros(
        (ntrain,)+target_shape+(nchannels,))
    batch_of_data[0,:,:,:] = img

    labels = []
    if labels_file:
        label = linecache.getline(
            labels_file,
            inds[0]+1).splitlines()[0]
        labels += [int(label)]

    # reading batch of images
    for i in range(1,ntrain):
        img_path = linecache.getline(
            imgs_path_file, 
            inds[i]+1).splitlines()[0]
        img = np.float64(cv2.imread(img_path))
        img = cv2.resize(img,target_shape)
        if mean:
            img -= mean
        batch_of_data[i,:,:,:] = img
        
        if labels_file:
            label = linecache.getline(
                labels_file,
                inds[i]+1).splitlines()[0]
            labels += [int(label)]
            
    return (batch_of_data, labels)

def gen_batch_inds(data_size, batch_size):
    """Generating a list of random indices 
    to extract batches
    """
    
    # determine size of the batches
    quot, rem = np.divmod(data_size, 
                          batch_size)
    batches = list()
    
    # random permutation of indices
    rand_perm = np.random.permutation(
        data_size).tolist()
    
    # assigning indices to batches
    for i in range(quot):
        this_batch = rand_perm[
            slice(i*batch_size, 
                  (i+1)*batch_size)]
        batches += [this_batch]
        
    # if there is remainder, add them
    # separately
    if rem>0:
        batches += [rand_perm[-rem:]]
        
    return batches

    
        
        
        
