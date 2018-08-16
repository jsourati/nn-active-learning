import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import f1_score
import linecache
import copy
import h5py
import pdb
import sys
import cv2
import os

import NNAL_tools
import PW_NN
import AL

read_file_path = "/home/ch194765/repos/atlas-active-learning/"
sys.path.insert(0, read_file_path)
import prep_dat

read_file_path = "/home/ch194765/repos/atlas-active-learning/AlexNet"
sys.path.insert(0, read_file_path)
import alexnet
from alexnet import AlexNet


class CNN(object):
    """Class of CNN models
    """
    
    def __init__(self, 
                 x, 
                 layer_dict, 
                 name,
                 skips=[],
                 feature_layer=None,
                 dropout=None,
                 probes=[[],[]]):
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
            
                    {layer_name: [layer_type, layer_specs, operation_order]}

                Layer's specifications, in turn, contains two or more
                items dependending on the type of the layer. At this time
                this class supports only three types of layers:
               
                - Convolutional:   [# output channel, kernel size, strides, padding]
                - Transpose Convolutional:
                                   [# output channel, kernel size, strides]
                - Fully-connected: [# output channel]
                - max-pooling:     [pool size]
                
                 Note that "kernel size", "strides" and "pool size" are a list with 
                 two elements, and "padding" is a string. For now the class
                 only supports `SAME` padding for transpose 2D-convolution; also
                 the second and third elements of `strides` for this layer
                 specifies height and width of the output tensor as 
                 `width=strides[1]*input.shape[1]`
                 `height=strides[2]*input.shape[2]`
        
            **name**: string
                Name of Tensorflow scope of all the variables defined in
                this class.

            **skips** : list
                List of skip connections; each element is a list of 
                three elements: [layer_index, list_of_destinations, skip_type]
                which indices that output of the layer layer_index should be
                connected to the input of all layers with indices in 
                list_of_destinations; skip_type specifies the type of
                connection: summing up if `skip_type=='sum'` and concatenating
                if `skip_type=='con'`

                CAUTIOUS: all elements in list_of_destinatins are assumed
                          to be larger than the (source) layer_index
        
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
        self.batch_size = tf.shape(x)[0]  # to be used in conv2d_transpose

        self.layer_type = []
        self.name = name
        self.skips = skips
        
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

        self.probes = []
        sources_idx = [skips[i][0] for i in range(len(skips))]
        sources_output = []

        self.layer_names = list(layer_dict.keys())

        self.output = x
        with tf.name_scope(name):
            for i, layer_name in enumerate(layer_dict):
                
                # before adding a layer check if this layer
                # is a destination layer, i.e. it has to
                # consider the output of a previous layer
                # in its input 
                combine_layer_outputs(self, 
                                      i, 
                                      skips, 
                                      sources_output)

                if i in probes[0]:
                    self.probes += [self.output]

                layer = layer_dict[layer_name]
                if len(layer)==2:
                    layer += ['M']
                # layer[0]: layer type
                # layer[1]: layer specs
                # layer[2]: order of operations, default: 'MA'
                self.add_layer(
                    layer_name, layer[0], layer[1], layer[2])
                
                if i in probes[1]:
                    self.probes += [self.output]

                # dropping out the output layers if the layer
                # is in the list of dropped-out layers
                if i in self.dropout_layers:
                    self.output = tf.nn.dropout(
                        self.output, self.keep_prob)
                
                if i in sources_idx:
                    sources_output += [self.output]

                # set the output of the layer one before last as 
                # the features that the network will extract
                if i==feature_layer:
                    self.feature_layer = self.output
                
                # flatenning the output of the current layer
                # if it is 'conv' or 'pool' and the next
                # layer is 'fc' (hence needs flattenned input)
                if i<len(layer_dict)-1:
                    next_layer_type = layer_dict[
                        self.layer_names[i+1]][0]
                    if (layer[0]=='conv' or layer[0]=='pool') \
                       and next_layer_type=='fc':
                        # flattening the output
                        out_size = np.prod(
                            self.output.get_shape()[1:]).value
                        self.output = tf.reshape(
                            tf.transpose(self.output), 
                            [out_size, -1])
                        
            # creating the label node
            if len(self.output.shape)==2:
                # posterior
                posteriors = tf.nn.softmax(
                    tf.transpose(self.output))
                self.posteriors = tf.transpose(
                    posteriors, name='Posteriors')
                # prediction node
                self.prediction = tf.argmax(
                    self.posteriors, 0, name='Prediction')
                c = self.output.get_shape()[0].value
                self.y_ = tf.placeholder(tf.float32, 
                                         [c, None],
                                         name='labels')
            else:
                w = self.output.shape[1].value
                h = self.output.shape[2].value
                c = self.output.shape[3].value
                self.y_ = tf.placeholder(tf.float32,
                                         [None,h,w,c])
            
    def add_layer(self, 
                  layer_name,
                  layer_type, 
                  layer_specs, 
                  layer_op_order):
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
        
        with tf.name_scope(layer_name):
            for op in layer_op_order:

                if op=='M':
                    # main operation
                    if layer_type=='conv':
                        self.add_conv(layer_name, 
                                      layer_specs)
                    elif layer_type=='conv_transpose':
                        self.add_conv_transpose(layer_name, 
                                                layer_specs)
                    elif layer_type=='fc':
                        self.add_fc(layer_name, 
                                    layer_specs)
                    elif layer_type=='pool':
                        self.add_pool(layer_specs)
                    else:
                        raise ValueError(
                            "Layer's type should be either 'fc'" + 
                            ", 'conv', 'conv_transpose' or 'pool'.")

                elif op=='B':

                    # batch normalization
                    self.add_BN(layer_name, layer_specs)

                elif op=='A':
                    # activation (ReLU)
                    self.output = tf.nn.relu(self.output)

                else:
                    raise ValueError(
                        "Operations should be either 'M'" + 
                        ", 'B' or 'A'.")
                
    def add_conv(self, 
                 layer_name,
                 layer_specs):
        """Adding a convolutional layer to the graph given 
        the specifications

        The layer specifications should have the following
        order:
        - [num. of kernel, dim. of kernel, strides, padding]

        It should containt at least the first two elements,
        the other two have default values.

        CAUTIOUS: when giving default values be careful
        about the order of the parameters, e.g. you cannot
        use default values for strides, when padding is 
        assigned a value. The 3rd value will ALWAYS be
        assigned to strides 
        """

        # creating necessary TF variables
        kernel_num = layer_specs[0]
        kernel_dim = layer_specs[1]
        if len(layer_specs)==2:
            strides = [1,1]
            padding='SAME'
        elif len(layer_specs)==3:
            strides = layer_specs[2]
            padding='SAME'

        prev_depth = self.output.get_shape()[-1].value
        new_vars = [weight_variable([kernel_dim[0], 
                                     kernel_dim[1], 
                                     prev_depth, 
                                     kernel_num], 
                                    name='Weight'),
                    bias_variable([kernel_num],
                                  name='Bias')
        ]

        # there may have already been some variables
        # created for this layer (through BN)
        if layer_name in self.var_dict:
            self.var_dict[layer_name] += new_vars
        else:
            self.var_dict.update({layer_name: new_vars})


        # output of the layer
        self.output = tf.nn.conv2d(
            self.output, 
            self.var_dict[layer_name][-2], 
            strides= [1,] + strides + [1,],
            padding=padding) + self.var_dict[layer_name][-1]
    
    def add_fc(self, 
               layer_name,
               layer_specs):
        """Adding a fully-connected layer with a given 
        specification to the graph
        """
        prev_depth = self.output.get_shape()[0].value
        new_vars = [weight_variable([layer_specs[0], prev_depth], 
                                    name='Weight'),
                    bias_variable([layer_specs[0], 1],
                                  name='Bias')
        ]

        if layer_name in self.var_dict:
            self.var_dict[layer_name] += new_vars
        else:
            self.var_dict.update({layer_name: new_vars})

        # output of the layer
        self.output = tf.matmul(
            self.var_dict[layer_name][-2], 
            self.output) + self.var_dict[layer_name][-1]
            
    def add_pool(self, layer_specs):
        """Adding a (max-)pooling layer with given specifications
        """
        pool_size = layer_specs
        self.output = max_pool(self.output, 
                               pool_size[0], 
                               pool_size[1])

    def add_BN(self, 
               layer_name,
               layer_specs):
        """Adding batch normalization layer
        """

        # defining required variables (scale, offset)
        output_shape = self.output.shape
        if len(output_shape)==2:
            ax = [1]
            shape = [output_shape[0].value]
        else:
            ax = [0]
            shape = [output_shape[i].value for i 
                     in range(1,len(output_shape))]
        # new_vars = [ gamma ,  beta  ]
        #            [(scale), (offset)
        new_vars = [tf.Variable(tf.constant(1., shape=shape), 
                                name='Scale'),
                    tf.Variable(tf.constant(0., shape=shape), 
                                name='Offset')]

        if layer_name in self.var_dict:
            self.var_dict[layer_name] += new_vars
        else:
            self.var_dict.update({layer_name: new_vars})

        mu_B, Sigma_B = tf.nn.moments(self.output, axes=ax,
                                      keep_dims=False)

        if len(output_shape)==2:

            self.output = tf.transpose(tf.nn.batch_normalization(
                tf.transpose(self.output), mu_B, Sigma_B, 
                self.var_dict[layer_name][-1],
                self.var_dict[layer_name][-2],
                1e-6))

        else:

            self.output = tf.nn.batch_normalization(
                self.output, mu_B, Sigma_B, 
                self.var_dict[layer_name][-1],
                self.var_dict[layer_name][-2],
                1e-6)

    def add_conv_transpose(self, 
                           layer_name,
                           layer_specs):
        """Adding a transpose convolutional layer
        
        Any transpose convolution is indeed the backward 
        direction of a convolution layer with some 
        ambiguity on the output's size (not fully clear)

        Number of elements in `layer_specs` of this
        layer should be EXACTLY three
        """
        kernel_num = layer_specs[0]
        kernel_dim = layer_specs[1]
        strides = layer_specs[2]

        # adding new variables
        prev_depth = self.output.get_shape()[-1].value
        new_vars = [weight_variable([kernel_dim[0],
                                     kernel_dim[1],
                                     kernel_num, 
                                     prev_depth], 
                                    name=layer_name+'_weight'),
                    bias_variable([kernel_num],
                                  name=layer_name+'_bias')]
        # there may have already been some variables
        # created for this layer (through BN)
        if layer_name in self.var_dict:
            self.var_dict[layer_name] += new_vars
        else:
            self.var_dict.update({layer_name: new_vars})


        # output of the layer
        input_size = [self.output.shape[i].value for
                      i in range(1,3)]
        output_shape = [self.batch_size, 
                        strides[0]*input_size[0],
                        strides[1]*input_size[1],
                        kernel_num]
        strides = [1,] + strides + [1,]
        # padding will stay `SAME` for now
        self.output = tf.nn.conv2d_transpose(
            self.output, self.var_dict[layer_name][-2],
            output_shape, strides) + \
            self.var_dict[layer_name][-1] 
        

        # last line is adding zeros with the same size of the 
        # output (except batch size) only in order to
        # remove the size ambiguities that is created by 
        # tf.nn.conv2d_transpsoe
        if output_shape[-1]>1:
            self.output += tf.constant(1., shape=[1,]+output_shape[1:])
        else:
            output_shape[-1]=2
            self.output += tf.constant(1., shape=[1,]+output_shape[1:])
            self.output = self.output[:,:,:,:1]


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

        # initializing of variables of only this model
        model_vars = list(np.concatenate(
            [self.var_dict[layer_name] for layer_name
             in self.var_dict]))

        session.run(tf.variables_initializer(model_vars))
                    

    def save_weights(self, file_path, save_moments=False):
        """Saving only the parameter values of the 
        current model into a .h5 file
        
        The file will have as many groups as the number
        of layers in the model (which is equal to the
        number of keys in `self.var_dict`. Each group has
        two datasets, one for the weight W, and one for
        the bias b.
        """
        
        f = h5py.File(file_path, 'w')
        for layer_name in list(self.var_dict):
            L = f.create_group(layer_name)
            layer_vars = self.var_dict[layer_name]

            if save_moments:
                # create three groups for parameter
                # values, first and second moments
                L0 = L.create_group('Values')
                L1 = L.create_group('Moments1')
                L2 = L.create_group('Moments2')
                for var in layer_vars:
                    var_name = var.name.split('/')[-1][:-2]
                    # last line, [:-2] accounts for ':0' in 
                    # TF variables
                    L0.create_dataset(var_name, data=var.eval())
                    L1.create_dataset(
                        var_name, 
                        data=self.optimizer.get_slot(var,'m').eval())
                    L2.create_dataset(
                        var_name, 
                        data=self.optimizer.get_slot(var,'v').eval())
            else:
                for var in layer_vars:
                    var_name = var.name.split('/')[-1][:-2]
                    # last line, [:-2] accounts for ':0' in 
                    # TF variables
                    L.create_dataset(var_name, data=var.eval())
            
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
        model_real_name = self.output.name.split('/')[0]

        for layer_name in list(self.var_dict):
            var_names = list(f[layer_name].keys())
            for var_name in var_names:
                var_value = np.array(f[layer_name][var_name])
                full_var_name = '/'.join([model_real_name,
                                          layer_name,
                                          var_name])
                tf_var = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, 
                    scope=full_var_name)[0]
                session.run(tf_var.assign(var_value))
            
    def add_assign_ops(self):
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

        The function defines a new attribute called `assign_dict`,
        a dictionary of layer names as the keys. Each item
        of `assign_dict` is itself a dictionary with keys
        equal to the variables names of that layer (from
        among `Weight`, `Bias`, `Scale` and `Offset`). Each
        item of this dictionary then has two elements: 

            - the assigning operation for the variable
              with the given name
            - the placeholder that will carry the value
              to be assigned to this variable

        In summary `assign_dict` has the following structure:

            {'layer_name_1': 
                             {'Weight': [<assign_op>,
                                         <placeholder>],
                              'Bias':   [<assign_op>,
                                         <placeholder>],
                                 :
                              }
                 :
             }
        
        """

        self.assign_dict = {}
        for layer_name in list(self.var_dict):
            layer_vars = self.var_dict[layer_name]

            layer_dict = {}

            for var in layer_vars:
                var_name = var.name.split('/')[-1][:-2]

                # value placeholder
                var_placeholder = tf.placeholder(var.dtype,
                                                 var.get_shape())
                # assigning ops
                assign_op = var.assign(var_placeholder)
                layer_dict.update({var_name:[assign_op,
                                             var_placeholder]})

            self.assign_dict.update({layer_name: layer_dict})

    def add_assign_ops_AdamMoments(self):

        self.assign_dict_moments1 = {}
        self.assign_dict_moments2 = {}
        for layer_name in list(self.var_dict):
            layer_vars = self.var_dict[layer_name]

            if (len(self.train_layers)>0) and \
               not(layer_name in self.train_layers):
                continue

            layer_dict_m = {}
            layer_dict_v = {}
            for var in layer_vars:
                var_name = var.name.split('/')[-1][:-2]

                # value placeholder
                var_m_placeholder = tf.placeholder(var.dtype,
                                                   var.get_shape())
                var_v_placeholder = tf.placeholder(var.dtype,
                                                   var.get_shape())
                # assigning ops
                assign_op_m = self.optimizer.get_slot(
                    var,'m').assign(var_m_placeholder)
                assign_op_v = self.optimizer.get_slot(
                    var,'v').assign(var_v_placeholder)

                layer_dict_m.update({var_name:[assign_op_m,
                                               var_m_placeholder]})
                layer_dict_v.update({var_name:[assign_op_v,
                                               var_v_placeholder]})

            self.assign_dict_moments1.update({layer_name: 
                                              layer_dict_m})
            self.assign_dict_moments2.update({layer_name: 
                                              layer_dict_v})


    def perform_assign_ops(self,file_path,sess):
        """Performing assignment operations that have
        been created by `self.add_assign_ops`.

        This function, together with `self.add_assign_ops`
        will be used instead of `self.load_weights` when 
        value assignment needs to be done repeatedly after
        finalizing the graph.
        """

        model_real_name = self.output.name.split('/')[0]

        f = h5py.File(file_path)        

        # preparing the operation list to be performed
        # and the necessary `feed_dict`
        feed_dict={}
        ops_list = []
        for layer_name in list(self.assign_dict):
            if 'Values' in list(f[layer_name].keys()):
                var_names = list(f[layer_name]['Values'].keys())
                for var_name in var_names:
                    # for Values
                    var_value = np.array(f[layer_name]['Values'][var_name])
                    ops_list += [self.assign_dict[layer_name][var_name][0]]
                    feed_dict.update({
                        self.assign_dict[layer_name][var_name][1]: var_value})

                    if (len(self.train_layers)>0) and \
                       not(layer_name in self.train_layers):
                        continue
                    # for Moment 1
                    var_value = np.array(f[layer_name]['Moments1'][var_name])
                    ops_list += [self.assign_dict_moments1[
                        layer_name][var_name][0]]
                    feed_dict.update({self.assign_dict_moments1[
                        layer_name][var_name][1]: var_value})
                    # for Moment 2
                    var_value = np.array(f[layer_name]['Moments2'][var_name])
                    ops_list += [self.assign_dict_moments2[
                        layer_name][var_name][0]]
                    feed_dict.update({self.assign_dict_moments2[
                        layer_name][var_name][1]: var_value})
            else:
                var_names = list(f[layer_name].keys())
                for var_name in var_names:
                    var_value = np.array(f[layer_name][var_name])
                    # adding the operation
                    ops_list += [self.assign_dict[layer_name][var_name][0]]
                    # adding the corresponding value into feed_dict
                    feed_dict.update({
                        self.assign_dict[layer_name][var_name][1]: var_value})

        sess.run(ops_list, feed_dict=feed_dict)


    def get_optimizer(self, 
                      learning_rate, 
                      loss_name='CE',
                      optimizer_name='SGD',
                      train_layers=[],
                      **kwargs):
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

        self.learning_rate = learning_rate

        if len(self.output.shape)==2:
            get_loss_scalar_output(self, loss_name)
        else:
            get_loss_2d_output(self, loss_name)

        self.train_layers = train_layers
        get_optimizer(self, optimizer_name, loss_name, **kwargs)
        
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
        # in binary classification, get only
        # gradient of the first class

        for j in range(c):
            self.grad_posts.update(
                {str(j): tf.gradients(
                    tf.log(self.posteriors[j, 0]),
                    gpars, name='score_class_%d'% j)
             }
            )

def combine_layer_outputs(model,
                          layer_index,
                          skips,
                          sources_output):
    """Preparing the inputs to a given layer
    considering the skip connections that may
    indicate the layer needs inputs from the
    previous layers 
    """

    sources_for_sink = []
    for j in range(len(skips)):
        if layer_index in skips[j][1]:
            sources_for_sink += [j]

    if len(sources_for_sink)==0:
        return
    else:
        for j in sources_for_sink:
            if skips[j][2]=='sum':
                model.output = tf.add(model.output,
                                sources_output[j])
            elif skips[j][2]=='con':
                model.output = concat_outputs(
                    model.output, sources_output[j])


def concat_outputs(curr_output, prev_output):
    """Concatenating output of a layer in the 
    network with the output of some previous
    layers
    """

    assert len(curr_output.shape)==len(prev_output.shape), \
        'The outputs to concatenate should have the same' + \
        ' dimensionalities.'

    d = len(curr_output.shape)
    if d==2:
        output = tf.concat((prev_output, curr_output),
                           axis=0)
    else:
        output = tf.concat((prev_output, curr_output),
                           axis=d-1)

    return output


def add_loss_grad(model, pars=[]):
    """Adding the gradient of the loss
    with respect to parameters if necessary
    """

    model.loss_grads = {}
    for layer in model.var_dict:
        layer_grads_dict = {}

        for var in model.var_dict[layer]:
            var_name = var.name.split('/')[-1][:-2]
            grad = tf.gradients(model.loss, var)
            layer_grads_dict.update({var_name: grad})

        model.loss_grads.update({layer:layer_grads_dict})

def get_loss_scalar_output(model, loss_name='CE'):

    with tf.name_scope(model.name):
            
        # Loss 
        # (for now, only cross entropy)
        if loss_name=='CE':
            model.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.transpose(model.y_), 
                    logits=tf.transpose(model.output)),
                name='CE_Loss')

def get_loss_2d_output(model, loss_name='CE'):
    
    with tf.name_scope(model.name):
            
        # Loss 
        # (for now, only cross entropy)
        if loss_name=='CE':
            model.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=model.y_, logits=model.output, 
                    dim=-1),
                name='Loss')


def get_optimizer(model, 
                  optimizer_name='SGD',
                  loss_name='CE',
                  **kwargs):
    """Creating an optimizer (if needed) together with
    training step for a given loss
    """

    # optimizer
    if not(hasattr(model, 'optmizer')):
        if optimizer_name=='SGD':
            model.optimizer = tf.train.GradientDescentOptimizer(
                model.learning_rate)
        elif optimizer_name=='Adam':
            kwargs.setdefault('beta1', 0.9)
            kwargs.setdefault('beta2', 0.999)
            model.__dict__.update(kwargs)
            model.optimizer = tf.train.AdamOptimizer(
                model.learning_rate, 
                model.beta1, 
                model.beta2)

    # training step
    if len(model.train_layers)==0:
        loss_train_step = model.optimizer.minimize(
            model.loss, name=loss_name+'_train_step')
    else:
        # if some layers are specified, only
        # modify these layers in the training
        var_list = []
        for layer in model.train_layers:
            var_list += model.var_dict[layer]
        loss_train_step = model.optimizer.minimize(
            model.loss, var_list=var_list, 
            name=loss_name+'_train_step')

    if loss_name=='CE':
        model.train_step = loss_train_step
    else:
        setattr(model, loss_name+'_train_step', 
                loss_train_step)

def get_LwF(model):
    """Taking for which a loss has been already defined,
    and modifying it to LwF (learning without forgetting)

    REMARK: this function needs model.get_optimizer() to be
        called beforehand 

    CAUTIOUS: modify it for FCNs
    """

    # needs introducing two hyper-parameters to model
    model.lambda_o = tf.placeholder(tf.float32)
    model.T = tf.placeholder(tf.float32)

    # defining output of the previous model
    model.y__ = tf.placeholder(tf.float32, model.y_.get_shape())

    # knowledge distillation (soft soft-max)
    soft_target = tf.nn.softmax(tf.transpose(
        tf.divide(model.y__, model.T)))
    loss_old_term = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=soft_target, 
            logits=tf.transpose(tf.divide(model.output, model.T))))

    model.LwF_loss = tf.add(model.loss, 
                            tf.multiply(loss_old_term,
                                        model.lambda_o))

    if len(model.train_layers)==0:
        model.LwF_train_step = model.optimizer.minimize(
            model.LwF_loss)
    else:
        var_list = []
        for layer in model.train_layers:
            var_list += model.var_dict[layer]
        model.LwF_train_step = model.optimizer.minimize(
            model.LwF, var_list=var_list)

def LLFC_hess(model,sess,feed_dict):
    """Explicit Hessian matrix of the loss with 
    respect to the last (FC) layer when the loss
    is the soft-max and the last layer does not
    have any additional activation except this
    soft-max
    """

    # input to the last layer (u)
    u = sess.run(model.feature_layer,
                 feed_dict=feed_dict)
    d = u.shape[0]

    # the class probabilities
    pi = sess.run(model.posteriors,
                  feed_dict=feed_dict)

    # A(pi)
    c = pi.shape[0]
    repM = np.repeat(pi,c,axis=1) - np.eye(c)
    A = np.diag(pi[:,0]) @ repM.T

    # Hessian
    H = np.zeros(((d+1)*c, (d+1)*c))
    H[:c*d,:c*d] = np.kron(A, np.outer(u,u))
    H[:c*d,c*d:] = np.kron(A,u)
    H[c*d:,:c*d] = np.kron(A,u.T)
    H[c*d:,c*d:] = A

    return H

def LLFC_grads(model, sess, feed_dict, labels=None):
    """General module for computing gradients
    of the log-loss with respect to parameters
    of the (FC) last layer of the network
    """

    # posteriors (pi)
    pies = sess.run(model.posteriors,
                    feed_dict=feed_dict)
    c,n = pies.shape

    # input to the last layer (u)
    U = sess.run(model.feature_layer,
                 feed_dict=feed_dict)
    d = U.shape[0]

    # term containing [pi_1.u_1 ,..., pi_1.u_d,
    #                  pi_2.u_1 ,..., pi_2.u_d,...]
    rep_pies = np.repeat(pies, d, axis=0)
    rep_U = np.tile(U, (c,1))
    pies_dot_U = rep_pies * rep_U

    flag=0
    if labels is None:
        labels = sess.run(model.prediction,
                          feed_dict=feed_dict)
        flag = 1
    hot_labels = np.zeros((c,n))
    for j in range(c):
        hot_labels[j,labels==j]=1

    # sparse term containing columns
    #         [0,...,0, u_1,...,u_d, 0,...,0].T
    #                   |____ ____|
    #                        v
    #                   y*-th block
    sparse_term = np.repeat(
        hot_labels, d, axis=0) * rep_U

    # dJ/dW
    dJ_dW = sparse_term - pies_dot_U

    # dJ/db
    dJ_db = hot_labels - pies

    if flag==1:
        return np.concatenate(
            (dJ_dW,dJ_db),axis=0), labels
    else:
        return np.concatenate(
            (dJ_dW,dJ_db),axis=0)

def PW_LLFC_grads(model, sess, 
                  expr,
                  all_padded_imgs,
                  img_inds,
                  labels):
    """Computing gradients of the log-likelihoods
    with respect to the parameters of the last
    layer of a given model

    Given labels are not necessarily the true
    labels of the indexed sampels (i.e. not
    necessarily those based on the mask image
    present in `all_padded_imgs`)
    """

    s = len(img_inds)
    n = np.sum([len(img_inds[i]) for i in range(s)])
    d = model.feature_layer.shape[0].value
    c = expr.nclass

    all_pies = np.zeros((c,n))
    all_a = np.zeros((d,n))

    # loading patches
    patches,_ = patch_utils.get_patches_multimg(
        all_padded_imgs, img_inds, 
        expr.pars['patch_shape'], 
        expr.train_stats)

    cnt=0
    for i in range(s):
        # posteriors pie's
        pies = sess.run(model.posteriors,
                        feed_dict={model.x:patches[i],
                                   model.keep_prob:1.})
        all_pies[:,cnt:cnt+len(img_inds[i])] = pies

        # last layer's inputs a^{n1-1}
        a_s = sess.run(model.feature_layer,
                       feed_dict={model.x:patches[i],
                                  model.keep_prob:1.})
        all_a[:,cnt:cnt+len(img_inds[i])] = a_s

        cnt += len(img_inds[i])

    # repeating the matrices
    rep_pies = np.repeat(all_pies, d, axis=0)
    rep_a = np.tile(all_a, (c,1))
    pies_dot_as = rep_pies * rep_a

    # forming dJ / dW_(nl-1)
    term_1 = np.zeros((c*d, n))
    multinds = (np.zeros(n*d, dtype=int), 
                np.zeros(n*d, dtype=int))
    for i in range(n):
        multinds[0][i*d:(i+1)*d] = np.arange(
            labels[i]*d,(labels[i]+1)*d)
        multinds[1][i*d:(i+1)*d] = i
    term_1[multinds] = np.ravel(a_s.T)

    dJ_dW = term_1 - pies_dot_as

    # appending with dJ / db_{nl-1}
    term_1 = np.zeros((c,n))
    multinds = (np.array(labels),
                np.arange(n))
    term_1[multinds] = 1.
    dJ_db = term_1 - pies
    
    # final gradient vectors
    grads = np.concatenate((dJ_dW,dJ_db), axis=0)

    return grads


def create_model(model_name,
                 dropout_rate, 
                 nclass,
                 learning_rate, 
                 grad_layers=[],
                 train_layers=[],
                 optimizer_name='SGD',
                 patch_shape=None):
    
    if model_name=='Alex':
        model = create_Alex(dropout_rate, 
                            nclass,
                            learning_rate, 
                            starting_layer)
    elif model_name=='VGG19':
        model = create_VGG19(dropout_rate, 
                             learning_rate,
                             nclass, 
                             grad_layers,
                             train_layers)
        
    elif model_name=='PW':
        model = create_PW1(nclass,
                            dropout_rate,
                            learning_rate,
                            optimizer_name,
                            patch_shape)
        
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

def create_PW1(nclass,
               dropout_rate,
               learning_rate,
               optimizer_name,
               patch_shape):
    """Creating a model for patch-wise
    segmentatio of medical images
    """

    pw_dict = {'conv1':[24, 'conv', [5,5]],
               'conv2':[32, 'conv', [5,5]],
               'max1': [[2,2], 'pool'],
               'conv3':[48, 'conv', [3,3]],
               'conv4':[96, 'conv', [3,3]],
               'max2' :[[2,2], 'pool'],
               'fc1':[4096,'fc'],
               'fc2':[4096,'fc'],
               'fc3':[nclass,'fc']}
    
    dropout = [[6,7,8], dropout_rate]
    x = tf.placeholder(
        tf.float32,
        [None, 
         patch_shape[0],
         patch_shape[1],
         patch_shape[2]],
        name='input')
    feature_layer = len(pw_dict) - 2
    probes = [5]
    
    # the model
    model = CNN(x, pw_dict, 'PatchWise', 
                feature_layer, 
                dropout, probes)
    # optimizers
    model.get_optimizer(learning_rate, [],
                        optimizer_name)
    # gradients
    model.get_gradients()
    
    return model

def CNN_layers(W_dict, b_dict, x):
    """Creating the output of CNN layers 
    and return them as TF variables
    
    Each layer consists of a convolution, 
    following by a max-pooling and
    a ReLu activation.
    The number of channels of the input, 
    should match the number of
    input channels to the first layer based 
    on the parameter dictionary.
    """
    
    L = len(W_dict)
    
    output = x
    for i in range(L):
        output = tf.nn.conv2d(
            output, W_dict[str(i)], 
            strides=[1, 1, 1, 1], 
            padding='SAME') + b_dict[str(i)]
        output = tf.nn.relu(output)
        output = max_pool(output, 2, 2)
        
    return output
    

def CNN_variables(kernel_dims, layer_list):
    """Creating the CNN variables
    
    We should have `depth_lists[0] = in_channels`.
    In the i-th layer, dimensionality 
    of the kernel `W` would be
    `(kernel_dims[i],kernel_dims[i])`, and the 
    number of them (that is, the number
     of filters) would be `layer_list[i+1]`. 
    Moreover, the number
    of its input channels is `layer_list[i]`.
    """
    
    if not(len(layer_list)==len(kernel_dims)+1):
        raise ValueError(
            "List of  layers should have one more"+
            "element than the list of kernel dimensions.")
    
    W_dict = {}
    b_dict = {}
    
    layer_num = len(layer_list)
    # size of W should be 
    # [filter_height, filter_width, 
    # in_channels, out_channels]
    # here, filter_height = 
    #       filter_width = 
    #       kernel_dim
    for i in range(layer_num-1):
        W_dict.update(
            {str(i):weight_variable(
                [kernel_dims[i], 
                 kernel_dims[i], 
                 layer_list[i], 
                 layer_list[i+1]])})
        b_dict.update(
            {str(i): bias_variable(
                [layer_list[i+1]])})
        
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
    

def test_model(model, sess, test_dat):
    
    b = 1000
    X_test, Y_test = test_dat
    n = Y_test.shape[1]
    Y_test = np.argmax(Y_test, axis=0)
    batches = gen_batch_inds(n,b)
    preds = np.nan*np.zeros(n)

    for batch_inds in batches:
        if len(X_test.shape)==2:
            batch_X = X_test[:,batch_inds]
            if len(model.x.shape)>2:
                batch_X = np.reshape(batch_X.T, 
                                     (len(batch_inds),28,28,1))
        else:
            batch_X = X_test[batch_inds,:,:,:]

        feed_dict={model.x:batch_X, model.keep_prob:1.}
        batch_preds = sess.run(
            model.prediction,
            feed_dict=feed_dict)
        preds[batch_inds] = batch_preds
        
    # (multi-class) F1 score
    F1 = f1_score(y_true=Y_test,
                  y_pred=preds,
                  average='weighted')
    return F1


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
    
    
