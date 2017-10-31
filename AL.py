import numpy as np
import shutil
import pickle
import yaml
import pdb
import os

import tensorflow as tf
import NNAL_tools

class Experiment(object):
    """class of an active learning experiments
    """
    
    def __init__(self, root_dir, img_path_list, 
                 labels, pars={}):
        """Constructor
        
        It needs the root directory of the experiment, 
        which will contain all runs' folders, and the
        path to the data that will be used for all
        different active learning experiments. This 
        data will be partitioned randomly in each run
        of the experiments into test, training and 
        unlabeled pool samples.
        
        If a set of parameters are given, they will
        be saved in the root. Otherwise, we just leave
        it there.
        """
        
        self.root_dir = root_dir
        # create the directory if not existed
        if not(os.path.exists(root_dir)):
            os.mkdir(root_dir)
        
        img_path_list_f = os.path.join(
            root_dir,'img_path_list.txt')
        labels_f = os.path.join(
            root_dir,'labels.txt')
        
        if os.path.isfile(img_path_list_f):
            # load the data and ignore the input
            with open(img_path_list_f, 'r') as f:
                self.img_path_list = f.readlines()
        else:
            self.img_path_list = img_path_list
            with open(img_path_list_f, 'a') as f:
                # writing paths into a file
                for path in img_path_list:
                    f.write('%s\n'% path)
        
        if os.path.isfile(labels_f):
            labels = np.int32(np.loadtxt(labels_f))
            if labels.ndim>1:
                self.labels=labels
                return

        # if labels are not in a hot-one form
        # transform it
        if np.array(labels).ndim==1:
            symbols = np.unique(labels)
            c = len(symbols)
            hot_labels = np.zeros((c, len(labels)))
            for i in range(len(labels)):
                label_ind = np.where(
                    symbols==labels[i])[0]
                hot_labels[label_ind,i] = 1.

            self.labels = hot_labels
        else:
            self.labels = labels
                
        # writing the labels into a file
        np.savetxt(labels_f, self.labels, fmt='%d')
        
        # if there are parameter given, write them
        # into a text file to be used later
        if len(pars)>0:
            self.save_parameters(pars)
        else:
            print("No parameters have been saved..")
            print("Save some by calling " + 
                  "save_parameters() before continuing")

    def save_parameters(self, pars):
        """Saving a given dictionary of parameters
        into a text file in the root folder of the
        experiment
        """
        
        with open(os.path.join(
                self.root_dir,
                'parameters.txt'),'w') as f:
            
            self.pars = pars
            yaml.dump(pars, f)
                    
    def load_parameters(self):
        """Loading the parameters that are saved
        into the text file into the local variables
        """
        
        with open(os.path.join(
                self.root_dir,
                'parameters.txt'),'r') as f:

            self.pars = yaml.load(f)

                        
    def get_runs(self):
        """List all the runs that this experiment has
        so far
        """
        
        # assuming that the root directory has only 
        # folders of the runs
        return [
            d for d in os.listdir(self.root_dir) 
            if os.path.isdir(
                os.path.join(self.root_dir,d))
            ]
        
    def remove_run(self, run):
        """Removing a given run by deleting the folder,
        and renaming all the folders
        """
        
        shutil.rmtre(os.path.join(self.root_dir, run))
        self.organize_runs()

    def organize_runs(self):
        """Organizing run folders such that they have 
        names from 1 to n

        Having such organized folder for the runs, makes
        it much easier to add a new run.
        """
        
        run_dirs = np.sort(os.listdir(self.root_dir))
        for name,i in enumerate(run_dirs):
            if not(i==int(name)):
                os.rename(os.path.join(self.root_dir, name),
                          os.path.join(self.root_dir, str(i)))
                
    def load_model(self, sess, model_path):
        """Loading a network model that is already saved
        """
        
        if hasattr(self, 'pars'):
            self.load_parameters()

        # create a model if nothing already exists
        tf_vars = tf.trainable_variables()        
        if len(tf_vars)<self.nvars:
            model = NN.create_Alex(
                self.pars['dropout_rate'], 
                nclass, 
                self.pars['learning_rate'], 
                self.pars['starting_layer'])

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        
    def add_run(self):
        """Adding a run to this experiment
        
        Each run will have its own random partitioning
        of data, and model initializing (if needed)
        """
        
        # create a folder for the new run
        curr_runs = self.get_runs()
        # when organized (from 0 to n-1), name of the 
        # new folder could be `n`
        n = len(curr_runs)
        run_path = os.path.join(self.root_dir, str(n))
        os.mkdir(run_path)
        
        if hasattr(self, 'pars'):
            self.load_parameters()
        
        # preparing the indices
        # -----------------------
        # test-training partitioning
        train_inds, test_inds = NNAL_tools.test_training_part(
            self.labels, self.pars['test_ratio'])
        
        # getting the initial and pool indices
        ntrain = len(train_inds)
        rand_inds = np.random.permutation(ntrain)
        init_train_inds = train_inds[
            rand_inds[:self.pars['init_size']]]
        pool_inds = train_inds[
            rand_inds[self.pars['init_size']:]]
        
        # saving indices into the run's folder
        np.savetxt('%s/train_inds.txt'% run_path, 
                   train_inds, fmt='%d')
        np.savetxt('%s/test_inds.txt'% run_path, 
                   test_inds, fmt='%d')
        np.savetxt('%s/init_inds.txt'% run_path, 
                   train_inds, fmt='%d')
        np.savetxt('%s/pool_inds.txt'% run_path, 
                   pool_inds, fmt='%d')
        
        # creating an initial initial model
        # -------------------------
        print('Initializing a model for this run..')
        # create the NN model
        nclass = self.labels.shape[0]
        model = NN.create_Alex(
            self.pars['dropout_rate'], 
            nclass, 
            self.pars['learning_rate'], 
            self.pars['starting_layer'])

        saver = tf.train.Saver()
        # start a session to do the training
        with tf.Session() as sess:
            # training from initial training data
            model.initialize_graph(
                sess, self.pars['pre_weights_path'])
            model.train_graph_one_epoch(
                self, 
                init_train_inds, 
                self.pars['batch_size'], 
                sess)
            
            # compute the initial accuracy and store it
            init_acc = model.evaluate(
                self, test_inds, 
                self.pars['batch_size'], sess)
            np.savetxt(os.path.join(
                    run_path,'init_acc.txt'),init_acc)
            # save the initial model into the run's folder
            saver.save(sess, 
                       os.path.join(run_path, 'init_model.ckpt'))
            
    def add_method(self, method_name, run):
        """Adding a method to a given run of the experiment
        """
        
        # check if the method already exists in this run
        if os.path.exists(os.path.join(self.root_dir, 
                                       str(run), 
                                       method_name)):
            print("This method already exists in run %s"% run)
            print("Nothing more to do..")
            return
        
        # create a directory for the method
        os.mkdir(os.path.exists(
                os.path.join(self.root_dir, 
                             str(run), 
                             method_name)))
        # create a directory for the queries
        os.mkdir(os.path.exists(
                os.path.join(self.root_dir, 
                             str(run), 
                             method_name,
                             'queries')))

        # copying the following files:
        # init_train_inds -->   curr_train
        # pool_inds       -->   curr_pool
        # init_model      -->   curr_model
        run_path = os.path.join(self.root_dir, str(run))
        method_path = os.path.join(run_path, method_name)

        shutil.copyfile(
            os.path.join(run_path,'init_inds.txt'),
            os.path.join(method_path,'curr_train.txt')
            )
        shutil.copyfile(
            os.path.join(run_path,'pool_inds.txt'),
            os.path.join(method_path,'curr_pool.txt')
            )
        shutil.copyfile(
            os.path.join(run_path,'init_model.ckpt'),
            os.path.join(method_path,'curr_model.ckpt')
            )
        
        # create results.txt and put the initial accuracy
        # as the first value in it
        init_acc = np.loadtxt(os.path.join(
                run_path, 'init_acc.txt'))
        np.savetxt(os.path.join(
                method_path, 'accs.txt'),init_acc)
        
    def run_method(self, method_name, run, max_queries):
        """Running a querying method in a run until a 
        given number of queries are drawn
        """
        
        run_path = os.path.join(self.root_dir,str(run))
        method_path = os.path.join(run_path, method_name)
        
        # count how many queries have been 
        # selected before
        n_oldqueries = len(os.listdir(
            os.path.join(method_path, 'queries')))
        
        # preparing the indices
        test_inds = np.loadtxt(os.path.join(
                run_path, 'test_inds.txt'))
        curr_train = np.loadtxt(os.path.join(
                run_path, 'curr_train.txt'))
        curr_pool = np.loadtxt(os.path.join(
                run_path, 'curr_pool.txt'))
        
        if hasattr(self, 'pars'):
            self.load_parameters()
        
        # load the stored model before the iterations
        with tf.Session() as sess:
            self.load_model(sess, 
                            os.path.join(method_path,
                                         'curr_model.ckpt'))
        
            # starting the iterations
            print("Starting the iterations for %s"%
                  method_name)
            nqueries = 0
            iter_cnt = 0
            while nqueries < max_queries:
                # do the querying
                #Q_inds = ...
                # save the queries
                np.savetxt(os.path.join(
                        method_path, 
                        'queries',
                        'Q_%d.txt'% (
                            n_oldqueries+iter_cnt)
                        ), Q_inds)
                
                # update the indices
                
                # update the loop variables
                nqueries += len(Q_inds)
                iter_cnt += 1
        

def paths_n_labels(path, label_name):
    """Preparing a list containing the path to all individual
    images in a given path, and reading the label text file
    
    The second argument is the file name of the labels.
    """
    
    files = os.listdir(path)
    if not(label_name in path):
        raise ValueError('The label file is not in the given directory')
    
    files.remove(label_name)
    labels = np.int32(np.loadtxt(label_name))
    
    return files, labels
    
def create_hot_labels(path):
    """Given the path of a data set where each category (class)
    has its own separate folder, a hot-label vector will be created
    accordingly

    We assume here that the only existing directories in the given
    path are the categories, each of which only has data files,
    just as in the folder of Caltech-101 data set.
    """
    
    classes = os.listdir(path)
    
        
        
    
