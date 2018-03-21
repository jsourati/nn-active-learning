#from matplotlib import pyplot as plt
import numpy as np
import linecache
import shutil
import pickle
import scipy
import yaml
import pdb
import os

import tensorflow as tf
import NNAL_tools
import NNAL
import NN

class Experiment(object):
    """class of an active learning experiments
    """
    
    def __init__(self, 
                 root_dir,
                 imgs_path_file,
                 labels_file,
                 pars={}):
        """Constructor
        
        It needs the root directory of the experiment, 
        which will contain all runs' folders, and the
        path to the data that will be used for all
        different active learning experiments. This 
        data will be partitioned randomly in each run
        of the experiments into test, training and 
        unlabeled pool samples.
        
        If the experiment does not exist, variables
        `img_path_list` and `labels` have to be given.
         
        If a set of parameters are given, they will
        be saved in the root. Otherwise, we just leave
        it there.
        """
        
        self.root_dir = root_dir
        if not(os.path.exists(root_dir)):
            os.mkdir(root_dir)
            self.imgs_path_file = imgs_path_file
            self.labels_file = labels_file
            # writing 
            with open(os.path.join(
                    root_dir,'data_dirs.txt'),'w') as f:
                f.write(self.imgs_path_file+'\n')
                f.write(self.labels_file+'\n')
        else:
            with open(os.path.join(
                    root_dir,'data_dirs.txt'),'r') as f:
                dirs = f.read().splitlines()
            self.imgs_path_file = dirs[0]
            self.labels_file = dirs[1]
            
        labels = np.loadtxt(self.labels_file)
        self.nclass = int(labels.max()+1)
        # if there are parameter given, write them
        # into a text file to be used later
        if len(pars)>0:
            self.save_parameters(pars)
            
    def modify_parameters(self, mod_dict):
        """Modifying parameters of a given 
        experiment according to a given 
        dictionary which has a subset of keys
        and the corresponding modified values
        
        CAUTIOUS: only use this method for testing.
        Never change parameters of an experiment
        whose runs are completed
        """
        
        if not(hasattr(self, 'pars')):
            self.load_parameters()
            
        for var, value in mod_dict.items():
            self.pars[var] = value
            
        # saving the modified parameters
        self.save_parameters(self.pars)
        
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
        
        run_dirs = self.get_runs()
        for i,name in enumerate(run_dirs):
            if not(i==int(name)):
                os.rename(os.path.join(self.root_dir, name),
                          os.path.join(self.root_dir, str(i)))
                
        
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
        
        if not(hasattr(self, 'pars')):
            self.load_parameters()
        
        # preparing the indices
        # -----------------------
        # test-training partitioning
        labels = np.loadtxt(self.labels_file)
        train_inds, test_inds = NNAL_tools.test_training_part(
            labels, self.pars['test_ratio'])
        
        # getting the initial and pool indices
        ntrain = len(train_inds)
        rand_inds = np.random.permutation(ntrain)
        init_inds = train_inds[
            rand_inds[:self.pars['init_size']]]
        pool_inds = train_inds[
            rand_inds[self.pars['init_size']:]]
        
        
        # saving indices into the run's folder
        np.savetxt('%s/train_inds.txt'% run_path, 
                   train_inds, fmt='%d')
        np.savetxt('%s/test_inds.txt'% run_path, 
                   test_inds, fmt='%d')
        np.savetxt('%s/init_inds.txt'% run_path, 
                   init_inds, fmt='%d')
        np.savetxt('%s/pool_inds.txt'% run_path, 
                   pool_inds, fmt='%d')

        # creating an initial initial model
        # -------------------------
        print('Initializing a model for this run..')
            
        # create the NN model
        tf.reset_default_graph()
        model = NN.create_model(
            self.pars['model_name'],
            self.pars['dropout_rate'],
            self.nclass,
            self.pars['learning_rate'],
            self.pars['grad_layers'],
            self.pars['train_layers'])

        # start a session to do the training
        with tf.Session() as sess:
            # training from initial training data
            model.initialize_graph(
                sess, self.pars['pre_weights_path'])
            
            merged_summ = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                os.path.join(
                    '/common/external/rawabd/Jamshid/train_log/All/'),sess.graph)
            TB_opt = {'summs':merged_summ,
                      'writer': train_writer,
                      'epoch_id': 0,
                      'tag': 'initial'}

            for i in range(self.pars['epochs']):
                model.train_graph_one_epoch(
                    self,
                    init_inds,
                    sess)
                TB_opt['epoch_id'] += 1
                print('%d'% i, end=',')
            
            # get a prediction of the test samples
            predicts = model.predict(
                self, 
                test_inds, 
                sess)
                
            # save the predictions 
            np.savetxt(os.path.join(
                run_path, 'init_predicts.txt'), 
                       predicts)
            # save the initial weights
            model.save_weights(os.path.join(
                run_path,'init_weights.h5'))
            
    def add_method(self, method_name, run):
        """Adding a method to a given run of the experiment
        """
        
        # check if the method already exists in this run
        run_path = os.path.join(self.root_dir, str(run))
        if os.path.exists(os.path.join(run_path, 
                                       method_name)):
            print("This method already exists in run %s"% run)
            print("Nothing else to do..")
            return
        
        # create a directory for the method
        os.mkdir(os.path.join(run_path,
                              method_name))
        # create a directory for the queries
        os.mkdir(os.path.join(run_path,
                              method_name,
                              'queries'))

        # copying the following files:
        # init_train_inds -->   curr_train
        # pool_inds       -->   curr_pool
        # init_predicts   -->   predicts
        # saved_model/    -->   curr_model/
        method_path = os.path.join(run_path, method_name)

        shutil.copy(
            os.path.join(run_path,'init_inds.txt'),
            os.path.join(method_path,'curr_train.txt')
            )
        shutil.copy(
            os.path.join(run_path,'pool_inds.txt'),
            os.path.join(method_path,'curr_pool.txt')
            )
        shutil.copy(
            os.path.join(run_path,'init_predicts.txt'),
            os.path.join(method_path,'predicts.txt')
            )
        shutil.copy(
            os.path.join(run_path,'init_weights.h5'),
            os.path.join(method_path,'curr_weights.h5')
            )
        
        # also, computing the first accuracy for the 
        # method
        test_inds = np.int32(np.loadtxt(
            os.path.join(run_path, 'test_inds.txt')))
        predicts = np.loadtxt(
            os.path.join(method_path,'predicts.txt'))
        init_acc = get_accuracy(
            predicts, self.labels_file, test_inds)
        np.savetxt(
            os.path.join(method_path,'accs.txt'),
            [init_acc])
        
    def run_method(self, method_name, run, max_queries):
        """Running a querying method in a run until a 
        given number of queries are drawn
        """
        
        run_path = os.path.join(self.root_dir,str(run))
        method_path = os.path.join(run_path, method_name)
        
        # count how many queries have been 
        # selected before
        n_oldqueries = 0
        iter_cnt = 0
        Q_path = os.path.join(method_path,'queries')
        Q_files = os.listdir(Q_path)
        for f in Q_files:
            Qs = np.loadtxt(os.path.join(
                Q_path, f))
            n_oldqueries += len(Qs)
            iter_cnt += 1
        
        # preparing the indices
        test_inds = np.int32(
            np.loadtxt(os.path.join(
                run_path, 'test_inds.txt')
                   ))
        curr_train = np.int32(
            np.loadtxt(os.path.join(
                method_path, 'curr_train.txt')
                   ))
        curr_pool = np.int32(
            np.loadtxt(os.path.join(
                method_path, 'curr_pool.txt')
                   ))

        tf.reset_default_graph()
        
        if not(hasattr(self, 'pars')):
            self.load_parameters()
            
        
        """ Loading the model """
        print("Loading the current model..")
        # create a model-holder
        model = NN.create_model(
            self.pars['model_name'],
            self.pars['dropout_rate'], 
            self.nclass, 
            self.pars['learning_rate'], 
            self.pars['grad_layers'],
            self.pars['train_layers'])
        model.add_assign_ops(os.path.join(
            method_path, 'curr_weights.h5'))
        
        
        if self.pars['model_name']=='Alex':
            # for AlexNet there are two main
            # differences: name of keep-
            # probability variable is KEEP_PROB
            # and the output is row-wise,
            # hence the column flag (col_flag)
            # should be False
            extra_feed_dict = {
                model.KEEP_PROB: 1.}
            col_flag = False
        else:
            extra_feed_dict = {
                model.keep_prob: 1.}
            col_flag = True
        
        # printing the accuracies so far:
        curr_accs = np.loadtxt(os.path.join(
            method_path, 'accs.txt'))
        if curr_accs.size==1:
            curr_accs = [curr_accs]
        print("Current accuracies: ", end='')
        print(*curr_accs, sep=', ')

        #merged_summ = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(
        #    os.path.join(
        #        '/common/external/rawabd/Jamshid/train_log'),sess.graph)
        #TB_opt = {'summs':merged_summ,
        #          'writer': train_writer,
        #          'epoch_id': 0,
        #          'tag': 'initial'}
        TB_opt = {}
        
        with tf.Session() as sess:
            # loading the stored weights
            model.initialize_graph(sess)
            sess.graph.finalize()

            # starting the iterations
            print("Starting the iterations for %s"%
                  method_name)
            nqueries = 0
            #iter_cnt = 0
            while nqueries < max_queries:
                model.perform_assign_ops(sess)
                
                print("Iter. %d: "% iter_cnt,
                      end='\n\t')
                """ querying """
                Q_inds = NNAL.CNN_query(
                    model, 
                    self,
                    curr_pool,
                    method_name,
                    sess, 
                    col_flag,
                    extra_feed_dict
                )

                # save the queries
                np.savetxt(os.path.join(
                        method_path, 
                        'queries',
                        '%d.txt'% (
                            iter_cnt)
                        ), curr_pool[Q_inds])
                
                # preparing the new training sampels
                old_ratio = 1.
                nold_train = int(np.floor(
                    len(curr_train)*old_ratio))
                rand_inds = np.random.permutation(
                    len(curr_train))[:nold_train]
                old_train = curr_train[rand_inds]
                update_inds = np.append(
                    old_train, curr_pool[Q_inds])

                # update the indices
                curr_train = np.append(
                    curr_train, curr_pool[Q_inds])
                curr_pool = np.delete(
                    curr_pool, Q_inds)
                
                """ updating the model """
                for i in range(self.pars['epochs']):
                    model.train_graph_one_epoch(
                        self,
                        update_inds,
                        sess,
                        TB_opt)
                    print('%d'% i, end=',')
                    
                """ evluating the updated model """
                predicts = model.predict(
                    self, test_inds, sess)
                # loading the previous predictions,
                # appending the new ones to them,
                # and save them back
                curr_predicts = np.loadtxt(
                    os.path.join(method_path, 
                                 'predicts.txt'))
                if curr_predicts.ndim<2:
                    curr_predicts = np.expand_dims(
                        curr_predicts, axis=0)
                new_predicts = np.append(
                    curr_predicts, 
                    np.expand_dims(predicts, axis=0),
                    axis=0)
                np.savetxt(os.path.join(
                    method_path, 
                    'predicts.txt'), new_predicts)

                # computing the accuracies
                acc = get_accuracy(
                    predicts, 
                    self.labels_file,
                    test_inds)
                                   
                with open(os.path.join(
                        method_path, 
                        'accs.txt'), 'a') as f:
                    f.write('%f\n'% acc)
                
                # update the loop variables
                nqueries += len(Q_inds)
                iter_cnt += 1
                
                print('\n\t', end='')
                print("Total queries: %d"% 
                      (nqueries + n_oldqueries),
                      end='\n\t')
                print("Accuracy: %.2f"% acc)
                
            # when querying is done..
            # save the current training and pool
            np.savetxt(os.path.join(
                method_path, 'curr_pool'), 
                       curr_pool,
                       fmt='%d')
            np.savetxt(os.path.join(
                method_path, 'curr_train'), 
                       curr_train,
                       fmt='%d')
            # save the current weights
            model.save_weights(
                os.path.join(
                    method_path,
                    'curr_weights.h5'))
            
    def reset_method(self, method_name, run):
        """ Resetting a given run/method, 
        by deleting its folder and add
        another method with the same name
        """
        
        run_path = os.path.join(self.root_dir, 
                                str(run))
        method_path = os.path.join(run_path,
                                   method_name)
        shutil.rmtree(method_path)
        
        # re-add the same method
        self.add_method(method_name, run)
        
    def read_queries(self, method_name, run):
        """Reading queries of a method in the experiment's
        run
        """
        
        run_path = os.path.join(self.root_dir, str(run))
        method_path = os.path.join(run_path, method_name)
        
        queries = []
        Q_files = os.listdir(os.path.join(
            method_path,'queries'))
        for f in Q_files:
            file_path = os.path.join(
                method_path,'queries',f)
            queries += [len(np.loadtxt(file_path))]
            
        return queries
        
    def eval_run(self, run, eval_method, save=True):
        """Evaluating methods of a given run by
        comparing the predictions in different iterations
        with the ground-truth labels
        """
        
        run_path = os.path.join(
            self.root_dir, str(run))
        
        test_inds = np.int32(np.loadtxt(os.path.join(
            run_path, 'test_inds.txt')))
        test_labels = self.labels[:,test_inds]
        # existing methods in the run    
        subdirs = [subdir for subdir in 
                   os.listdir(run_path) 
                   if os.path.isdir(os.path.join(
                           run_path,subdir))]
        if "saved_model" in subdirs:
            subdirs.remove('saved_model')
            
        """Computing the evaluation metrics"""
        """ ------------------------------ """
        eval_dict={method:[] for method in subdirs}
        for method in subdirs:
            # load all the predictions
            Yhat = np.loadtxt(os.path.join(
                run_path, method, 'predicts.txt'))

            if eval_method=='accuracy':
                eval_crit = np.zeros(Yhat.shape[0])
                for i in range(Yhat.shape[0]):
                    eval_crit[i] = get_accuracy(
                        Yhat[i,:], test_labels)
                if save:
                    np.savetxt(os.path.join(
                        run_path, method, 
                        'accs.txt'), eval_crit)

            elif eval_method=='PR':
                eval_crit = np.zeros((2,Yhat.shape[0]))
                for i in range(Yhat.shape[0]):
                    P,R = get_multi_PR(
                        Yhat[i,:], test_labels)
                    eval_crit[0,i] = P
                    eval_crit[1,i] = R
                if save:
                    np.savetxt(os.path.join(
                        run_path, method, 
                        'accs.txt'), eval_crit)
            
            eval_dict[method] = eval_crit
            
        return eval_dict
            
        
    def read_run(self, run):
        """Reading results of different methods
        in a given run of the experiment
        """
        
        # methods that exist in this run
        run_path = os.path.join(self.root_dir,
                                str(run))
        dirs = [dir for dir in os.listdir(run_path)
                if os.path.isdir(os.path.join(
                        run_path,dir))]
        # throw away folder of 'saved_models'
        if 'saved_model' in dirs:
            dirs.remove('saved_model')
        
        # dictionary of accuracies
        accs = {}
        for method in dirs:
            method_path = os.path.join(run_path,
                                       method)
            accs.update({method: np.loadtxt(
                os.path.join(method_path,'accs.txt'))})
            
        # also returning the number of queries for
        # FI-sum algorithm
        fi_queries = []
        if 'fi' in dirs:
            Q_files = os.listdir(os.path.join(
                run_path,'fi','queries'))
            for f in Q_files:
                file_path = os.path.join(
                    run_path,'fi','queries',f)
                fi_queries += [len(np.loadtxt(file_path))]

        return accs, fi_queries
        
    def visualize_run(self, run, tags={}, interp=True):
        """Visualizing results of a specific run in the
        experiment
        """
        
        # first read the results
        accs, fi_queries = self.read_run(run)
        
        # load parameters
        if not(hasattr(self, 'pars')):
            self.load_parameters()
            
        

        # compute the common maximum queries
        max_queries = np.sum(
            self.read_queries('random',run))
        xq = np.arange(0,max_queries+self.pars['k'],
                       self.pars['k'])
        if 'fi' in accs:
            if interp:
                interp_fi_accs = np.zeros(len(xq))
                f = scipy.interpolate.interp1d(
                    np.cumsum([0]+fi_queries), 
                    accs['fi'], kind='nearest')
                accs['fi'] = f(xq)
        # make sure the order will be the same
        all_methods = ['fi','random',
                       'entropy','rep-entropy']
        for method_name in all_methods:
            if not(method_name in accs):
                continue
                
            if (method_name=='fi' and not(interp)):
                fi_xq = np.cumsum([0] + fi_queries)
                plt.plot(fi_xq, 
                         accs[method_name],
                         label=method_name,
                         marker='*')
            else:
                plt.plot(xq, 
                         accs[method_name], 
                         label=method_name,
                         marker='*')
            
        plt.xlabel('# Queries', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.legend(fontsize=15)
        #plt.xticks(np.arange(
        #    0,max_queries+1,k))
        plt.xlim([-5,max_queries+5])
        plt.grid()
        
    def summarize_all(self, viz=True, std=False):
        """Visualizing the average results of 
        all existing runs (assuming that they
        all have same number of queries)
        """
        
        # list the runs
        runs = self.get_runs()
        
        # load parameters
        if not(hasattr(self, 'pars')):
            self.load_parameters()
        
        # read the first run to compute the 
        # common maximum queries
        max_queries = np.sum(
            self.read_queries('random',runs[0]))
        xq = np.arange(0,max_queries+self.pars['k'],
                       self.pars['k'])
        
        # reading everything 1-by-1
        all_methods = ['fi','random',
                       'entropy','rep-entropy']

        total_accs = {method: 
                      np.zeros((len(runs), len(xq))) 
                      for method in all_methods}

        for i in range(len(runs)):
            accs, fi_queries = self.read_run(runs[i])
            
            # interpolation
            interp_fi_accs = np.zeros(len(xq))
            f = scipy.interpolate.interp1d(
                np.cumsum([0]+fi_queries), 
                accs['fi'], kind='nearest')
            accs['fi'] = f(xq)
                
            for method in all_methods:
                total_accs[method][i,:] = accs[method]
                
        # visualizing the mean/std, if necessary
        if std:
            for method in all_methods:
                plt.errorbar(
                    xq, np.mean(total_accs[method], axis=0), 
                    yerr=np.std(total_accs[method], axis=0), 
                    label=method, marker='*')
        else:
            for method in all_methods:
                plt.plot(
                    xq, np.mean(total_accs[method], axis=0),
                    label=method, marker='*')
        plt.xlabel('# Queries', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid()
        return total_accs
        

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
    
def make_onehot(labels,c):
    """Make a one-hot label matrix out of
    a 1D array of labels
    """
    if np.array(labels).ndim>1:
        raise ValueError(
            "The input for one-hot conversion"+
            " be a 1-D array.. %d-D is given"% 
            (labels.ndim))
    
    one_hot = np.zeros((c, len(labels)))
    given_labels = np.unique(labels)
    for label in given_labels:
        label_inds = np.where(
            labels==label)[0]
        one_hot[label,label_inds] = 1.
        
    return one_hot

def onehot_to_classid(labels):
    """Covnerting a one-hot label matrix into
    an array of class ID, such that the i'th
    element of the output array indicates the 
    class ID--which corresponds to the row ID
    in the input one-hot vector--of the i'th
    column of the input.
    """
    
    if np.array(labels).ndim<2:
        raise ValueError(
            "The given label does not seem to"+
            "be a one-hot vector..")
        
    one_indics = np.where(labels>0)
    class_ids = one_indics[0][
        np.argsort(one_indics[1])]
    
    return class_ids
        
        
def get_accuracy(predicts, labels_file, inds):
    """Computing accuracy of a set of predictions
    based on a given ground-truth labels
    
    The predictions should be in form
    of integers, where each integer represents a
    class label.
    """
    
    n = len(predicts)
    
    # if labels are in one-hot format
    cnt = 0
    for i in range(len(inds)):
        label = linecache.getline(
            labels_file,
            inds[i]+1).splitlines()[0]
        if predicts[i]==int(label):
            cnt += 1
        
    # now compare the integer class labels
    acc = float(cnt) / float(n)
    
    return acc
    

def get_multi_PR(predicts, labels, hot=True):
    """Computing Precision-Recall of a multiclass
    predictions and ground-truth

    This function is an example-based method explained
    in http://ieeexplore.ieee.org/document/6471714/
    """
    
    n=len(predicts)

    # if labels are in one-hot vector format
    if hot:
        labels = np.where(labels>0)[0]
    
    # number of classes:
    C = len(np.unique(labels))
    PRs = np.zeros((2,C))
    for i in range(C):
        # compute PR for this class versus rest
        bin_predicts = predicts==i
        bin_labels = labels==i
        if all(~bin_predicts):
            continue
        
        (P,R) = get_PR(bin_predicts, bin_labels)
        PRs[0,i] = P
        PRs[1,i] = R
    
    return np.mean(PRs, axis=1)


def get_PR(bin_predicts, bin_labels):
    """Computing Precision-Recall metric for a given
    set of binary predictions and ground-truth
    """
    
    TP = np.logical_and(bin_predicts, bin_labels)
    FP = np.logical_and(bin_predicts, ~bin_labels)
    FN = np.logical_and(~bin_predicts, bin_labels)
    
    # precision = TP / (TP+FP)
    P = float(np.sum(TP)) / float(np.sum(TP) + np.sum(FP))
        
    # recall = TP / (TP+FN)
    R = float(np.sum(TP)) / float(np.sum(TP) + np.sum(FN))
    
    return (P,R)
