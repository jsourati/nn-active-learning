from matplotlib import pyplot as plt
import numpy as np
import linecache
import shutil
import pickle
import scipy
import nrrd
import yaml
import pdb
import os

import tensorflow as tf
import patch_utils
import PW_NNAL
import PW_NN
import NNAL
import NN

class Experiment(object):
    """class of an active learning experiments
    """
    
    def __init__(self,
                 root_dir,
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
                
        
    def add_run(self, 
                pool_imgs, test_imgs,
                pool_ratio, test_ratio):
        """Adding a run to this experiment
        
        Each run will have its pool and test
        image indices, which will be sampled
        to get the pool and test data sets
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
        img_addrs, mask_addrs = patch_utils.extract_newborn_data_path()
        inds_opath = os.path.join(
            run_path,'inds.txt')
        label_opath = os.path.join(
            run_path,'labels.txt')
        pool_inds, test_inds = newborn_prep_dat(
            img_addrs, mask_addrs,
            pool_imgs, test_imgs, 
            pool_ratio, test_ratio,
            inds_opath, label_opath,
            self.pars['mask_ratio'])
        
        # saving indices into the run's folder
        np.savetxt('%s/test_inds.txt'% run_path, 
                   test_inds, fmt='%d')
        np.savetxt('%s/pool_inds.txt'% run_path, 
                   pool_inds, fmt='%d')

        # evaluating the initial performance
        # -------------------------
        # create the NN model
        tf.reset_default_graph()
        model = NN.create_model(
            self.pars['model_name'],
            self.pars['dropout_rate'],
            self.nclass,
            self.pars['learning_rate'],
            self.pars['grad_layers'],
            self.pars['train_layers'],
            self.pars['patch_shape'])

        # start a session to do the training
        with tf.Session() as sess:
            # training from initial training data
            model.initialize_graph(sess)
            model.load_weights(
                self.pars['init_weights_path'], sess)
                        
            # get a prediction of the test samples
            ts_preds = batch_eval_winds(
                self,
                n,
                model,
                test_inds,
                'prediction',
                sess)
                
            # save the predictions 
            np.savetxt(os.path.join(run_path, 
                                    'init_predicts.txt'), 
                       np.expand_dims(ts_preds,axis=0),
                       fmt='%d')
            
            # initial, performance evaluation
            ts_labels = read_label_lines(
                label_opath, test_inds)
            Fmeas = PW_NN.get_Fmeasure(ts_preds, 
                                       ts_labels)
            perf_eval_path = os.path.join(
                run_path, 'init_perf_eval.txt')
            with open(perf_eval_path, 'w') as f:
                f.write('%f\n'% Fmeas)
                
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
        # pool_inds            -->   curr_pool
        # init_predicts        -->   predicts
        # `init_weights_path`  -->   curr_weights.h5
        # init_perf_eval       -->   perf_evals
        method_path = os.path.join(run_path, method_name)

        shutil.copy(
            os.path.join(run_path,'pool_inds.txt'),
            os.path.join(method_path,'curr_pool.txt')
            )
        shutil.copy(
            os.path.join(run_path,'init_predicts.txt'),
            os.path.join(method_path,'predicts.txt')
            )
        shutil.copy(
            self.pars['init_weights_path'],
            os.path.join(method_path,'curr_weights.h5')
            )
        shutil.copy(
            os.path.join(run_path,'init_perf_eval.txt'),
            os.path.join(method_path,'perf_evals.txt')
            )
        
    def run_method(self, method_name, run, max_queries):
        """Running a querying method in a run until a 
        given number of queries are drawn
        """
        
        run_path = os.path.join(self.root_dir,
                                str(run))
        method_path = os.path.join(run_path, 
                                   method_name)
        labels_path = os.path.join(run_path, 
                                   'labels.txt')
        
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
        
        # preparing the (line) indices
        test_inds = np.int32(
            np.loadtxt(os.path.join(
                run_path, 'test_inds.txt')
                   ))
        train_path = os.path.join(
            method_path, 'curr_train.txt')
        if os.path.exists(train_path):
            curr_train = np.int32(
                np.loadtxt(train_path))
        else:
            curr_train = []
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
            self.pars['train_layers'],
            self.pars['patch_shape'])
        model.add_assign_ops(os.path.join(
            method_path, 'curr_weights.h5'))
        
        
        #extra_feed_dict = {
        #    model.keep_prob: 1.}
        #col_flag = True
        
        # printing the accuracies so far:
        curr_fmeas = np.loadtxt(os.path.join(
            method_path, 'perf_evals.txt'))
        if curr_fmeas.size==1:
            curr_fmeas = [curr_fmeas]
        print("Current F-measures: ", end='')
        print(*curr_fmeas, sep=', ')

        
        with tf.Session() as sess:
            # loading the stored weights
            model.initialize_graph(sess)
            sess.graph.finalize()

            # starting the iterations
            print("Starting the iterations for %s"%
                  method_name)
            nqueries = 0
            while nqueries < max_queries:
                model.perform_assign_ops(sess)
                
                print("Iter. %d: "% iter_cnt,
                      end='\n\t')
                """ querying """
                Q_inds = PW_NNAL.CNN_query(
                    self,
                    run,
                    model,
                    curr_pool,
                    method_name,
                    sess)

                if self.pars['k']==1:
                    Q = [curr_pool[Q_inds]]
                else:
                    Q = curr_pool[Q_inds]

                # save the queries
                np.savetxt(os.path.join(
                        method_path, 
                        'queries',
                        '%d.txt'% (
                            iter_cnt)
                        ), Q, fmt='%d')
                
                # update the indices
                if len(curr_train)==0:
                    curr_train = Q
                else:
                    curr_train = np.append(
                        curr_train, Q)
                curr_pool = np.delete(
                    curr_pool, Q)
                
                """ updating the model """
                for i in range(self.pars['epochs']):
                    PW_train_epoch_winds(
                        model,
                        self,
                        run,
                        curr_train,
                        sess)
                    print('%d'% i, end=',')
                    
                """ evluating the updated model """
                ts_preds = batch_eval_winds(self,
                                            run,
                                            model,
                                            test_inds,
                                            'prediction',
                                            sess)
                # saving the predictions
                curr_predicts = np.loadtxt(
                    os.path.join(method_path, 
                                 'predicts.txt'))
                # loading the previous predictions,
                # appending the new ones to them,
                # and save them back
                if curr_predicts.ndim<2:
                    curr_predicts = np.expand_dims(
                        curr_predicts, axis=0)
                new_predicts = np.append(
                    curr_predicts, 
                    np.expand_dims(ts_preds, axis=0),
                    axis=0)
                np.savetxt(os.path.join(
                    method_path, 'predicts.txt'), 
                           new_predicts, fmt='%d')
            
                # performance evaluation
                ts_labels = read_label_lines(
                    labels_path, test_inds)
                Fmeas = PW_NN.get_Fmeasure(
                    ts_preds, ts_labels)
                                   
                with open(os.path.join(
                        method_path, 
                        'perf_evals.txt'), 'a') as f:
                    f.write('%f\n'% Fmeas)
                    
                    # update the loop variables
                    nqueries += len(Q_inds)
                    iter_cnt += 1
                
                    print('\n\t', end='')
                    print("Total queries: %d"% 
                          (nqueries + n_oldqueries),
                          end='\n\t')
                    print("F-measure: %.4f"% Fmeas)
                
            # when querying is done..
            # save the current training and pool
            np.savetxt(os.path.join(
                method_path, 'curr_pool'), 
                       curr_pool,
                       fmt='%d')
            np.savetxt(train_path, 
                       curr_train,
                       fmt='%d')
            # save the current weights
            model.save_weights(
                os.path.join(
                    method_path,
                    'curr_weights.h5'))

def newborn_prep_dat(img_addrs, mask_addrs,
                     pool_imgs, test_imgs, 
                     pool_ratio, test_ratio,
                     dat_opath, label_opath,
                     mask_ratio):
    """Preparing test and pool image data
    sets, by giving separate images for 
    test-sampling and pool-sampling
    
    The image indices are in terms of the
    newborn data
    
    """
    
    npool_img = len(pool_imgs)
    ntest_img = len(test_imgs)

    # data set
    sel_img_addrs = [img_addrs[i] 
                for i in pool_imgs+test_imgs]
    sel_mask_addrs = [mask_addrs[i] 
                for i in pool_imgs+test_imgs]

    D = patch_utils.PatchBinaryData(
        sel_img_addrs, sel_mask_addrs)

    # sampling from test images
    pinds_dict, pmask_dict = D.generate_samples(
        np.arange(npool_img), test_ratio, 
        mask_ratio, 'axial')

    # sampling from pool images
    tsinds_dict, tsmask_dict = D.generate_samples(
        np.arange(npool_img, npool_img+ntest_img), 
        pool_ratio, mask_ratio, 'axial')
    
    """write the text files"""
    cnt = 0
    # pool
    with open(dat_opath,'a') as f, open(
            label_opath,'a') as g:
        for path in list(pinds_dict.keys()):
            for i in range(len(pinds_dict[path])):
                # determining type of this sample
                stype,_ = patch_utils.get_sample_type(
                    pool_ratio, i)
                # data file
                f.write(
                    '%s, %d, %s\n'
                    % (path,
                       pinds_dict[path][i],
                       stype))
                # label file
                g.write(
                    '%d\n'%(pmask_dict[path][i]))
                cnt += 1
    npool = cnt

    # test
    with open(dat_opath,'a') as f, open(
            label_opath,'a') as g:
        for path in list(tsinds_dict.keys()):
            for i in range(len(tsinds_dict[path])):
                # determining type of this sample
                stype,_ = patch_utils.get_sample_type(
                    test_ratio, i)
                # data file
                f.write(
                    '%s, %d, %s\n'
                    % (path,
                       tsinds_dict[path][i],
                       stype))
                # label file
                g.write(
                    '%d\n'%(tsmask_dict[path][i]))
                cnt += 1
    
    return np.arange(npool)+1, np.arange(npool,cnt)+1

def load_patches(expr, 
                 run, 
                 inds, 
                 label_flag=False):
    """Loading a set of patches and their
    labels that are specified by the line
    numbers where they are saved in the
    corresponding PW-AL experiment's run
    """
        
    inds_path = os.path.join(
        expr.root_dir,str(run),'inds.txt')

    img_paths = []
    inds_array = np.zeros(len(inds), dtype=int)

    for i in range(len(inds)):
        line = linecache.getline(
            inds_path, inds[i]).splitlines()[0]
        img_paths += [line.split(',')[0]]
        inds_array[i] = int(line.split(',')[1])
        
    # start loading the patches in the same order
    # as is determined in `inds`
    patches = np.zeros((len(inds),) + 
                       expr.pars['patch_shape'])
    # load patches of same images at the 
    # same time
    upaths = np.unique(img_paths)
    for path in upaths:
        indics = np.where(
            np.array(img_paths)==path)[0]
        img,_ = nrrd.read(path)
        img_patches = patch_utils.get_patches(
            img, inds_array[indics],
            expr.pars['patch_shape'])
        patches[indics,:,:,:] = img_patches

    if label_flag:
        labels = np.zeros(len(inds), 
                          dtype=bool)
        for i in range(len(inds)):
            labels[i] = linecache.getline(
                inds_path, 
                inds[i]).splitlines()[0]
            
        return patches, labels
        
    # if labels_flag is off, only returns
    # the patches
    return patches
    
def create_dict(inds_path,
                line_inds,
                labels_path=None):
    """Creating a dictionary of
    sample indices, from specific lines
    of a given index-file (that is built
    in a PW-experiment's run,
    """
    img_paths = []
    inds_array = np.zeros(len(line_inds), dtype=int)
    labels_array = np.zeros(len(line_inds), dtype=int)
    
    for i in range(len(line_inds)):
        line = linecache.getline(
            inds_path, 
            line_inds[i]).splitlines()[0]
        img_paths += [line.split(',')[0]]
        inds_array[i] = int(line.split(',')[1])
        if labels_path:
             labels_array[i] = int(linecache.getline(
                labels_path, 
                line_inds[i]).splitlines()[0])

    upaths = np.unique(img_paths)
    inds_dict = {}
    labels_dict = {}
    # also need the location of samples in each
    # image, within the original given line-indices
    locs_dict = {} 
    for path in upaths:
        indics = np.where(
            np.array(img_paths)==path)[0]
        img_inds = inds_array[indics]
        inds_dict[path] = img_inds
        locs_dict[path] = indics
        if labels_path:
            img_labels = labels_array[indics]
            labels_dict[path] = img_labels

    if labels_path:
        return inds_dict, labels_dict, locs_dict
    else:
        return inds_dict, locs_dict

def batch_eval_winds(expr,
                     run,
                     model,
                     line_inds,
                     varname,
                     sess):
    """Batch-evaluation of network
    variables (such as predictions,
    and posteriors) using for samples
    that are specified by line-numbers 
    of saved indices in the corresponding
    experiment's run
    
    CAUTION: for now, we assume only a 
    single variable is given, that is 
    `len(varname)` is one
    """
    
    inds_path = os.path.join(
        expr.root_dir, str(run), 'inds.txt')
    
    # we will be using a function that is
    # already written in PW_NN, that is
    # `PW_NN.batch_eval()`, which is 
    # capable of evaluating different
    # network variables with batches. But
    # it needs a dictionary of indices, 
    # not a bunch of line indices.
    # -----
    # Hence, create the dictionary of 
    # indices, from the sample line
    # numbers
    if varname=='loss':
        labels_path = os.path.join(
            expr.root_dir, 
            str(run), 
            'labels.txt')
        inds_dict, labels_dict, locs_dict = create_dict(
            inds_path, line_inds, labels_path)
        
        eval_dict = PW_NN.batch_eval(
            model,
            inds_dict,
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            sess,
            varname,
            labels_dict)[0]
    else:
        inds_dict, locs_dict = create_dict(
            inds_path, line_inds)
        
        eval_dict = PW_NN.batch_eval(
            model,
            inds_dict,
            expr.pars['patch_shape'],
            expr.pars['ntb'],
            expr.pars['stats'],
            sess,
            varname)[0]
        
    # and finally, copy the resulting values
    # in their right places that corresponds to
    # the original array of line-indices
    eval_array = np.zeros(len(line_inds))
    for path in list(eval_dict.keys()):
        locs = locs_dict[path]
        eval_array[locs] = eval_dict[path]

    return eval_array
    
def PW_train_epoch_winds(model,
                        expr,
                        run,
                        line_inds,
                        sess):
    """Running one epoch of trianing 
    with training (line) indices with respect
    to a run of pw-experiment
    """
    
    inds_path = os.path.join(
        expr.root_dir, str(run), 'inds.txt')
    labels_path = os.path.join(
        expr.root_dir, str(run), 'labels.txt')
    
    inds_dict, labels_dict, locs_dict = create_dict(
        inds_path, line_inds, labels_path)
    
    # now that we have the index- and labels-
    # dictionaries, we can feed it to 
    # PW_NN.PW_train_epoch()
    PW_NN.PW_train_epoch(
        model,
        expr.pars['dropout_rate'],
        inds_dict,
        labels_dict,
        expr.pars['patch_shape'],
        expr.pars['b'],
        expr.pars['stats'],
        sess)
    

def read_label_lines(labels_path, line_inds):
    """Reading several lines of a label
    file, which is stored in a format consistent
    with that of pw-experiment run's labels.txt
    """
    
    labels_array = np.zeros(len(line_inds),
                            dtype=int)
    for i in range(len(line_inds)):
        labels_array[i] = int(linecache.getline(
            labels_path, 
            line_inds[i]).splitlines()[0])
        
    return labels_array
