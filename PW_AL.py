from skimage.measure import regionprops
from skimage.segmentation import slic
#from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import linecache
import shutil
import pickle
import scipy
import nrrd
import yaml
import copy
import pdb
import os

import PW_analyze_results
import patch_utils
import NNAL_tools
import PW_NNAL
import PW_NN
import NNAL
import NN

class Experiment(object):
    """class of an active learning experiments with 
    voxel-wise querying
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
        self.nclass=2
        if not(os.path.exists(root_dir)):
            os.mkdir(root_dir)
        
        # if there are parameter given, write them
        # into a text file to be used later
        if len(pars)>0:
            if os.path.exists(os.path.join(
                    root_dir, 
                    'parameters.txt')):
                print("Some parameters already exist")
            else:
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
            
            self.pars = copy.deepcopy(pars)
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
                

    def prep_data(self):
        """Adding a run to this experiment
        
        Each run will have its pool and test
        image indices, which will be sampled
        to get the pool and test data sets
        """
                
        if not(hasattr(self, 'pars')):
            self.load_parameters()
        
        # preparing the indices
        # -----------------------
        prep_AL_data(self)
        
        # get the test indices for initial
        # performance evaluation
        test_inds = read_ints(os.path.join(
            self.root_dir,'test_inds.txt'))
        test_labels = np.array(read_ints(os.path.join(
            self.root_dir,'test_labels.txt')))

        # evaluating the initial performance
        # -------------------------
        # create the NN model
        tf.reset_default_graph()
        m = len(self.pars['img_paths'])
        patch_shape = self.pars['patch_shape'][:2] + \
                      (m*self.pars['patch_shape'][2],)
        model = NN.create_model(
            self.pars['model_name'],
            self.pars['dropout_rate'],
            self.nclass,
            self.pars['learning_rate'],
            self.pars['grad_layers'],
            self.pars['train_layers'],
            self.pars['optimizer_name'],
            patch_shape)

        #  computing pool statistics
        #self.pars['stats'] = [mu, sigma] #[65., 54.5]
        #self.save_parameters(self.pars)

        # start a session to do the training
        with tf.Session() as sess:
            # training from initial training data
            model.initialize_graph(sess)
            if 'init_weights_path' in self.pars:
                model.load_weights(
                    self.pars['init_weights_path'], 
                    sess)
                        
            # get a prediction of the test samples
            test_preds = PW_NN.batch_eval(
                model,
                sess,
                self.pars['img_paths'],
                test_inds,
                self.pars['patch_shape'],
                self.pars['ntb'],
                self.pars['stats'],
                'prediction')[0]
                
            # save the predictions 
            np.savetxt(os.path.join(self.root_dir, 
                                    'init_predicts.txt'), 
                       np.expand_dims(test_preds,axis=0),
                       fmt='%d')
            
            # initial, performance evaluation
            Fmeas = PW_analyze_results.get_Fmeasure(
                test_preds, test_labels)
            print("Initial F-measure: %f"% Fmeas)
            perf_eval_path = os.path.join(
                self.root_dir, 'init_perf_eval.txt')
            with open(perf_eval_path, 'w') as f:
                f.write('%f\n'% Fmeas)
                
    def add_method(self, method_name):
        """Adding a method to a given run of the experiment
        """
        
        # check if the method already exists in this run
        if os.path.exists(os.path.join(self.root_dir, 
                                       method_name)):
            print("This method already exists")
            print("Nothing else to do..")
            return
        
        # create a directory for the method
        os.mkdir(os.path.join(self.root_dir,
                              method_name))
        # create a directory for the queries
        os.mkdir(os.path.join(self.root_dir,
                              method_name,
                              'queries'))

        # copying the following files:
        # pool_inds            -->   curr_pool
        # init_predicts        -->   predicts
        # `init_weights_path`  -->   curr_weights.h5
        # init_perf_eval       -->   perf_evals
        method_path = os.path.join(self.root_dir, 
                                   method_name)

        shutil.copy(
            os.path.join(self.root_dir,'init_pool_inds.txt'),
            os.path.join(method_path,'pool_inds.txt')
            )
        shutil.copy(
            os.path.join(self.root_dir,'init_pool_labels.txt'),
            os.path.join(method_path,'pool_labels.txt')
            )
        shutil.copy(
            os.path.join(self.root_dir,'init_predicts.txt'),
            os.path.join(method_path,'predicts.txt')
            )
        shutil.copy(
            self.pars['init_weights_path'],
            os.path.join(method_path,'curr_weights.h5')
            )
        shutil.copy(
            os.path.join(self.root_dir,'init_perf_eval.txt'),
            os.path.join(method_path,'perf_evals.txt')
            )
        
    def run_method(self, method_name, max_queries):
        """Running a querying method in a run until a 
        given number of queries are drawn
        """

        # read all the modalities and pad them
        rads = np.zeros(3,dtype=int)
        for i in range(3):
            rads[i] = int(
                (self.pars['patch_shape'][i]-1)/2.)

        padded_imgs = []
        for path in self.pars['img_paths']:
            img,_ = nrrd.read(path)
            padded_img = np.pad(
                img, 
                ((rads[0],rads[0]),
                 (rads[1],rads[1]),
                 (rads[2],rads[2])),
                'constant')
            padded_imgs += [padded_img]
        mask,_ = nrrd.read(self.pars['mask_path'])

        # set up the paths
        method_path = os.path.join(self.root_dir, 
                                   method_name)
        labels_path = os.path.join(self.root_dir, 
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
        
        # preparing the indices
        test_inds = read_ints(os.path.join(
            self.root_dir,'test_inds.txt'))
        test_labels = np.array(read_ints(os.path.join(
            self.root_dir,'test_labels.txt')))
        pool_inds = np.array(read_ints(os.path.join(
            method_path, 'pool_inds.txt')))
        pool_labels = read_ints(os.path.join(
            method_path, 'pool_labels.txt'))
        # for training
        train_path = os.path.join(
            method_path, 'train_inds.txt')
        if os.path.exists(train_path):
            train_inds = np.int32(
                np.loadtxt(train_path))
        else:
            train_inds = []

        print('Test-size: %d'% (len(test_inds)))
        print('Pool-size: %d'% (len(pool_inds)))
        print('Train-size: %d'% (len(train_inds)))
        
        if not(hasattr(self, 'pars')):
            self.load_parameters()
            
        
        """ Loading the model """
        print("Loading the current model..")
        tf.reset_default_graph()
        # create a model-holder
        m = len(self.pars['img_paths'])
        patch_shape = self.pars['patch_shape'][:2] + \
                      (m*self.pars['patch_shape'][2],)
        model = NN.create_model(
            self.pars['model_name'],
            self.pars['dropout_rate'], 
            self.nclass, 
            self.pars['learning_rate'], 
            self.pars['grad_layers'],
            self.pars['train_layers'],
            self.pars['optimizer_name'],
            patch_shape)
        
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
            model.load_weights(os.path.join(
                method_path,'curr_weights.h5'), sess)
            sess.graph.finalize()

            # starting the iterations
            print("Starting iterations of %s"%
                  method_name)
            nqueries = 0

            while nqueries < max_queries:
                
                print("Iter. %d: "% iter_cnt,
                      end='\n\t')
                """ querying """
                # decide the number of queries 
                # for this iteration (only to 
                # be used fro non-fi algorithms)
                if 'iter_k' in self.pars:
                    self.pars['k'] = self.pars[
                        'iter_k'][iter_cnt]
                
                Q_inds = PW_NNAL.CNN_query(
                    self,
                    model,
                    sess,
                    padded_imgs,
                    pool_inds,
                    train_inds,
                    method_name)

                if self.pars['k']==1:
                    Q = [pool_inds[Q_inds]]
                else:
                    Q = pool_inds[Q_inds]

                # save the queries
                np.savetxt(os.path.join(
                        method_path, 
                        'queries',
                        '%d.txt'% (
                            iter_cnt)
                        ), Q, fmt='%d')
                
                # update the indices
                if len(train_inds)==0:
                    train_inds = Q
                else:
                    train_inds = np.append(
                        train_inds, Q)
                pool_inds = np.delete(
                    pool_inds, Q_inds)
                                
                """ updating the model """
                for i in range(self.pars['epochs']):
                    finetune(model,
                             sess,
                             self,
                             padded_imgs,
                             mask,
                             train_inds)
                    print('%d'% i, end=',')
                    
                """ evluating the updated model """
                test_preds = PW_NN.batch_eval(
                    model,
                    sess,
                    padded_imgs,
                    test_inds,
                    self.pars['patch_shape'],
                    self.pars['ntb'],
                    self.pars['stats'],
                    'prediction')[0]

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
                    np.expand_dims(test_preds, axis=0),
                    axis=0)
                np.savetxt(os.path.join(
                    method_path, 'predicts.txt'), 
                           new_predicts, fmt='%d')
            
                # performance evaluation
                Fmeas = PW_analyze_results.get_Fmeasure(
                    test_preds, test_labels)
                                   
                with open(os.path.join(
                        method_path, 
                        'perf_evals.txt'), 'a') as f:
                    f.write('%f\n'% Fmeas)
                    
                    # update the loop variables
                    nqueries += len(Q_inds)
                    iter_cnt += 1
                
                    print('\n\t', end='')
                    print("Total queries: %d"% 
                          (len(train_inds)),
                          end='\n\t')
                    print("F-measure: %.4f"% Fmeas)
                
                # when querying is done..
                # save the current training and pool
                np.savetxt(os.path.join(
                    method_path, 'pool_inds.txt'), 
                           pool_inds,
                           fmt='%d')
                np.savetxt(train_path, 
                           train_inds,
                           fmt='%d')
                # save the current weights
                model.save_weights(
                    os.path.join(
                        method_path,
                        'curr_weights.h5'))

    def finetune_wpool(self, run, 
                       save_names=[],
                       train_lines_path=[],
                       full=False,
                       tb_files=[]):
        """Finetuning the initial model of an
        experiment with all the pool samples of
        a given run
        
        
        """
        
        if len(train_lines_path)>0:
            train_lines = np.int32(np.loadtxt(
                train_lines_path))
        else:
            train_lines = np.int32(np.loadtxt(
                os.path.join(self.root_dir,str(run),
                             'init_pool_lines.txt')))

        test_lines = np.int32(np.loadtxt(
            os.path.join(self.root_dir,str(run),
                         'test_lines.txt')))
        
        pool_Fmeas, model = finetune_winds(
            self, run,
            train_lines,
            test_lines,
            tb_files)
        
        print('Final F-measure: %f'% pool_Fmeas)
        if save_names:
            save_path = os.path.join(
                self.root_dir, str(run),
                '%s.txt'% (save_names[0]))
            with open(save_path, 'w') as f:
                f.write('%f\n'% pool_Fmeas)
                
            save_path = os.path.join(
                self.root_dir, str(run),
                '%s.txt'% (save_names[1]))

            model.save_weights(save_path)


    def load_results(self, run):
        """Loading performance evaluations
        for all the methods in a given
        run of the experiment
        """
        
        methods = os.listdir(os.path.join(
            self.root_dir, str(run)))
        methods = [
            f for f in methods 
            if os.path.isdir(os.path.join(
                    self.root_dir,
                    str(run),f))]
        
        # load performance evaluations
        # together with number of queries
        # in each method
        Q_lens = []
        perf_evals = []
        for method in methods:
            method_path = os.path.join(
                self.root_dir, str(run),
                method)
            # performance evaluation
            F = np.loadtxt(os.path.join(
                method_path, 'perf_evals.txt'))
            perf_evals += [F]
            # length of the queries
            Q_path = os.path.join(
                method_path,'queries')
            Q_files = os.listdir(Q_path)
            L = [0]
            for f in Q_files:
                Qs = np.loadtxt(os.path.join(
                    Q_path, f))
                L += [len(Qs)]
            Q_lens += [L]

        return perf_evals, Q_lens, methods


class SuPixExperiment(Experiment):
    """Active learning experiment with 
    super-pixel queries
    """
    
    def __init__(self, root_dir, 
                 pars={}):
        """Constructor
        """
        
        Experiment.__init__(self, 
                            root_dir,
                            pars)
        
    def add_run(self):
        """Adding a run to the
        experiment
        """
        
        Experiment.add_run(self)
        nrun = len(self.get_runs())-1
        
        # now preparing the super-pixels
        pool_lines = np.int32(np.loadtxt(
            os.path.join(self.root_dir,
                         str(nrun),
                         'init_pool_lines.txt')))
        pool_multinds = get_multinds(
            self, nrun, pool_lines)
        pool_slices = np.unique(pool_multinds[2])

        img_path,_ = get_expr_paths(self)
        img,_ = nrrd.read(img_path)
        # over-segmentation has the same 
        # shape as the original image, but
        # includes oversegmentation only in
        # pool slices (and not test)
        overseg_img = np.zeros(img.shape,
                               dtype=int)
        for z in pool_slices:
            slice_ = img[:,:,z]
            if slice_.max()>0:
                slice_ /= slice_.max()

            # exclude label zero from the
            # oversegmentation image
            overseg_img[:,:,z] = slic(
                slice_, 
                n_segments=750, 
                compactness=1e-1, 
                max_iter=50, 
                sigma=0) + 1
            
        # saving the image in the run
        np.save(os.path.join(
            self.root_dir, str(nrun),
            'oversegs.npy'), overseg_img)

    def run_method(self, 
                   method_name, 
                   run,
                   max_queries):
        """running a querying method 
        in a run
        """
        
        run_path = os.path.join(self.root_dir,
                                str(run))
        inds_path = os.path.join(run_path,
                                 'inds.txt')
        labels_path = os.path.join(run_path,
                                 'labels.txt')
        method_path = os.path.join(run_path, 
                                   method_name)
        
        # loading the image's oversegmentation
        overseg_img = np.load(os.path.join(
            run_path, 'oversegs.npy'))
        
        # count the previously queries super-
        # pixels and zeroing them out from the 
        # oversegmented image
        iter_cnt = 0  # querying iterations
        nSuPix = 0 # labeled super-pixels
        nPix = 0   # labeled pixels
        Q_path = os.path.join(method_path,
                              'queries')
        Q_files = []#os.listdir(Q_path)
        for f in Q_files:
            qSuPix = np.int32(np.loadtxt(
                os.path.join(Q_path, f)))
            iter_cnt += 1
            nSups = qSuPix.shape[1]
            nSuPix += nSups
            # zero-ing out
            for i in range(nSups):
                tmp = overseg_img[:,:,
                                  qSuPix[0,i]]
                nPix += np.sum(tmp==qSuPix[1,i])
                tmp[tmp==qSuPix[1,i]] = 0
            
        """Prearing the Indices"""
        test_lines = np.int32(
            np.loadtxt(os.path.join(
                run_path, 'test_lines.txt')))

        train_path = os.path.join(
            method_path, 'train_inds.txt')
        if os.path.exists(train_path):
            train_inds = np.int32(
                np.loadtxt(train_path))
        else:
            train_inds = np.array([],
                                  dtype=int)

        nPix = len(train_inds)
        pool_lines = np.int32(
            np.loadtxt(os.path.join(
                method_path, 'pool_lines.txt')))
        # for the pool, get the pixel indices
        _,img_path,mask_path = get_expr_data_info(
            self)
        pool_inds,pool_locs = create_dict(
            inds_path, pool_lines)
        pool_inds = pool_inds[img_path]
        pool_locs = pool_locs[img_path]
        mask,_ = nrrd.read(mask_path)

        print('Pool-size: %d'% (len(pool_lines)))
        print('Test-size: %d'% (len(test_lines)))

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
            self.pars['optimizer_name'],
            self.pars['patch_shape'])
        model.add_assign_ops(os.path.join(
            method_path, 'curr_weights.h5'))
        
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

            """ Querying iterations """
            print("Starting iterations of %s"%
                  method_name)
            # separete counts for this session
            newSuPix = 0
            newPix   = 0
            model.perform_assign_ops(sess)
            while newSuPix < max_queries:
                print("Iter. %d: "% iter_cnt,
                      end='\n\t')

                # decide the number of queries 
                # for this iteration (only to 
                # be used fro non-fi algorithms)
                if 'iter_k' in self.pars:
                    self.pars['k'] = self.pars[
                        'iter_k'][iter_cnt]
                
                """ Querying Super-pixels"""
                (qSuPix,
                 qSuPix_inds) = PW_NNAL.SuPix_query(
                     self,
                     run,
                     model,
                     pool_lines,
                     train_inds,
                     overseg_img,
                     method_name,
                     sess)
                
                newSuPix += qSuPix.shape[1]
                
                """ Updating the Indices """
                pool_set = set(pool_inds)
                pool_lines = list(pool_lines)
                new_train_inds = []
                for i, SPinds in enumerate(qSuPix_inds):
                    # updating training indices
                    #train_inds = np.append(train_inds,
                    #                       SPinds)
                    # zeroing out the superpixel
                    tmp = overseg_img[:,:,qSuPix[0,i]]
                    newPix += np.sum(tmp==qSuPix[1,i])
                    tmp[tmp==qSuPix[1,i]] = 0
                    # remove the indices from 
                    # curr_pool (because this is
                    # what is fed to the querying
                    # function)
                    SP_gridpts = set.intersection(
                        set(SPinds), pool_set)
                    for pts in SP_gridpts:
                        new_train_inds += [pts]
                        # find location of this point;
                        # this indiex is the same for 
                        # pixel and line indices as
                        # there is only one image in
                        # this experiment
                        indic = np.where(
                            pool_inds==pts)[0][0]
                        pool_inds = np.delete(
                            pool_inds, indic)
                        del pool_lines[indic]

                pool_lines = np.array(pool_lines)
                train_inds = np.append(train_inds,
                                       new_train_inds)

                newPix = len(train_inds)

                """ Updating the model """
                labels = read_label_winds(
                    mask, train_inds)
                inds_dict = {img_path: train_inds}
                labels_dict = {img_path: labels}
                for i in range(self.pars['epochs']):
                    PW_NN.PW_train_epoch(
                        model,
                        self.pars['dropout_rate'],
                        inds_dict,
                        labels_dict,
                        self.pars['patch_shape'],
                        self.pars['b'],
                        self.pars['stats'],
                        sess)
                    print('%d'% i, end=',')

                """ Evluating the updated model """
                ts_preds = batch_eval_wlines(self,
                                            run,
                                            model,
                                            test_lines,
                                            'prediction',
                                            sess)
                # Saving the predictions: 
                # first, loading whatever we had
                # before
                curr_predicts = np.loadtxt(
                    os.path.join(method_path, 
                                 'predicts.txt'))
                # appending the new ones to them,
                # and save them back
                if curr_predicts.ndim<2:
                    curr_predicts = np.expand_dims(
                        curr_predicts, axis=0)
                new_predicts = np.append(
                    curr_predicts, 
                    np.expand_dims(ts_preds, axis=0),
                    axis=0)
                # saving the appended prediction
                np.savetxt(os.path.join(
                    method_path, 'predicts.txt'), 
                           new_predicts, fmt='%d')

                # performance evaluation
                ts_labels = read_label_lines(
                    labels_path, test_lines)
                Fmeas = PW_analyze_results.get_Fmeasure(
                    ts_preds, ts_labels)
                                   
                with open(os.path.join(
                        method_path, 
                        'perf_evals.txt'), 'a') as f:
                    f.write('%f\n'% Fmeas)
                
                print('\n\t', end='')
                print("Total Super-pixels: %d+%d"% 
                      (nSuPix, newSuPix),
                      end='\n\t')
                print("Total Pixels: %d+%d"% 
                      (nPix, newPix),
                      end='\n\t')
                print("F-measure: %.4f"% Fmeas)

                """ Storing the Indices """
                # super-pixel queries
                np.savetxt(os.path.join(
                    method_path, 'queries',
                    '%d.txt'% iter_cnt), qSuPix,
                           fmt='%d')
                np.savetxt(os.path.join(
                    method_path, 'queries',
                    'inds_%d.txt'% iter_cnt), 
                           list(train_inds),
                           fmt='%d')
                # update the loop variables
                iter_cnt += 1
                # the current training and pool
                np.savetxt(os.path.join(
                    method_path, 'pool_lines.txt'), 
                           pool_lines,
                           fmt='%d')
                np.savetxt(train_path, 
                           train_inds,
                           fmt='%d')
                # save the current weights
                model.save_weights(
                    os.path.join(
                        method_path,
                        'curr_weights.h5'))

class Experiment_MultiImg(Experiment):
    """Active learning experiments that use multiple
    images as for training/pool and test data set
    (to be used for Universal active learning)
    """

    def __init__(self, root_dir, 
                 pars={},
                 train_paths={},
                 test_paths={}):

        Experiment.__init__(self, 
                            root_dir,
                            pars)

        if not(hasattr(self, 'pars')):
            self.load_parameters()

        # saving paths to train and test data
        tr_file = os.path.join(self.root_dir, 
                               'train_paths.txt')
        if not(os.path.exists(tr_file)):
            with open(tr_file, 'w') as f:
                yaml.dump(train_paths, f)
            self.train_paths = train_paths
        else:
            with open(tr_file, 'r') as f:
                self.train_paths = yaml.load(f)
                
        test_file = os.path.join(self.root_dir, 
                               'test_paths.txt')
        if not(os.path.exists(test_file)):
            with open(test_file, 'w') as f:
                yaml.dump(test_paths, f)
            self.test_paths = test_paths
        else:
            with open(test_file, 'r') as f:
                self.test_paths = yaml.load(f)

        # take care of the statistics
        if os.path.exists(os.path.join(
                self.root_dir, 'train_stats.txt')):
            self.read_stats()
        else:
            self.get_stats()

        # compute initial prediction if necessary
        init_eval_path = os.path.join(self.root_dir,
                                      'init_eval.txt')
        if False:#not(os.path.exists(init_eval_path)):
            m = len(self.train_paths[0])-1
            patch_shape = self.pars['patch_shape'][:2] + \
                          (m*self.pars['patch_shape'][2],)
            model = NN.create_model(
                self.pars['model_name'],
                self.pars['dropout_rate'], 
                self.nclass, 
                self.pars['learning_rate'], 
                self.pars['grad_layers'],
                self.pars['train_layers'],
                self.pars['optimizer_name'],
                patch_shape)

            with tf.Session() as sess:
                model.initialize_graph(sess)
                model.load_weights(
                    self.pars['init_weights_path'], sess)
                F1 = self.test_eval(model,sess)
                np.savetxt(init_eval_path, [F1])

    def get_stats(self):
        
        # training
        m = len(self.train_paths[0])-1
        n = len(self.train_paths)
        train_stats=np.zeros((n, 2*m))
        for i, dat_paths in enumerate(self.train_paths):
            for j in range(m):
                img,_ = nrrd.read(dat_paths[j])
                train_stats[i,j*m] = np.mean(img)
                train_stats[i,j*m+1] = np.std(img)
        np.savetxt(os.path.join(
            self.root_dir,'train_stats.txt'),train_stats)

        # test
        m = len(self.test_paths[0])-1
        n = len(self.test_paths)
        test_stats=np.zeros((n, 2*m))
        for i, dat_paths in enumerate(self.test_paths):
            for j in range(m):
                img,_ = nrrd.read(dat_paths[j])
                test_stats[i,j*m] = np.mean(img)
                test_stats[i,j*m+1] = np.std(img)
        np.savetxt(os.path.join(
            self.root_dir,'test_stats.txt'),test_stats)

        self.train_stats = train_stats
        self.test_stats  = test_stats

    def read_stats(self):        
        self.train_stats = np.loadtxt(os.path.join(
            self.root_dir, 'train_stats.txt'))
        self.test_stats = np.loadtxt(os.path.join(
            self.root_dir, 'test_stats.txt'))

    def test_eval(self, model, sess,
                  test_inds=[], test_labels=[]):

        m = len(self.test_paths[0])-1
        if len(test_inds)==0:
            test_inds, test_labels = gen_multimg_inds(
                self.test_paths, self.pars['grid_spacing'])
            
        tP,tTP,tFP = 0,0,0
        for i in range(len(test_inds)):
            stats = []
            for j in range(m):
                stats += [[self.test_stats[i,2*j],
                           self.test_stats[i,2*j+1]]]
            test_preds = PW_NN.batch_eval(
                model,
                sess,
                self.test_paths[i][:-1],
                test_inds[i],
                self.pars['patch_shape'],
                self.pars['ntb'],
                stats,
                'prediction')[0]

            (P,N,TP,
             FP,TN,FN) = PW_analyze_results.get_preds_stats(
                 test_preds, np.array(test_labels[i]))
            tP  += P
            tTP += TP
            tFP += FP
        # compute total Pr/Rc and F1
        Pr = tTP / (tTP + tFP)
        Rc = tTP / tP
        F1 = 2. / (1/Pr + 1/Rc)

        return F1

    def add_method(self, method_name):

        method_path = os.path.join(self.root_dir, 
                                   method_name)
        if not(os.path.exists(method_path)):
            os.mkdir(method_path)
            os.mkdir(os.path.join(method_path,
                                  'queries'))
        
    def run_method(self, method_name, max_queries):
        
        method_path = os.path.join(self.root_dir,
                                   method_name)
        
        """ Pool/Training Indices """
        # first load up all the indices for
        # pool images
        pool_inds,_ = gen_multimg_inds(
            self.train_paths, self.pars['grid_spacing'])
        test_inds, test_labels = gen_multimg_inds(
            self.test_paths, self.pars['grid_spacing'])

        # moving already selected queries into
        # the training set
        training_inds = [[] for i in 
                         range(len(pool_inds))]

        Q_path = os.path.join(method_path,
                              'queries')
        Q_files = os.listdir(Q_path)
        iters = len(Q_files)
        for f in Q_files:
            Qs = np.int32(np.loadtxt(os.path.join(
                Q_path, f)))
            for ind in np.unique(Qs[:,1]):
                I = Qs[Qs[:,1]==ind,0]
                training_inds[ind] += I.tolist()
                # get the locations to use for labels
                #I_locs = [pool_inds[ind].index(i) 
                #          for i in I]
                #sorted_inds = np.argsort(-np.array(I_locs))
                #I_locs.sort()
                [pool_inds[ind].remove(i) for i in I]

        evals_path = os.path.join(method_path, 
                                  'perf_evals.txt')
        # check if initial prediction is already 
        # available, if not copy
        if False:#not(os.path.exists(eval_path)):
            shutil.copy(
                os.path.join(self.root_dir,'init_eval.txt'),
                os.path.join(method_path,'perf_evals.txt')
            )

            curr_fmeas = np.loadtxt(evals_path)
            if curr_fmeas.size==1:
                curr_fmeas = [curr_fmeas]
                print("Current F-measures: ", end='')
                print(*curr_fmeas, sep=', ')
        

        """ Load and Pad Training Images """
        rads = np.zeros(3, dtype=int)
        for i in range(3):
            rads[i] = int(
                (self.pars['patch_shape'][i]-1)/2.)

        all_padded_imgs = []
        m = len(self.train_paths[0])-1
        for sub_paths in self.train_paths:
            padded_imgs = []
            for i,path in enumerate(sub_paths):
                img,_ = nrrd.read(path)
                # if mask, don't pad it
                if i==m:
                    padded_imgs += [img]
                    continue
                padded_img = np.pad(
                    img,
                    ((rads[0],rads[0]),
                     (rads[1],rads[1]),
                     (rads[2],rads[2])),
                    'constant')
                padded_imgs += [padded_img]

            all_padded_imgs += [padded_imgs]

        """ Loading the Model """
        tf.reset_default_graph()
        # create a model-holder
        m = len(self.train_paths[0])-1
        patch_shape = self.pars['patch_shape'][:2] + \
                      (m*self.pars['patch_shape'][2],)
        model = NN.create_model(
            self.pars['model_name'],
            self.pars['dropout_rate'], 
            self.nclass, 
            self.pars['learning_rate'], 
            self.pars['grad_layers'],
            self.pars['train_layers'],
            self.pars['optimizer_name'],
            patch_shape)

        with tf.Session() as sess:
            model.initialize_graph(sess)
            model.load_weights(
                self.pars['init_weights_path'], sess)
            sess.graph.finalize()

            """ Start Generating the Queries """
            nqueries = 0
            while nqueries < max_queries:
                print("Iter. %d: "% iters,end='\n\t')

                Q_inds = PW_NNAL.query_multimg(
                    self, model, sess, 
                    all_padded_imgs, 
                    pool_inds,training_inds,
                    method_name)

                # moving Qs from pool --> training
                nQ = np.sum([len(qind) for qind in Q_inds])
                nqueries += nQ
                # Q_mat:  1st column=voxel indices
                #         2nd column=training image index
                Q_mat = np.zeros((nQ,2))
                q_file = os.path.join(self.root_dir,
                                      method_name,
                                      'queries/%d'% iters)
                cnt = 0
                for ind in range(len(Q_inds)):
                    if len(Q_inds[ind]>0):
                        Q_mat[cnt:cnt+len(Q_inds[ind]),
                              0] = np.array(pool_inds[ind])[Q_inds[ind]]
                        Q_mat[cnt:cnt+len(Q_inds[ind]),
                              1] = ind
                        cnt += len(Q_inds[ind])
                        # ading to the training
                        training_inds[ind] += list(np.array(
                            pool_inds[ind])[Q_inds[ind]])
                        # remove from the pool
                        sorted_inds = -np.sort(-Q_inds[ind])
                        [pool_inds[ind].pop(i) for i in sorted_inds]
                        
                np.savetxt(q_file, Q_mat, fmt='%d')
                iters += 1

                """ Finetuning the Model """
                finetune_multimg(self,
                                 model, sess,
                                 all_padded_imgs,
                                 training_inds)

                """ Evaluating on Test Images """
                #F1 = self.test_eval(model,sess,
                #                    test_inds,test_labels)
                #with open(perf_eval_path, 'a') as f:
                #    f.write('%f\n'% F1)

                # save the current weights
                model.save_weights(
                    os.path.join(self.root_dir, method_name,
                                 'curr_weights_%d.h5'% iters))


def gen_multimg_inds(dat_paths, grid_spacing):
    """Gegenerating inidices from a set of
    images
    
    The input contains a list of image paths,
    where each path is another list of at
    least two elements: raw image path(s), 
    and the corresponding mask path as the
    last element.
    """

    # number of test subjects
    n = len(dat_paths)
    
    all_inds = []
    all_labels = []
    for i in range(n):
        mask,_ = nrrd.read(dat_paths[i][-1])
        # forming the 2D grid
        s = mask.shape
        Y, X = np.meshgrid(np.arange(s[1]),
                           np.arange(s[0]))
        X = np.ravel(X)
        Y = np.ravel(Y)
        grid_locs = np.logical_and(
            X%grid_spacing==0,
            Y%grid_spacing==0)
        grid_X = np.array(X[grid_locs])
        grid_Y = np.array(Y[grid_locs])
        # forming the 3D grid
        inds = []
        labels = []
        for i in range(s[2]):
            grid_Z = np.ones(
                len(grid_X),dtype=int)*i
            grid_3D = np.ravel_multi_index(
                (grid_X, grid_Y, grid_Z), s)
            inds += list(grid_3D)

            slice_labels = mask[
                grid_X, grid_Y, grid_Z]
            labels += list(slice_labels)

        all_inds += [inds]
        all_labels += [labels]

    return all_inds, all_labels
        

def prep_AL_data(expr):
    """Preparing the target data set, including
    unlabeled pool and test samples for running
    an active learning experiment, based on a 
    single subject
    
    The pool and test will be tried to be selected
    from similar slices. Hence, one strategy is to
    draw samples from the even slices for the pool,
    and from the odd slices for the test data set.
    """

    # assuming that all image modalities have the
    # same shape, we use the first modality to
    # prepare grid indices
    img_addr = expr.pars['img_paths'][0]
    mask_addr = expr.pars['mask_path']
    
    """Sampling from the slices"""
    # grid sampling
    inds, labels, types = patch_utils.generate_grid_samples(
        img_addr, mask_addr,
        expr.pars['grid_spacing'], 
        expr.pars['grid_offset'])

    img,_ = nrrd.read(img_addr)
    multinds = np.unravel_index(inds, img.shape)
    # take the even slices
    even_slices = np.where(multinds[2]%2==0)[0]
    pool_inds = np.array(inds)[even_slices]
    pool_labels = np.array(labels)[even_slices]

    odd_slices = np.where(multinds[2]%2==1)[0]
    test_inds = np.array(inds)[odd_slices]
    test_labels = np.array(labels)[odd_slices]
    
    """write the text files"""
    np.savetxt(os.path.join(
        expr.root_dir, 'init_pool_inds.txt'),
               pool_inds, fmt='%d')
    np.savetxt(os.path.join(
        expr.root_dir, 'init_pool_labels.txt'),
               pool_labels, fmt='%d')
    np.savetxt(os.path.join(
        expr.root_dir, 'test_inds.txt'),
               test_inds, fmt='%d')
    np.savetxt(os.path.join(
        expr.root_dir, 'test_labels.txt'),
               test_labels, fmt='%d')

    
def finetune(model,
             sess,
             expr,
             padded_imgs,
             mask,
             train_inds):
    """Fine-tuning a given model for
    a number of epochs; written mainly
    to be used within querying iterations

    This function basically does same thin
    as `PW_NN.PW_train_epoch_MultiModal`,
    but is little handier and more brief:
    it only uses indices of a single imagel,
    there is no option for saving variables
    in tensorboard, and also the images are
    given as input arguments, hence no need
    to load them separately here
    """

    n = len(train_inds)
    m = len(padded_imgs)
    b = expr.pars['b']
    patch_shape = expr.pars['patch_shape']
    stats = expr.pars['stats']

    for t in range(expr.pars['epochs']):
        # batch-ify the data
        batch_inds = NN.gen_batch_inds(n, b)

        for i in range(len(batch_inds)):
            img_inds = train_inds[batch_inds[i]]
            patches, labels = patch_utils.\
                              get_patches(
                                  padded_imgs,
                                  img_inds,
                                  patch_shape,
                                  True,
                                  mask)
            # hot-one vector for labels
            hot_labels = np.zeros(
                (2, len(labels)))
            hot_labels[0,labels==0]=1
            hot_labels[1,labels==1]=1

            # normalizing the patches
            for j in range(m):
                patches[:,:,:,j] = (
                    patches[:,:,:,j]-stats[
                        j][0])/stats[j][1] 

            # perform this iteration
            # batch gradient step
            sess.run(
                model.train_step,
                feed_dict={
                    model.x: patches,
                    model.y_: hot_labels,
                    model.keep_prob:model.dropout_rate})


def finetune_multimg(expr,
                     model,
                     sess,
                     all_padded_imgs,
                     training_inds):

    s = len(training_inds)
    img_ind_sizes = [len(training_inds[i]) for i 
                     in range(s)] 
    n = np.sum(img_ind_sizes)
    m = len(all_padded_imgs[0]) - 1
    b = expr.pars['b']
    d3 = expr.pars['patch_shape'][2]

    for t in range(expr.pars['epochs']):
        batch_inds = NN.gen_batch_inds(n, b)

        for i in range(len(batch_inds)):
            """ Preparing the Patches/Labels """

            # batch indices are global indices,
            # extract local indices for each image
            local_inds = patch_utils.global2local_inds(
                batch_inds[i], img_ind_sizes)
            # local indices --> image (voxel) indices
            img_inds = [np.array(training_inds[j])[
                local_inds[j]] for j in range(s)]

            b_patches, b_labels = patch_utils.get_patches_multimg(
                all_padded_imgs, img_inds,
                expr.pars['patch_shape'], 
                expr.train_stats)
            
            # stitching patches and labels
            b_patches = [b_patches[j] for j in range(len(img_inds))
                         if len(img_inds[j])>0]
            b_patches = np.concatenate(b_patches,
                                       axis=0)
            b_labels = [b_labels[j] for j in range(len(img_inds))
                        if len(img_inds[j])>0]
            b_labels = np.concatenate(b_labels)


            # converting to hot-one vectors
            hot_labels = np.zeros((2,len(b_labels)))
            hot_labels[0,b_labels==0] = 1
            hot_labels[1,b_labels==1] = 1

            """ Doing an Optimization Iteration """
            # finally we are ready to take 
            # optimization step
            sess.run(
                model.train_step,
                feed_dict={
                    model.x: b_patches,
                    model.y_: hot_labels,
                    model.keep_prob:model.dropout_rate})



def read_ints(file_path):
    """Reading several lines of a text
    file containing integer values (either
    sample indices or 
    """
    
    ints_array = np.array(linecache.getlines(
        file_path))

    # remove \n from the lines and 
    # conveting strings to integers
    ints_array = [int(L[:-1]) for 
                  L in ints_array]
        
    return ints_array


def get_SuPix_inds(overseg_img,
                   SuPix_codes):
    """Getting pixel indices of a give
    set of super-pixels in an 
    oversegmentation image
    
    Output pixel indices are generated
    in a list with the same order as
    the given codes of superpixels
    """

    s = overseg_img.shape

    SuPix_inds = [[]] * SuPix_codes.shape[1]
    SuPix_slices = np.unique(SuPix_codes[0,:])
    for z in SuPix_slices:
        slice_ = overseg_img[:,:,z]
        slice_props = regionprops(slice_)
        slice_labels = np.unique(slice_)
        if 0 in slice_labels:
            slice_labels = slice_labels[1:]

        # take all superpixels of a slice
        # and store their order in the input
        # super-pixel codes
        SP_locinds = np.array(np.where(
            SuPix_codes[0,:]==z)[0])
        for ind in SP_locinds:
            label = SuPix_codes[1, ind]
            # extracting 2D multi-indices
            # of the current super-pixel
            # -------------------------
            # ASSUMPTION:
            # the order of labels in the 
            # region-properties list 
            # (output of regionprops)
            # is the same as the order 
            # of `np.unique` of the 
            # oversegmentation slice
            prop_ind = np.where(
                slice_labels==label)[0][0]
            if not(slice_props[prop_ind][
                    'label']==label):
                raise ValueError(
                    "The super-pixel's label"+
                    " is different than the "+
                    "extracted property's class")

            # 2D multi-index
            multinds_2D = slice_props[
                prop_ind]['coords']
            # 2D multi-index --> 2D index
            inds_2D = np.ravel_multi_index(
                (multinds_2D[:,0],
                 multinds_2D[:,1]), s[:2])
            # 2D index --> 3D index
            inds_3D = patch_utils.\
                      expand_raveled_inds(
                          inds_2D, z, 2, s)
            
            SuPix_inds[ind] = list(
                np.int64(inds_3D))

    return SuPix_inds


def get_expr_paths(expr):
    """Getting path where the 
    experiment's image and mask is 
    saved
    """

    if expr.pars['data']=='newborn':
        img_addrs, mask_addrs = patch_utils.\
                              extract_newborn_data_path()
    elif expr.pars['data']=='adults':
        img_addrs, mask_addrs = patch_utils.\
                              extract_Hakims_data_path()

    img_path = img_addrs[expr.pars[
        'indiv_img_ind']]
    mask_path = mask_addrs[expr.pars[
        'indiv_img_ind']]

    return img_path, mask_path

def get_expr_data_info(expr, base_dir=None):
    """Getting path where the 
    experiment's image and mask is 
    saved using `inds.txt`
    
    If the base directory is given, return 
    the paths as sub-directory of the 
    base directory.
    """

    # path to image
    img_path = linecache.getline(os.path.join(
        expr.root_dir,'0/inds.txt'),1).split(',')[0]
    split_path = img_path.split('/')

    # the item that contains subject ID
    sub_indic = np.where([
        'sub' in S for S in split_path])[0][0]
    sub_id = split_path[sub_indic][8:]
    
    # change the root if a sub-directory
    # is given
    if base_dir:
        # the base directory is considered to
        # the path where all the patiend 
        # folders lie
        split_path = [base_dir] + \
                     split_path[sub_indic:]
        # reconstruct the path to get the
        # new path to image 
        img_path = '/'.join(split_path)

    # preparing path to mask
    split_path[-2] = '03-ICC'
    split_path[-1] = split_path[-1][:-12] + \
                     'ICC.nrrd'
    mask_path = '/'.join(split_path)

    return sub_id, img_path, mask_path


def sequential_AL(base_expr,
                  target_img_inds,
                  seq_base_dir):
    """Doing sequential active learning starting
    from the model in the last iteration of the
    FI-based querying iterations, and repeat the
    querying iterations on another images; 
    this sequence will be repeated for multiple
    images and the results will be saved in the
    corresponding experiments' folders

    For now, it is mostly compatible with adults
    data set.
    """

    #base_ind = base_expr.pars['indiv_img_ind']

    # For each target image, create a new
    # experiment to do the sequential AL
    pars = copy.deepcopy(base_expr.pars)
    prev_expr_dir = base_expr.root_dir
    for i, ind in enumerate(target_img_inds):
        print('Running AL of experiment '+
              '%s on image %s'% 
              (prev_expr_dir, ind))
        
        # creating a new experiment
        pars['indiv_img_ind'] = ind
        pars['init_weights_path'] = os.path.join(
            prev_expr_dir, '0/fi/curr_weights.h5')
        
        root_dir = os.path.join(
            seq_base_dir, 
            'N1017_%d'% (ind))
        E = Experiment(root_dir, pars)
        E.add_run()

        # starting the FI-based iteration
        E.add_method('fi',0)
        E.run_method('fi', 0, 1500)
        
        # when it is done, modify the path
        # to the previous experiment
        prev_expr_dir = E.root_dir
