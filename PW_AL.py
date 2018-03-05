from skimage.measure import regionprops
from skimage.segmentation import slic
from matplotlib import pyplot as plt
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
                

    def add_run(self):
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
        inds_path = os.path.join(
            run_path,'inds.txt')
        labels_path = os.path.join(
            run_path,'labels.txt')
        # OUT-DATED because of adding types_dict to
        # the Experiment.generate_sample()
        #pool_inds, test_inds = newborn_prep_dat(
        #    img_addrs, mask_addrs,
        #    self.pars['pool_img_inds'],
        #    self.pars['test_img_inds'],
        #    self.pars['pool_ratio'],
        #    self.pars['test_ratio'],
        #    inds_path, labels_path,
        #    self.pars['mask_ratio'])
        pool_lines, test_lines, sub_id = prep_target_indiv(
            self,
            inds_path, 
            labels_path)
        self.subject_id = sub_id
        
        # saving indices into the run's folder
        np.savetxt('%s/test_lines.txt'% run_path, 
                   test_lines, fmt='%d')
        np.savetxt('%s/init_pool_lines.txt'% run_path, 
                   pool_lines, fmt='%d')

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
            self.pars['optimizer_name'],
            self.pars['patch_shape'])

        #  computing pool statistics
        mu, sigma = get_statistics(
            self, n)
        #self.pars['stats'] = [mu, sigma] #[65., 54.5]
        #self.save_parameters(self.pars)

        # start a session to do the training
        with tf.Session() as sess:
            # training from initial training data
            model.initialize_graph(sess)
            model.load_weights(
                self.pars['init_weights_path'], sess)
                        
            # get a prediction of the test samples
            ts_preds = batch_eval_wlines(
                self,
                n,
                model,
                test_lines,
                'prediction',
                sess)
                
            # save the predictions 
            np.savetxt(os.path.join(run_path, 
                                    'init_predicts.txt'), 
                       np.expand_dims(ts_preds,axis=0),
                       fmt='%d')
            
            # initial, performance evaluation
            ts_labels = read_label_lines(
                labels_path, test_lines)
            Fmeas = PW_analyze_results.get_Fmeasure(
                ts_preds, ts_labels)
            print("Initial F-measure: %f"% Fmeas)
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
            os.path.join(run_path,'init_pool_lines.txt'),
            os.path.join(method_path,'pool_lines.txt')
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
        test_lines = np.int32(
            np.loadtxt(os.path.join(
                run_path, 'test_lines.txt')
                   ))
        train_path = os.path.join(
            method_path, 'train_lines.txt')
        if os.path.exists(train_path):
            train_lines = np.int32(
                np.loadtxt(train_path))
        else:
            train_lines = []
        pool_lines = np.int32(
            np.loadtxt(os.path.join(
                method_path, 'pool_lines.txt')
                   ))
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

            # starting the iterations
            print("Starting iterations of %s"%
                  method_name)
            nqueries = 0
            model.perform_assign_ops(sess)
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
                    run,
                    model,
                    pool_lines,
                    train_lines,
                    method_name,
                    sess)

                if self.pars['k']==1:
                    Q = [pool_lines[Q_inds]]
                else:
                    Q = pool_lines[Q_inds]

                # save the queries
                np.savetxt(os.path.join(
                        method_path, 
                        'queries',
                        '%d.txt'% (
                            iter_cnt)
                        ), Q, fmt='%d')
                
                # update the indices
                if len(train_lines)==0:
                    train_lines = Q
                else:
                    train_lines = np.append(
                        train_lines, Q)
                pool_lines = np.delete(
                    pool_lines, Q_inds)
                                
                """ updating the model """
                for i in range(self.pars['epochs']):
                    PW_train_epoch_wlines(
                        model,
                        self,
                        run,
                        train_lines,
                        sess)
                    print('%d'% i, end=',')
                    
                """ evluating the updated model """
                ts_preds = batch_eval_wlines(self,
                                            run,
                                            model,
                                            test_lines,
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
                    labels_path, test_lines)
                Fmeas = PW_analyze_results.get_Fmeasure(
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
                          (len(train_lines)),
                          end='\n\t')
                    print("F-measure: %.4f"% Fmeas)
                
                # when querying is done..
                # save the current training and pool
                np.savetxt(os.path.join(
                    method_path, 'curr_pool.txt'), 
                           pool_lines,
                           fmt='%d')
                np.savetxt(train_path, 
                           train_lines,
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
            

def target_prep_dat(img_addrs, mask_addrs,
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

def prep_target_indiv(expr,
                      dat_opath, 
                      label_opath):
    """Preparing the target data set, including
    unlabeled pool and test samples for running
    an active learning experiment, based on a 
    single subject
    
    The pool and test will be tried to be selected
    from similar slices. Hence, one strategy is to
    draw samples from the even slices for the pool,
    and from the odd slices for the test data set.
    """
    
    if expr.pars['data']=='adults':
        img_addrs, mask_addrs = patch_utils.extract_Hakims_data_path()
    elif expr.pars['data']=='newborn':
        img_addrs, mask_addrs = patch_utils.extract_newborn_data_path()

    img_addr = img_addrs[expr.pars['indiv_img_ind']]
    mask_addr = mask_addrs[expr.pars['indiv_img_ind']]
    subject_id = img_addr.split('/')[6]
    
    D = patch_utils.PatchBinaryData(
        [img_addr], [mask_addr])

    """Sampling from the slices"""
    # grid sampling
    inds_dict, mask_dict, types_dict = patch_utils.generate_grid_samples(
        img_addr, mask_addr, 
        expr.pars['grid_spacing'], 
        expr.pars['grid_offset'])

    # divide slices
    img,_ = nrrd.read(img_addr)
    multinds = np.unravel_index(inds_dict[img_addr],
                                img.shape)
    # take the even slices
    even_slices = np.where(multinds[2]%2==0)[0]
    odd_slices = np.where(multinds[2]%2==1)[0]
    
    """write the text files"""
    with open(dat_opath,'w') as f, open(
            label_opath,'w') as g:
        for i in range(len(inds_dict[img_addr])):
            # data file
            f.write(
                '%s, %d, %s\n'
                % (img_addr,
                   inds_dict[img_addr][i],
                   types_dict[img_addr][i]))
            # label file
            g.write(
                '%d\n'%(mask_dict[img_addr][i]))

    return even_slices+1, odd_slices+1, subject_id

def load_patches(expr, 
                 run, 
                 line_inds,
                 label_flag=False):
    """Loading a set of patches and their
    labels that are specified by the line
    numbers where they are saved in the
    corresponding PW-AL experiment's run
    """
        
    inds_path = os.path.join(
        expr.root_dir,str(run),'inds.txt')

    img_paths = []
    inds_array = np.zeros(len(line_inds), dtype=int)

    for i in range(len(line_inds)):
        #pdb.set_trace()
        line = linecache.getline(
            inds_path, line_inds[i]).splitlines()[0]
        img_paths += [line.split(',')[0]]
        inds_array[i] = int(line.split(',')[1])
        
    # start loading the patches in the same order
    # as is determined in `inds`
    patches = np.zeros((len(line_inds),) + 
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
        labels = np.zeros(len(line_inds), 
                          dtype=bool)
        for i in range(len(line_inds)):
            labels[i] = linecache.getline(
                inds_path, 
                line_inds[i]).splitlines()[0]
            
        return patches, labels
        
    # if labels_flag is off, only returns
    # the patches
    return patches

def get_statistics(expr, run, hist_flag=False):
    """Getting statistics of intensity values
    of the pool samples in an experiment's 
    run
    """

    pool_lines = np.int32(np.loadtxt(
        os.path.join(expr.root_dir,
                     str(run),
                     'init_pool_lines.txt')))
    pool_patches = load_patches(
        expr, run, pool_lines)

    # mean and std
    mu = np.mean(pool_patches)
    sigma = np.std(pool_patches)
    
    if hist_flag:
        npool = np.prod(pool_patches.shape)
        # preparing histogram bins
        M = pool_patches.max()
        nbin = 100
        bin_seq = np.linspace(0, M, nbin)

        # histogram of intensities
        hist = np.histogram(
            pool_patches, bin_seq)[0]/npool

        return mu, sigma, bin_seq, hist

    return mu, sigma
    
def create_dict(inds_path,
                line_inds,
                labels_path=None):
    """Creating a dictionary of
    sample indices, from specific lines
    of a given index-file (that is built
    in a PW-experiment's run,
    """
    img_paths = []
    inds_array = np.zeros(len(line_inds), 
                          dtype=int)
    labels_array = np.zeros(len(line_inds), 
                            dtype=int)
    
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

def batch_eval_wlines(expr,
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
    if varname=='feature_layer':
        fdim = model.feature_layer.shape[0].value
        eval_array = np.zeros((fdim, len(line_inds)))
        for path in list(eval_dict.keys()):
            locs = locs_dict[path]
            eval_array[:,locs] = eval_dict[path]
    else:
        eval_array = np.zeros(len(line_inds))
        for path in list(eval_dict.keys()):
            locs = locs_dict[path]
            eval_array[locs] = eval_dict[path]

    return eval_array

def FC_gradnorms_wlines(expr,
                        run,
                        model,
                        line_inds,
                        sess):
    """Computing norm of gradients of FC layers
    of a model for a specific line inds (w.r.t.
    index file of the experiemt's run)
    
    The reason that this part is not included
    inside `batch_eval_wlines()` was that it 
    has a different procedure of being calculated
    """

    # preparing batch indices
    n = len(line_inds)
    b = expr.pars['ntb']
    batch_ends = np.arange(0,n,b)
    if not(batch_ends[-1]==n):
        batch_ends = np.append(
            batch_ends, n)

    # going through batches
    gradnorms = np.zeros(n)
    for i in range(1,len(batch_ends)):
        # getting the chunk of indices
        batch_inds = np.arange(
            batch_ends[i-1],batch_ends[i])
        
        # load patches corresponding to
        # the current batch of lines
        X = load_patches(
            expr, run,
            line_inds[batch_inds])
        X = (X - expr.pars['stats'][0]) / \
            expr.pars['stats'][1]
        
        S = NNAL_tools.FC_gradnorms_batch(
            model, X, sess)
        gradnorms[batch_inds] = np.sum(
            S, axis=0)
        
    return gradnorms

    
def PW_train_epoch_wlines(model,
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
    
    # convert the paths to the ones used
    # in Wombat
    if True:
        base_dir = '/common/data/raw/Hakim/For_Christine'+\
                   '/Mrakotsky_IBD_Brain/Processed/'
        
        old_path = list(inds_dict.keys())[0]
        new_path = base_dir + old_path[31:]
        c_inds_dict = {
            new_path: inds_dict[old_path]}
        c_labels_dict = {
            new_path: labels_dict[old_path]}
        inds_dict = c_inds_dict
        labels_dict = c_labels_dict

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

def PW_train_epoch_winds_wconf(model,
                               expr,
                               run,
                               trline_inds,
                               conf_inds,
                               conf_labels,
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
        inds_path, trline_inds, labels_path)
    
    # adding the confident samples
    # with given labels (not the true ones)
    conf_inds_dict, conf_locs_dict = create_dict(
        inds_path, conf_inds)

    for path in list(conf_inds_dict.keys()):
        if not(path in inds_dict):
            inds_dict[path] = []
            labels_dict[path] = []
        else:
            inds_dict[path] = list(
                inds_dict[path])
            labels_dict[path] = list(
                labels_dict[path])

        inds_dict[path] += list(
            conf_inds_dict[path])
        labels_dict[path] += list(
            conf_labels[conf_locs_dict[path]])

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

def read_label_winds(mask, inds_3D):
    """Reading labels of several pixels
    given with their D indices
    """
    
    # 3D inds --> 3D multi-inds
    multinds = np.unravel_index(
        inds_3D, mask.shape)
    
    return mask[multinds]

def finetune_winds(expr, run,
                  tr_lines,
                  ts_lines,
                  tb_files=[]):
    """Finetuning a given model, with a given set
    of indices, and then evaluate the resulting
    model on a set of test samples
    """
    
    # preparing the model
    model = NN.create_model(
        expr.pars['model_name'],
        expr.pars['dropout_rate'], 
        expr.nclass, 
        expr.pars['learning_rate'], 
        expr.pars['grad_layers'],
        expr.pars['train_layers'],
        expr.pars['optimizer_name'],
        expr.pars['patch_shape'])
    
    # labels of test and training samples
    labels_path = os.path.join(
        expr.root_dir, str(run), 
        'labels.txt')
    ts_labels = read_label_lines(
        labels_path, ts_lines)
    tr_labels = read_label_lines(
        labels_path, tr_lines)

    with tf.Session() as sess:
        # loading the stored weights
        model.initialize_graph(sess)
        model.load_weights(
            expr.pars['init_weights_path'],
            sess)
        sess.graph.finalize()

        if len(tb_files)>0:
            tb_writers = [
                tf.summary.FileWriter(tb_files[0]),
                tf.summary.FileWriter(tb_files[1])]

        # finetuning epochs
        for i in range(expr.pars['epochs']+1):
            """ TensorBaord variables """
            if len(tb_files)>0:
                # training/loss
                tr_losses = batch_eval_wlines(
                    expr,
                    run,
                    model,
                    tr_lines,
                    'loss',
                    sess)
                loss_summ = tf.Summary()
                loss_summ.value.add(
                    tag='Loss',
                    simple_value=np.mean(tr_losses))
                tb_writers[0].add_summary(
                    loss_summ, i)
                # training/F-measure
                tr_preds = batch_eval_wlines(
                    expr,
                    run,
                    model,
                    tr_lines,
                    'prediction',
                    sess)
                Fmeas = PW_analyze_results.get_Fmeasure(
                    tr_preds, tr_labels)
                Fmeas_summ = tf.Summary()
                Fmeas_summ.value.add(
                    tag='F-measure',
                    simple_value=Fmeas)
                tb_writers[0].add_summary(
                    Fmeas_summ, i)
                # test/loss
                ts_losses = batch_eval_wlines(
                    expr,
                    run,
                    model,
                    ts_lines,
                    'loss',
                    sess)
                loss_summ = tf.Summary()
                loss_summ.value.add(
                    tag='Loss',
                    simple_value=np.mean(ts_losses))
                tb_writers[1].add_summary(
                    loss_summ, i)
                # test/F-measure
                ts_preds = batch_eval_wlines(
                    expr,
                    run,
                    model,
                    ts_lines,
                    'prediction',
                    sess)
                Fmeas = PW_analyze_results.get_Fmeasure(
                    ts_preds, ts_labels)
                Fmeas_summ = tf.Summary()
                Fmeas_summ.value.add(
                    tag='F-measure',
                    simple_value=Fmeas)
                tb_writers[1].add_summary(
                    Fmeas_summ, i)
                
            if i==expr.pars['epochs']:
                break

            PW_train_epoch_wlines(
                model,
                expr,
                run,
                tr_lines,
                sess)
            print('%d'% i, end=',')

        # final evaluation over test 
        ts_preds = batch_eval_wlines(
            expr,
            run,
            model,
            ts_lines,
            'prediction',
            sess)
        Fmeas = PW_analyze_results.get_Fmeasure(
            ts_preds, ts_labels)
        
    return Fmeas, model

def wholepool_train(expr_names,
                    img_addrs,
                    mask_addrs,
                    weights_path):
    """Finetuning a given model, with a given set
    of indices, and then evaluate the resulting
    model on a set of test samples
     """

    # preparing the model
    root_dir = expr_names[0]
    E = Experiment(root_dir)
    E.load_parameters()
    model = NN.create_model(
        E.pars['model_name'],
        E.pars['dropout_rate'],
        E.nclass,
        E.pars['learning_rate'],
        E.pars['grad_layers'],
        E.pars['train_layers'],
        E.pars['optimizer_name'],
        E.pars['patch_shape'])

    with tf.Session() as sess:
        model.initialize_graph(sess)
        model.add_assign_ops(weights_path)

        sess.graph.finalize

        for root_dir in expr_names:
            # initializing 
            model.perform_assign_ops(sess)

            # preparing whole-pool training
            print(root_dir)
            E = Experiment(root_dir)
            E.load_parameters()
            labels_path = os.path.join(
                E.root_dir,
                '0/labels.txt')

            pool_lines = np.int32(np.loadtxt(os.path.join(
                E.root_dir, '0/init_pool_lines.txt')))

            for i in range(60):
                PW_train_epoch_wlines(
                    model,
                    E,
                    0,
                    pool_lines,
                    sess)
                #print('%d'% i, end=',')

            save_dir = os.path.join(
                E.root_dir, '0/wholepool')
            if not(os.path.exists(save_dir)):
                os.mkdir(save_dir)

            model.save_weights(os.path.join(
                save_dir,
                'final_weights.h5'))

            print('Predicting...', end=' ')
            img_path = img_addrs[
                E.pars['indiv_img_ind']]
            mask_path = mask_addrs[
                E.pars['indiv_img_ind']]
            slice_inds = np.arange(
                1, img.shape[2], 2)
            _,F1 = PW_analyze_results.full_model_eval(
                E, model, sess, img_path, mask_path,
                slice_inds, save_dir)
            print('F1: %.5f'% F1)


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


def get_multinds(expr, run, 
                  line_inds):
    """Returning slice indices of 
    a given set of indices (in terms
    of line numbers of `inds.txt`)
    """

    inds_path = os.path.join(
        expr.root_dir, str(run), 
        'inds.txt')
    inds_dict,_ = create_dict(
        inds_path, line_inds)
    
    # turning to multiple
    img_path = list(
        inds_dict.keys())[0]
    inds = inds_dict[img_path]
    img,_ = nrrd.read(img_path)
    multinds = np.unravel_index(
        inds, img.shape)

    return multinds

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
        
