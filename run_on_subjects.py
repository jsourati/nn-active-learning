import PW_analyze_results
import numpy as np
import patch_utils
import PW_AL
import sys
import os


def AL(subjects, exp_names, pars_template):
    img_addrs,_ = patch_utils.extract_newborn_data_path()
    img_ids = [img_addrs[i].split('/')[6][8:] for i in 
               range(len(img_addrs))]

    for i, subject in enumerate(subjects):
        sub_ind = np.where(np.array(subjects)==subject)[0][0]
        pars_template['indiv_img_ind'] = sub_ind
        root_dir = exp_names[i]
        
        E = PW_AL.Experiment(root_dir, pars_template)
        E.nclass=2
        E.add_run()
        E.save_parameters(E.pars)
        
        if not(os.path.exists(os.path.join(
                E.root_dir, '0', 'fi'))):
            E.add_method('fi',0)
            E.run_method('fi',0,1500)

        # get the number of queries for 
        # each iteration of FI algorithm
        Qs = PW_analyze_results.get_queries(
            E, 0, 'fi')
        Qsizes = [len(Q) for Q in Qs]
        E.pars['iter_k'] = Qsizes
        E.save_parameters(E.pars)

        E.add_method('entropy',0)
        E.run_method('entropy',0,1500)
        
        E.add_method('random',0)
        E.run_method('random',0,1500)
        
