import numpy as np
import shutil
import yaml
import sys
import pdb
import os

import AL


def do_expr(root_dir, 
            run,
            method_name,
            nqueries):
    """Running experiment in a given
    run
    """
    
    E = AL.Experiment(root_dir)

    # add a method to this run, if 
    # it's not there
    E.add_method(method_name, run)
    
    # and now run the querying
    E.run_method(method_name, 
                 run, nqueries)

def create_expr(base_dir, 
                data_dir,
                target_classes):
    """Creating an experiment with the
    given data directory and target classes
    """
    
    img_path_list, labels = prepare_data(
        data_dir, target_classes)
    
    A = AL.Experiment(base_dir,
                      img_path_list,
                      labels)

def create_run(root_dir):
    """Add a new run to an experiment
    with the given root directory, and
    return the ID of the new run
    """
    
    E = AL.Experiment(root_dir)
    run = len(E.get_runs())
    E.add_run()
    
    return run

def prepare_data(data_dir,
                 target_classes):
    """Preparing data for creating an 
    experiment
    
    If classes are to be filtered, the
    target classes should be specified by 
    saving indices of target classes into
    a text file.
    """
    
    all_files = os.listdir(data_dir)
    all_files.remove('labels.txt')
    img_path_list = [
        os.path.join(data_dir, path)
        for path in all_files]
    labels = np.loadtxt(os.path.join(
        data_dir,'labels.txt'))

    if not(target_classes=='NA'):
        
        targets = np.int32(
            np.loadtxt(target_classes))
        
        inds = np.zeros(len(labels), dtype=bool)
        for i in targets:
            inds = np.logical_or(inds, labels==i)
        inds = np.where(inds)[0]

        labels = labels[inds]
        img_path_list = [img_path_list[idx] 
                         for idx in inds]
        
    return img_path_list, labels
    

def set_parameters(par_temp, root_dir, optpars):
    """Creating parameters of the experiment
    by giving a template for the parameters
    and additional changes (if needed)
    """
    
    if len(optpars)==2:
        shutil.copy(
            par_temp,
            os.path.join(root_dir,'parameters.txt'))
    elif len(optpars)>2:
        # if additional parameters are given
        # load them and save the modified set
        # of parameters
        optpars = optpars[1:-1].split(',')
        with open(par_temp, 'r') as f:
            pars = yaml.load(f)

        for item in optpars:
            subitems = item.split('=')
            key = subitems[0]
            val = subitems[1]
            
            if type(pars[key])==int:
                pars[key] = int(val)
            elif type(pars[key])==float:
                pars[key] = float(val)
            elif type(pars[key])==str:
                pars[key] = val
        with open(os.path.join(
                root_dir, 'parameters.txt'),'w') as f:
            yaml.dump(pars, f)
        

if __name__=="__main__":
    do_expr(sys.argv[1],
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]))
