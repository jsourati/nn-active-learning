import os
import numpy as np

def extract_Hakims_data_path():
    """Preparing addresses pointing to the
    raw images and masks of brain data
    that Hakim has labeled
    
    """
    
    ids = ['00%d'% i for i in 
           np.arange(1,10)] + [
               '0%d'% i for i in 
               np.arange(10,67)]

    root_dir = '/data/Jamshid/Hakim_adolescents/'
    mask_rest_of_path = 'scan01/common-processed/anatomical/03-ICC/'
    T1_rest_of_path = 'scan01/common-processed/anatomical/01-t1w-ref/'
    T2_rest_of_path = 'scan01/common-processed/anatomical/02-coregistration/'

    mask_addrs =[]
    T1_addrs = []
    T2_addrs = []
    Orig_addrs = []
    for idx in ids:
        name = os.listdir(
            os.path.join(
                root_dir,'Case%s'% idx))[0]
        mask_addrs += [
            os.path.join(
                root_dir,'Case%s'% idx,
                name,mask_rest_of_path,
                'c%s_s01_ICC.nrrd'% idx)]
        T1_addrs += [
            os.path.join(
                root_dir,'Case%s'% idx,
                name,T1_rest_of_path,
                'c%s_s01_t1w_ref.nrrd'% idx)]
        T2_addrs += [
            os.path.join(
                root_dir,'Case%s'% idx,
                name,T2_rest_of_path,
                'c%s_s01_t2w_r.nrrd'% idx)]

        Orig_addrs += [
            os.path.join(
                root_dir,'Case%s'% idx,
                name,mask_rest_of_path,
                'c%s_s01_BrainMask.nrrd'% idx)]
        
    return {'T1': T1_addrs, 'T2': T2_addrs}, mask_addrs

def extract_newborn_data_path():
    """Preparing addresses pointing to
    the raw images and masks of 
    T1-weighted MRI of brains on newborn
    subjects. The masks are manually 
    modified by Hakim.
    """

    # common root directory 
    root_dir = '/data/Jamshid/Hakim_newborn/'
    
    # common sub-directories
    # (except the data files which include
    # the subject and session codes in their
    # names)
    T1_rest_of_path = 'common-processed' +\
                       '/anatomical/01-t1w-ref'
    T2_rest_of_path = 'common-processed' +\
                       '/anatomical/02-coregistration'
    mask_rest_of_path = 'common-processed' +\
                        '/anatomical/03-ICC'


    # subject-specific sub-directories
    dirs = get_subdirs(root_dir)
    T1_addrs = []
    T2_addrs = []
    mask_addrs = []
    sub_codes = []
    for i, d in enumerate(dirs):
        if not('sub-CC00' in d):
            continue
            
        # there are two levels of subject-
        # specific sub-directories
        subdir = get_subdirs(os.path.join(
            root_dir, d))[0]
        subsubdir = get_subdirs(os.path.join(
            root_dir, d, subdir))[0]
        
        # we need the codes for accessing
        # to names of the data
        sub_code = d[4:]
        sub_codes += [sub_code]
        sess_code = subsubdir[4:]
            
        # subject-specific sub-directories
        subdirs = os.path.join(
            root_dir,
            d,
            subdir,
            subsubdir)

        """Putting everything together"""
        T1_addrs += [
            os.path.join(
                root_dir,
                subdirs,
                T1_rest_of_path,
                'c%s_s%s_t1w_ref.nrrd'% 
                (sub_code, sess_code))]

        T2_addrs += [
            os.path.join(
                root_dir,
                subdirs,
                T2_rest_of_path,
                'c%s_s%s_t2w_r.nrrd'% 
                (sub_code, sess_code))]
        
        mask_addrs += [
            os.path.join(
                root_dir,
                subdirs,
                mask_rest_of_path,
                'c%s_s%s_ICC.nrrd'% 
                (sub_code, sess_code))]

    sort_inds = np.argsort(sub_codes)
    sub_codes = list(np.array(sub_codes)[sort_inds])
    T1_addrs = list(np.array(T1_addrs)[sort_inds])
    T2_addrs = list(np.array(T2_addrs)[sort_inds])
    mask_addrs = list(np.array(mask_addrs)[sort_inds])

        
    return {'T1': T1_addrs, 'T2': T2_addrs}, mask_addrs


def extract_lesion_data_path(group='ACE', scans=[]):

    # common directory
    if group=='ACE':
        root_dir = '/data/Jamshid/Xavi_ACE/'
    elif group=='TSCR':
        root_dir = '/data/Jamshid/Xavi_TSCR'

    # common sub-directories
    # there are four scans, specify which
    # one to use
    scan_idx = 1
    if len(scans)==0:
        T1_rest_of_path = 'scan0%d/t1w.nrrd'% scan_idx
        T2_rest_of_path = 'scan0%d/t2w.nrrd'% scan_idx
        mask_rest_of_path='scan0%d/Manual-ICC.nrrd'% scan_idx

    # subject-specific sub-directories
    dirs = get_subdirs(root_dir)
    dirs = list(np.sort(np.array(dirs)))
    T1_addrs = []
    T2_addrs = []
    mask_addrs = []
    sub_codes = []

    for i,dir in enumerate(dirs):

        if len(scans)==0:
            T1_rest_of_path = 'scan0%d/t1w.nrrd'% scan_idx
            T2_rest_of_path = 'scan0%d/t2w.nrrd'% scan_idx
            mask_rest_of_path='scan0%d/Manual-ICC.nrrd'% scan_idx
        else:
            T1_rest_of_path = 'scan0%d/t1w.nrrd'% (scans[i])
            T2_rest_of_path = 'scan0%d/t2w.nrrd'% (scans[i])
            mask_rest_of_path='scan0%d/Manual-ICC.nrrd'% (scans[i])

        sub_codes += [dir]

        T1_path = os.path.join(
            root_dir,dir,T1_rest_of_path)
        T1_addrs += [T1_path]

        T2_path = os.path.join(
            root_dir,dir,T2_rest_of_path)
        T2_addrs += [T2_path]

        mask_path = os.path.join(
            root_dir,dir,mask_rest_of_path)
        mask_addrs += [mask_path]

    return {'T1': T1_addrs, 'T2': T2_addrs}, mask_addrs
    

def extract_NVM_data_path():

    root_dir = '/fileserver/external/rawabd/Jamshid/' + \
               'PWNNAL_results/unimodal_NVM/preprop_data/'

    # get data codes
    files = os.listdir(root_dir)
    sub_codes = np.array([f[:4] for f in files])
    sub_codes = np.sort(np.unique(sub_codes))

    # construct full paths
    T1_rest_of_path = '-t1w.nrrd'
    parc_rest_of_path = '-parcellation.nrrd'
    mask_rest_of_path = '-mask-wnan.nrrd'

    T1_addrs = []
    parc_addrs = []
    mask_addrs = []
    for code in sub_codes:
        T1_addrs += [os.path.join(
            root_dir,code+T1_rest_of_path)]

        parc_addrs += [os.path.join(
            root_dir,code+parc_rest_of_path)]

        mask_addrs += [os.path.join(
            root_dir,code+mask_rest_of_path)]

    return T1_addrs, parc_addrs, mask_addrs, list(sub_codes)

def extract_ISBI2015_MSLesion_data_path(test_or_training='train'):

    modalities = ['flair', 'mprage', 'pd', 't2']

    root_dir = '/data/Lesion/ISBI2015/'
    if test_or_training=='training':
        root_dir = os.path.join(root_dir, 'training')
    else:
        root_dir = os.path.join(root_dir, 'testdata_website')

    subdirs = np.sort(get_subdirs(root_dir, test_or_training))

    img_addrs = {mod.upper(): [] for mod in modalities}
    mask_addrs = {'mask1': [], 'mask2': []}
    for sbd in subdirs:
        
        full_img_dir = os.path.join(root_dir, sbd, 'preprocessed')
        all_img_files = os.listdir(full_img_dir)
        if test_or_training=='training':
            full_mask_dir = os.path.join(root_dir, sbd, 'masks')
            all_mask_files = os.listdir(full_mask_dir)
        # there are lot of irrelevant files inside the
        # subject diroctaries
        for mod in modalities:
            fnames = [name for name in all_img_files if 
                      (test_or_training in name) and 
                      (mod in name)]
            # sorting the file based on time-point index
            tp_inds = [int(name.split('_')[1]) for name in fnames]
            sort_inds = np.argsort(tp_inds)
            mod_names = list(np.array(fnames)[sort_inds])
            img_addrs[mod.upper()] += [os.path.join(full_img_dir, name) for  
                               name in mod_names]

        if test_or_training=='training':
            mask1_names = [name for name in all_mask_files if 
                           (test_or_training in name) and 
                           ('mask1' in name)]
            tp_inds = [int(name.split('_')[1]) for name in mask1_names]
            sort_inds = np.argsort(tp_inds)
            mask1_names = list(np.array(mask1_names)[sort_inds])
            mask_addrs['mask1'] += [os.path.join(full_mask_dir, name) for
                                    name in mask1_names]

            mask2_names = [name for name in all_mask_files if 
                           (test_or_training in name) and 
                           ('mask2' in name)]
            tp_inds = [int(name.split('_')[1]) for name in mask2_names]
            sort_inds = np.argsort(tp_inds)
            mask1_names = list(np.array(mask2_names)[sort_inds])
            mask_addrs['mask2'] += [os.path.join(full_mask_dir, name) for
                                    name in mask2_names]

    if test_or_training=='training':
        return img_addrs, mask_addrs

    else:
        return img_addrs


def get_subdirs(path, common_term=None):
    """returning all sub-directories of a 
    given path
    """
    
    subdirs = [d for d in os.listdir(path)
               if os.path.isdir(os.path.join(
                       path,d))]

    if common_term is not None:
        subdirs = [subdir for subdir in subdirs 
                   if common_term in subdir]
    
    return subdirs

