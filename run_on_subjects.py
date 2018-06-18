# model: given
# save_dir for savign results: given

for ind in good_imgs:
    img_T1,_ = nrrd.read(T1_addrs[ind])
    img_T2,_ = nrrd.read(T2_addrs[ind])
    mask,_ = nrrd.read(mask_addrs[ind])

    stats = [[np.mean(img_T1[~np.isnan(mask)]), 
              np.std(img_T1[~np.isnan(mask)])],
             [np.mean(img_T2[~np.isnan(mask)]), 
              np.std(img_T2[~np.isnan(mask)])]]

    img_paths = [T1_addrs[ind], T2_addrs[ind]]
    slice_inds = np.arange(mask.shape[2])
    
    slice_preds = PW_analyze_results.full_slice_eval(
        model,sess,img_paths,slice_inds,(25,25,1),
        10000, stats, 'prediction')

    sub_save_dir = os.path.join(save_dir,'%d'%ind)
    if not(os.path.exists(sub_save_dir)):
        os.mkdir(sub_save_dir)
    nrrd.write(os.path.join(sub_save_dir,'segs.nrrd'), slice_preds)


""" Saving grid-level predictions for universal experiments"""
if False:

    base_dir = os.path.join(E.root_dir, method)

    save_dir = os.path.join(base_dir,'preds')
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)

    for ind in good_imgs:
        save_path = os.path.join(save_dir,'%s.txt'% (sub_codes[ind]))
        dat = [[T1_addrs[ind],  mask_addrs[ind]]]
        inds,labels = PW_AL.gen_multimg_inds([dat[0]], 
                                              E.pars['grid_spacing'])

        # stats
        img,_ = nrrd.read(T1_addrs[ind])
        mask,_ = nrrd.read(mask_addrs[ind])
        stats = [[np.mean(img[~np.isnan(mask)]), 
                  np.std(img[~np.isnan(mask)])]]

        Qs = PW_analyze_results.get_queries(E, method)
        L = len(Qs)
        subject_preds = np.zeros((L+1, len(inds[0])))

        for i in range(L+1):
            if i==0:
                weights_path = E.pars['init_weights_path']
            else:
                weights_path = os.path.join(E.root_dir,
                                            method,
                                            'curr_weights_%d.h5'% i)

            model.perform_assign_ops(weights_path, sess)
            preds = PW_NN.batch_eval(model,sess,dat[0][:1],inds[0],
                             E.pars['patch_shape'],E.pars['ntb'],
                             stats,'prediction')[0]
            subject_preds[i,:] = preds

            np.savetxt(save_path, subject_preds)


""" Retraining/Evaluating modelas on all slices (not only odds)"""
if False:
    base_dir = '/fileserver/external/rawabd/Jamshid/PWNNAL_results/bimodal_10/AL/newborns'

    for ind in good_imgs:
        print('Woring on image %d'% ind)
        E = PW_AL.Experiment(os.path.join(base_dir,'%d'% ind))
        E.load_parameters()

        E.pars['img_paths'] = [T1_addrs[ind], T2_addrs[ind]]
        E.pars['mask_path'] = mask_addrs[ind]

        dat = [E.pars['img_paths'] + [E.pars['mask_path']]]
        dat[0] += [[]]
        inds,labels = PW_AL.gen_multimg_inds([dat[0][:-1]],E.pars['grid_spacing'])

        scores = []
        model.perform_assign_ops(init_weights_path, sess)
        preds = PW_NN.batch_eval(model,sess,dat[0][:-2],inds[0],
                         E.pars['patch_shape'],E.pars['ntb'],
                         E.pars['stats'],'prediction')[0]
        scores += [F1_scores(preds, np.array(labels[0]))]

        Qs = PW_analyze_results.get_queries(E, method)
        stats = np.array(E.pars['stats'][0] + E.pars['stats'][1])
        stats = np.expand_dims(stats, axis=0)
        for i, Q in enumerate(Qs):
            dat[0][-1] += Q.tolist()
            PW_NN.PW_train_epoch_MultiModal(
                model,sess,dat,
                E.pars['epochs'],E.pars['patch_shape'],
                E.pars['b'],E.pars['ntb'],stats)

            preds = PW_NN.batch_eval(model,sess,dat[0][:-2],inds[0],
                                     E.pars['patch_shape'],E.pars['ntb'],
                                     E.pars['stats'],'prediction')[0]
            scores += [F1_scores(preds, np.array(labels[0]))]

            np.savetxt(os.path.join(E.root_dir,'%s/PostRev_perf_evals.txt'% method), scores)






        
