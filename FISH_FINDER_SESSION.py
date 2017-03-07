"""This is the session call associated with GRAPH.py"""

logs_path = os.getcwd()+'/TB_logs/'+version_ID

with tf.Session(graph = fish_finder) as session :

    # TODO : refactor to look at meta data folder in order to decide to initate or retrieve weights
    if restore_model == False :
        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        epochs_completed = 0
        meta_dict = {'Init':  { 'Num_epochs' : epochs_completed,
                                'checkpoint_directory' :  os.getcwd()+'/model_checkpoints/'+version_ID}}

        with open(version_ID+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

    elif restore_model == True :
        restorer = tf.train.Saver()
        print("Restorer initialized!")
        restorer.restore(session, tf.train.latest_checkpoint(directory))
        print("Weights and biases retrieved from {} in {}\n".format(last_version, directory))
        # TODO : lookup meta_data file

    print("Loading high-resolution image dictionary...")
    with open('image_dictionary.pickle', 'rb') as handle :
        image_dictionary = pickle.load(handle)
    print("`image_dictionary` loaded!")

    saver = tf.train.Saver()
    print("Checkpoint saver initialized!\n")
    writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(logs_path))

    print("Training model...\n")


    # Helper Functions +++++++++
    def prepare_batch(dictionary, training_set_list, batch_size) :
        """
        Retrieves fovea from a dictionary that contains filname, coordinates of
        fovea, fovea_label, pre-scale float.  As fovea are added to the batch, they
        are removed from training_set_list to avoid duplicate use.
        """
        # TODO assert that dictionary contains filename ('f') and 'coordinates' keys
        def retrieve_fovea(key) :
            """Convenience function to retrive the fovea array with one-hot label"""
            f_dict = dictionary.get(key)
            f = f_dict['f']
            scale = f_dict['scale']
            y_off, x_off = f_dict['coordinates']['y_offset'], f_dict['coordinates']['x_offset']
            fovea = np.expand_dims(
                        misc.imresize(
                            misc.imread(f, mode = 'RGB'),
                                size = scale)[y_off:(y_off+fov_size), x_off:(x_off+fov_size), :],
                                [0])
            try :
                label = f_dict['fovea_label'].replace(onehot_dict)  # TODO : make into pandas for replace to work as expected
            except :
                label = 'Unknown'
            return fovea, label, key

        fovea, label, key = retrieve_fovea(
                            training_set_list.pop(
                                np.random.randint(0, len(training_set_list), 1)))

        X_batch = fovea
        y_batch = label
        keys = key

        while y_batch.shape[0] != batch_size :
            fovea, label, key = retrieve_fovea(
                            training_set_list.pop(
                                np.random.randint(0, len(training_set_list), 1)))
            X_batch.concatenate(fovea, 0)
            y_batch.concatenate(label, 0)
            keys.append(key)

        return X_batch, y_batch, keys

    def fovea_generation(image_dictionary, num_fovea = 100) :
        """
        Function for random sampling of high-resolution image files, followed by
        random fovea generation.
        """
        new_fovea_dict = {}
        f_list = [x for x in image_dictionary]
        samples_f_list = np.random.choice(f_list, num_fovea)

        while len(samples_f_list) > 0 :
            f = samples_f_list.pop(np.random.randint(0,len(samples_f_list),1))
            scale = 0.4 * np.random.rand() + 0.4
            shape = misc.imresize(misc.imread(f, mode = 'RGB'), size = scale).shape
            y_offset = np.random.randint(0, shape[0]-fov_size, 1)
            x_offset = np.random.randint(0, shape[1]-fov_size, 1)

            new_fovea_dict[f] = {'scale': scale,
                                 'coordinates' : {'y_offset' : y_offset, 'x_offset' : x_offset},
                                 'image_label' : image_dictionary[f]['image_label'],
                                 'staged_steps' : 0 }
        return new_fovea_dict


        def stage_set_supervisor(stgd_lgts, staged_dictionary, training_set_dictionary, keys) :
            """
            Fn will destage, keep staged, or commit fovea from the stage dictionary based on set of criteria :
                * Destage
                    * None of the softmax logit labels were predicted with confidence exceeding some threshold
                    * Predicted an impossible fovea label (i.e. 'ALB' for a high-resolution image of a 'NoF')
                        * Note : It is okay to predict 'NoF' fovea from a high-resolution with a fish label
                * Remain in stage_dictionary
                    * label is predicted with a confidence that exceeds the threshold set
                    * label is not impossible
                    * prediction has not yet exceeded the stability requirement for commitment to training set
                * Commit to training set
                    * fovea has remained in the stage_dictionary for greater than a threshold of epochs, indicating stability in the model prediction

            """
            for i, key in enumerate(keys) :
                pred = stgd_lgts[i,:]
                keep_threshold = 0.98   # TODO : change to file that can be amended during the run
                commit_threshold = 5    # TODO : change to file load that can be amended during run
                commit, keep = False, False
                if pred[np.argmax(pred)] < keep_threshold :
                    keep = False
                elif np.argmax(pred) != np.argmax( staged_dictionary.get(key)['image_label'].replace(onehot_dict) ) :
                    if np.argmax(pred) == 4 : # NoF label
                        keep = True
                        staged_dictionary.get(key)['stage_steps'] += 1
                        if staged_dictionary.get(key)['stage_steps'] == commit_threshold :
                            commit = True
                    else :
                        keep = False
                else :
                    print("ERROR in stage_set_supervisor parsing")

                if commit :
                    training_set_dictionary.append( {key : staged_dictionary.pop(key)} )
                elif keep :
                    pass
                else :
                    _ = staged_dictionary.pop(key)
            return staged_dictionary, training_set_dictionary

    # Helper functions ---------------------


    total_fovea = 0
    while open('stop.txt', 'r').read().strip() == 'False' :
        with open('training_dictionary.pickle', 'rb') as handle :
            training_set_dictionary = pickle.load(handle)
        training_set_list = [x for x in training_set_dictionary]  # full of the keys from training set dictionary

        while len(training_set_list) > batch_size :
            batch_X, batch_y, _ = prepare_batch(training_set_dictionary, training_set_list, batch_size = batch_size)


            feed_dict = {   train_images : batch_X,
                            train_labels : batch_y,
                            learning_rate : float(open('learning_rate.txt', 'r').read().strip())
                        }
            total_fovea += batch_size

            if (total_fovea % (batch_size*summary_rate)) == 0 :
                _ , summary_fetch = session.run([train_op, summaries], feed_dict = feed_dict)
                writer.add_summary(summary_fetch,total_fovea )
            else :
                _ = session.run(train_op, feed_dict = feed_dict)




        new_fovea_dict = fovea_generation(image_dictionary, num_fovea = num_fovea)

        try :
            with open('staged_dictionary.pickle', 'rb') as handle :
                staged_dictionary = pickle.load(handle)
            staged_dictionary = staged_dictionary + new_fovea_dict
        except : # initiate staged_dictionary in first instance
            staged_dictionary = new_fovea_dict
            with open('staged_dictionary.pickle', 'wb') as fsd:
                pickle.dump(staged_dictionary, fsd)


        staged_set_list = [x for x in staged_dictionary]
        while len(staged_set_list) >= batch_size :
            staged_X, staged_y, stg_keys = prepare_batch(staged_dictionary, staged_set_list, batch_size = batch_size)

            feed_dict = {staged_set : staged_X}
            stgd_lgts = session.run(staged_logits, feed_dict = feed_dict)

            staged_dictionary, training_set_dictionary = staged_set_supervisor(stgd_lgts, staged_dictionary, training_set_dictionary, keys = stg_keys)


        with open('staged_dictionary.pickle', 'wb') as fsd: # whatever is left after while loop will be saved into the staged set for next epoch.
            pickle.dump(staged_dictionary, fsd)
        with open('training_dictionary.pickle', 'wb') as ftd:
            pickle.dump(training_set_dictionary, ftd)

        epochs_completed += 1

        saver.save(session, 'model_checkpoints/'+version_ID, global_step = total_fovea)
        print("="*40)
        print("Model checkpoint created after {} images consumed".format((batch_num+1)*batch_size))
        print("Model can be restored from: \n\n{}\n".format(os.getcwd()+'/model_checkpoints/'+version_ID))

        meta_dict[epochs_completed] = {'Num_epochs' : epochs_completed,
                                       'training_set' : len(training_set_dictionary),
                                       'stage_set' : len(staged_dictionary),
                                       'fovea_trained' : total_fovea,
                                       'checkpoint_directory' :  os.getcwd()+'/model_checkpoints/'+version_ID}

        with open(version_ID+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)
