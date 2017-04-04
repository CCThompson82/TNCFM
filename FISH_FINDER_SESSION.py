"""This is the session call associated with FISH_FINDER_GRAPH.py"""


with tf.Session(graph = fish_finder) as session :

    # check for metadata dictionary
    if 'meta_dictionary.pickle' in os.listdir(md) and initiate_model != True:
        print("Loading FishFinder version {}".format(version_ID))
        with open(md+'/meta_dictionary.pickle', 'rb') as  handle :
            meta_dict = pickle.load(handle)
        print("Metadata dictionary loaded!")
        total_fovea = meta_dict.get(np.max([key for key in meta_dict])).get('fovea_trained')
        epochs_completed = meta_dict.get(np.max([key for key in meta_dict])).get('Num_epochs')
        restorer = tf.train.Saver()
        print("Initializing restorer...")
        restorer.restore(session, tf.train.latest_checkpoint(md))
        print("Weights and biases retrieved!  Picking up at {} epochs completed : {} training images observed".format(epochs_completed, total_fovea))

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")

    else :
        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        epochs_completed = 0
        total_fovea = 0
        meta_dict = {0 : {  'version_ID' : version_ID,
                            'fovea_trained' : total_fovea,
                            'Num_epochs' : epochs_completed}
                    }

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)

    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_path, graph = tf.get_default_graph())
    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(tensorboard_path))

    print("\nTRAINING FISHFINDER...")
    while open('stop.txt', 'r').read().strip() == 'False' :
        # NOTE : fovea_dictionary is mutatable and must be loaded at the beginning of every epoch
        with open(md+'/fovea_dictionary.pickle', 'rb') as handle :
            fovea_dictionary = pickle.load(handle)


        # Generate an epoch fovea dicionary to aid in batch image loading
        epoch_dictionary, epoch_keys = fd.generate_epoch_dictionary(fovea_dictionary = fovea_dictionary)

        while len(epoch_keys) > batch_size :
            # Choose batch_size number of keys
            batch_key_list = []
            for _ in range(0, batch_size, 1) :
                batch_key_list.append( epoch_keys.pop(np.random.randint(0,len(epoch_keys))))


            img_arr, label_arr = fd.generate_batch(batch_key_list, epoch_dictionary, fov_size = fov_size, label_dict = label_dict)
            if len(epoch_keys) > batch_size :
                feed_dict = {train_images : img_arr,
                             train_labels : label_arr,
                             learning_rate : float(open('learning_rate.txt', 'r').read().strip())
                             }

                _ = session.run([train_op, cross_entropy], feed_dict = feed_dict)
                total_fovea += batch_size
            else : # NOTE : This is the last batch before epoch ends ; summarize to tensorboard
                feed_dict = {train_images : img_arr,
                             train_labels : label_arr,
                             learning_rate : float(open('learning_rate.txt', 'r').read().strip()),
                             epoch_size : len(epoch_dictionary)
                             }
                _ , summary_fetch = session.run([train_op, summaries], feed_dict = feed_dict)

                total_fovea += batch_size
                writer.add_summary(summary_fetch, total_fovea)


        epochs_completed += 1
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
        print("Epoch {} completed : {} fovea observed. Model checkpoint created!".format(epochs_completed, total_fovea))
        meta_dict[epochs_completed] = {'Num_epochs' : epochs_completed,
                               'fovea_trained' : total_fovea,
                               'fov_pixels' : fov_size,
                               'checkpoint_directory' :  os.getcwd()+'/model_checkpoints/'+version_ID}

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)


        if open('run_train_stack_clf.txt', 'r').read().strip() == 'True' :
            print("Running fovea stack predictions for training set...")
            train_pred_dict = {}
            # list of all images in train set
            train_f_list = fd.generate_filenames_list(subdirectory = 'data/train/', subfolders = True)
            # loop through all images
            for f in train_f_list :
                arr_stack_list = fd.generate_fovea(f, fov_size = fov_size, scale = pred_scales, y_bins = bins_y, x_bins = bins_x, pixel_norm = 'centre')
                # loop through arr_stack_list
                for i, arr_stack in enumerate(arr_stack_list) :
                    # stack fovea and input to predictor
                    # fetch array of predictions, concatenate for each image
                    cursor = 0
                    while cursor < pred_batch :
                        batch_stack = arr_stack[cursor:cursor+pred_batch, :, :, :]
                        if cursor == 0 :
                            stacked_preds = session.run(stack_prediction, feed_dict = {img_stack : batch_stack})
                        else :
                            stacked_preds = np.concatenate([stacked_preds, session.run(stack_prediction, feed_dict = {img_stack : batch_stack})])
                        cursor += pred_batch

                    train_pred_dict[f] = {pred_scales[i] : stacked_preds}

            with open(md+'/train_prediction_stacks_dictionary.pickle', 'wb') as ftrainpsd :
                pickle.dump(train_pred_dict, ftrainpsd)
            print("Prediction stacks saved!")

        if open('run_test_stack_clf.txt', 'r').read().strip() == 'True' :
            print("Running fovea stack predictions for test set...")
            test_pred_dict = {}
            test_f_list = fd.generate_filenames_list(subdirectory = 'data/test_stg1/', subfolders = False)

            for f in test_f_list :
                arr_stack_list = fd.generate_fovea(f, fov_size = fov_size, scale = pred_scales, y_bins = bins_y , x_bins = bins_x, pixel_norm = 'centre')
                # loop through arr_stack_list
                for i, arr_stack in enumerate(arr_stack_list) :
                    # stack fovea and input to predictor
                    # fetch array of predictions, concatenate for each image
                    cursor = 0
                    while cursor < pred_batch :
                        batch_stack = arr_stack[cursor:cursor+pred_batch, :, : ,:]
                        if cursor == 0 :
                            stacked_preds = session.run(stack_prediction, feed_dict = {img_stack : batch_stack})
                        else :
                            stacked_preds = np.concatenate([stacked_preds, session.run(stack_prediction, feed_dict = {img_stack : batch_stack})])
                        cursor += pred_batch
                    test_pred_dict[f] = {pred_scales[i] : stacked_preds}

            with open(md+'/test_prediction_stacks_dictionary.pickle', 'wb') as ftpsd :
                pickle.dump(test_pred_dict, ftpsd)
            print("Prediction stacks saved!")
