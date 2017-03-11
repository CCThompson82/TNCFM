"""This is the session call associated with GRAPH.py"""

wd = os.getcwd()
md = wd+'/FISHFINDER_MODELS/'+version_ID
if not os.path.exists(md) :
    os.makedirs(md)
tensorboard_path = md+'/Tensorboard_logs'



with tf.Session(graph = fish_finder) as session :

    # check for metadata dictionary
    if 'meta_dictionary.pickle' in os.listdir(md) and initiate_model != True:
        print("Loading FISHFINDER version {}".format(version_ID))
        with open(md+'/meta_dictionary.pickle', 'rb') as  handle :
            meta_dict = pickle.load(handle)
        print("Metadata dictionary loaded!")
        epochs_completed = meta_dict.get(np.max([key for key in meta_dict])).get('Num_epochs')
        total_fovea = meta_dict.get(np.max([key for key in meta_dict])).get('fovea_trained')

        restorer = tf.train.Saver()
        print("Initializing restorer...")
        restorer.restore(session, tf.train.latest_checkpoint(md))
        print("Weights and biases retrieved!  Picking up at {} epochs completed : {} training images observed".format(epochs_completed, total_fovea))

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")

    else :
        with open('training_dictionary.pickle', 'rb') as handle : #training_dictionary is mutatble and must be loaded at the beginning of every epoch
            seed_training_set_dictionary = pickle.load(handle)
        with open(md+'/training_dictionary.pickle', 'wb') as ftd:
            pickle.dump(seed_training_set_dictionary, ftd)

        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        meta_dict = {0 : { 'Num_epochs' : 0,
                            'version_ID' : version_ID,
                            'fovea_trained' : 0}
                    }

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")
        epochs_completed = 0
        total_fovea = 0
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_path, graph = tf.get_default_graph())
    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(tensorboard_path))

    # Image dictionary - immutable
    print("Loading high-resolution image dictionary...")
    with open('image_dictionary.pickle', 'rb') as handle :
        image_dictionary = pickle.load(handle)
    print("`image_dictionary` loaded!")



    print("\nTRAINING FISHFINDER...")
    while open('stop.txt', 'r').read().strip() == 'False' :
        with open(md+'/training_dictionary.pickle', 'rb') as handle : #training_dictionary is mutatble and must be loaded at the beginning of every epoch
            training_set_dictionary = pickle.load(handle)
        training_set_list = [x for x in training_set_dictionary]  # full of the keys from training set dictionary
        # TODO : calculate the fovea label frequencies
        fov_labels_for_freq = []
        for key in training_set_list :
            fov_labels_for_freq.append(training_set_dictionary[key].get('fovea_label'))
        counts = pd.Series(fov_labels_for_freq).value_counts()
        counts = counts.reindex(['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
        counts = np.array(counts)
        count_freq = np.array(counts) / np.sum(counts)
        count_weights = np.expand_dims((1 / count_freq),0)


        while len(training_set_list) >= batch_size :
            batch_X, batch_y, _ = fd.prepare_batch(training_set_dictionary, training_set_list, batch_size = batch_size, fov_size = fov_size, label_dictionary = label_dict, return_label = 'onehot')

            feed_dict = {   train_images : batch_X,
                            train_labels : batch_y,
                            learning_rate : float(open('learning_rate.txt', 'r').read().strip()),
                            beta_weights : float(open('beta_weights.txt', 'r').read().strip()),
                            frequency_weights : count_weights
                        }

            if (total_fovea % (batch_size*summary_rate)) == 0 :
                _ , summary_fetch = session.run([train_op, summaries], feed_dict = feed_dict)
                writer.add_summary(summary_fetch, total_fovea)
            else :
                _ = session.run(train_op, feed_dict = feed_dict)

            total_fovea += batch_size

        new_fovea_dict = fd.fovea_generation(image_dictionary, num_fovea = num_fovea, fov_size = fov_size)


        if epochs_completed == 0 :
            staged_dictionary = new_fovea_dict
            with open(md+'/staged_dictionary.pickle', 'wb') as fsd:
                pickle.dump(staged_dictionary, fsd)
        else :
            with open(md+'/staged_dictionary.pickle', 'rb') as handle :
                staged_dictionary = pickle.load(handle)
            staged_dictionary.update(new_fovea_dict)

        staged_set_list = [x for x in staged_dictionary]
        while len(staged_set_list) >= batch_size :
            staged_X, stg_keys = fd.prepare_batch(staged_dictionary, staged_set_list, batch_size = batch_size, fov_size = fov_size, label_dictionary = label_dict, return_label = False)

            feed_dict = {staged_set : staged_X}
            stgd_lgts = session.run(staged_logits, feed_dict = feed_dict)

            staged_dictionary, training_set_dictionary = fd.stage_set_supervisor(stgd_lgts, staged_dictionary, training_set_dictionary, keys = stg_keys, label_dict = label_dict, reverse_label_dict = reverse_label_dict)


        with open(md+'/staged_dictionary.pickle', 'wb') as fsd: # whatever is left after while loop will be saved into the staged set for next epoch.
            pickle.dump(staged_dictionary, fsd)
        if open('commit_threshold.txt', 'r').read().strip() != 'Manual' :  # when in manual mode, the original training set will be loaded at teh beginning of every epoch until the user saves over it in the backdoor function.  When this is changed to numeric threshold, the training_set_dictionary will be updated and saved over automatically
            with open(md+'/training_dictionary.pickle', 'wb') as ftd:
                pickle.dump(training_set_dictionary, ftd)

        epochs_completed += 1
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
        print("Epoch {} completed : {} fovea observed. Model checkpoint created!".format(epochs_completed, total_fovea))
        meta_dict[epochs_completed] = {'Num_epochs' : epochs_completed,
                                       'training_set' : len(training_set_dictionary),
                                       'count_weights' : count_weights,
                                       'stage_set' : len(staged_dictionary),
                                       'fovea_trained' : total_fovea,
                                       'fov_pixels' : fov_size,
                                       'keep_threshold' : float(open('keep_threshold.txt', 'r').read().strip()) ,
                                       'commit_threshold' : open('keep_threshold.txt', 'r').read().strip(),
                                       'checkpoint_directory' :  os.getcwd()+'/model_checkpoints/'+version_ID}

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)
