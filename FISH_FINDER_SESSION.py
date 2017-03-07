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
        epochs_completed = meta_dict.get(np.max([key for key in meta_dict])).get('Epochs_completed')
        total_fovea = meta_dict.get(np.max([key for key in meta_dict])).get('fovea_consumed')

        restorer = tf.train.Saver()
        print("Initializing restorer...")
        restorer.restore(session, tf.train.latest_checkpoint(md+'/checkpoint'))
        print("Weights and biases retrieved!  Picking up at {} epochs completed : {} training images observed".format(epochs_completed, total_fovea))

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")

    else :
        tf.global_variables_initializer().run()
        print("Weight and bias variables initialized!\n")
        meta_dict = {0 : { 'Epochs_completed' : 0,
                            'version_ID' : version_ID,
                            'fovea_consumed' : 0}
                    }

        with open(md+'/meta_dictionary.pickle', 'wb') as fmd :
            pickle.dump(meta_dict, fmd)

        saver = tf.train.Saver()
        print("Checkpoint saver initialized!\n")
        epochs_completed = 0
        saver.save(session, md+'/checkpoint', global_step = epochs_completed)
    # Tensorboard writer
    writer = tf.summary.FileWriter(tensorboard_path, graph = tf.get_default_graph())
    print("Tensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(tensorboard_path))

    # Image dictionary - immutable
    print("Loading high-resolution image dictionary...")
    with open('image_dictionary.pickle', 'rb') as handle :
        image_dictionary = pickle.load(handle)
    print("`image_dictionary` loaded!")
    print("\nCOMMENCE TRAINING...")
    total_fovea = 0
    while open('stop.txt', 'r').read().strip() == 'False' :
        with open('training_dictionary.pickle', 'rb') as handle : #training_dictionary is mutatble and must be loaded at the beginning of every epoch
            training_set_dictionary = pickle.load(handle)
        training_set_list = [x for x in training_set_dictionary]  # full of the keys from training set dictionary

        while len(training_set_list) > batch_size :
            batch_X, batch_y, _ = fd.prepare_batch(training_set_dictionary, training_set_list, batch_size = batch_size, fov_size = fov_size, label_dictionary = label_dict)


            feed_dict = {   train_images : batch_X,
                            train_labels : batch_y,
                            learning_rate : float(open('learning_rate.txt', 'r').read().strip())
                        }

            if (total_fovea % (batch_size*summary_rate)) == 0 :
                _ , summary_fetch = session.run([train_op, summaries], feed_dict = feed_dict)
                writer.add_summary(summary_fetch, total_fovea)
            else :
                _ = session.run(train_op, feed_dict = feed_dict)

            total_fovea += batch_size


        new_fovea_dict = fd.fovea_generation(image_dictionary, num_fovea = num_fovea)

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
            staged_X, staged_y, stg_keys = fd.prepare_batch(staged_dictionary, staged_set_list, batch_size = batch_size, fov_size = fov_size, label_dictionary = label_dict)

            feed_dict = {staged_set : staged_X}
            stgd_lgts = session.run(staged_logits, feed_dict = feed_dict)

            staged_dictionary, training_set_dictionary = fd.staged_set_supervisor(stgd_lgts, staged_dictionary, training_set_dictionary, keys = stg_keys)


        with open('staged_dictionary.pickle', 'wb') as fsd: # whatever is left after while loop will be saved into the staged set for next epoch.
            pickle.dump(staged_dictionary, fsd)
        with open('training_dictionary.pickle', 'wb') as ftd:
            pickle.dump(training_set_dictionary, ftd)

        epochs_completed += 1

        saver.save(session, metadata_path, global_step = epochs_completed)
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
