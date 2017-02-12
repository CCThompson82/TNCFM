"""This is the session call associated with GRAPH.py"""

logs_path = os.getcwd()+'/TB_logs/'+version_ID

with tf.Session(graph = graph) as session :
    tf.global_variables_initializer().run()
    print("Weight and bias variables initialized!\n")
    saver = tf.train.Saver()
    print("Checkpoint saver initialized!\n")
    writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
    #valid_writer = tf.summary.FileWriter(logs_path+'/test')
    print("\nTensorboard initialized!\nTo view your tensorboard dashboard summary, run the following on the command line:\n\ntensorboard --logdir='{}'\n".format(logs_path))

    print("Training model...\n")
    batch_num = 0
    batches_in_epoch = len(files_train) // batch_size

    while  batch_num < (batches_in_epoch * num_epochs) :
        offset = (batch_num*(batch_size)) - ((batch_num*batch_size // len(files_train)) * len(files_train))

        X, y = fd.process_batch(files_train, y_train, offset = offset, batch_size = batch_size,
                        std_size = std_size, crop_size = crop_size, crop_mode = 'random', normalize = 'custom',
                        pixel_offset = 100.0, pixel_factor = 100.0,
                        mutation = True, verbose = False)
        feed_dict = { train_images : X , train_labels : y}

        if (batch_num % checkpoint_interval) == 0 :
            _, summary, vlog, vlab = session.run([training_op, summaries, validation_logits, validation_labels], feed_dict = feed_dict)

            saver.save(session, 'model_checkpoints/'+version_ID, global_step = batch_num * batch_size)
            print("="*40)
            print("Model checkpoint created after {} images consumed".format((batch_num+1)*batch_size))
            print("Model can be restored from: \n\n{}\n".format(os.getcwd()+'/model_checkpoints/'+version_ID))
            print("Validation set classification report:")
            print(classification_report(np.argmax(vlab,1),np.argmax(vlog,1), target_names = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'] ))
            print("="*40)

        elif  batch_num < validate_interval :
            _, summary = session.run([training_op, summaries], feed_dict = feed_dict)
            writer.add_summary(summary, batch_num*batch_size)

        elif (batch_num % validate_interval) == 0 :
            _, summary = session.run([training_op, summaries], feed_dict = feed_dict)
            writer.add_summary(summary, batch_num*batch_size)

        else :
            _ = session.run(training_op, feed_dict)
        batch_num += 1

    print("\nTRAINING FINISHED!\n\nSaving final model...")
    saver.save(session, 'FINAL_MODELS/'+version_ID)
    print("Model saved!\n\nRunning Test set predictions...")
    for i in range(len(test_filenames)) :
        X, _ = fd.process_batch(test_filenames, labels = None, offset = i, batch_size = 1, std_size = std_size, crop_size = crop_size, crop_mode = 'centre', normalize = 'custom', pixel_offset = 100.0, pixel_factor = 100.0, mutation = False, verbose = False)

        test_lgts = session.run(test_logits, {test_images : X})

        try :
            test_scores = np.concatenate([test_scores, test_lgts], 0)
        except :
            test_scores = test_lgts

    test_df = pd.DataFrame(test_scores, columns = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    test_df = pd.concat([pd.Series([ x[15:] for x in test_filenames], name = 'image'), test_df], axis = 1)
    test_df.to_csv('Test_predictions/'+version_ID+'.csv', header=True, index = False)

    print("Test Predictions Stored at {}".format('Test_predictions/'+version_ID+'.csv'))

    print("Submit file at:\n\n   https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/submit")
