"""This is the session call associated with GRAPH.py"""


with tf.Session(graph = graph) as session :
    #Initialize session
    writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
    #valid_writer = tf.summary.FileWriter(logs_path+'/test')
    tf.global_variables_initializer().run()
    print("Initialized!\n")

    print("\nRun the command line:\ntensorboard --logdir='{}'\n\n to view your tensorboard dashboard summary!".format(logs_path))

    non_record_counter = valid_every
    record_counter = y_valid.shape[0] // batch_size # in order to iterate  through the whole validation set
    # NOTE : if y_valid.shape[0] is not divisible wholely by batch_size, then the final remainder examples in the validation set will never be used.
    assert (y_valid.shape[0] % batch_size) == 0, 'Validation size is not wholely divisible by batch_size, please ammend to batch_size that is a factor of {}'.format(y_valid.shape[0])
    record = True
    for batch_number in range(int(num_epochs * y_train.shape[0] // batch_size)) :
        print(batch_number+1)
        # Determine offset for training batch collection
        offset = (batch_number * batch_size) - (y_train.shape[0]*(batch_number * batch_size) // y_train.shape[0])
        # generate standardized training set
        X_batch = fd.make_batch(X_train_filenames, offset, batch_size, std_y, std_x)
        y_batch = fd.make_label(y_train, offset, batch_size)

        if record == True :
            v_offset = ((y_valid.shape[0] // batch_size) - record_counter) * batch_size
            feed_dict= {    training_data : X_batch,
                            training_labels : y_batch,
                            keep_prob_convs : kp_convs,
                            keep_prob_hidden : kp_hidden,
                            valid_data : X_valid[v_offset:(v_offset+batch_size), :, :, :],
                            valid_labels : y_valid[v_offset:(v_offset+batch_size), :] }

            _, summary = session.run([training_op, summaries], feed_dict = feed_dict)
            writer.add_summary(summary, batch_number*batch_size)
            record_counter -= 1
            if record_counter == 0 :
                record = False
                print("record switched to false")
                record_counter = y_valid.shape[0] // batch_size # reset the counter

        else :
            feed_dict = {   training_data : X_batch,
                            training_labels : y_batch,
                            keep_prob_convs : kp_convs,
                            keep_prob_hidden : kp_hidden }
            _ = session.run(training_op, feed_dict = feed_dict)
            non_record_counter -= 1
            if non_record_counter == 0 :
                record = True
                print("record switch to True")
                non_record_counter = valid_every # reset the non_record_counter





    print("\nFINISHED TRAINING!")
