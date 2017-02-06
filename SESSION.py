"""This is the session call associated with GRAPH.py"""


with tf.Session(graph = graph) as session :
    tf.global_variables_initializer().run()
    print("Initialized!\n")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
    #valid_writer = tf.summary.FileWriter(logs_path+'/test')
    print("\nTo view your tensorboard dashboard summary, run the following on the command line:\ntensorboard --logdir='{}'\n".format(logs_path))

    batch_num = 0
    while  batch_num < ((len(files_train) // batch_size) * num_epochs) :
        c, _ = session.run([train_cross_entropy, training_op])
        if (batch_num % 10) == 0 :
            summary, vce = session.run([summaries, validation_cross_entropy])
            print("Batch number: {}".format(batch_num+1))
            print("     Training_mean_cross_entropy: {}".format(c))
            print("     Valid_mean_cross_entropy: {}".format(vce))
            writer.add_summary(summary, batch_num*batch_size)
        batch_num += 1
    coord.request_stop()
    coord.join(threads)












    """

    # NOTE : if y_valid.shape[0] is not divisible wholely by batch_size, then the final remainder examples in the validation set will never be used.
    assert (y_valid.shape[0] % batch_size) == 0, 'Validation size is not wholely divisible by batch_size, please ammend to batch_size that is a factor of {}'.format(y_valid.shape[0])
    non_record_counter = valid_every
    record_counter = y_valid.shape[0] // batch_size # in order to iterate  through the whole validation set

    record = True
    for batch_number in range(int(num_epochs * len(X_filenames) // batch_size)) :
        # Determine offset for training batch collection
        offset = (batch_number * batch_size) - (len(X_filenames)*(batch_number * batch_size) // len(X_filenames))
        # generate standardized training set

        X_batch = fd.make_batch(X_filenames, offset, batch_size, std_y, std_x, mutate = True)
        y_batch = fd.make_label(X_filenames, offset, batch_size)

        if record == True :
            v_offset = ((y_valid.shape[0] // batch_size) - record_counter) * batch_size

            feed_dict= {    training_data : X_batch,
                            training_labels : y_batch,
                            valid_data : X_valid[v_offset:(v_offset+batch_size), :, :, :],
                            valid_labels : y_valid[v_offset:(v_offset+batch_size), :] }

            _, summary = session.run([training_op, summaries], feed_dict = feed_dict)
            writer.add_summary(summary, batch_number*batch_size)
            record_counter -= 1
            if record_counter == 0 :
                if batch_number > 10 * y_valid.shape[0] / batch_size :  # Measure the validation set 'X' times prior to allowing the record variable to be switched off.  After this, breaks in validation are given to save on the overall run time.  Should save time in knowing whether to kill a run or not
                    record = False
                record_counter = y_valid.shape[0] // batch_size # reset the counter

        else :
            feed_dict = {   training_data : X_batch,
                            training_labels : y_batch }
            _ = session.run(training_op, feed_dict = feed_dict)
            non_record_counter -= 1
            if non_record_counter == 0 :
                record = True
                non_record_counter = valid_every # reset the non_record_counter





    print("\nFINISHED TRAINING!")

"""
