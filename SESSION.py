"""This is the session call associated with GRAPH.py"""


with tf.Session(graph = graph) as session :
    #Initialize session
    train_writer = tf.summary.FileWriter(logs_path+'/train', graph = tf.get_default_graph())
    valid_writer = tf.summary.FileWriter(logs_path+'/test')
    tf.global_variables_initializer().run()
    print("Initialized!\n")

    print("\nRun the command line:\n tensorboard --logdir='{}'\n\n to view the tensorboard summary".format(logs_path))


    for batch_number in range(int(num_epochs * y_train.shape[0] // batch_size)) :
        # Determine offset for training batch collection
        offset = (batch_number * batch_size)
        # generate standardized training set
        X_batch = fd.make_batch(X_train_filenames, offset, batch_size, std_y, std_x)
        y_batch = fd.make_label(y_train, offset, batch_size)

        feed_dict = {training_data : X_batch, training_labels : y_batch}
        _, summary = session.run([training_op, train_summaries] , feed_dict = feed_dict)
        train_writer.add_summary(summary, batch_number*batch_size)


        if (batch_number % valid_every) == 0 :
            v_offset = 0
            validation_loss = []
            validation_accuracy = []
            while v_offset != valid_size :
                v_sum = session.run(valid_summaries,
                                    feed_dict = {   valid_data : X_valid[v_offset:(v_offset+batch_size), :, :, :],
                                                    valid_labels : y_valid[v_offset:(v_offset+batch_size), :] })
                valid_writer.add_summary(v_sum, ((batch_number*batch_size) + 10*(v_offset/valid_size)))
                v_offset += batch_size

    print("\nFINISHED TRAINING!")
