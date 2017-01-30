"""This is the session call associated with GRAPH.py"""


with tf.Session(graph = graph) as session :
    #Initialize session
    #writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())
    tf.global_variables_initializer().run()
    print("Initialized!\n")


    for batch_number in range(int(num_epochs * y_train.shape[0] // batch_size)) :
        # Determine offset for training batch collection
        offset = (batch_number * batch_size)
        # generate standardized training set
        X_batch = fd.make_batch(X_train_filenames, offset, batch_size, std_y, std_x)
        y_batch = fd.make_label(y_train, offset, batch_size)
        feed_dict = {training_data : X_batch, training_labels : y_batch}
        _, l = session.run([training_op, cross_entropy] , feed_dict = feed_dict)

        print(l)

        if (batch_number % 1) == 0 :
            print("Validation calcs...")
            print(valid_loss.eval())
