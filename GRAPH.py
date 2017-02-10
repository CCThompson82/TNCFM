"""This is the graph for the NCF Fish Classification Kaggle Competition"""



graph = tf.Graph()

with graph.as_default() :
    with tf.name_scope('Training_input') :

        train_q = tf.train.slice_input_producer([files_train, y_train], shuffle = True, capacity = 100)
        train_label = train_q[1]
        train_image = fd.decode_image(tf.read_file(train_q[0]), size = std_sizes, mutate = True, crop = 'random', crop_size = crop_size)

        train_images, train_labels= tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=batch_size, capacity = batch_size * 4
                                    )
    # Variables

    with tf.variable_scope('Variables') :
        with tf.name_scope('Batch_step') :
            steps = tf.Variable(0, trainable = False)
        with tf.name_scope('Label_weights') :
            label_weights = tf.constant(np.sum(y_train,0))
        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(tf.truncated_normal([kernel_sizes[0], kernel_sizes[0], num_channels, conv_depths[0]], stddev = stddev))
            tf.summary.histogram('W_conv1', W_conv1)
            b_conv1 = tf.Variable(tf.zeros([conv_depths[0]]))
            tf.summary.histogram('b_conv1', b_conv1)

            W_conv2 = tf.Variable(tf.truncated_normal([kernel_sizes[1], kernel_sizes[1], conv_depths[0], conv_depths[1]], stddev = stddev))
            tf.summary.histogram('W_conv2', W_conv2)
            b_conv2 = tf.Variable(tf.ones([conv_depths[1]]))
            tf.summary.histogram('b_conv2', b_conv2)

            W_conv3 = tf.Variable(tf.truncated_normal([kernel_sizes[2], kernel_sizes[2], conv_depths[1], conv_depths[2]], stddev = stddev))
            tf.summary.histogram('W_conv3', W_conv3)
            b_conv3 = tf.Variable(tf.zeros([conv_depths[2]]))
            tf.summary.histogram('b_conv3', b_conv3)

            W_conv4 = tf.Variable(tf.truncated_normal([kernel_sizes[3], kernel_sizes[3], conv_depths[2], conv_depths[3]], stddev = stddev))
            tf.summary.histogram('W_conv4', W_conv4)
            b_conv4 = tf.Variable(tf.ones([conv_depths[3]]))
            tf.summary.histogram('b_conv4', b_conv4)

            W_conv5 = tf.Variable(tf.truncated_normal([kernel_sizes[4], kernel_sizes[4], conv_depths[3], conv_depths[4]], stddev = stddev))
            tf.summary.histogram('W_conv5', W_conv5)
            b_conv5 = tf.Variable(tf.ones([conv_depths[4]]))
            tf.summary.histogram('b_conv5', b_conv5)


        with tf.variable_scope('Fully_connected') :
            W_fc1 = tf.Variable(tf.truncated_normal(
                            [nodes_exit_convolution, fc1_depth], # from PARAMETERS.py
                            stddev = stddev ))
            tf.summary.histogram('W_fc1', W_fc1)
            b_fc1 = tf.Variable(tf.zeros([fc1_depth]))
            tf.summary.histogram('b_fc1', b_fc1)
            W_fc2 = tf.Variable(tf.truncated_normal([fc1_depth, fc2_depth], stddev = stddev ))
            tf.summary.histogram('W_fc2', W_fc2)
            b_fc2 = tf.Variable(tf.zeros([fc2_depth]))
            tf.summary.histogram('b_fc2', b_fc2)


        with tf.variable_scope('Softmax') :
            W_softmax = tf.Variable(tf.truncated_normal([fc2_depth, num_labels], stddev = stddev))
            tf.summary.histogram('W_softmax', W_softmax)
            b_softmax = tf.Variable(tf.zeros([num_labels]))
            tf.summary.histogram('b_softmax', b_softmax)


    def nn(data, keep_prob_hidden) :
        with tf.name_scope('Convolution') :
            # TODO : local_response_normalization of RELU layers
            c1 = tf.nn.max_pool(
                    tf.nn.local_response_normalization(
                        tf.nn.relu(
                            tf.nn.conv2d(data, filter = W_conv1,
                                strides = [1, conv_strides[0],conv_strides[0], 1],
                                padding = 'SAME') +
                            b_conv1),
                        depth_radius = 5, bias = 2, alpha = 10e-4, beta = 0.75),
                    ksize = [1,pool_kernels[0],pool_kernels[0],1], strides = [1,pool_strides[0], pool_strides[0],1],
                    padding ='VALID')
            c2 = tf.nn.max_pool(
                    tf.nn.local_response_normalization(
                        tf.nn.relu(
                            tf.nn.conv2d(c1, filter = W_conv2,
                                strides = [1,conv_strides[1], conv_strides[1],1],
                                padding = 'SAME') +
                            b_conv2),
                        depth_radius = 5, bias = 2, alpha = 10e-4, beta = 0.75),
                    ksize = [1,pool_kernels[1],pool_kernels[1],1], strides = [1,pool_strides[1],pool_strides[1],1],
                    padding = 'VALID')
            c3 = tf.nn.relu(
                    tf.nn.conv2d(c2, filter = W_conv3,
                        strides = [1,conv_strides[2], conv_strides[2],1],
                        padding = 'SAME') +
                    b_conv3)
            c4 = tf.nn.relu(
                    tf.nn.conv2d(c3, filter = W_conv4,
                        strides = [1,conv_strides[3], conv_strides[3],1],
                        padding = 'SAME') +
                    b_conv4)
            c5 = tf.nn.max_pool(
                    tf.nn.relu(
                        tf.nn.conv2d(c4, filter = W_conv5,
                            strides = [1,conv_strides[4], conv_strides[4],1],
                            padding = 'SAME') +
                        b_conv5),
                    ksize = [1, pool_kernels[2], pool_kernels[2],1], strides = [1,pool_strides[2],pool_strides[2],1],
                    padding = 'VALID')

        with tf.name_scope('Full_connections') :
            flatten = tf.contrib.layers.flatten(c5)
            fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1), keep_prob_hidden)
            fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2), keep_prob_hidden)
        with tf.name_scope('Softmax_Classification') :
            logits = tf.matmul(fc2, W_softmax) + b_softmax
        #print(c1, c2, c3, c4, c5, flatten, fc1, fc2, logits)
        return logits


    with tf.name_scope('Training') :
        logits = nn(train_images, kp)
        weighted_logits = tf.div(logits, label_weights)

    with tf.name_scope('BackProp') :
        train_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(weighted_logits, train_labels))

        # TODO : momentum decay

        learning_rate = tf.train.exponential_decay(init_rate, global_step = steps*batch_size, decay_steps = per_steps, decay_rate = decay_rate, staircase = False)
        training_op = tf.train.AdamOptimizer(learning_rate).minimize(train_cross_entropy, global_step = steps)


    with tf.name_scope('Validation') :
        with tf.name_scope('Validation_input') :
            val_q = tf.train.slice_input_producer([files_val, y_val], shuffle = False, capacity = valid_size)
            val_label = val_q[1]
            val_image = fd.decode_image(tf.read_file(val_q[0]), size = std_sizes, mutate = False, crop = 'centre', crop_size = crop_size)
            val_images, val_labels= tf.train.batch(
                                        [val_image, val_label],
                                        batch_size=batch_size,
                                        capacity = batch_size * 2)

        with tf.name_scope('Validation_Management') :
            batch_valid_logits = nn(val_images, 1.0)
            validation_logits, validation_labels = tf.train.batch(
                [batch_valid_logits, val_labels], batch_size = valid_size,
                enqueue_many = True, capacity = valid_size)

            validation_cross_entropy = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(validation_logits, validation_labels)
                                )

    with tf.name_scope('Summaries') :
            training_acc = tf.reduce_mean(
                            tf.cast(
                                tf.equal(
                                    tf.argmax(logits, 1),tf.argmax(train_labels,1)),
                            tf.float32))

            valid_acc = tf.reduce_mean(
                            tf.cast(
                                tf.equal(
                                    tf.argmax(validation_logits, 1), tf.argmax(validation_labels,1)),
                            tf.float32))

            tf.summary.scalar('Training_Cross_entropy', train_cross_entropy)
            tf.summary.scalar('Training_Accuracy', training_acc)
            tf.summary.scalar('Validation_Cross_entropy', validation_cross_entropy)
            tf.summary.scalar('Validation_Accuracy', valid_acc)
            tf.summary.scalar('Learning_rate', learning_rate)
            summaries = tf.summary.merge_all()


    with tf.name_scope('Test') :
        with tf.name_scope('Test_set_input') :
            test_q = tf.train.slice_input_producer([test_filenames], shuffle = False, capacity = len(test_filenames))
            test_image = fd.decode_image(tf.read_file(test_q[0]), size = std_sizes, mutate = False, crop = 'centre', crop_size = crop_size)
            test_images = tf.train.batch([test_image], batch_size = batch_size, capacity = batch_size * 2)

        with tf.name_scope('Test_Management') :
            batch_test_logits = nn(test_images, 1.0)
            test_logits = tf.nn.softmax(tf.train.batch([batch_test_logits], batch_size = len(test_filenames), enqueue_many = True, capacity = len(test_filenames)))
