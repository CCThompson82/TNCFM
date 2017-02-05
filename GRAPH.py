"""This is the graph for the NCF Fish Classification Kaggle Competition"""



graph = tf.Graph()

with graph.as_default() :
    with tf.name_scope('Training_input') :

        train_q = tf.train.slice_input_producer([files_train, y_train], shuffle = False)
        train_label = train_q[1]
        train_image = fd.decode_image(tf.read_file(train_q[0]), size = [std_y, std_x], mutate = False)

        train_images, train_labels= tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=batch_size
                                    #,num_threads=1
                                    )





    with tf.name_scope('Valid_input') :
        val_q = tf.train.slice_input_producer([files_val, y_val], shuffle = False)
        val_label = val_q[1]
        val_image = fd.decode_image(tf.read_file(val_q[0]), size = [std_y, std_x], mutate = False)
        val_images, val_labels= tf.train.batch(
                                    [val_image, val_label],
                                    batch_size=batch_size
                                    #,num_threads=1
                                    )



    # Variables

    with tf.variable_scope('Variables') :
        with tf.name_scope('Batch_step') :
            steps = tf.Variable(0, trainable = False)
        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(tf.truncated_normal([kernel_sizes[0], kernel_sizes[0], num_channels, conv_depths[0]], stddev = stddev))
            sw1 = tf.summary.histogram('W_conv1', W_conv1)
            W_conv2 = tf.Variable(tf.truncated_normal([kernel_sizes[1], kernel_sizes[1], conv_depths[0], conv_depths[1]], stddev = stddev))
            sw2 = tf.summary.histogram('W_conv2', W_conv2)
            W_conv3 = tf.Variable(tf.truncated_normal([kernel_sizes[2], kernel_sizes[2], conv_depths[1], conv_depths[2]], stddev = stddev))
            sw3 = tf.summary.histogram('W_conv3', W_conv3)
            W_conv4 = tf.Variable(tf.truncated_normal([kernel_sizes[3], kernel_sizes[3], conv_depths[2], conv_depths[3]], stddev = stddev))
            sw4 = tf.summary.histogram('W_conv4', W_conv4)
            W_conv5 = tf.Variable(tf.truncated_normal([kernel_sizes[4], kernel_sizes[4], conv_depths[3], conv_depths[4]], stddev = stddev))
            sw5 = tf.summary.histogram('W_conv5', W_conv5)
            W_conv6 = tf.Variable(tf.truncated_normal([kernel_sizes[5], kernel_sizes[5], conv_depths[4], conv_depths[5]], stddev = stddev))
            sw6 = tf.summary.histogram('W_conv6', W_conv6)

        with tf.variable_scope('Fully_connected') :
            W_fc1 = tf.Variable(tf.truncated_normal(
                            [fd.count_nodes(std_y, std_x, pool_steps = 4, final_depth = final_depth ), fc1_depth],
                            stddev = stddev ))
            sf1 = tf.summary.histogram('W_fc1', W_fc1)
            b_fc1 = tf.Variable(tf.zeros([fc1_depth]))
            sb1 = tf.summary.histogram('b_fc1', b_fc1)
            W_fc2 = tf.Variable(tf.truncated_normal([fc1_depth, fc2_depth], stddev = stddev ))
            sf2 = tf.summary.histogram('W_fc2', W_fc2)
            b_fc2 = tf.Variable(tf.zeros([fc2_depth]))
            sb2 = tf.summary.histogram('b_fc2', b_fc2)


        with tf.variable_scope('Softmax') :
            W_softmax = tf.Variable(tf.truncated_normal([fc2_depth, num_labels], stddev = stddev))
            swsm = tf.summary.histogram('W_softmax', W_softmax)
            b_softmax = tf.Variable(tf.zeros([num_labels]))
            sbsm = tf.summary.histogram('b_softmax', b_softmax)


    def nn(data, keep_prob_hidden) :
        with tf.name_scope('Convolution') :
            c1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(data, filter = W_conv1, strides = [1, 3, 3, 1], padding = 'SAME')),ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
            c2 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(c1, filter = W_conv2, strides = [1,1,1,1], padding = 'SAME')), ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
            c3 = tf.nn.relu(tf.nn.conv2d(c2, filter = W_conv3, strides = [1,1,1,1], padding = 'SAME'))
            c4 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(c3, filter = W_conv4, strides = [1,1,1,1], padding = 'SAME')), ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
            c5 = tf.nn.relu(tf.nn.conv2d(c4, filter = W_conv5, strides = [1,1,1,1], padding = 'SAME'))
            c6 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(c5, filter = W_conv6, strides = [1,1,1,1], padding = 'SAME')), ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

        with tf.name_scope('Full_connections') :
            flatten = tf.nn.dropout(tf.contrib.layers.flatten(c6), keep_prob_hidden)
            fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1), keep_prob_hidden)
            fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, W_fc2) +b_fc2), keep_prob_hidden)
        with tf.name_scope('Softmax_Classification') :
            logits = tf.matmul(fc2, W_softmax) + b_softmax

        return logits

    def regularize_weights() :
        return (beta * (
                tf.nn.l2_loss(W_conv1) +
                tf.nn.l2_loss(W_conv2) +
                tf.nn.l2_loss(W_conv3) +
                tf.nn.l2_loss(W_conv4) +
                tf.nn.l2_loss(W_conv5) +
                tf.nn.l2_loss(W_conv6) +
                tf.nn.l2_loss(W_fc1) +
                tf.nn.l2_loss(W_fc2) +
                tf.nn.l2_loss(W_softmax)))

    with tf.name_scope('Training') :
        logits = nn(train_images, kp)

    with tf.name_scope('BackProp') :
        train_cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, train_labels))
        training_loss = train_cross_entropy + regularize_weights()
        #learning_rate = tf.train.exponential_decay(init_rate, global_step = steps*batch_size, decay_steps = per_steps, decay_rate = decay_rate, staircase = True)
        training_op = tf.train.AdagradOptimizer(init_rate).minimize(training_loss, global_step = steps)


    with tf.name_scope('Validation') :
        valid_logits = nn(val_images, 1.0)
        valid_cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(valid_logits, val_labels))
        valid_loss = valid_cross_entropy + regularize_weights()


    """
    with tf.name_scope('Summaries') :
            training_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(training_labels,1)), tf.float32))

            sc = tf.summary.scalar('Training_Cross_entropy', train_cross_entropy)
            sa = tf.summary.scalar('Training_Accuracy', training_acc)
            sl = tf.summary.scalar('Training_Loss', training_loss)

            valid_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(valid_logits, 1), tf.argmax(valid_labels,1)), tf.float32))
            vc = tf.summary.scalar('Validation_Cross_entropy', valid_cross_entropy)
            va = tf.summary.scalar('Validation_Accuracy', valid_acc)
            vl = tf.summary.scalar('Validation_Loss', valid_loss)

            summaries = tf.summary.merge_all()
    """
