"""This is the graph for the NCF Fish Classification Kaggle Competition"""



graph = tf.Graph()

with graph.as_default() :
    with tf.name_scope('Training_input') :
        training_data = tf.placeholder(dtype = tf.float32, shape = (batch_size, std_y, std_x, num_channels))
        training_labels = tf.placeholder(dtype = tf.int32, shape = (batch_size, num_labels))
    with tf.name_scope('Valid_input') :
        valid_data = tf.placeholder(dtype = tf.float32, shape = (batch_size, std_y, std_x, num_channels))
        valid_labels = tf.placeholder(dtype = tf.int32, shape = (batch_size, num_labels))
    with tf.name_scope('keep_probabilities') :
        keep_prob_convs = tf.placeholder(dtype = tf.float32, shape = ())
        keep_prob_hidden = tf.placeholder(dtype = tf.float32, shape = ())

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
            W_conv7 = tf.Variable(tf.truncated_normal([kernel_sizes[6], kernel_sizes[6], conv_depths[5], conv_depths[6]], stddev = stddev))
            sw7 = tf.summary.histogram('W_conv7', W_conv7)
            W_conv8 = tf.Variable(tf.truncated_normal([kernel_sizes[7], kernel_sizes[7], conv_depths[6], conv_depths[7]], stddev = stddev))
            sw8 =tf.summary.histogram('W_conv8', W_conv8)
            W_conv9 = tf.Variable(tf.truncated_normal([kernel_sizes[8], kernel_sizes[8], conv_depths[7], conv_depths[8]], stddev = stddev))
            sw9 = tf.summary.histogram('W_conv9', W_conv9)
            W_conv10 = tf.Variable(tf.truncated_normal([kernel_sizes[9], kernel_sizes[9], conv_depths[8], conv_depths[9]], stddev = stddev))
            sw10 = tf.summary.histogram('W_conv10', W_conv10)
            W_conv11 = tf.Variable(tf.truncated_normal([kernel_sizes[10], kernel_sizes[10], conv_depths[9], conv_depths[10]], stddev = stddev))
            sw11 = tf.summary.histogram('W_conv11', W_conv11)
            W_conv12 = tf.Variable(tf.truncated_normal([kernel_sizes[11], kernel_sizes[11], conv_depths[10], conv_depths[11]], stddev = stddev))
            sw12 = tf.summary.histogram('W_conv12', W_conv12)


        with tf.variable_scope('Fully_connected') :
            W_fc1 = tf.Variable(tf.truncated_normal(
                            [fd.count_nodes(std_y, std_x, pool_steps = 6, final_depth = final_depth ), fc1_depth],
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


    def nn(data, keep_prob_convs, keep_prob_hidden) :
        with tf.name_scope('Convolution') :
            c1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(data, filter = W_conv1, strides = [1, stride, stride, 1], padding = 'SAME')), keep_prob_convs)
            c2 = tf.nn.relu(tf.nn.conv2d(c1, filter = W_conv2, strides = [1, stride, stride, 1], padding = 'SAME'))
            c2_pool = tf.nn.max_pool(c2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
            c3 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(c2_pool, filter = W_conv3, strides = [1, stride, stride, 1], padding = 'SAME')), keep_prob_convs)
            c4 = tf.nn.relu(tf.nn.conv2d(c3, filter = W_conv4, strides = [1, stride, stride, 1], padding = 'SAME'))
            c4_pool = tf.nn.max_pool(c4, ksize = [1,2,2,1], strides = [1, 2,2,1], padding = 'VALID')
            c5 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(c4_pool, filter = W_conv5, strides = [1, stride, stride, 1], padding = 'SAME')), keep_prob_convs)
            c6 = tf.nn.relu(tf.nn.conv2d(c5, filter = W_conv6, strides = [1, stride, stride, 1], padding = 'SAME'))
            c6_pool = tf.nn.max_pool(c6, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
            c7 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(c6_pool, filter = W_conv7, strides = [1,stride,stride,1], padding = 'SAME')), keep_prob_convs)
            c8 = tf.nn.relu(tf.nn.conv2d(c7, filter = W_conv8, strides = [1, stride, stride, 1], padding = 'SAME'))
            c8_pool = tf.nn.max_pool(c8, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
            c9 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(c8_pool, filter = W_conv9, strides = [1,stride,stride,1], padding = 'SAME')), keep_prob_convs)
            c10 = tf.nn.relu(tf.nn.conv2d(c9, filter = W_conv10, strides = [1, stride, stride, 1], padding = 'SAME'))
            c10_pool = tf.nn.max_pool(c10, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
            c11 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(c10_pool, filter = W_conv11, strides = [1,stride,stride,1], padding = 'SAME')), keep_prob_convs)
            c12 = tf.nn.relu(tf.nn.conv2d(c11, filter = W_conv12, strides = [1, stride, stride, 1], padding = 'SAME'))
            c12_pool = tf.nn.max_pool(c12, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
        with tf.name_scope('Full_connections') :
            flatten = tf.nn.dropout(tf.contrib.layers.flatten(c12_pool), keep_prob_hidden)
            fc1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1), keep_prob_hidden)
            fc2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc1, W_fc2) +b_fc2), keep_prob_hidden)
        with tf.name_scope('Softmax_Classification') :
            logits = tf.matmul(fc2, W_softmax) + b_softmax

        return logits

    with tf.name_scope('Training') :
        logits = nn(training_data, keep_prob_convs, keep_prob_hidden)

    with tf.name_scope('BackProp') :
        training_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, training_labels))
        learning_rate = tf.train.exponential_decay(init_rate, global_step = steps*batch_size, decay_steps = per_steps, decay_rate = 0.5, staircase = True)
        training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_loss, global_step = steps)

    with tf.name_scope('Validation') :
        valid_logits = nn(valid_data, keep_prob_convs, keep_prob_hidden)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, valid_labels))


    with tf.name_scope('Summaries') :
            training_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(training_labels,1)), tf.float32))

            sc = tf.summary.scalar('Training_Cross_entropy', training_loss)
            slr = tf.summary.scalar('Learning_rate', learning_rate)
            sa = tf.summary.scalar('Training_Accuracy', training_acc)
            train_summaries = tf.summary.merge([sw1, sw2, sw3, sw4, sw5, sw6, sw7, sw8, sw9, sw10, sw11, sw12, sf1, sf2, sb1, sb2, swsm, sbsm, sc, slr, sa])

            valid_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(valid_logits, 1), tf.argmax(valid_labels,1)), tf.float32))
            vl = tf.summary.scalar('Validation_Cross_entropy', valid_loss)
            va = tf.summary.scalar('Validation_Accuracy', valid_acc)
            valid_summaries = tf.summary.merge([vl,va])
