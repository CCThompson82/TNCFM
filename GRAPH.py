"""This is the graph for the NCF Fish Classification Kaggle Competition"""



graph = tf.Graph()

with graph.as_default() :

    training_data = tf.placeholder(dtype = tf.float32, shape = (batch_size, std_y, std_x, num_channels))
    valid_data = tf.placeholder(dtype = tf.float32, shape = (batch_size, std_y, std_x, num_channels)) # called from global env

    training_labels = tf.placeholder(dtype = tf.int32, shape = (batch_size, num_labels))
    valid_labels = tf.placeholder(dtype = tf.int32, shape = (batch_size, num_labels)) # called from global env

    # Variables

    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[0], kernel_sizes[0], num_channels, conv_depths[0]], stddev = stddev))
            W_conv2 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[1], kernel_sizes[1], conv_depths[0], conv_depths[1]], stddev = stddev))
            W_conv3 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[2], kernel_sizes[2], conv_depths[1], conv_depths[2]], stddev = stddev))
            W_conv4 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[3], kernel_sizes[3], conv_depths[2], conv_depths[3]], stddev = stddev))
            W_conv5 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[4], kernel_sizes[4], conv_depths[3], conv_depths[4]], stddev = stddev))
            W_conv6 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[5], kernel_sizes[5], conv_depths[4], conv_depths[5]], stddev = stddev))
            W_conv7 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[6], kernel_sizes[6], conv_depths[5], conv_depths[6]], stddev = stddev))
            W_conv8 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[7], kernel_sizes[7], conv_depths[6], conv_depths[7]], stddev = stddev))
            W_conv9 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[8], kernel_sizes[8], conv_depths[7], conv_depths[8]], stddev = stddev))
            W_conv10 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[9], kernel_sizes[9], conv_depths[8], conv_depths[9]], stddev = stddev))
            W_conv11 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[10], kernel_sizes[10], conv_depths[9], conv_depths[10]], stddev = stddev))
            W_conv12 = tf.Variable(
                        tf.truncated_normal([kernel_sizes[11], kernel_sizes[11], conv_depths[10], conv_depths[11]], stddev = stddev))


        with tf.variable_scope('Fully_connected') :
            W_fc1 = tf.Variable(
                        tf.truncated_normal(
                            [fd.count_nodes(std_y, std_x, pool_steps = 7, final_depth = final_depth ), fc1_depth],
                            stddev = stddev ))
            b_fc1 = tf.Variable(tf.zeros([fc1_depth]))
            W_fc2 = tf.Variable(
                        tf.truncated_normal([fc1_depth, fc2_depth], stddev = stddev ))
            b_fc2 = tf.Variable(tf.zeros([fc2_depth]))


        with tf.variable_scope('Softmax') :
            W_softmax = tf.Variable(
                            tf.truncated_normal([fc2_depth, num_labels],
                            stddev = stddev))
            b_softmax = tf.Variable(tf.zeros([num_labels]))



    def nn(data, dropout = False) :

        c1 = tf.nn.conv2d(data, filter = W_conv1, strides = [1, stride, stride, 1], padding = 'SAME')
        c1_pool = tf.nn.max_pool(c1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        c2 = tf.nn.conv2d(c1_pool, filter = W_conv2, strides = [1, stride, stride, 1], padding = 'SAME')
        c2_pool = tf.nn.max_pool(c2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
        c3 = tf.nn.conv2d(c2_pool, filter = W_conv3, strides = [1, stride, stride, 1], padding = 'SAME')
        c4 = tf.nn.conv2d(c3, filter = W_conv4, strides = [1, stride, stride, 1], padding = 'SAME')
        c4_pool = tf.nn.max_pool(c4, ksize = [1,2,2,1], strides = [1, 2,2,1], padding = 'VALID')
        c5 = tf.nn.conv2d(c4_pool, filter = W_conv5, strides = [1, stride, stride, 1], padding = 'SAME')
        c6 = tf.nn.conv2d(c5, filter = W_conv6, strides = [1, stride, stride, 1], padding = 'SAME')
        c6_pool = tf.nn.max_pool(c6, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
        c7 = tf.nn.conv2d(c6_pool, filter = W_conv7, strides = [1,stride,stride,1], padding = 'SAME')
        c8 = tf.nn.conv2d(c7, filter = W_conv8, strides = [1, stride, stride, 1], padding = 'SAME')
        c8_pool = tf.nn.max_pool(c8, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
        c9 = tf.nn.conv2d(c8_pool, filter = W_conv9, strides = [1,stride,stride,1], padding = 'SAME')
        c10 = tf.nn.conv2d(c9, filter = W_conv10, strides = [1, stride, stride, 1], padding = 'SAME')
        c10_pool = tf.nn.max_pool(c10, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')
        c11 = tf.nn.conv2d(c10_pool, filter = W_conv11, strides = [1,stride,stride,1], padding = 'SAME')
        c12 = tf.nn.conv2d(c11, filter = W_conv12, strides = [1, stride, stride, 1], padding = 'SAME')
        c12_pool = tf.nn.max_pool(c12, ksize = [1,2,2,1], strides = [1,2,2,1], padding ='VALID')


        flatten = tf.contrib.layers.flatten(c12_pool)
        fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1)
        fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) +b_fc2)
        logits = tf.matmul(fc2, W_softmax) + b_softmax

        return logits

    with tf.name_scope('Training') :
        logits = nn(training_data, dropout = True)


        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, training_labels))
    #print(logits)

        training_op = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)

    # Validation
    with tf.name_scope('Validation') :
        valid_logits = nn(valid_data, valid_labels)
        valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, valid_labels))
