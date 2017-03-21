"""This is the graph for the NCF Fish Classification Kaggle Competition"""



fish_finder = tf.Graph()

with fish_finder.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            with tf.name_scope('Convolution_0') :
                W_conv0 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, num_channels, conv_depth[-1]], stddev = stddev))
                b_conv0 = tf.Variable(tf.zeros([conv_depth[-1]]))
                tf.summary.histogram('W_conv0', W_conv0)
                tf.summary.histogram('b_conv0', b_conv0)
            with tf.name_scope('Convolution_1') :
                W_conv1 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[-1], conv_depth[0]], stddev = stddev))
                b_conv1 = tf.Variable(tf.zeros([conv_depth[0]]))
                tf.summary.histogram('W_conv1', W_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            with tf.name_scope('Convolution_2') :
                W_conv2 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[0], conv_depth[1]], stddev = stddev))
                b_conv2 = tf.Variable(tf.zeros([conv_depth[1]]))
                tf.summary.histogram('W_conv2', W_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
            with tf.name_scope('Convolution_3') :
                W_conv3 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[1], conv_depth[2]], stddev = stddev))
                b_conv3 = tf.Variable(tf.zeros([conv_depth[2]]))
                tf.summary.histogram('W_conv3', W_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
            with tf.name_scope('Convolution_4') :
                W_conv4 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[2], conv_depth[3]], stddev = stddev))
                b_conv4 = tf.Variable(tf.zeros([conv_depth[3]]))
                tf.summary.histogram('W_conv4', W_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
            with tf.name_scope('Convolution_5') :
                W_conv5 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[3], conv_depth[4]], stddev = stddev))
                b_conv5 = tf.Variable(tf.zeros([conv_depth[4]]))
                tf.summary.histogram('W_conv5', W_conv5)
                tf.summary.histogram('b_conv5', b_conv5)
            with tf.name_scope('Convolution_6') :
                W_conv6 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[4], conv_depth[5]], stddev = stddev))
                b_conv6 = tf.Variable(tf.zeros([conv_depth[5]]))
                tf.summary.histogram('W_conv6', W_conv6)
                tf.summary.histogram('b_conv6', b_conv6)
            with tf.name_scope('Convolution_7') :
                W_conv7 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[5], conv_depth[6]], stddev = stddev))
                b_conv7 = tf.Variable(tf.zeros([conv_depth[6]]))
                tf.summary.histogram('W_conv7', W_conv7)
                tf.summary.histogram('b_conv7', b_conv7)
            with tf.name_scope('Convolution_8') :
                W_conv8 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[6], conv_depth[7]], stddev = stddev))
                b_conv8 = tf.Variable(tf.zeros([conv_depth[7]]))
                tf.summary.histogram('W_conv8', W_conv8)
                tf.summary.histogram('b_conv8', b_conv8)
            with tf.name_scope('Convolution_9') :
                W_conv9 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[7], conv_depth[8]], stddev = stddev))
                b_conv9 = tf.Variable(tf.zeros([conv_depth[8]]))
                tf.summary.histogram('W_conv9', W_conv9)
                tf.summary.histogram('b_conv9', b_conv9)
            with tf.name_scope('Convolution_10') :
                W_conv10 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[8], conv_depth[9]], stddev = stddev))
                b_conv10 = tf.Variable(tf.zeros([conv_depth[9]]))
                tf.summary.histogram('W_conv10', W_conv10)
                tf.summary.histogram('b_conv10', b_conv10)
        with tf.variable_scope('Dense_layers') :
            with tf.name_scope('dense_11') :
                W_11 = tf.Variable(tf.truncated_normal([nodes_after_conv, fc_depth[0]], stddev = stddev ))
                b_11 = tf.Variable(tf.zeros([fc_depth[0]]))
                tf.summary.histogram('W_11', W_11)
                tf.summary.histogram('b_11', b_11)
            with tf.name_scope('dense12') :
                W_12 = tf.Variable(tf.truncated_normal([fc_depth[0], fc_depth[1]], stddev = stddev ))
                b_12 = tf.Variable(tf.zeros([fc_depth[1]]))
                tf.summary.histogram('W_12', W_12)
                tf.summary.histogram('b_12', b_12)
            with tf.name_scope('dense13') :
                W_13 = tf.Variable(tf.truncated_normal([fc_depth[1], fc_depth[2]], stddev = stddev ))
                b_13 = tf.Variable(tf.zeros([fc_depth[2]]))
                tf.summary.histogram('W_13', W_13)
                tf.summary.histogram('b_13', b_13)
            with tf.name_scope('dense14') :
                W_14 = tf.Variable(tf.truncated_normal([fc_depth[2], fc_depth[3]], stddev = stddev ))
                b_14 = tf.Variable(tf.zeros([fc_depth[3]]))
                tf.summary.histogram('W_14', W_14)
                tf.summary.histogram('b_14', b_14)
        with tf.variable_scope('Classifier') :
            W_clf = tf.Variable(tf.truncated_normal([fc_depth[3], num_labels], stddev = stddev))
            b_clf = tf.Variable(tf.zeros([num_labels]))
            tf.summary.histogram('W_clf', W_clf)
            tf.summary.histogram('b_clf', b_clf)



    def convolutions(data) :
        """
        Performs the convolution steps of FISHFINDER model.
        """
        with tf.name_scope('Convolution') :
            def conv_conv_pool(data, W1, b1, W2, b2, conv_stride, pool_stride) :
                """
                Convenience function for iteration of conv-conv-pool
                network architecture
                """
                i1 = tf.nn.relu(
                        tf.nn.conv2d(data, filter = W1,
                            strides = [1, conv_stride, conv_stride, 1],
                            padding = 'SAME') + b1)
                i2 = tf.nn.max_pool(
                        tf.nn.relu(
                            tf.nn.conv2d(i1, filter = W2,
                                strides = [1, conv_stride, conv_stride, 1],
                                padding = 'SAME') + b2),
                        ksize = [1, pool_kernel, pool_kernel,1],
                        strides = [1, pool_stride, pool_stride, 1],
                        padding ='VALID')
                return i1, i2

            c0 = tf.nn.relu(
                    tf.nn.conv2d(data, filter = W_conv0,
                        strides = [1, 2, 2, 1],
                        padding = 'SAME') + b_conv0)
            c1, c2 = conv_conv_pool(c0, W_conv1, b_conv1, W_conv2, b_conv2, conv_stride, pool_stride[0])
            c3, c4 = conv_conv_pool(c2, W_conv3, b_conv3, W_conv4, b_conv4, conv_stride, pool_stride[0])
            c5, c6 = conv_conv_pool(c4, W_conv5, b_conv5, W_conv6, b_conv6, conv_stride, pool_stride[0])
            c7, c8 = conv_conv_pool(c6, W_conv7, b_conv7, W_conv8, b_conv8, conv_stride, pool_stride[0])
            c9, c10 = conv_conv_pool(c8, W_conv9, b_conv9, W_conv10, b_conv10, conv_stride, pool_stride[1])

        return c10

    def dense_layers(data, drop_prob) :
        """
        Executes a series of dense layers.
        """
        def fc(data, W, b) :
            """Convenience function for dense layer with dropout"""
            fc = tf.nn.dropout(
                    tf.nn.relu(
                        tf.matmul(data, W) + b),
                    drop_prob)
            return fc

        d11 = fc(data, W_11, b_11)
        d12 = fc(d11, W_12, b_12)
        d13 = fc(d12, W_13, b_13)
        d14 = fc(d13, W_14, b_14)
        return d14




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :
            train_images = tf.placeholder(tf.float32, shape = [batch_size, fov_size, fov_size, num_channels])
            train_labels = tf.placeholder(tf.float32, shape = [batch_size, num_labels])
            learning_rate = tf.placeholder(tf.float32, shape = () )
            beta_weights = tf.placeholder(tf.float32, shape = () )
            frequency_weights = tf.placeholder(tf.float32, shape = [1, num_labels])
        with tf.name_scope('Network') :
            conv_output = convolutions(train_images)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input, drop_prob = drop_prob)
        with tf.name_scope('Classifier') :
            logits = tf.matmul(dense_output, W_clf) + b_clf
        with tf.name_scope('Backpropigation') :
            weight_per_label = beta_weights*tf.transpose(tf.matmul(train_labels, tf.transpose(frequency_weights)))
            xent = tf.nn.softmax_cross_entropy_with_logits(
                        logits = logits, labels = train_labels)
            cross_entropy = tf.reduce_mean(xent)
            cost = cross_entropy + tf.reduce_mean(tf.mul(weight_per_label, xent)) # penalty weights for unbalanced data - penalizes for missing underrepresented labels to force model to pay attention to them

            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('Staged_prediction') :
        with tf.name_scope('Input') :
            staged_set = tf.placeholder(tf.float32, shape = [batch_size, fov_size, fov_size, num_channels])
        with tf.name_scope('Network') :
            staged_conv_output = convolutions(staged_set)
            staged_dense_input = tf.contrib.layers.flatten(staged_conv_output)
            staged_dense_output = dense_layers(staged_dense_input, drop_prob = 1.0)
        with tf.name_scope('Prediction') :
            staged_logits = tf.nn.softmax(tf.matmul(staged_dense_output, W_clf) + b_clf)


    with tf.name_scope('Summaries') :
        tf.summary.scalar('Cross_entropy', cross_entropy)
        tf.summary.scalar('Cost', cost)
        tf.summary.scalar('beta_weights', beta_weights)
        tf.summary.scalar('Learning_rate', learning_rate)
        summaries = tf.summary.merge_all()


    with tf.name_scope('Prediction') :
        with tf.name_scope('Input') :
            img_stack = tf.placeholder(tf.float32, shape = [bins_y*bins_x, fov_size, fov_size, num_channels])
        with tf.name_scope('Network') :
            stack_conv_output = convolutions(img_stack)
            stack_dense_input = tf.contrib.layers.flatten(stack_conv_output)
            stack_dense_output = dense_layers(stack_dense_input, drop_prob = 1.0)
        with tf.name_scope('Classifier') :
            stack_prediction = tf.nn.softmax(tf.matmul(dense_output, W_clf) + b_clf)
