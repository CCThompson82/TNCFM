"""This is the graph for the NCF Fish Classification Kaggle Competition"""



fish_finder = tf.Graph()

with fish_finder.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            with tf.name_scope('Convolution_1') :
                W_conv1 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, num_channels, conv_depth[0]], stddev = stddev))
                b_conv1 = tf.Variable(tf.zeros([conv_depths[0]))
                tf.summary.histogram('W_conv1', W_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            with tf.name_scope('Convolution_2') :
                W_conv2 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[0], conv_depth[1]], stddev = stddev))
                b_conv2 = tf.Variable(tf.zeros([conv_depths[1]))
                tf.summary.histogram('W_conv2', W_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
            with tf.name_scope('Convolution_3') :
                W_conv3 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[1], conv_depth[2]], stddev = stddev))
                b_conv3 = tf.Variable(tf.zeros([conv_depths[2]))
                tf.summary.histogram('W_conv3', W_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
            with tf.name_scope('Convolution_4') :
                W_conv4 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[2], conv_depth[3]], stddev = stddev))
                b_conv4 = tf.Variable(tf.zeros([conv_depths[3]))
                tf.summary.histogram('W_conv4', W_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
            with tf.name_scope('Convolution_5') :
                W_conv5 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[3], conv_depth[4]], stddev = stddev))
                b_conv5 = tf.Variable(tf.zeros([conv_depths[4]))
                tf.summary.histogram('W_conv5', W_conv5)
                tf.summary.histogram('b_conv5', b_conv5)
            with tf.name_scope('Convolution_6') :
                W_conv6 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[4], conv_depth[5]], stddev = stddev))
                b_conv6 = tf.Variable(tf.zeros([conv_depths[5]))
                tf.summary.histogram('W_conv6', W_conv6)
                tf.summary.histogram('b_conv6', b_conv6)
            with tf.name_scope('Convolution_7') :
                W_conv7 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[5], conv_depth[6]], stddev = stddev))
                b_conv7 = tf.Variable(tf.zeros([conv_depths[6]))
                tf.summary.histogram('W_conv7', W_conv7)
                tf.summary.histogram('b_conv7', b_conv7)
            with tf.name_scope('Convolution_8') :
                W_conv8 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[6], conv_depth[7]], stddev = stddev))
                b_conv8 = tf.Variable(tf.zeros([conv_depths[7]))
                tf.summary.histogram('W_conv8', W_conv8)
                tf.summary.histogram('b_conv8', b_conv8)
            with tf.name_scope('Convolution_9') :
                W_conv9 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[7], conv_depth[8]], stddev = stddev))
                b_conv9 = tf.Variable(tf.zeros([conv_depths[8]))
                tf.summary.histogram('W_conv9', W_conv9)
                tf.summary.histogram('b_conv9', b_conv9)
            with tf.name_scope('Convolution_10') :
                W_conv10 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[8], conv_depth[9]], stddev = stddev))
                b_conv10 = tf.Variable(tf.zeros([conv_depths[9])) 
                tf.summary.histogram('W_conv10', W_conv10)
                tf.summary.histogram('b_conv10', b_conv10)
        with tf.variable_scope('Dense_layers') :
            with tf.name_scope('dense_11') :
                W_11 = tf.Variable(tf.truncated_normal([nodes_after_conv, fc_depth[0]], stddev = stddev ))
                b_11 = tf.Variable(tf.zeros([fc_depth[0]]))
                tf.summary.histogram('W_11', W_11)
                tf.summary.histogram('b_11', b_11)
            with tf.name_scope('dense12') :
                W_12 = tf.Variable(tf.truncated_normal([fc_depth[1], fc_depth[2]], stddev = stddev ))
                b_12 = tf.Variable(tf.zeros([fc_depth[2]]))
                tf.summary.histogram('W_12', W_12)
                tf.summary.histogram('b_12', b_12)
            with tf.name_scope('dense13') :
                W_13 = tf.Variable(tf.truncated_normal([fc_depth[2], fc_depth[3]], stddev = stddev ))
                b_13 = tf.Variable(tf.zeros([fc_depth[3]]))
                tf.summary.histogram('W_13', W_13)
                tf.summary.histogram('b_13', b_13)
        with tf.variable_scope('Classifier') :
            W_clf = tf.Variable(tf.truncated_normal([fc_depth[3], num_labels], stddev = stddev))
            b_clf = tf.Variable(tf.zeros([num_labels]))
            tf.summary.histogram('W_clf', W_clf)
            tf.summary.histogram('b_clf', b_clf)



    def convolutions(data) :
        """ Convenience function to iterate through several rounds of convolution"""
        with tf.name_scope('Convolution') :
            def conv_conv_pool(data, W1, b1, W2, b2) :
                """ Convenience function for iteration of conv-conv-pool
                network architecture"""
                i1 = tf.nn.relu(
                        tf.nn.conv2d(data, filter = W1,
                            strides = [1, conv_stride, conv_stride, 1],
                            padding = 'SAME') + b1)
                i2 = f.nn.max_pool(
                        tf.nn.relu(
                            tf.nn.conv2d(i1, filter = W2,
                                strides = [1, conv_stride, conv_stride, 1],
                                padding = 'SAME') + b2),
                        ksize = [1, pool_kernel, pool_kernel,1],
                        strides = [1, pool_stride, pool_stride, 1],
                        padding ='VALID')
                return i1, i2

            c1, c2 = conv_conv_pool(data, W_conv1, b_conv1, W_conv2, b_conv2)
            c3, c4 = conv_conv_pool(c2, W_conv3, b_conv3, W_conv4, b_conv4)
            c5, c6 = conv_conv_pool(c4, W_conv5, b_conv5, W_conv6, b_conv6)
            c7, c8 = conv_conv_pool(c6, W_conv7, b_conv7, W_conv8, b_conv8)
            c9, c10 = conv_conv_pool(c8, W_conv9, b_conv9, W_conv10, b_conv10)
        return c10

    def dense_layers(data, drop_prob) :
        """Executes a series of dense layers"""
        def fc(data, W, b) :
            """Convenience function for dense layer with dropout"""
            fc = tf.nn.dropout(
                    tf.nn.relu(
                        tf.matmul(data, W) + b),
                    drop_prob)
            return fc

        d11 = fc(data, W11, b11)
        d12 = fc(d11, W12, b12)
        d13 = fc(d12, W13, b13)
        return d13




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :
            # TODO : Write session for this training set input
            # X
            # y
            # learning_rate
        with tf.name_scope('Network') :
            conv_output = convolutions(X)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input)
        with tf.name_scope('Classifier')
            logits = tf.matmul(dense_output, W_clf) + b_clf
        with tf.name_scope('Backpropigation')
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
            train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)


    with tf.name_scope('Summaries') :
        tf.summary.scalar('Cross_entropy', cross_entropy)
        tf.summary.scalar('Learning_rate', learning_rate)
        summaries = tf.summary.merge_all()
