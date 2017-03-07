"""This is the graph for the NCF Fish Classification Kaggle Competition"""



fish_finder = tf.Graph()

with fish_finder.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            with tf.name_scope('Convolution_1') :
                W_conv1 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, num_channels, conv_depth[0]], stddev = stddev))
                b_conv1 = tf.Variable(tf.zeros([conv_depths[0]]))
                tf.summary.histogram('W_conv1', W_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            with tf.name_scope('Convolution_2') :
                W_conv2 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[0], conv_depth[1]], stddev = stddev))
                b_conv2 = tf.Variable(tf.zeros([conv_depths[1]]))
                tf.summary.histogram('W_conv2', W_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
            with tf.name_scope('Convolution_3') :
                W_conv3 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depth[1], conv_depth[2]], stddev = stddev))
                b_conv3 = tf.Variable(tf.zeros([conv_depths[2]]))
                tf.summary.histogram('W_conv3', W_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
            with tf.name_scope('Convolution_4') :
                W_conv4 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[2], conv_depth[3]], stddev = stddev))
                b_conv4 = tf.Variable(tf.zeros([conv_depths[3]]))
                tf.summary.histogram('W_conv4', W_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
            with tf.name_scope('Convolution_5') :
                W_conv5 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[3], conv_depth[4]], stddev = stddev))
                b_conv5 = tf.Variable(tf.zeros([conv_depths[4]]))
                tf.summary.histogram('W_conv5', W_conv5)
                tf.summary.histogram('b_conv5', b_conv5)
            with tf.name_scope('Convolution_6') :
                W_conv6 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[4], conv_depth[5]], stddev = stddev))
                b_conv6 = tf.Variable(tf.zeros([conv_depths[5]]))
                tf.summary.histogram('W_conv6', W_conv6)
                tf.summary.histogram('b_conv6', b_conv6)
            with tf.name_scope('Convolution_7') :
                W_conv7 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[5], conv_depth[6]], stddev = stddev))
                b_conv7 = tf.Variable(tf.zeros([conv_depths[6]]))
                tf.summary.histogram('W_conv7', W_conv7)
                tf.summary.histogram('b_conv7', b_conv7)
            with tf.name_scope('Convolution_8') :
                W_conv8 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[6], conv_depth[7]], stddev = stddev))
                b_conv8 = tf.Variable(tf.zeros([conv_depths[7]]))
                tf.summary.histogram('W_conv8', W_conv8)
                tf.summary.histogram('b_conv8', b_conv8)
            with tf.name_scope('Convolution_9') :
                W_conv9 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[7], conv_depth[8]], stddev = stddev))
                b_conv9 = tf.Variable(tf.zeros([conv_depths[8]]))
                tf.summary.histogram('W_conv9', W_conv9)
                tf.summary.histogram('b_conv9', b_conv9)
            with tf.name_scope('Convolution_10') :
                W_conv10 = tf.Variable(tf.truncated_normal([conv_kernel, conv_kernel, conv_depths[8], conv_depth[9]], stddev = stddev))
                b_conv10 = tf.Variable(tf.zeros([conv_depths[9]]))
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
        with tf.variable_scope('Classifier') :
            W_clf = tf.Variable(tf.truncated_normal([fc_depth[2], num_labels], stddev = stddev))
            b_clf = tf.Variable(tf.zeros([num_labels]))
            tf.summary.histogram('W_clf', W_clf)
            tf.summary.histogram('b_clf', b_clf)



    def convolutions(data) :
        """ Function to iterate through several rounds of convolution.  An input
        image of size 224x224x3 will have the following tensor sizes,
        assuming conv_depths of [16, 16, 64, 64, 128, 128, 256, 256, 512, 512]:
            * data =    224x224x3  = 150528 ->> (strided) ->> 112x112x3
            * c1 =      112x112x32 = 401408
            * c2 =      112x112x32 = 401408 ->> (pooled) ->> 56x56x32   = 100352
            * c3 =      56x56x64   = 200704
            * c4 =      56x56x64   = 200704 ->> (pooled) ->> 27x27x64   =  46656
            * c5 =      27x27x128  =  93312
            * c6 =      27x27x128  =  93312 ->> (pooled) ->> 13x13x128  =  21632
            * c7 =      13x13x256  =  43264
            * c8 =      13x13x256  =  43264 ->> (pooled) ->> 7x7x256   =   12544
            * c9 =      7x7x512    =  25088
            * c10 =     7x7x512    =  25088 --> (pooled) ->> 3x3x512    =   4608

        """
        with tf.name_scope('Convolution') :
            def conv_conv_pool(data, W1, b1, W2, b2, conv_stride) :
                """ Convenience function for iteration of conv-conv-pool
                network architecture"""
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

            c1, c2 = conv_conv_pool(data, W_conv1, b_conv1, W_conv2, b_conv2, conv_stride)
            c3, c4 = conv_conv_pool(c2, W_conv3, b_conv3, W_conv4, b_conv4, conv_stride)
            c5, c6 = conv_conv_pool(c4, W_conv5, b_conv5, W_conv6, b_conv6, conv_stride)
            c7, c8 = conv_conv_pool(c6, W_conv7, b_conv7, W_conv8, b_conv8, conv_stride)
            c9, c10 = conv_conv_pool(c8, W_conv9, b_conv9, W_conv10, b_conv10, conv_stride)
        return c10

    def dense_layers(data, drop_prob) :
        """Executes a series of dense layers.  Tensor sizes :
            * fc11 =    4608 -> 1024
            * fc12 =    1024 ->  512
            * fc13 =    512  -> 256
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
        return d13




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :
            train_images = tf.placeholder(tf.float32, shape = [batch_size, fov_size, fov_size, num_channels])
            train_labels = tf.placeholder(tf.int32, shape = [batch_size, num_labels])
            learning_rate = tf.placeholder(tf.float32, shape = () )
        with tf.name_scope('Network') :
            conv_output = convolutions(train_images)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input, drop_prob = drop_prob)
        with tf.name_scope('Classifier') :
            logits = tf.matmul(dense_output, W_clf) + b_clf
        with tf.name_scope('Backpropigation') :
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = train_labels))
            train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)

    with tf.name_scope('Staged_prediction') :
        with tf.name_scope('Input') :
            staged_set = tf.placeholder(tf.float32, shape = [batch_size, fov_size, fov_size, num_channels])
        with tf.name_scope('Network') :
            staged_conv_output = convolutions(staged_set)
            staged_dense_input = tf.contrib.layers.flatten(staged_conv_output)
            staged_dense_output = dense_layers(staged_dense_input, drop_prob = 1.0)
        with tf.name_scope('Prediction') :
            staged_logits = tf.matmul(staged_dense_output, W_clf) + b_clf


    with tf.name_scope('Summaries') :
        tf.summary.scalar('Cross_entropy', cross_entropy)
        tf.summary.scalar('Learning_rate', learning_rate)
        summaries = tf.summary.merge_all()
