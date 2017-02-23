"""This is a file encoding the Graph for fovea autoencoding."""

autoencoder_depths = [32]
autoencoder_kernels = [5]
autoencoder_strides = [2]

graph = tf.Graph()

with graph.as_default() :
    # Variables
    with tf.variable_scope('Variables') :

        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(tf.truncated_normal([autoencoder_kernels[0], autoencoder_kernels[0], num_channels, autoencoder_depths[0]], stddev = stddev))
            tf.summary.histogram('W_conv1', W_conv1)
            b_conv1 = tf.Variable(tf.zeros([autoencoder_depths[0]])) # experiment with this value
            tf.summary.histogram('b_conv1', b_conv1)

        with tf.variable_scope('Transpose_Convolution') :
            W_deconv1 = tf.Variable(tf.truncated_normal([autoencoder_kernels[0], autoencoder_kernels[0], num_channels, autoencoder_depths[0], stddev = stddev))
            tf.summary.histogram('W_deconv1', W_deconv1)
            b_deconv1 = tf.Variable(tf.zeros([num_channels]))
            tf.summary.histogram('b_deconv1', b_deconv1)


    def encoder(data) :
        with tf.name_scope('Encoder') :
            e1 =  tf.nn.conv2d(data, filter = W_conv1,
                            strides = [1, autoencoder_strides[0], autoencoder_strides[0], 1],
                            padding = 'SAME') + b_conv1
        return e1

    def decoder(data) :
        with tf.name_scope('Decoder') :
            d1 = tf.sigmoid(
                    tf.nn.conv2d_transpose(
                        data,
                        filter = W_deconv1,
                        output_shape = [batch_size, fovea_size, fovea_size, num_channels],
                        strides = [1, autoencoder_strides[0], autoencoder_strides[0], 1],
                        padding = 'SAME')
                    + b_deconv1)
        return d1

    with tf.name_scope('Input_fovea') :
        alpha = tf.placeholder(tf.float32, shape = ())
        batch_fovea = tf.placeholder(tf.float32, shape = [batch_size, fovea_size, fovea_size, num_channels])

    with tf.name_scope('Encoding') :
        encoded = encoder(batch_fovea)
    with tf.name_scope('Decoding') :
        decoded = decoder(encoded)


    with tf.name_scope('Training') :
        logloss = tf.contrib.losses.log_loss(predictions= decoded, labels = batch_fovea)
        #sparcity_cost = (beta*tf.nn.l2_loss(encoded) + beta*tf.nn.l2_loss(e1))
        total_loss = logloss #+ sparcity_cost
        train_op = tf.train.GradientDescentOptimizer(alpha).minimize(total_loss)

    with tf.name_scope('Summaries') :
        tf.summary.histogram('encoded', encoded)
        tf.summary.scalar('log_loss', logloss)
        tf.summary.scalar('learning_rate', alpha)
        summaries = tf.summary.merge_all()


"""
            W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 64, 128], stddev = stddev))
            tf.summary.histogram('W_conv2', W_conv2)
            b_conv2 = tf.Variable(tf.zeros([128]))
            tf.summary.histogram('b_conv2', b_conv2)

                        W_deconv1 = tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev = stddev))
                        tf.summary.histogram('W_deconv1', W_deconv1)
                        b_deconv1 = tf.Variable(tf.zeros([64]))
                        tf.summary.histogram('b_deconv1', b_deconv1)

            e2 = tf.sigmoid(
                    tf.nn.conv2d(e1, filter = W_conv2,
                        strides = [1,2,2,1],
                        padding = 'SAME') +
                    b_conv2
                    )

                        d1 =  tf.sigmoid(
                                tf.nn.conv2d_transpose(
                                    data,
                                    filter = W_deconv1,
                                    output_shape = [batch_size, fovea_size // 4, fovea_size //4, 64],
                                    strides = [1,2,2,1],
                                    padding = 'SAME') + b_deconv1)
"""
