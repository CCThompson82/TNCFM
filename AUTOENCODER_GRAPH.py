"""This is a file encoding the Graph for fovea autoencoding."""

autoencoder_depths = [3, 3]
autoencoder_kernels = [5, 3]
autoencoder_strides = [2, 2]

graph = tf.Graph()

with graph.as_default() :
    # Variables
    with tf.variable_scope('Variables') :

        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(
                        tf.truncated_normal(
                            [autoencoder_kernels[0], autoencoder_kernels[0], num_channels, autoencoder_depths[0]],
                            stddev = stddev))
            b_conv1 = tf.Variable(tf.zeros([autoencoder_depths[0]])) # experiment with this value
            tf.summary.histogram('W_conv1', W_conv1)
            tf.summary.histogram('b_conv1', b_conv1)

            W_conv2 = tf.Variable(
                        tf.truncated_normal(
                            [autoencoder_kernels[1], autoencoder_kernels[1], autoencoder_depths[0], autoencoder_depths[1]],
                            stddev = stddev))
            b_conv2 = tf.Variable(tf.zeros([autoencoder_depths[1]])) # experiment with this value
            tf.summary.histogram('W_conv2', W_conv2)
            tf.summary.histogram('b_conv2', b_conv2)

        with tf.variable_scope('Transpose_Convolution') :
            W_deconv2 = tf.Variable(
                        tf.truncated_normal(
                            [autoencoder_kernels[1], autoencoder_kernels[1], autoencoder_depths[0], autoencoder_depths[1]],
                            stddev = stddev))
            b_deconv2 = tf.Variable(tf.zeros([autoencoder_depths[0]])) # experiment with this value
            tf.summary.histogram('W_deconv2', W_deconv2)
            tf.summary.histogram('b_deconv2', b_deconv2)


            W_deconv1 = tf.Variable(
                            tf.truncated_normal(
                                [autoencoder_kernels[1], autoencoder_kernels[1], num_channels, autoencoder_depths[1]],
                                stddev = stddev))
            b_deconv1 = tf.Variable(tf.zeros([num_channels]))
            tf.summary.histogram('W_deconv1', W_deconv1)
            tf.summary.histogram('b_deconv1', b_deconv1)


    def encoder(data) :
        with tf.name_scope('Encoder') :
            e1 =  tf.nn.conv2d(data, filter = W_conv1,
                            strides = [1, autoencoder_strides[0], autoencoder_strides[0], 1],
                            padding = 'SAME') + b_conv1
            e2 = tf.nn.conv2d(e1,
                                filter = W_conv1,
                                strides = [1, autoencoder_strides[1], autoencoder_strides[1], 1],
                                padding = 'SAME') + b_conv2
        return e1, e2

    def decoder(data) :
        with tf.name_scope('Decoder') :
            d2 = tf.nn.conv2d_transpose(
                    data,
                    filter = W_deconv2,
                    output_shape = [batch_size, fovea_size // autoencoder_strides[1], fovea_size // autoencoder_strides[1], num_channels],
                    strides = [1, autoencoder_strides[1], autoencoder_strides[1], 1],
                    padding = 'SAME') + b_deconv2



            d1 = tf.sigmoid(
                    tf.nn.conv2d_transpose(
                        d2,
                        filter = W_deconv1,
                        output_shape = [batch_size, fovea_size, fovea_size, num_channels],
                        strides = [1, autoencoder_strides[0], autoencoder_strides[0], 1],
                        padding = 'SAME')
                    + b_deconv1)
        return d1, d2

    with tf.name_scope('Input_fovea') :
        alpha = tf.placeholder(tf.float32, shape = ())
        batch_fovea = tf.placeholder(tf.float32, shape = [batch_size, fovea_size, fovea_size, num_channels])

    with tf.name_scope('Encoding') :
        e1, encoded = encoder(batch_fovea)
    with tf.name_scope('Decoding') :
        decoded, d2 = decoder(encoded)


    with tf.name_scope('Training') :
        logloss = tf.contrib.losses.log_loss(predictions= decoded, labels = batch_fovea)
        total_loss = logloss #+ sparcity_cost
        train_op = tf.train.GradientDescentOptimizer(alpha).minimize(total_loss)

    with tf.name_scope('Summaries') :
        tf.summary.histogram('encoded_final', encoded)
        tf.summary.histogram('encoded_layer1', e1)
        tf.summary.scalar('log_loss', logloss)
        tf.summary.scalar('learning_rate', alpha)
        summaries = tf.summary.merge_all()
