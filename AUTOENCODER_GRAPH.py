"""This is a file encoding the Graph for fovea autoencoding."""


graph = tf.Graph()

with graph.as_default() :
    # Variables
    with tf.variable_scope('Variables') :

        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(tf.truncated_normal([kernel_sizes[0], kernel_sizes[0], num_channels, conv_depths[0]], stddev = stddev))
            tf.summary.histogram('W_conv1', W_conv1)
            b_conv1 = tf.Variable(tf.zeros([conv_depths[0]])) # experiment with this value
            tf.summary.histogram('b_conv1', b_conv1)

            W_conv2 = tf.Variable(tf.truncated_normal([kernel_sizes[1], kernel_sizes[1], conv_depths[0], conv_depths[1]], stddev = stddev))
            tf.summary.histogram('W_conv2', W_conv2)
            b_conv2 = tf.Variable(tf.zeros([conv_depths[1]]))
            tf.summary.histogram('b_conv2', b_conv2)

        with tf.variable_scope('Transpose_Convolution') :
            W_deconv1 = tf.Variable(tf.truncated_normal([3,3,conv_depths[-1], 256], stddev = stddev))
            tf.summary.histogram('W_deconv1', W_deconv1)
            b_deconv1 = tf.Variable(tf.zeros([256]))

            W_deconv2 = tf.Variable(tf.truncated_normal([3,3,conv_depths[-2], conv_depths[-1]], stddev = stddev))
            tf.summary.histogram('W_deconv2', W_deconv2)
            b_deconv2 = tf.Variable(tf.zeros([96]))

            W_deconv3 = tf.Variable(tf.truncated_normal([3,3,3, conv_depths[-2]], stddev = stddev))
            tf.summary.histogram('W_deconv3', W_deconv3)
            b_deconv3 = tf.Variable(tf.zeros([3]))


    def encoder(data) :
        with tf.name_scope('Encoder') :
            e1 = tf.nn.max_pool(
                    tf.nn.relu(
                        tf.nn.conv2d(data, filter = W_conv1,
                            strides = [1, conv_strides[0],conv_strides[0], 1],
                            padding = 'SAME') +
                        b_conv1),
                    ksize = [1,pool_kernels[0],pool_kernels[0],1], strides = [1,pool_strides[0], pool_strides[0],1],
                    padding ='VALID')
            e2 = tf.nn.max_pool(
                    tf.nn.relu(
                        tf.nn.conv2d(e1, filter = W_conv2,
                            strides = [1,conv_strides[1], conv_strides[1],1],
                            padding = 'SAME') +
                        b_conv2),
                    ksize = [1,pool_kernels[1],pool_kernels[1],1], strides = [1,pool_strides[1],pool_strides[1],1],
                    padding = 'VALID')

        return e1, e2

    def decoder(data) :
        with tf.name_scope('Decoder') :
            d1 = tf.nn.relu(
                    tf.nn.conv2d_transpose(
                        data,
                        filter = W_deconv1,
                        output_shape = [batch_size, 27, 27, 256],
                        strides = [1,2,2,1],
                        padding = 'VALID') +
                    b_deconv1)
            d2 = tf.nn.relu(
                    tf.nn.conv2d_transpose(
                        d1,
                        filter = W_deconv2,
                        output_shape = [batch_size, 56, 56, 96],
                        strides = [1,2,2,1],
                        padding = 'VALID') +
                    b_deconv2)
            d3 = tf.sigmoid(
                    tf.nn.conv2d_transpose(
                        d2,
                        filter = W_deconv3,
                        output_shape = [batch_size, fovea_size, fovea_size, num_channels],
                        strides = [1,4,4,1],
                        padding = 'SAME') +
                    b_deconv3 )
        return d3

    with tf.name_scope('Input_fovea') :
        batch_fovea = tf.placeholder(tf.float32, shape = [batch_size, fovea_size, fovea_size, num_channels])

    with tf.name_scope('Encoding') :
        e1, e2 = encoder(batch_fovea)
        encoded = e2
    with tf.name_scope('Decoding') :
        decoded = decoder(encoded)


    with tf.name_scope('Training') :
        loss = tf.contrib.losses.mean_squared_error(predictions= decoded, labels = batch_fovea)
        cost = tf.reduce_sum(loss) + (beta*tf.nn.l2_loss(e1)) + (beta*tf.nn.l2_loss(e2))

        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('Summaries') :
        tf.summary.scalar('loss', tf.reduce_sum(loss))
        tf.summary.scalar('cost', cost)
        summaries = tf.summary.merge_all()
