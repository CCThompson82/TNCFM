"""This is a file encoding the Graph for fovea autoencoding."""


graph = tf.Graph()

with graph.as_default() :
    # Variables
    with tf.variable_scope('Variables') :

        with tf.variable_scope('Convolutions') :
            W_conv1 = tf.Variable(tf.truncated_normal([4, 4, num_channels, 32], stddev = stddev))
            tf.summary.histogram('W_conv1', W_conv1)
            b_conv1 = tf.Variable(tf.zeros([32])) # experiment with this value
            tf.summary.histogram('b_conv1', b_conv1)

        with tf.variable_scope('Transpose_Convolution') :
            W_deconv1 = tf.Variable(tf.truncated_normal([4,4,num_channels, 32], stddev = stddev))
            tf.summary.histogram('W_deconv1', W_deconv1)
            b_deconv1 = tf.Variable(tf.zeros([3]))
            tf.summary.histogram('b_deconv1', b_deconv1)


    def encoder(data) :
        with tf.name_scope('Encoder') :
            e1 = tf.sigmoid(
                        tf.nn.conv2d(data, filter = W_conv1,
                            strides = [1, 2,2, 1],
                            padding = 'SAME') +
                        b_conv1)
        return e1

    def decoder(data) :
        with tf.name_scope('Decoder') :
            d1 =    tf.nn.conv2d_transpose(
                        data,
                        filter = W_deconv1,
                        output_shape = [batch_size, fovea_size, fovea_size, num_channels],
                        strides = [1,2,2,1],
                        padding = 'SAME') + b_deconv1
        return d1

    with tf.name_scope('Input_fovea') :
        batch_fovea = tf.placeholder(tf.float32, shape = [batch_size, fovea_size, fovea_size, num_channels])

    with tf.name_scope('Encoding') :
        encoded = encoder(batch_fovea)
    with tf.name_scope('Decoding') :
        decoded = decoder(encoded)


    with tf.name_scope('Training') :
        mse = tf.contrib.losses.mean_squared_error(predictions= decoded, labels = batch_fovea)
        cost = (beta*tf.nn.l2_loss(encoded))
        loss = mse + cost
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.name_scope('Summaries') :
        tf.summary.histogram('encoded', encoded)
        tf.summary.scalar('mse', mse)
        tf.summary.scalar('loss', tf.reduce_sum(loss))
        tf.summary.scalar('cost', cost)
        summaries = tf.summary.merge_all()
