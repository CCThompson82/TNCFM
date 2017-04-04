"""This script is the FishFinder graph for the Kaggle Nature Conservancy Fishery
Competition.  It utilizes VGG-19 model as pre-trained weights during the
convolution steps"""


fish_finder = tf.Graph()

with fish_finder.as_default() :
    # Variables
    with tf.variable_scope('Variables') :
        with tf.variable_scope('Convolutions') :
            with tf.name_scope('Convolution_1') :
                W_conv1 = tf.Variable(np.load(pretrained_path+'W_conv_1.npy'), trainable = False)
                b_conv1 = tf.Variable(np.load(pretrained_path+'b_conv_1.npy'), trainable = False)
                tf.summary.histogram('W_conv1', W_conv1)
                tf.summary.histogram('b_conv1', b_conv1)
            with tf.name_scope('Convolution_2') :
                W_conv2 = tf.Variable(np.load(pretrained_path+'W_conv_2.npy'), trainable = False)
                b_conv2 = tf.Variable(np.load(pretrained_path+'b_conv_2.npy'), trainable = False)
                tf.summary.histogram('W_conv2', W_conv2)
                tf.summary.histogram('b_conv2', b_conv2)
            with tf.name_scope('Convolution_3') :
                W_conv3 = tf.Variable(np.load(pretrained_path+'W_conv_3.npy'), trainable = False)
                b_conv3 = tf.Variable(np.load(pretrained_path+'b_conv_3.npy'), trainable = False)
                tf.summary.histogram('W_conv3', W_conv3)
                tf.summary.histogram('b_conv3', b_conv3)
            with tf.name_scope('Convolution_4') :
                W_conv4 = tf.Variable(np.load(pretrained_path+'W_conv_4.npy'), trainable = False)
                b_conv4 = tf.Variable(np.load(pretrained_path+'b_conv_4.npy'), trainable = False)
                tf.summary.histogram('W_conv4', W_conv4)
                tf.summary.histogram('b_conv4', b_conv4)
            with tf.name_scope('Convolution_5') :
                W_conv5 = tf.Variable(np.load(pretrained_path+'W_conv_5.npy'), trainable = False)
                b_conv5 = tf.Variable(np.load(pretrained_path+'b_conv_5.npy'), trainable = False)
                tf.summary.histogram('W_conv5', W_conv5)
                tf.summary.histogram('b_conv5', b_conv5)
            with tf.name_scope('Convolution_6') :
                W_conv6 = tf.Variable(np.load(pretrained_path+'W_conv_6.npy'), trainable = False)
                b_conv6 = tf.Variable(np.load(pretrained_path+'b_conv_6.npy'), trainable = False)
                tf.summary.histogram('W_conv6', W_conv6)
                tf.summary.histogram('b_conv6', b_conv6)
            with tf.name_scope('Convolution_7') :
                W_conv7 = tf.Variable(np.load(pretrained_path+'W_conv_7.npy'), trainable = False)
                b_conv7 = tf.Variable(np.load(pretrained_path+'b_conv_7.npy'), trainable = False)
                tf.summary.histogram('W_conv7', W_conv7)
                tf.summary.histogram('b_conv7', b_conv7)
            with tf.name_scope('Convolution_8') :
                W_conv8 = tf.Variable(np.load(pretrained_path+'W_conv_8.npy'), trainable = False)
                b_conv8 = tf.Variable(np.load(pretrained_path+'b_conv_8.npy'), trainable = False)
                tf.summary.histogram('W_conv8', W_conv8)
                tf.summary.histogram('b_conv8', b_conv8)
            with tf.name_scope('Convolution_9') :
                W_conv9 = tf.Variable(np.load(pretrained_path+'W_conv_9.npy'), trainable = False)
                b_conv9 = tf.Variable(np.load(pretrained_path+'b_conv_9.npy'), trainable = False)
                tf.summary.histogram('W_conv9', W_conv9)
                tf.summary.histogram('b_conv9', b_conv9)
            with tf.name_scope('Convolution_10') :
                W_conv10 = tf.Variable(np.load(pretrained_path+'W_conv_10.npy'), trainable = False)
                b_conv10 = tf.Variable(np.load(pretrained_path+'b_conv_10.npy'), trainable = False)
                tf.summary.histogram('W_conv10', W_conv10)
                tf.summary.histogram('b_conv10', b_conv10)
            with tf.name_scope('Convolution_11') :
                W_conv11 = tf.Variable(np.load(pretrained_path+'W_conv_11.npy'), trainable = False)
                b_conv11 = tf.Variable(np.load(pretrained_path+'b_conv_11.npy'), trainable = False)
                tf.summary.histogram('W_conv11', W_conv11)
                tf.summary.histogram('b_conv11', b_conv11)
            with tf.name_scope('Convolution_12') :
                W_conv12 = tf.Variable(np.load(pretrained_path+'W_conv_12.npy'), trainable = False)
                b_conv12 = tf.Variable(np.load(pretrained_path+'b_conv_12.npy'), trainable = False)
                tf.summary.histogram('W_conv12', W_conv12)
                tf.summary.histogram('b_conv12', b_conv12)
            with tf.name_scope('Convolution_13') :
                W_conv13 = tf.Variable(np.load(pretrained_path+'W_conv_13.npy'), trainable = False)
                b_conv13 = tf.Variable(np.load(pretrained_path+'b_conv_13.npy'), trainable = False)
                tf.summary.histogram('W_conv13', W_conv13)
                tf.summary.histogram('b_conv13', b_conv13)
            with tf.name_scope('Convolution_14') :
                W_conv14 = tf.Variable(np.load(pretrained_path+'W_conv_14.npy'), trainable = False)
                b_conv14 = tf.Variable(np.load(pretrained_path+'b_conv_14.npy'), trainable = False)
                tf.summary.histogram('W_conv14', W_conv14)
                tf.summary.histogram('b_conv14', b_conv14)
            with tf.name_scope('Convolution_15') :
                W_conv15 = tf.Variable(np.load(pretrained_path+'W_conv_15.npy'), trainable = False)
                b_conv15 = tf.Variable(np.load(pretrained_path+'b_conv_15.npy'), trainable = False)
                tf.summary.histogram('W_conv15', W_conv15)
                tf.summary.histogram('b_conv15', b_conv15)
            with tf.name_scope('Convolution_16') :
                W_conv16 = tf.Variable(np.load(pretrained_path+'W_conv_16.npy'), trainable = False)
                b_conv16 = tf.Variable(np.load(pretrained_path+'b_conv_16.npy'), trainable = False)
                tf.summary.histogram('W_conv16', W_conv16)
                tf.summary.histogram('b_conv16', b_conv16)

        with tf.variable_scope('Dense_layers') :
            with tf.name_scope('dense_1') :
                W_fc1 = tf.Variable(tf.truncated_normal([nodes_after_conv, fc_depth[0]], stddev = stddev ))
                b_fc1 = tf.Variable(tf.zeros([fc_depth[0]]))
                tf.summary.histogram('W_fc1', W_fc1)
                tf.summary.histogram('b_fc1', b_fc1)
            with tf.name_scope('dense_2') :
                W_fc2 = tf.Variable(tf.truncated_normal([fc_depth[0], fc_depth[1]], stddev = stddev ))
                b_fc2 = tf.Variable(tf.zeros([fc_depth[1]]))
                tf.summary.histogram('W_fc2', W_fc2)
                tf.summary.histogram('b_fc2', b_fc2)
            with tf.name_scope('dense_3') :
                W_fc3 = tf.Variable(tf.truncated_normal([fc_depth[1], fc_depth[2]], stddev = stddev ))
                b_fc3 = tf.Variable(tf.zeros([fc_depth[2]]))
                tf.summary.histogram('W_fc3', W_fc3)
                tf.summary.histogram('b_fc3', b_fc3)
        with tf.variable_scope('Classifier') :
            W_clf = tf.Variable(tf.truncated_normal([fc_depth[2], num_labels], stddev = stddev))
            b_clf = tf.Variable(tf.zeros([num_labels]))
            tf.summary.histogram('W_clf', W_clf)
            tf.summary.histogram('b_clf', b_clf)





    def convolutions(data) :
        """
        Emulates VGG-19 architecture.
        """
        with tf.name_scope('Convolution') :
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(data, filter = W_conv1,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv1)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv2,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv2),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv3,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv3)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv4,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv4),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv5,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv5)

            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv6,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv6)
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv7,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv7)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv8,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv8),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv9,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv9)
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv10,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv10)
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv11,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv11)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv12,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv12),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv13,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv13)
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv14,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv14)
            conv_layer = tf.nn.relu(
                                tf.nn.conv2d(conv_layer, filter = W_conv15,
                                    strides = [1, conv_stride, conv_stride, 1],
                                    padding = 'SAME') + b_conv15)
            conv_layer = tf.nn.max_pool(
                                    tf.nn.relu(
                                        tf.nn.conv2d(conv_layer, filter = W_conv16,
                                            strides = [1, conv_stride, conv_stride, 1],
                                            padding = 'SAME') + b_conv16),
                                    ksize = [1, pool_kernel, pool_kernel,1],
                                    strides = [1, pool_stride, pool_stride, 1],
                                    padding ='VALID')
        return conv_layer

    def dense_layers(data, keep_prob) :
        """
        Executes a series of dense layers.
        """
        def fc(data, W, b, keep_prob = keep_prob) :
            """Convenience function for dense layer with dropout"""
            fc = tf.nn.dropout(
                    tf.nn.relu(
                            tf.matmul(data, W) + b,
                            ),
                    keep_prob)
            return fc

        fc_layer = fc(data, W_fc1, b_fc1, keep_prob[0])
        fc_layer = fc(fc_layer, W_fc2, b_fc2,  keep_prob[1])
        fc_layer = fc(fc_layer, W_fc3, b_fc3, keep_prob[2])
        return fc_layer




    with tf.name_scope('Training') :
        with tf.name_scope('Input') :
            train_images = tf.placeholder(tf.float32, shape = [batch_size, fov_size, fov_size, num_channels])
            train_labels = tf.placeholder(tf.float32, shape = [batch_size, num_labels])
            learning_rate = tf.placeholder(tf.float32, shape = () )
            epoch_size = tf.placeholder(tf.int32, shape = ())
        with tf.name_scope('Network') :
            conv_output = convolutions(train_images)
            dense_input = tf.contrib.layers.flatten(conv_output)
            dense_output = dense_layers(dense_input, keep_prob = keep_prob)
        with tf.name_scope('Classifier') :
            logits = tf.matmul(dense_output, W_clf) + b_clf
        with tf.name_scope('Backpropigation') :
            xent = tf.nn.softmax_cross_entropy_with_logits(
                        logits = logits, labels = train_labels)
            cross_entropy = tf.reduce_mean(xent)
            cost = cross_entropy # + regularization or other cost amendment?

            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

    """
    with tf.name_scope('Validation') :
        with tf.name_scope('Input') :
            valid_set = tf.placeholder(tf.float32, shape = [batch_size, fov_size, fov_size, num_channels])
            valid_labels = tf.placeholder(tf.float32, shape = [batch_size, num_labels])
        with tf.name_scope('Network') :
            valid_conv_output = convolutions(valid_set)
            valid_dense_input = tf.contrib.layers.flatten(valid_conv_output)
            valid_dense_output = dense_layers(valid_dense_input, keep_prob = 1.0)
        with tf.name_scope('Prediction') :
            valid_logits = tf.nn.softmax(tf.matmul(valid_dense_output, W_clf) + b_clf)
            valid_acc = tf.reduce_mean(tf.to_int32(tf.equal(tf.argmax(valid_logits, 1), tf.argmax(valid_labels, 1))))
        """

    with tf.name_scope('Summaries') :
        tf.summary.scalar('Epoch_size', epoch_size)
        tf.summary.scalar('Cross_entropy', cross_entropy)
        tf.summary.scalar('Learning_rate', learning_rate)
        #tf.summary.scalar('Valid_Accuracy', valid_acc)
        summaries = tf.summary.merge_all()

    with tf.name_scope('Prediction') :
        with tf.name_scope('Input') :
            img_stack = tf.placeholder(tf.float32, shape = [pred_batch, fov_size, fov_size, num_channels])
        with tf.name_scope('Network') :
            stack_conv_output = convolutions(img_stack)
            stack_dense_input = tf.contrib.layers.flatten(stack_conv_output)
            stack_dense_output = dense_layers(stack_dense_input, keep_prob = [1.0,1.0,1.0])
        with tf.name_scope('Classifier') :
            stack_prediction = tf.nn.softmax(tf.matmul(stack_dense_output, W_clf) + b_clf)
