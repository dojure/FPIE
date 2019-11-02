import tensorflow as tf


def resnet_baseline(input_image):
    with tf.variable_scope("generator"):
        W1 = weight_variable([9, 9, 3, 64], name="W1");
        b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2");
        b2 = bias_variable([64], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3");
        b3 = bias_variable([64], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4");
        b4 = bias_variable([64], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5");
        b5 = bias_variable([64], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # residual 3

        W6 = weight_variable([3, 3, 64, 64], name="W6");
        b6 = bias_variable([64], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7");
        b7 = bias_variable([64], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8");
        b8 = bias_variable([64], name="b8");
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9");
        b9 = bias_variable([64], name="b9");
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional

        W10 = weight_variable([3, 3, 64, 64], name="W10");
        b10 = bias_variable([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11");
        b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Final

        W12 = weight_variable([9, 9, 64, 3], name="W12");
        b12 = bias_variable([3], name="b12");
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced


def resnet(input_image, kernel_size, depth, blocks, parametric=False, s_conv=False):
    if parametric:
        relu = parametric_relu
    else:
        def relu(x, scope=None):
            return tf.nn.relu(x)
    with tf.variable_scope("generator"):
        W1 = weight_variable([9, 9, 3, depth], name="W1")
        b1 = bias_variable([depth], name="b1")
        x = relu(conv2d(input_image, W1) + b1, scope="r1")

        # residual blocks

        for i in range(blocks):
            x = residual_block(x, kernel_size, depth, name="residual_block_{}".format(i), relu=relu, s_conv=s_conv)

        # Convolutional

        W10 = weight_variable([kernel_size, kernel_size, depth, depth], name="W10")
        b10 = bias_variable([depth], name="b10")
        c10 = relu(conv2d(x, W10) + b10, scope="r10")

        W11 = weight_variable([kernel_size, kernel_size, depth, depth], name="W11")
        b11 = bias_variable([depth], name="b11")
        c11 = relu(conv2d(c10, W11) + b11, scope="r11")

        # Final

        W12 = weight_variable([9, 9, depth, 3], name="W12")
        b12 = bias_variable([3], name="b12")
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced


def convdeconv(input_image, depth, parametric=False, s_conv=False):
    if parametric:
        relu = parametric_relu
    else:
        def relu(x, scope=None):
            return tf.nn.relu(x)
    if s_conv:
        def conv(x, mult=1):
            return skip_conv(x, 3, depth * mult)
    else:
        def conv(x, mult=1):
            return x
    with tf.variable_scope("generator"):
        batch_size = tf.shape(input_image)[0]
        height = tf.shape(input_image)[1]
        width = tf.shape(input_image)[2]

        W1 = weight_variable([9, 9, 3, depth], name="W1");
        b1 = bias_variable([depth], name="b1");
        c1 = relu(conv2d(input_image, W1) + b1)

        # decrease

        level_1_shape = tf.stack([batch_size, height, width, depth])

        W2 = weight_variable([4, 4, depth, depth * 2], name="W2");
        b2 = bias_variable([depth * 2], name="b2");
        c2 = relu(_instance_norm(conv2d(c1, W2, stride=2) + b2))

        level_2_shape = tf.stack([batch_size, height // 2, width // 2, depth * 2])

        W3 = weight_variable([4, 4, depth * 2, depth * 4], name="W3");
        b3 = bias_variable([depth * 4], name="b3");
        c3 = relu(_instance_norm(conv2d(c2, W3, stride=2) + b3))

        level_3_shape = tf.stack([batch_size, height // 4, width // 4, depth * 4])

        print(level_1_shape, level_2_shape, level_3_shape)

        # residual 1

        W4 = weight_variable([3, 3, depth * 4, depth * 4], name="W4");
        b4 = bias_variable([depth * 4], name="b4");
        c4 = relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, depth * 4, depth * 4], name="W5");
        b5 = bias_variable([depth * 4], name="b5");
        c5 = relu(_instance_norm(conv2d(c4, W5) + b5)) + conv(c3, mult=4)

        # residual 2

        W6 = weight_variable([3, 3, depth * 4, depth * 4], name="W6");
        b6 = bias_variable([depth * 4], name="b6");
        c6 = relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, depth * 4, depth * 4], name="W7");
        b7 = bias_variable([depth * 4], name="b7");
        c7 = relu(_instance_norm(conv2d(c6, W7) + b7)) + conv(c5, mult=4)

        # increase

        W8 = weight_variable([4, 4, depth * 2, depth * 4], name="W8");
        b8 = bias_variable([depth * 2], name="b8");
        c8 = relu(_instance_norm(deconv2d(c7, W8, stride=2, output_shape=level_2_shape) + b8))

        W9 = weight_variable([4, 4, depth, depth * 2], name="W9");
        b9 = bias_variable([depth], name="b9");
        c9 = relu(_instance_norm(deconv2d(c8, W9, stride=2, output_shape=level_1_shape) + b9)) + conv(
            c1)  # skip connection

        # Convolutional

        W10 = weight_variable([3, 3, depth, depth], name="W10");
        b10 = bias_variable([depth], name="b10");
        c10 = relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, depth, depth], name="W11");
        b11 = bias_variable([depth], name="b11");
        c11 = relu(conv2d(c10, W11) + b11)

        # Final

        W12 = weight_variable([9, 9, depth, 3], name="W12");
        b12 = bias_variable([3], name="b12");
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced


def adversarial(image_):
    with tf.variable_scope("discriminator"):
        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn=False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)

        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)

    return adv_out


def parametric_relu(_x, scope=None):
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        tf.summary.histogram("alphas", alphas)

        return pos + neg


def residual_block(x, kernel_size, depth, name, relu=tf.nn.relu, s_conv=False):
    with tf.variable_scope(name):
        W = weight_variable([kernel_size, kernel_size, depth, depth])
        b = bias_variable([depth])
        c = relu(_instance_norm(conv2d(x, W) + b))

        W = weight_variable([kernel_size, kernel_size, depth, depth])
        b = bias_variable([depth])

        if s_conv:
            x = skip_conv(x, kernel_size, depth)

            # tf.summary.histogram("skip_weights", W_skip)
            # tf.summary.histogram("skip_biases", b_skip)

        return relu(_instance_norm(conv2d(c, W) + b)) + x


def skip_conv(x, kernel_size, depth):
    W_skip = weight_variable([kernel_size, kernel_size, depth, depth])
    b_skip = bias_variable([depth])
    return conv2d(x, W_skip) + b_skip


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, stride=1, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def deconv2d(x, W, stride=1, padding='SAME', output_shape=None):
    return tf.nn.conv2d_transpose(x, W, strides=[1, stride, stride, 1], padding=padding, output_shape=output_shape)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net


def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)

    return scale * normalized + shift


def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
