import tensorflow as tf
from operator import mul


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summary(x):
    """
    Summarize the value of a variable.
    """
    variable_name = x.op.name
    tf.histogram_summary(variable_name + '/values', x)


def dropout(t, keep_prob):
    """
    Apply dropout to a tensor with the given probability

    This will also reshape the output to fix a bug with dropout not keeping
    shape.
    """
    dropped = tf.nn.dropout(t, keep_prob)
    reshaped = tf.reshape(t, t.get_shape(), name=t.op.name + '/do')
    return reshaped


def deconv_bn_relu(t, kernel_size, stride, out_filters,
                   scope, summarize=False):
    '''
    Apply a deconv operation (performs the reverse of a conv op)
    '''

    shape = t.get_shape()
    assert len(shape) == 4, "Must be 4d input tensor"
    batch_size = shape[0].value
    w = shape[1].value
    h = shape[2].value
    in_filters = shape[3].value
    kernel_shape = kernel_size + [out_filters, in_filters]
    kernel_init = tf.random_normal_initializer(
        0.0, 2.0 / (reduce(mul, kernel_shape, 1)))
    stride_shape = [1] + stride + [1]
    out_shape = [batch_size, w * stride[0], h * stride[0], out_filters]
    # beta is the bias value to be added after batch norm
    beta_init = tf.constant_initializer(0.0)
    # gamma is the scalar value to be multiplied after batch norm
    gamma_init = tf.constant_initializer(1.0)

    with tf.variable_scope(scope) as scope:
        conv_kernel = tf.get_variable(
            "conv_kernel", kernel_shape, initializer=kernel_init)
        conv = tf.nn.deconv2d(t, conv_kernel, out_shape, stride_shape,
                              padding='SAME', name="conv_activations")
        mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
        beta = tf.get_variable("beta", [out_filters], initializer=beta_init)
        gamma = tf.get_variable("gamma", [out_filters], initializer=gamma_init)
        bn = tf.nn.batch_norm_with_global_normalization(
            conv, mean, var, beta, gamma, 2e-6, scale_after_normalization=True)
        relu = tf.nn.relu(bn, "relu")

        if summarize:
            _activation_summary(conv)
            _activation_summary(relu)
            _variable_summary(beta)
            _variable_summary(gamma)

    return relu


def max_pool_2x2(t):
    '''
    Run a 2x2 max pooling operation
    '''
    return tf.nn.max_pool(t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME', name=t.op.name + '/pool')


def conv_bn_relu(input_tensor, kernel_size, out_filters, scope,
                 summarize=False):
    '''
    Add a spatial convolution to input_tensor, with batch norm and ReLU.

    -------
    INPUTS:
    -------
    input_tensor: 4d tensor with shape bwhc (batch width height channels)
    out_filters: number of output convolutional filters
    scope: a string to name all the variables and ops created here
    summarize: whether to add 'summary' values for the relevant layers

    -------
    OUTPUT:
    -------
    4d tensor with shape bwhc, with c==out_filters

    '''

    shape = input_tensor.get_shape()
    assert len(shape) == 4, "Must be 4d input tensor"
    batch_size = shape[0].value
    w = shape[1].value
    h = shape[2].value
    in_filters = shape[3].value
    var_shape = kernel_size + [in_filters, out_filters]
    kernel_init = tf.random_normal_initializer(
        0.0, 2.0 / (reduce(mul, var_shape, 1)))

    # beta is the bias value to be added after batch norm
    beta_init = tf.constant_initializer(0.0)
    # gamma is the scalar value to be multiplied after batch norm
    gamma_init = tf.constant_initializer(1.0)

    with tf.variable_scope(scope) as scope:
        conv_kernel = tf.get_variable(
            "conv_kernel", var_shape, initializer=kernel_init)
        conv = tf.nn.conv2d(input_tensor, conv_kernel, [1, 1, 1, 1],
                            padding='SAME', name="conv_activations")
        mean, var = tf.nn.moments(conv, axes=[0, 1, 2])
        beta = tf.get_variable("beta", [out_filters], initializer=beta_init)
        gamma = tf.get_variable("gamma", [out_filters], initializer=gamma_init)
        bn = tf.nn.batch_norm_with_global_normalization(
            conv, mean, var, beta, gamma, 2e-6, scale_after_normalization=True)
        relu = tf.nn.relu(bn, "relu")

        if summarize:
            _activation_summary(conv)
            _activation_summary(relu)
            _variable_summary(beta)
            _variable_summary(gamma)

    return relu


def negative_iou(pred, truth):
    '''
    Truth and label tensors contain 0, 1
    Calculate IOU.
    '''
    epsilon = 1e-2
    intersection = tf.reduce_sum(pred*truth)
    union = tf.reduce_sum(tf.maximum(pred, truth))
    union = tf.maximum(union, epsilon)
    return -1.0*(intersection/union)
