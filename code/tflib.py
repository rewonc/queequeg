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
    variable_name = x.name
    tf.histogram_summary(variable_name + '/values', x)


def convBNReLU(input_tensor, kernel_size, out_filters, scope, summarize=False):
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

    with tf.variable_scope(scope, reuse=True) as scope:
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
