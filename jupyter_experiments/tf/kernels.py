import tensorflow as tf


def ard_rbf_kernel(x1, x2, lengthscales, alpha, jitter=1e-5):

    # x1 is N1 x D
    # x2 is N2 x D (and N1 can be equal to N2)

    # Must have same number of dimensions
    assert(x1.get_shape()[1] == x2.get_shape()[1])

    # Also must match lengthscales
    assert(lengthscales.get_shape()[0] == x1.get_shape()[1])

    # Use broadcasting
    # X1 will be (N1, 1, D)
    x1_expanded = tf.expand_dims(x1, axis=1)
    # X2 will be (1, N2, D)
    x2_expanded = tf.expand_dims(x2, axis=0)

    # These will be N1 x N2 x D
    sq_differences = (x1_expanded - x2_expanded)**2
    inv_sq_lengthscales = 1. / lengthscales**2

    # Use broadcasting to do a dot product
    exponent = tf.reduce_sum(sq_differences * inv_sq_lengthscales,
                             axis=2)

    kern = alpha**2 * tf.exp(-0.5 * exponent)

    # Jitter this a little bit
    kern = tf.matrix_set_diag(kern, tf.diag_part(kern) + jitter)

    return kern
