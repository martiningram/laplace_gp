import tensorflow as tf
from kernels import ard_rbf_kernel


class SingleOutputLaplaceGP(object):

    def __init__(self, n_features, n_data):

        self.build_model(n_features, n_data)
        self.define_log_posterior()
        self.define_log_marg_lik()

    def build_model(self, n_features, n_data):

        # These will be fed
        self.X = tf.placeholder(tf.float32, shape=(n_data, n_features))
        self.y = tf.placeholder(tf.float32, shape=(n_data,))

        self.n_data = n_data

        self.scales = tf.placeholder(tf.float32, shape=(n_features))
        self.alpha = tf.placeholder(tf.float32, shape=())

        self.f = tf.get_variable('f', shape=(n_data, 1),
            initializer=tf.zeros_initializer)

        self.kernel = ard_rbf_kernel(self.X, self.X, self.scales, self.alpha)

    def define_log_posterior(self):

        self.chol_kernel = tf.linalg.cholesky(self.kernel)
        self.inv_kernel = tf.linalg.cholesky_solve(
            self.chol_kernel, tf.eye(self.n_data))
        self.log_prior = -0.5 * tf.matmul(
            tf.matmul(tf.transpose(self.f), self.inv_kernel), self.f)

        self.log_lik = tf.reduce_sum(
            tf.expand_dims(self.y, 1) * self.f - tf.nn.softplus(self.f))

        self.log_posterior = tf.squeeze(self.log_prior) + self.log_lik

        self.optimiser = tf.contrib.opt.ScipyOptimizerInterface(
            -self.log_posterior, var_list=[self.f])

    def define_log_marg_lik(self):

        probs = tf.nn.sigmoid(self.f)
        W = -probs * (probs - 1)
        W_sqrt = tf.diag(tf.squeeze(tf.sqrt(W)))
        multiplied = tf.matmul(tf.matmul(W_sqrt, self.kernel), W_sqrt)

        # Add on the identity
        with_id = tf.eye(self.n_data) + multiplied
        det_term = tf.linalg.logdet(with_id)

        self.log_marg_lik = (tf.squeeze(self.log_prior) + self.log_lik
                             - 0.5 * det_term)

        # Get the gradient of the log marginal likelihood
        hyper_grads = tf.gradients(self.log_marg_lik, [self.scales, self.alpha])

        self.scale_grad = hyper_grads[0]
        self.alpha_grad = hyper_grads[1]

    def marg_lik_and_grad(self, scales, alpha, X, y, sess):

        feed_dict = {
            self.X: X,
            self.y: y,
            self.scales: scales,
            self.alpha: alpha
        }

        # Optimise with these
        self.optimiser.minimize(sess, feed_dict=feed_dict)

        # Now that we are at the optimum, calculate the marginal likelihood and
        # its gradient w.r.t. the hyperparameters

        log_marg_lik, scale_grad, alpha_grad = sess.run(
            [self.log_marg_lik, self.scale_grad, self.alpha_grad],
            feed_dict=feed_dict)

        return {
            'log_marg_lik': log_marg_lik,
            'scale_grad': scale_grad,
            'alpha_grad': alpha_grad
        }
