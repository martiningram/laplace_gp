import numpy as np
import scipy.sparse as sps
from scipy.optimize import fmin_cg, minimize
from sksparse.cholmod import cholesky
from sparse_gp.inference.inference import Inference


class LaplaceInference(Inference):

    def __init__(self, kernel, likelihood):

        # Currently, support only diagonal hessians.
        assert(likelihood.hessian_is_diagonal)

        super(LaplaceInference, self).__init__(
            kernel=kernel, likelihood=likelihood)

    @staticmethod
    def find_mode(K, likelihood, y, f_init=None):

        if f_init is None:
            f = np.zeros(K.shape[0])
        else:
            f = f_init

        old_f = np.ones_like(f)

        while np.sum((f - old_f)**2) > 1e-8:

            grad_log_y = likelihood.log_likelihood_grad(y, f)
            W = -likelihood.log_likelihood_hessian(y, f)
            W_sqrt = np.sqrt(W)
            multiplied = W_sqrt.dot(K).dot(W_sqrt)

            B = multiplied + sps.eye(K.shape[0], format='csc')
            L = cholesky(B)
            b = W.dot(f) + grad_log_y

            first_solve = L.solve_L(W_sqrt.dot(K.dot(b)), False)
            second_solve = L.solve_Lt(first_solve, False)

            a = b - W_sqrt.dot(second_solve)
            old_f = f
            f = K.dot(a)

        # Calculate the log marginal likelihood
        log_lik = likelihood.log_likelihood(y, f)
        log_marg_lik = (-0.5 * a.T.dot(f) +
                        log_lik - np.log(L.L().diagonal()).sum())

        intermediate_quantities = {
            'B': B, 'L': L, 'b': b, 'a': a, 'W_sqrt': W_sqrt,
            'grad_log_y': grad_log_y
        }

        return f, log_marg_lik, intermediate_quantities

    def log_marg_lik_and_grad(self, hyperparameters, x, y):

        self.kernel.set_flat_hyperparameters(hyperparameters)
        hyper_grads = self.kernel.get_flat_gradients(x, x)
        kern = self.kernel.compute(x, x)

        # FIXME: Not a huge fan of having all the intermediate quantities in
        # "s", although it should save computation.
        f, log_marg_lik, s = self.find_mode(kern, self.likelihood, y)

        third_grad = self.likelihood.log_likelihood_third_deriv(y, f)

        Z = s['W_sqrt'].dot(s['L'].solve_Lt(
            s['L'].solve_L(s['W_sqrt'], False), False))

        C = s['L'].solve_L(s['W_sqrt'].dot(kern), False)

        diag_diff = sps.csc_matrix(np.diag(
            kern.diagonal() - C.transpose().dot(C).diagonal()))

        s2 = -0.5 * diag_diff.dot(third_grad)

        grads = list()

        for C in hyper_grads:

            # TODO: I think the trace can be computed more efficiently here.
            inter = 0.5 * s['a'].dot(C.dot(s['a']))
            s1 = inter - 0.5 * np.sum(Z.dot(C).diagonal())

            b = C.dot(s['grad_log_y'])
            s3 = b - kern.dot((Z.dot(b)))
            grads.append(s1 + s2.T.dot(s3))

        return log_marg_lik, np.array(grads), f

    def fit(self, x, y):

        # Get the kernel hyperparameters (initial values)
        hyperparams = self.kernel.get_flat_hyperparameters()
        to_minimise = lambda h: [
            -x for x in self.log_marg_lik_and_grad(h, x, y)[0:2]]
        result = minimize(to_minimise, x0=hyperparams, jac=True,
                          method='CG', options={'gtol': 1e-2})
        self.kernel.set_flat_hyperparameters(result.x)

        # Compute mode with parameters for prediction
        final_k = self.kernel.compute(x, x)
        f, marg_lik, quantities = self.find_mode(final_k, self.likelihood, y)

        self.f_hat = f
        self.L = quantities['L']
        self.W_sqrt = quantities['W_sqrt']
        self.grad_y = quantities['grad_y']
        self.x = x

        return result

    def predict(self, x_star):

        # TODO: Finish this off!

        # Compute mean
        # Compute variance
        kern_result = self.kernel.compute(x_star, x_star)
        v = self.solve_L(self.W_sqrt.dot(kern_result), False)
        var = kern_result - v.dot(v.T)
