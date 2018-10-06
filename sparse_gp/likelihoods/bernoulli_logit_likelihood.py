import numpy as np
import scipy.sparse as sps
from scipy.special import expit
from sparse_gp.likelihoods.likelihood import Likelihood


class BernoulliLogitLikelihood(Likelihood):

    def __init__(self):

        super(BernoulliLogitLikelihood, self).__init__(
            hessian_is_diagonal=True)

    def log_likelihood(self, y, f):

        per_point = (y * f) - np.log1p(np.exp(f))
        # Make sure size matches (no mistaken broadcasting)
        assert(per_point.size == y.size and per_point.size == f.size)
        return np.sum(per_point)

    def log_likelihood_grad(self, y, f):

        result = y - expit(f)
        assert(result.size == y.size and result.size == f.size)
        return result

    def log_likelihood_hessian(self, y, f):

        probs = expit(f)
        hess = sps.csc_matrix(np.diag(probs * (probs - 1)))
        return hess

    def log_likelihood_third_deriv(self, y, f):

        return expit(f) - 3 * expit(f)**2 + 2 * expit(f)**3
