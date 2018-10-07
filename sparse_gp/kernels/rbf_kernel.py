import numpy as np
from sparse_gp.kernels.kernel import Kernel
import scipy.sparse as sps


class RBFKernel(Kernel):

    def __init__(self, input_dims, stdev=None, lengthscales=None, jitter=1e-5):

        if lengthscales is None:
            # Initialise to ones
            self.lengthscales = np.ones(len(input_dims))
        else:
            assert(lengthscales.shape[0] == len(input_dims))
            self.lengthscales = lengthscales

        if stdev is None:
            self.stdev = np.ones(())
        else:
            # Make sure we were passed a numpy array of the correct size
            assert stdev.shape == tuple()
            self.stdev = stdev

        self.jitter = jitter

        super(RBFKernel, self).__init__(input_dims=input_dims)

    def compute_sq_differences(self, X1, X2):

        # x1 is N1 x D
        # x2 is N2 x D (and N1 can be equal to N2)

        X1 = X1[:, self.input_dims]
        X2 = X2[:, self.input_dims]

        # Must have same number of dimensions
        assert(X1.shape[1] == X2.shape[1])

        # Also must match lengthscales
        assert(self.lengthscales.shape[0] == X1.shape[1])

        # Use broadcasting
        # X1 will be (N1, 1, D)
        x1_expanded = np.expand_dims(X1, axis=1)
        # X2 will be (1, N2, D)
        x2_expanded = np.expand_dims(X2, axis=0)

        # These will be N1 x N2 x D
        sq_differences = (x1_expanded - x2_expanded)**2

        return sq_differences

    def compute_exponent(self, X1, X2):

        sq_differences = self.compute_sq_differences(X1, X2)

        inv_sq_lengthscales = 1. / self.lengthscales**2

        # Use broadcasting to do a dot product
        exponent = np.sum(sq_differences * inv_sq_lengthscales, axis=2)
        exponentiated = np.exp(-0.5 * exponent)

        return exponentiated

    def compute(self, X1, X2):

        exponentiated = self.compute_exponent(X1, X2)

        kern = self.stdev**2 * exponentiated
        diag_indices = np.diag_indices(np.min(kern.shape[:2]))
        kern[diag_indices] = kern[diag_indices] + self.jitter

        return sps.csc_matrix(kern)

    def gradients(self, X1, X2):

        # FIXME: This wastes some computation, since compute_exponent also
        # calculates the square differences. Could fix.
        sq_differences = self.compute_sq_differences(X1, X2)
        exponentiated = self.compute_exponent(X1, X2)

        # Find gradients
        # Gradient with respect to stdev:
        stdev_grad = 2 * self.stdev * exponentiated

        # Add a dimension on so that it is [N1 x N2 x 1]
        stdev_grad = np.expand_dims(stdev_grad, axis=2)

        # Gradient with respect to lengthscales
        # Square differences should be [N1 x N2 x D]
        lengthscale_grads = (
            self.stdev**2 * np.expand_dims(exponentiated, axis=2) *
            sq_differences / (self.lengthscales**3))

        return {
            'stdev': stdev_grad,
            'lengthscales': lengthscale_grads
        }

    def get_flat_gradients(self, X1, X2):

        # Compute the gradients as normal
        gradients = self.gradients(X1, X2)

        # Now, squash them together.
        flat_grads = np.concatenate([gradients['lengthscales'],
                                     gradients['stdev']], axis=2)

        return [sps.csc_matrix(flat_grads[:, :, i]) for i in
                range(flat_grads.shape[2])]

    def get_flat_hyperparameters(self):

        # Squash the hyperparameters together
        flat_hypers = np.concatenate(
            [self.lengthscales, np.expand_dims(self.stdev, axis=0)], axis=0)

        return flat_hypers

    def set_flat_hyperparameters(self, flat_hyperparameters):

        # Take the hyperparameters back apart
        lengthscales = flat_hyperparameters[:-1]
        stdev = flat_hyperparameters[-1]

        assert(lengthscales.shape[0] == self.lengthscales.shape[0])

        self.lengthscales = lengthscales
        self.stdev = stdev

    def __str__(self):

        return ('RBF Kernel with lengthscales {} and stdev {:.2f}.'.format(
            np.array2string(self.lengthscales, precision=2), self.stdev))
