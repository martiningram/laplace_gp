from abc import ABC, abstractmethod


class Likelihood(ABC):

    def __init__(self, hessian_is_diagonal):

        self.hessian_is_diagonal = hessian_is_diagonal

    @abstractmethod
    def log_likelihood(self, y, f):
        """The log likelihood of y under f.

        Args:
            y [np.array]: The data; assumed to be a vector.
            f [np.array]: The latent value; assumed to be a vector.

        Returns:
            float: The summed log likelihood.
        """
        pass

    @abstractmethod
    def log_likelihood_grad(self, y, f):
        """The gradient of the log likelihood with respect to f.

        Args:
            y [np.array]: The data; assumed to be a vector.
            f [np.array]: The latent value; assumed to be a vector.

        Returns:
            np.array: The vector of gradient values.
        """
        pass

    @abstractmethod
    def log_likelihood_hessian(self, y, f):
        """The Hessian of the log likelihood with respect to f.

        Args:
            y [np.array]: The data; assumed to be a vector.
            f [np.array]: The latent value; assumed to be a vector.

        Returns:
            csc_matrix: A sparse matrix of second derivatives.
        """
        pass

    def log_likelihood_third_deriv(self, y, f):
        # TODO: Is this the right way to handle this?
        raise Exception('Third derivative not implemented!')
