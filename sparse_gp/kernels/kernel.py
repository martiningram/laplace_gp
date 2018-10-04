# Abstract base class for Kernels.

from abc import ABC, abstractmethod


class Kernel(ABC):

    def __init__(self, input_dims, **kwargs):

        self.input_dims = input_dims

    @abstractmethod
    def compute(self, X1, X2):
        """Compute the kernel.

        Args:
            X1 (np.array): An [N1 x D] matrix.
            X2 (np.array): An [N2 x D] matrix.

        Returns:
            np.array: An [N1 x N2] kernel matrix.
        """
        pass

    @abstractmethod
    def gradients(self, X1, X2):
        """Compute the gradients of the kernel with respect to its
        hyperparameters.

        Args:
            X1 (np.array): An [N1 x D] matrix.
            X2 (np.array): An [N2 x D] matrix.

        Returns:
            Dict[str -> np.array]: A dictionary mapping the names of
            hyperparameters to their gradients.
        """
        pass

    @abstractmethod
    def get_flat_gradients(self, X1, X2):
        """Same as `gradients`, but must return the gradients as a single
        Matrix."""
        pass

    @abstractmethod
    def get_flat_hyperparameters(self):
        """Returns the hyperparameters of this kernel as a flat vector."""
        pass

    @abstractmethod
    def set_flat_hyperparameters(self, flat_hyperparameters):
        """Sets the hyperparameters of this kernel using a flat vector."""
        pass
