from abc import ABC, abstractmethod


class Inference(ABC):

    def __init__(self, kernel, likelihood):

        self.kernel = kernel
        self.likelihood = likelihood

    @abstractmethod
    def fit(self, x, y):
        """Fits kernel hyperparameters and GP latent values.

        Args:
            x (np.array): Input data [a matrix].
            y (np.array): Output data [a vector].

        Returns:
            Nothing, but fits the GP.
        """
        pass

    @abstractmethod
    def predict(self, x):
        """Returns predictions at function values x."""
        # FIXME: Work out what this should return.
        pass
