import numpy as np
import scipy.sparse as sps
from sparse_gp.kernels.kernel import Kernel


class IndependentOutputKernel(Kernel):

    def __init__(self, input_dims, kernels):

        # We must have exactly one input dimension
        assert(len(input_dims) == 1)
        self.kernels = kernels

        super(IndependentOutputKernel, self).__init__(input_dims)

    def compute_agnostic(self, X1, X2):

        # Let's do this naively first, then optimise.
        input_dim = self.input_dims[0]

        output = sps.lil_matrix((X1.shape[0], X2.shape[0]))

        # Pick out the relevant column:
        x1_kernels = X1[:, input_dim]
        x2_kernels = X2[:, input_dim]

        for i, cur_kernel in enumerate(self.kernels):

            x1_relevant = x1_kernels == i
            x2_relevant = x2_kernels == i

            x1_rows = X1[x1_relevant]
            x2_rows = X2[x2_relevant]

            if np.sum(x1_relevant) == 0 or np.sum(x2_relevant) == 0:
                # Nothing to do
                continue

            # Otherwise, compute the kernel for these rows
            computed_kernel = cur_kernel.compute(x1_rows, x2_rows)

            # Get the output entries via broadcasting
            # TODO: This may be super-inefficient. May need to improve.
            to_fill = x1_relevant.reshape(-1, 1) & x2_relevant.reshape(1, -1)

            output[to_fill] = computed_kernel.reshape(1, -1)

        return sps.csc_matrix(output)

    @profile
    def compute(self, X1, X2):

        if np.array_equal(X1, X2):

            print('Running efficient version.')

            # We can do something cleverer.
            input_dim = self.input_dims[0]
            ind_sorted = np.argsort(X1[:, input_dim])

            # FIXME: Quietly assuming that the input dim is all zeros, then all
            # 1s, and so on. This needs to be asserted somehow!

            sub_kernels = list()

            # Now, we can just compute the kernels individually and then put
            # them together to be block diagonal.
            for i, cur_kernel in enumerate(self.kernels):

                relevant = X1[:, input_dim] == i

                if np.sum(relevant) == 0:
                    # Nothing to do
                    continue

                relevant_rows = X1[relevant, :]

                # Otherwise, compute the kernel for these rows
                computed_kernel = cur_kernel.compute(
                    relevant_rows, relevant_rows)

                sub_kernels.append(computed_kernel)

            return sps.block_diag(sub_kernels, format='csc')

        else:

            print('Running agnostic version.')
            return self.compute_agnostic(X1, X2)


    def gradients(self, X1, X2):
        raise Exception('Only flat gradients supported!')

    def get_flat_gradients_agnostic(self, X1, X2):

        # Let's do this naively first, then optimise.
        input_dim = self.input_dims[0]

        output_grads = list()

        # Pick out the relevant column:
        x1_kernels = X1[:, input_dim]
        x2_kernels = X2[:, input_dim]

        for i, cur_kernel in enumerate(self.kernels):

            x1_relevant = x1_kernels == i
            x2_relevant = x2_kernels == i

            x1_rows = X1[x1_relevant]
            x2_rows = X2[x2_relevant]

            if np.sum(x1_relevant) == 0 or np.sum(x2_relevant) == 0:
                # Nothing to do
                continue

            # Otherwise, compute the kernel for these rows
            computed_grads = cur_kernel.get_flat_gradients(
                x1_rows, x2_rows)

            # Fill the big kernel with these
            to_fill = x1_relevant.reshape(-1, 1) & x2_relevant.reshape(1, -1)

            for cur_element in computed_grads:
                cur_output = sps.csc_matrix((X1.shape[0], X2.shape[0]))
                cur_output[to_fill] = cur_element.todense().reshape(1, -1)
                output_grads.append(cur_output)

        return output_grads

    @profile
    def get_flat_gradients(self, X1, X2):

        if np.array_equal(X1, X2):

            print('Running efficient gradients')

            # We can do something cleverer.
            input_dim = self.input_dims[0]

            # FIXME: Quietly assuming that the input dim is all zeros, then all
            # 1s, and so on. This needs to be asserted somehow!
            sub_kernels = list()

            num_kernels = len(self.kernels)

            # Now, we can just compute the kernels individually and then put
            # them together to be block diagonal.
            for i, cur_kernel in enumerate(self.kernels):

                relevant = X1[:, input_dim] == i

                if np.sum(relevant) == 0:
                    # Nothing to do
                    continue

                relevant_rows = X1[relevant, :]

                computed_kernel = cur_kernel.get_flat_gradients(
                    relevant_rows, relevant_rows)

                # Now have the relevant gradient matrices for this kernel.
                # We're assuming that these are all the same shape.
                for cur_sub_kernel in computed_kernel:

                    all_kernels = [
                        sps.csc_matrix((cur_sub_kernel.shape[0],
                                        cur_sub_kernel.shape[1]))
                        for _ in range(num_kernels)]
                    all_kernels[i] = cur_sub_kernel
                    cur_computed = sps.block_diag(all_kernels, format='csc')
                    sub_kernels.append(cur_computed)

            return sub_kernels

        else:

            return self.get_flat_gradients_agnostic(X1, X2)

    def get_flat_hyperparameters(self):

        all_hypers = list()

        for cur_kernel in self.kernels:
            all_hypers.extend(cur_kernel.get_flat_hyperparameters().tolist())

        return np.array(all_hypers)

    def set_flat_hyperparameters(self, flat_hyperparameters):

        for cur_kernel in self.kernels:

            # Get the current hypers
            cur_hypers = cur_kernel.get_flat_hyperparameters()

            # See how many there are
            num_hypers = cur_hypers.shape[0]

            # Take these off the flat hyperparameters
            to_assign = flat_hyperparameters[:num_hypers]

            cur_kernel.set_flat_hyperparameters(to_assign)

            # Update the flat ones
            flat_hyperparameters = flat_hyperparameters[num_hypers:]
