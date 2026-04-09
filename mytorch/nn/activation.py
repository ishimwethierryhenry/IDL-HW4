import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        # numerically stable softmax: subtract max along the target dim before exp
        # this prevents overflow when values are large
        handel_max = np.max(Z, axis=self.dim, keepdims=True)
        handel_shifted = Z - handel_max

        handel_exp = np.exp(handel_shifted)

        # sum denominator along the right dimension
        henry_denom = np.sum(handel_exp, axis=self.dim, keepdims=True)

        # store the result -- we need A in backward for the Jacobian
        self.A = handel_exp / henry_denom

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # the trick here: move the softmax dim to the last position,
        # flatten everything else into a 2D matrix, compute Jacobian row by row,
        # then move dims back. this makes the per-slice Jacobian easy to handle.

        # figure out which dim we actually used (handle negative indexing)
        thierry_ndim   = len(self.A.shape)
        thierry_dim    = self.dim % thierry_ndim  # convert negative dim to positive

        # step 1: move softmax dimension to the last position
        # e.g. if shape is (N, C, H, W) and dim=1, move axis 1 to last -> (N, H, W, C)
        handel_A_moved    = np.moveaxis(self.A,   thierry_dim, -1)
        handel_dLdA_moved = np.moveaxis(dLdA, thierry_dim, -1)

        # step 2: flatten all leading dims into one batch dimension
        # so (N, H, W, C) becomes (N*H*W, C)
        henry_flat_shape = (-1, handel_A_moved.shape[-1])
        handel_A_2d    = handel_A_moved.reshape(henry_flat_shape)
        handel_dLdA_2d = handel_dLdA_moved.reshape(henry_flat_shape)

        # step 3: compute Jacobian and gradient for each sample in the batch
        # Jacobian J[m,n] = a_m*(1-a_m) if m==n else -a_m*a_n
        # gradient = dLdA @ J  (right multiply)
        # doing this for all batch elements at once with einsum

        # J = diag(a) - a^T @ a  for each row
        # so dLdZ = dLdA @ J = dLdA * a - (dLdA @ a^T) * a
        # let's compute it step by step so it's clear

        # first: element-wise product along the class dimension
        thierry_term1 = handel_dLdA_2d * handel_A_2d  # (batch, C)

        # second: dot product of dLdA with a for each sample, then scale by a
        # sum(dLdA * a) gives a scalar per sample, then multiply by a
        thierry_dot   = np.sum(thierry_term1, axis=-1, keepdims=True)  # (batch, 1)
        thierry_term2 = thierry_dot * handel_A_2d  # (batch, C)

        # gradient = term1 - term2
        handel_dLdZ_2d = thierry_term1 - thierry_term2

        # step 4: reshape back to the moved shape
        handel_dLdZ_moved = handel_dLdZ_2d.reshape(handel_A_moved.shape)

        # step 5: move the last axis back to its original position
        dLdZ = np.moveaxis(handel_dLdZ_moved, -1, thierry_dim)

        return dLdZ
