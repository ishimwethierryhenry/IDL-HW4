import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # store the original shape so backward can unflatten gradients later
        self.A = A
        handel_original_shape = A.shape

        # collapse everything except the last dim into one batch dimension
        # e.g. (B, T, in_features) -> (B*T, in_features)
        handel_flat = A.reshape(-1, handel_original_shape[-1])

        # affine transform: Z = A @ W^T + b
        # same as hw1, just now the "batch" might have come from multiple dims
        henry_flat_out = handel_flat @ self.W.T + self.b

        # unflatten back: last dim becomes out_features, rest stays the same
        thierry_out_shape = handel_original_shape[:-1] + (self.W.shape[0],)
        Z = henry_flat_out.reshape(thierry_out_shape)

        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # grab original input shape so we can restore it at the end
        handel_original_shape = self.A.shape

        # flatten the incoming gradient to 2D: (batch_size, out_features)
        henry_dLdZ_flat = dLdZ.reshape(-1, self.W.shape[0])

        # flatten stored input to 2D: (batch_size, in_features)
        henry_A_flat = self.A.reshape(-1, handel_original_shape[-1])

        # dL/dA = dL/dZ @ W  -- gradient flows back through the weights
        thierry_dLdA_flat = henry_dLdZ_flat @ self.W

        # dL/dW = (dL/dZ)^T @ A  -- each weight accumulates grad from all positions
        self.dLdW = henry_dLdZ_flat.T @ henry_A_flat

        # dL/db = sum over batch -- bias gets gradient summed across all tokens
        self.dLdb = henry_dLdZ_flat.sum(axis=0)

        # reshape dL/dA back to original input shape before returning
        self.dLdA = thierry_dLdA_flat.reshape(handel_original_shape)

        return self.dLdA
