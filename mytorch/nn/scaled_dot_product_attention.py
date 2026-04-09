import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # softmax goes over the last dimension (the source sequence length S)
        # because each query position attends over all key positions
        self.eps = 1e10  # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) where True = ignore this position
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # dk is the embedding dimension of Q and K -- we divide by sqrt(dk) to stabilize gradients
        handel_dk = Q.shape[-1]

        # step 1: compute raw dot product scores Q @ K^T
        # Q shape: (..., L, E), K shape: (..., S, E)
        # we transpose K on the last two dims to get (..., E, S)
        # result shape: (..., L, S)
        henry_scores = Q @ np.swapaxes(K, -2, -1)

        # step 2: scale by 1/sqrt(dk) to keep gradients from getting too large
        thierry_scaled = henry_scores / np.sqrt(handel_dk)

        # step 3: apply mask before softmax if provided
        # where mask is True, we subtract a huge number so those positions
        # become essentially zero after softmax -- the model can't attend there
        if mask is not None:
            thierry_scaled = thierry_scaled - self.eps * mask.astype(float)

        # step 4: softmax over last dimension (S) to get attention weights
        # store these for the backward pass
        self.attention_scores = self.softmax.forward(thierry_scaled)

        # step 5: weighted sum of values
        # attention_scores shape: (..., L, S), V shape: (..., S, Ev)
        # output shape: (..., L, Ev)
        handel_output = self.attention_scores @ V

        # also store Q, K, V for backward
        self.Q = Q
        self.K = K
        self.V = V

        return handel_output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        handel_dk = self.Q.shape[-1]

        # gradient with respect to V
        # output = attention_scores @ V, so dV = attention_scores^T @ d_output
        # attention_scores shape: (..., L, S), d_output shape: (..., L, Ev)
        # dV shape: (..., S, Ev)
        henry_dV = np.swapaxes(self.attention_scores, -2, -1) @ d_output

        # gradient with respect to attention scores
        # output = A @ V, so dA = d_output @ V^T
        # d_output shape: (..., L, Ev), V shape: (..., S, Ev)
        # dA shape: (..., L, S)
        thierry_dA = d_output @ np.swapaxes(self.V, -2, -1)

        # gradient through softmax
        # this calls the softmax backward we implemented
        handel_dS = self.softmax.backward(thierry_dA)

        # gradient with respect to Q and K, with the 1/sqrt(dk) scaling
        # scaled = QK^T / sqrt(dk), so dQ = (dS / sqrt(dk)) @ K
        henry_dQ = (handel_dS / np.sqrt(handel_dk)) @ self.K

        # dK = (dS / sqrt(dk))^T @ Q
        # note the transpose on dS here -- gradient flows back through K^T
        thierry_dK = np.swapaxes(handel_dS / np.sqrt(handel_dk), -2, -1) @ self.Q

        return henry_dQ, thierry_dK, henry_dV
