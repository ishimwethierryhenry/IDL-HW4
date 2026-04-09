from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # one SDPA instance handles all heads at once because we include H in the tensor dims
        self.attention = ScaledDotProductAttention()

        # three input projections (embed_dim -> embed_dim) and one output projection
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where True indicates positions to ignore
        :param attn_mask: (L, S) where True indicates positions to ignore
        :return: (N, L, E)
        """
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # project Q, K, V -- each goes through its own learned linear layer
        # output shape still (N, L/S, E)
        handel_q = self.q_proj.forward(query)
        handel_k = self.k_proj.forward(key)
        handel_v = self.v_proj.forward(value)

        # split into H heads -- this is where the multi-head magic happens
        # (N, L, E) -> (N, H, L, E/H)
        handel_q = self._split_heads(handel_q)
        handel_k = self._split_heads(handel_k)
        handel_v = self._split_heads(handel_v)

        # merge padding mask and causal mask into a single (N, H, L, S) mask
        henry_mask = self._merge_masks(key_padding_mask, attn_mask)

        # run scaled dot product attention across all heads at once
        # Q shape: (N, H, L, E/H), K shape: (N, H, S, E/H)
        # output shape: (N, H, L, E/H)
        thierry_attn_out = self.attention.forward(handel_q, handel_k, handel_v, henry_mask)

        # concatenate heads back: (N, H, L, E/H) -> (N, L, E)
        handel_concat = self._concat_heads(thierry_attn_out)

        # final output projection to mix information across heads
        output = self.out_proj.forward(handel_concat)

        return output

    def backward(self, d_output):
        """
        Backward pass for multi-head attention.
        """
        # step 1: backprop through output projection
        # d_output shape: (N, L, E)
        handel_d_concat = self.out_proj.backward(d_output)

        # step 2: split the gradient back into heads
        # (N, L, E) -> (N, H, L, E/H)
        thierry_d_attn_out = self._split_heads(handel_d_concat)

        # step 3: backprop through scaled dot product attention
        henry_dq_heads, henry_dk_heads, henry_dv_heads = self.attention.backward(thierry_d_attn_out)

        # step 4: concatenate head gradients back to full embedding dim
        # (N, H, L, E/H) -> (N, L, E) for each of Q, K, V
        handel_dq = self._concat_heads(henry_dq_heads)
        handel_dk = self._concat_heads(henry_dk_heads)
        handel_dv = self._concat_heads(henry_dv_heads)

        # step 5: backprop through input projections
        thierry_dq = self.q_proj.backward(handel_dq)
        thierry_dk = self.k_proj.backward(handel_dk)
        thierry_dv = self.v_proj.backward(handel_dv)

        return thierry_dq, thierry_dk, thierry_dv

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge padding mask and attention mask into one combined mask.
        Both inputs are 2D but attention has 4 dims, so we expand for broadcasting.
        """
        handel_combined = None

        if key_padding_mask is not None:
            # key_padding_mask shape: (N, S)
            # we need (N, 1, 1, S) so it broadcasts over H and L dims
            henry_key_mask = key_padding_mask[:, np.newaxis, np.newaxis, :]
            handel_combined = henry_key_mask

        if attn_mask is not None:
            # attn_mask shape: (L, S)
            # we need (1, 1, L, S) so it broadcasts over N and H dims
            thierry_attn_mask = attn_mask[np.newaxis, np.newaxis, :, :]
            if handel_combined is None:
                handel_combined = thierry_attn_mask
            else:
                # OR the two masks together -- a position is masked if either mask says so
                handel_combined = handel_combined | thierry_attn_mask

        return handel_combined

    def _split_heads(self, x):
        """
        Reshape (N, seq_len, E) into (N, H, seq_len, E/H) for multi-head attention.
        """
        # get dimensions
        handel_N   = x.shape[0]
        handel_seq = x.shape[1]
        handel_d_k = self.embed_dim // self.num_heads

        # reshape: split the last dim into (num_heads, d_k)
        # then transpose to put num_heads right after batch
        x = x.reshape(handel_N, handel_seq, self.num_heads, handel_d_k)
        x = x.transpose(0, 2, 1, 3)  # (N, H, seq_len, d_k)

        return x

    def _concat_heads(self, x):
        """
        Reverse of _split_heads: (N, H, seq_len, d_k) -> (N, seq_len, E).
        """
        # x shape: (N, H, seq_len, d_k)
        handel_N   = x.shape[0]
        handel_seq = x.shape[2]

        # transpose back: (N, seq_len, H, d_k)
        x = x.transpose(0, 2, 1, 3)

        # merge last two dims back into E = H * d_k
        x = x.reshape(handel_N, handel_seq, self.embed_dim)

        return x
