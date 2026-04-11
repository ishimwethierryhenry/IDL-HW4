import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

'''
Transformer Encoder Layer for HW4P2.

The encoder processes the full speech sequence and does NOT use a causal mask.
Every position can attend to every other position - this is bidirectional attention,
which makes sense for speech because we have access to the whole utterance at once.

Compare this to the decoder's SelfAttentionDecoderLayer which uses a causal mask
to prevent attending to future tokens. The encoder has no such restriction.

SelfAttentionEncoderLayer:
  - self_attn : SelfAttentionLayer (no causal mask passed in forward)
  - ffn       : FeedForwardLayer

Attribute names self_attn and ffn are checked by the test suite directly.
'''


class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer.

    Each encoder layer is just:
        self-attention (bidirectional, no causal mask) -> feedforward

    Both sublayers use pre-norm and residual connections internally,
    so this class just calls them in sequence and passes through the padding mask.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Args:
            d_model   : model dimension
            num_heads : number of attention heads
            d_ff      : inner dimension of the feedforward sublayer
            dropout   : dropout probability used in both sublayers
        '''
        super().__init__()

        # bidirectional self-attention sublayer
        # no causal mask will be applied - the encoder sees the full sequence
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # position-wise feedforward sublayer
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            x                : (B, T, d_model) - encoder input (speech embeddings)
            key_padding_mask : (B, T) bool mask - True means this is a padding position

        Returns:
            x            : (B, T, d_model) - encoder output
            attn_weights : (B, T, T) - self-attention weights

        Note: We do NOT pass an attn_mask (causal mask) here.
        That is the key difference from the decoder self-attention layer.
        The encoder is free to look at all positions in both directions.
        '''
        # self-attention with padding mask only, no causal restriction
        # attn_mask=None means "attend everywhere" - that is what we want for the encoder
        henry_enc_x, thierry_enc_weights = self.self_attn(
            x=x,
            key_padding_mask=key_padding_mask,
            attn_mask=None   # no causal mask for encoder - bidirectional attention
        )

        # feedforward sublayer
        henry_enc_x = self.ffn(henry_enc_x)

        return henry_enc_x, thierry_enc_weights
