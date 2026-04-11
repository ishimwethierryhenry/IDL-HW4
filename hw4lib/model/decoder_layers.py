import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

'''
Transformer Decoder Layers for HW4P1 and HW4P2.

Two decoder layer types live here:

1. SelfAttentionDecoderLayer (P1, decoder-only transformer like GPT)
   - masked self-attention -> feedforward
   - uses causal mask so each token only sees previous tokens

2. CrossAttentionDecoderLayer (P2, encoder-decoder transformer)
   - masked self-attention -> cross-attention -> feedforward
   - self-attention: causal, decoder attends to its own previous tokens
   - cross-attention: decoder queries attend to encoder key/value pairs
   - this is how the decoder conditions on the speech encoder output

Attribute names checked by test suite:
  SelfAttentionDecoderLayer  : self_attn, ffn
  CrossAttentionDecoderLayer : self_attn, cross_attn, ffn
'''


## -------------------------------------------------------------------------------------------------
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer for the decoder-only transformer (HW4P1).

    Flow: masked self-attention -> feedforward
    The causal mask is passed in from outside (not created here).
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Args:
            d_model   : model dimension
            num_heads : number of attention heads
            d_ff      : feedforward inner dimension
            dropout   : dropout rate
        '''
        super().__init__()

        # causally masked self-attention
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # position-wise feedforward
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            x                : (B, T, d_model) - decoder input
            key_padding_mask : (B, T) - padding mask for decoder input
            attn_mask        : (T, T) - causal mask

        Returns:
            x            : (B, T, d_model) - output
            attn_weights : (B, T, T) - self-attention weights
        '''
        # self-attention with causal mask
        handel_x, handel_weights = self.self_attn(
            x=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )

        # feedforward
        handel_x = self.ffn(handel_x)

        return handel_x, handel_weights


## -------------------------------------------------------------------------------------------------
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer for the encoder-decoder transformer (HW4P2).

    Flow: masked self-attention -> cross-attention -> feedforward

    The self-attention is still causal (decoder can only see previous tokens).
    The cross-attention lets the decoder query the encoder output to condition
    its predictions on the speech features.

    This is the core layer that makes encoder-decoder work. The decoder
    "reads" the speech through cross-attention at every layer.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Args:
            d_model   : model dimension
            num_heads : number of attention heads
            d_ff      : feedforward inner dimension
            dropout   : dropout rate
        '''
        super().__init__()

        # sublayer 1: causally masked self-attention on the decoder sequence
        self.self_attn = SelfAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # sublayer 2: cross-attention - decoder queries encoder key/value pairs
        self.cross_attn = CrossAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # sublayer 3: position-wise feedforward
        self.ffn = FeedForwardLayer(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        dec_key_padding_mask: Optional[torch.Tensor] = None,
        enc_key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Args:
            x                    : (B, T_dec, d_model) - decoder input sequence
            enc_output           : (B, T_enc, d_model) - encoder hidden states
            dec_key_padding_mask : (B, T_dec) - padding mask for decoder positions
            enc_key_padding_mask : (B, T_enc) - padding mask for encoder positions
            attn_mask            : (T_dec, T_dec) - causal mask for self-attention

        Returns:
            x                : (B, T_dec, d_model) - decoder output
            self_attn_weights  : (B, T_dec, T_dec) - decoder self-attention weights
            cross_attn_weights : (B, T_dec, T_enc) - cross-attention weights
                                  These tell us which speech frames the decoder
                                  was attending to when generating each token.
        '''
        # step 1: causally masked self-attention on decoder tokens
        # each decoder token can only look at tokens before it (enforced by attn_mask)
        ishimwe_x, ishimwe_self_weights = self.self_attn(
            x=x,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask
        )

        # step 2: cross-attention - decoder queries into encoder output
        # Q comes from decoder (ishimwe_x), K and V come from encoder (enc_output)
        # enc_key_padding_mask prevents attending to padded encoder frames
        ishimwe_x, ishimwe_cross_weights = self.cross_attn(
            x=ishimwe_x,
            y=enc_output,
            key_padding_mask=enc_key_padding_mask,
            attn_mask=None   # no causal constraint on cross-attention
        )

        # step 3: feedforward
        ishimwe_x = self.ffn(ishimwe_x)

        return ishimwe_x, ishimwe_self_weights, ishimwe_cross_weights
