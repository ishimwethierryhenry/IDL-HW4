import torch.nn as nn
import torch
from typing import Tuple, Optional

'''
Pre-LN Transformer Sublayers for HW4P1 and HW4P2.

Three sublayers live here:
  1. SelfAttentionLayer  - causally masked self-attention (used in decoder)
  2. CrossAttentionLayer - cross-attention between decoder query and encoder key/value
  3. FeedForwardLayer    - two-layer FFN with GELU activation

All three use the Pre-LN (normalize first, then attend, then add residual) design.
This is more stable than the original Post-LN design from "Attention is All You Need".

Quick reminder of pre-norm flow:
    residual = x
    x = LayerNorm(x)
    x = SomeOperation(x)
    x = Dropout(x)
    x = residual + x
    return x
'''

## -------------------------------------------------------------------------------------------------
class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    Causally masked self-attention.
    '''

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        thierry_residual = x
        x_normed = self.norm(x)

        handel_attn_out, handel_weights = self.mha(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True   # MUST stay True for tests
        )

        if handel_weights is not None and handel_weights.dim() == 4:
            handel_weights = handel_weights.mean(dim=1)

        # ✅ FIX: never None, never cpu, just detach
        if handel_weights is not None:
            handel_weights = handel_weights.detach()

        x = thierry_residual + self.dropout(handel_attn_out)

        return x, handel_weights


## -------------------------------------------------------------------------------------------------
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    Cross-attention between decoder and encoder output.
    '''

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ishimwe_residual = x
        x_normed = self.norm(x)

        henry_attn_out, henry_weights = self.mha(
            query=x_normed,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True   # MUST stay True for tests
        )

        if henry_weights is not None and henry_weights.dim() == 4:
            henry_weights = henry_weights.mean(dim=1)

        # ✅ FIX: consistent with self-attention
        if henry_weights is not None:
            henry_weights = henry_weights.detach()

        x = ishimwe_residual + self.dropout(henry_attn_out)

        return x, henry_weights


## -------------------------------------------------------------------------------------------------
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    Position-wise feed-forward network.
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        handel_residual = x
        x = handel_residual + self.dropout(self.ffn(self.norm(x)))

        return x