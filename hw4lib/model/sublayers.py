import torch.nn as nn
import torch 
from typing import Tuple, Optional

'''
TODO: Implement these Modules.

The file contains three key sublayers used in transformer decoders:
1. SelfAttentionLayer: For masked self-attention
2. CrossAttentionLayer: For cross-attention between encoder and decoder
3. FeedForwardLayer: For position-wise feed-forward processing

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Normalization is applied before the main operation
- A residual connection wraps around the operation
'''

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    
    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply self-attention with masking
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights    
    ''' 
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the SelfAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # multi-head attention -- this is PyTorch's nn.MultiheadAttention, not our mytorch one
        # batch_first=True means input shape is (B, T, d_model) rather than (T, B, d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # layer norm goes before the attention (pre-norm architecture)
        self.norm = nn.LayerNorm(d_model)

        # dropout applied after the attention output before adding residual
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)   
            key_padding_mask (Optional[torch.Tensor]): Padding mask, shape (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (seq_len, seq_len)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): Attention weights, shape (batch_size, seq_len, seq_len)   
        '''
        # store the original input for the residual connection
        handel_residual = x

        # pre-norm: normalize before passing into attention
        henry_normed = self.norm(x)

        # self-attention: Q, K, V all come from the same input (hence self-attention)
        # need_weights=True so we get attention weights back for visualization
        thierry_attn_out, mha_attn_weights = self.mha(
            henry_normed,
            henry_normed,
            henry_normed,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False  # keep per-head weights
        )

        # residual connection with dropout
        x = handel_residual + self.dropout(thierry_attn_out)

        return x, mha_attn_weights

    
## -------------------------------------------------------------------------------------------------  
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    This layer is responsible for the cross-attention mechanism between encoder and decoder.
    
    Steps to implement:
    1. Initialize the multi-head attention with proper parameters
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply cross-attention (query from decoder, key/value from encoder)
       d. Apply residual connection with dropout
       e. Return the output tensor and attention weights (both are needed)    
    '''     
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the CrossAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # multi-head attention for cross-attention
        # Q comes from decoder, K and V come from encoder
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # pre-norm applied to the decoder input (not the encoder output)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionLayer.
        Args:
            x (torch.Tensor): Decoder input, shape (batch_size, seq_len, d_model)   
            y (torch.Tensor): Encoder input, shape (batch_size, seq_len, d_model)
            key_padding_mask (Optional[torch.Tensor]): Padding mask, shape (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (seq_len, seq_len)

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): Attention weights, shape (batch_size, seq_len, seq_len)   
        '''
        # save decoder input for residual
        handel_residual = x

        # pre-norm on decoder input only
        henry_normed = self.norm(x)

        # cross-attention: Q from decoder (normed), K and V from encoder output
        # key_padding_mask is for the encoder side (masks encoder padding)
        thierry_attn_out, mha_attn_weights = self.mha(
            henry_normed,
            y,
            y,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False
        )

        # residual connection
        x = handel_residual + self.dropout(thierry_attn_out)

        return x, mha_attn_weights

    
## -------------------------------------------------------------------------------------------------  
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    
    Steps to implement:
    1. Initialize the feed-forward network as a Sequential with:
       a. First linear layer: d_model -> d_ff
       b. GELU activation
       c. Dropout
       d. Second linear layer: d_ff -> d_model
    2. Initialize layer normalization for d_model dimensionality
    3. Initialize dropout with specified rate
    4. In forward pass:
       a. Store residual connection
       b. Apply pre-normalization
       c. Apply feed-forward network with dropout
       d. Add residual connection
       e. Return the output tensor
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer. 
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # the FFN: expand to d_ff, apply GELU, dropout, then compress back to d_model
        # GELU is smoother than ReLU and works better in transformers
        # the test suite specifically checks that self.ffn is a nn.Sequential
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model)
        )

        # pre-norm layer
        self.norm = nn.LayerNorm(d_model)

        # dropout applied after the FFN before adding residual
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)   

        Returns:
            x (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
        '''
        # save for residual connection
        handel_residual = x

        # pre-norm then FFN then residual
        henry_normed = self.norm(x)
        thierry_ffn_out = self.ffn(henry_normed)
        x = handel_residual + self.dropout(thierry_ffn_out)

        return x
