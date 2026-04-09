import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        # build an empty table: (max_len, d_model)
        # each row is the positional encoding for one time step
        handel_pe = torch.zeros(max_len, d_model)

        # position indices: shape (max_len, 1) -- one per row
        henry_positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # the division term for the sinusoidal frequencies
        # 10000^(2i/d_model) for i in 0..d_model/2
        # we compute it in log space for numerical stability
        thierry_div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )

        # even indices get sine, odd indices get cosine
        # P[t, 2i]   = sin(t / 10000^(2i/d))
        # P[t, 2i+1] = cos(t / 10000^(2i/d))
        handel_pe[:, 0::2] = torch.sin(henry_positions * thierry_div_term)
        handel_pe[:, 1::2] = torch.cos(henry_positions * thierry_div_term)

        # unsqueeze to (1, max_len, d_model) so it broadcasts across the batch dimension
        pe = handel_pe.unsqueeze(0)

        # register as buffer -- not a learnable parameter, but saved with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # Step 1: Get sequence length from input tensor
        seq_len = x.shape[1]

        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")

        # Step 3: Add positional encodings to input
        # self.pe has shape (1, max_len, d_model), slice to (1, seq_len, d_model)
        # it broadcasts automatically across the batch dimension
        return x + self.pe[:, :seq_len, :]
