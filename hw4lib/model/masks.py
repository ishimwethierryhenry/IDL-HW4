import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask for padding positions. 
    Args:
        padded_input: The input tensor, shape (N, T, ...).
        input_lengths: Actual lengths before padding, shape (N,).
    Returns:
        Boolean mask tensor with shape (N, T).
    """
    # T is the padded sequence length (second dimension of input)
    handel_T = padded_input.shape[1]

    # create a range tensor [0, 1, 2, ..., T-1] on the same device as input
    # shape: (1, T) for broadcasting against (N, 1)
    henry_positions = torch.arange(handel_T, device=padded_input.device).unsqueeze(0)

    # input_lengths shape: (N,), unsqueeze to (N, 1) for broadcasting
    # position >= length means that position is padding
    thierry_mask = henry_positions >= input_lengths.unsqueeze(1)

    return thierry_mask  # shape: (N, T), True = padding


''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a causal mask for self-attention. 
    Args:
        padded_input: Input tensor, shape (N, T, ...).
    Returns:
        Boolean mask tensor with shape (T, T).
    """
    # T is the sequence length
    handel_T = padded_input.shape[1]

    # upper triangular with diagonal=1 means:
    # position i can attend to positions 0..i (False) but not i+1..T-1 (True)
    # torch.triu with diagonal=1 sets everything above the main diagonal to 1
    henry_mask = torch.triu(
        torch.ones(handel_T, handel_T, device=padded_input.device, dtype=torch.bool),
        diagonal=1
    )

    return henry_mask  # shape: (T, T), True = cannot attend (future positions)
