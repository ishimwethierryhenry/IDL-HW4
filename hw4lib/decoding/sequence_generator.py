import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
Sequence generation strategies for transformer models.

1. Greedy Search: always picks the highest probability next token
2. Beam Search: keeps top-k candidate sequences at each step
3. Sampling: probabilistic sampling with temperature/top-k/top-p filters
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        handel_batch_size = x.size(0)
        # cumulative log-prob scores for each sequence
        thierry_scores   = torch.zeros(handel_batch_size, device=x.device)
        # track which sequences have hit EOS already
        ishimwe_finished = torch.zeros(handel_batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if ishimwe_finished.all():
                break

            # get logits for the next token from the model
            henry_logits = self.score_fn(x)                                          # (B, vocab)
            henry_logits = self._apply_repeat_penalty(henry_logits, x, repeat_penalty)
            henry_logits = henry_logits / temperature
            henry_log_probs = torch.log_softmax(henry_logits, dim=-1)                # (B, vocab)

            # greedily pick the highest probability token
            handel_next_tokens  = henry_log_probs.argmax(dim=-1)                     # (B,)
            handel_token_scores = henry_log_probs.gather(
                1, handel_next_tokens.unsqueeze(1)
            ).squeeze(1)                                                               # (B,)

            # only accumulate score for sequences that are not yet done
            thierry_scores = torch.where(
                ishimwe_finished, thierry_scores, thierry_scores + handel_token_scores
            )

            # append the new token
            x = torch.cat([x, handel_next_tokens.unsqueeze(1)], dim=1)               # (B, T+1)

            # mark sequences that just produced EOS as finished
            ishimwe_finished = ishimwe_finished | (handel_next_tokens == self.tokenizer.eos_id)

        return x, thierry_scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length)
             - scores is of shape (batch_size, beam_width)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")

        handel_batch_size = x.size(0)
        handel_vocab_size = None   # will be set after first score_fn call

        # --- step 1: get initial logits and pick top beam_width tokens ---
        henry_logits = self.score_fn(x)                                               # (B, vocab)
        handel_vocab_size = henry_logits.size(-1)

        henry_logits = self._apply_repeat_penalty(henry_logits, x, repeat_penalty)
        henry_logits = henry_logits / temperature
        henry_log_probs = torch.log_softmax(henry_logits, dim=-1)                    # (B, vocab)

        # pick top beam_width tokens to start each beam
        # scores shape: (B, beam_width)
        thierry_scores, handel_next_tokens = henry_log_probs.topk(beam_width, dim=-1)

        # expand x to beam dimension: (B, seq_len) -> (B, beam_width, seq_len)
        # then append the chosen first tokens
        x = x.unsqueeze(1).expand(-1, beam_width, -1)                                # (B, W, T)
        x = torch.cat([x, handel_next_tokens.unsqueeze(-1)], dim=-1)                 # (B, W, T+1)

        # finished flags per beam: (B, beam_width)
        ishimwe_finished = (handel_next_tokens == self.tokenizer.eos_id)

        # --- step 2: expand beams step by step ---
        for _ in range(1, self.max_length - x.size(2) + 1):
            if ishimwe_finished.all():
                break

            # score each beam independently
            handel_beam_logits = []
            for w in range(beam_width):
                # x[:, w, :] is (B, T) - the current sequence for beam w
                beam_logits = self.score_fn(x[:, w, :])                              # (B, vocab)
                handel_beam_logits.append(beam_logits)

            # stack into (B, beam_width, vocab)
            handel_beam_logits = torch.stack(handel_beam_logits, dim=1)

            # apply repeat penalty and temperature
            handel_beam_logits = self._apply_repeat_penalty(
                handel_beam_logits, x, repeat_penalty
            )
            handel_beam_logits = handel_beam_logits / temperature
            handel_beam_log_probs = torch.log_softmax(handel_beam_logits, dim=-1)    # (B, W, vocab)

            # cumulative scores: add current step log-probs to existing beam scores
            # thierry_scores is (B, W), expand to (B, W, vocab) for addition
            ishimwe_cum_scores = thierry_scores.unsqueeze(-1) + handel_beam_log_probs # (B, W, vocab)

            # flatten beam x vocab to pick global top-beam_width candidates
            ishimwe_cum_scores_flat = ishimwe_cum_scores.view(handel_batch_size, -1)  # (B, W*vocab)

            # select top beam_width candidates
            thierry_scores, handel_top_indices = ishimwe_cum_scores_flat.topk(
                beam_width, dim=-1
            )                                                                          # (B, W)

            # recover which beam and which token each top index came from
            handel_beam_indices = handel_top_indices // handel_vocab_size              # (B, W)
            handel_token_ids    = handel_top_indices  % handel_vocab_size              # (B, W)

            # reorder x according to selected beam indices
            # gather the right beam sequences for each batch item
            handel_beam_indices_exp = handel_beam_indices.unsqueeze(-1).expand(
                -1, -1, x.size(2)
            )                                                                          # (B, W, T)
            x = x.gather(1, handel_beam_indices_exp)                                  # (B, W, T)

            # append the newly chosen tokens
            x = torch.cat([x, handel_token_ids.unsqueeze(-1)], dim=-1)               # (B, W, T+1)

            # update finished flags - reorder to match new beam order then check EOS
            ishimwe_finished = ishimwe_finished.gather(1, handel_beam_indices)
            ishimwe_finished = ishimwe_finished | (handel_token_ids == self.tokenizer.eos_id)

        # sort beams in descending order of score so beam 0 is always the best
        thierry_scores, handel_sort_idx = thierry_scores.sort(dim=-1, descending=True)
        handel_sort_idx_exp = handel_sort_idx.unsqueeze(-1).expand(-1, -1, x.size(2))
        x = x.gather(1, handel_sort_idx_exp)

        return x, thierry_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            next_scores = self.score_fn(x)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

            scores = torch.where(finished, scores, scores + token_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)

            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        eos_mask = seq == tokenizer.eos_id
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]