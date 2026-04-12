import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary

'''
Two transformer architectures:

1. DecoderOnlyTransformer (HW4P1)
   GPT-style. Takes a sequence of token ids, predicts next token at every position.
   Uses causal masking so each position only attends to previous positions.

2. EncoderDecoderTransformer (HW4P2)
   Original transformer style. Encoder processes speech features, decoder
   generates text tokens autoregressively using cross-attention into the encoder.
   Includes a CTC auxiliary head on the encoder output for faster alignment learning.
'''


## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer (HW4P1)
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    Pre-LN Decoder-Only Transformer. Same basic idea as GPT-2.
    Used for causal language modeling in HW4P1.
    '''
    def __init__(
            self,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float,
            max_len: int,
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        '''
        Args:
            num_layers      : number of decoder layers
            d_model         : embedding dimension
            num_heads       : attention heads per layer
            d_ff            : feedforward inner dimension
            dropout         : dropout rate
            max_len         : max sequence length (for positional encoding)
            num_classes     : vocabulary size
            weight_tying    : tie embedding weights with final linear projection
            layer_drop_rate : probability of randomly skipping a layer during training
        '''
        super().__init__()

        # store these as attributes - the trainer accesses them
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers

        # stack of decoder layers
        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # token embedding: maps token ids to d_model-dimensional vectors
        self.target_embedding = nn.Embedding(num_classes, d_model)

        # sinusoidal positional encoding, added to embeddings
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # final linear projection: d_model -> num_classes (vocabulary logits)
        self.final_linear = nn.Linear(d_model, num_classes, bias=False)

        # dropout applied after embedding + positional encoding
        self.dropout = nn.Dropout(dropout)

        # final layer norm applied before the projection (pre-norm architecture)
        self.norm = nn.LayerNorm(d_model)

        # weight tying: the embedding matrix and the output projection share weights
        # this reduces parameters and often improves perplexity
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(
        self,
        padded_targets: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass. Used during training only. Processes all tokens in parallel.

        Args:
            padded_targets : (B, T) - right-padded token id sequences
            target_lengths : (B,) - actual (non-padded) lengths of each sequence

        Returns:
            seq_out    : (B, T, num_classes) - logit predictions for each position
            runnint_att: dict of attention weights from each layer
        '''
        # must have lengths during training so we can make the padding mask
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        # padding mask: True where the token is padding (should be ignored)
        # shape (B, T), True = padding position
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths).to(padded_targets.device)

        # causal mask: True where attention should be blocked (upper triangle)
        # shape (T, T)
        # thierry_seq_len = padded_targets.size(1)
        # causal_mask = CausalMask(thierry_seq_len).to(padded_targets.device)
        causal_mask = CausalMask(padded_targets).to(padded_targets.device)

        # embed tokens and add positional encoding
        # embedding: (B, T) -> (B, T, d_model)
        x = self.dropout(self.positional_encoding(self.target_embedding(padded_targets)))

        # pass through decoder layers
        runnint_att = {}
        for i in range(self.num_layers):
            # LayerDrop: randomly skip layers during training for regularization
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue

            x, handel_attn = self.dec_layers[i](
                x=x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask
            )
            runnint_att[f'layer{i+1}_dec_self'] = handel_attn

        # final norm and project to vocabulary logits
        seq_out = self.final_linear(self.norm(x))

        return seq_out, runnint_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score function used during generation (greedy/beam search).
        Takes a batch of prompts and returns the logits for the NEXT token only.
        No padding mask is applied - prompts are assumed to be of equal length and unpadded.

        Args:
            batch_prompts : (B, T) - current sequence of tokens

        Returns:
            logits : (B, num_classes) - logits for the next token
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        # forward with no lengths (no padding mask)
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)

        # return only the last position's logits for next-token prediction
        return seq_out[:, -1, :]


## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer (HW4P2)
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    Pre-LN Encoder-Decoder Transformer for ASR (HW4P2).

    The encoder processes speech features (80-dim filterbanks) through:
        SpeechEmbedding (time reduction) -> PositionalEncoding -> N x EncoderLayer -> LayerNorm

    The decoder generates text tokens through:
        Embedding -> PositionalEncoding -> N x CrossAttentionDecoderLayer -> LayerNorm -> Linear

    The CTC head on the encoder output provides an auxiliary training signal
    that helps the model learn alignment between speech frames and text tokens.
    This usually speeds up convergence significantly.
    '''
    def __init__(
            self,
            input_dim: int,
            time_reduction: int,
            reduction_method: Literal['lstm', 'conv', 'both'],
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int,
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float,
            max_len: int,
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            skip_encoder_pe: bool = False,
            skip_decoder_pe: bool = False,
    ):
        '''
        Args:
            input_dim          : dimension of input speech features (80 for filterbanks)
            time_reduction     : stride factor for the SpeechEmbedding time downsampling
            reduction_method   : 'conv', 'lstm', or 'both' - how SpeechEmbedding reduces time
            num_encoder_layers : number of encoder layers
            num_encoder_heads  : number of heads in encoder self-attention
            d_ff_encoder       : feedforward inner dimension for encoder
            num_decoder_layers : number of decoder layers
            num_decoder_heads  : number of heads in decoder attention
            d_ff_decoder       : feedforward inner dimension for decoder
            d_model            : shared model dimension for encoder and decoder
            dropout            : dropout rate
            max_len            : maximum sequence length for positional encoding
            num_classes        : vocabulary size
            weight_tying       : tie target embedding weights with final linear weights
            layer_drop_rate    : probability of dropping a layer during training
            skip_encoder_pe    : skip positional encoding for encoder (useful with LSTM embedding)
            skip_decoder_pe    : skip positional encoding for decoder
        '''
        super().__init__()

        # store as attributes - the progressive trainer and from_pretrained_decoder need these
        self.max_len            = max_len
        self.layer_drop_rate    = layer_drop_rate
        self.num_classes        = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe    = skip_encoder_pe
        self.skip_decoder_pe    = skip_decoder_pe

        # encoder stack: N layers of unmasked self-attention + feedforward
        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(
                d_model=d_model,
                num_heads=num_encoder_heads,
                d_ff=d_ff_encoder,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])

        # decoder stack: N layers of masked self-attention + cross-attention + feedforward
        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(
                d_model=d_model,
                num_heads=num_decoder_heads,
                d_ff=d_ff_decoder,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])

        # speech embedding: converts raw filterbank features to d_model dimension
        # also applies time reduction (downsampling along time axis)
        # input: (B, T, input_dim), output: (B, T', d_model) where T' = T / time_reduction
        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout
        )

        # text token embedding for the decoder
        self.target_embedding = nn.Embedding(num_classes, d_model)

        # one shared positional encoding module, used for both encoder and decoder
        # the skip flags control whether we actually apply it (useful when LSTM already
        # captures position information from the speech embedding)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # final linear projection for the decoder output
        # d_model -> num_classes (vocabulary logits)
        self.final_linear = nn.Linear(d_model, num_classes, bias=False)

        # dropout applied after embedding (both encoder and decoder side)
        self.dropout = nn.Dropout(dropout)

        # separate layer norms for encoder and decoder output
        # applied before the CTC head / final linear respectively
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # CTC head: projects encoder output to vocabulary logits + log_softmax
        # nn.CTCLoss expects log-probabilities, so we apply LogSoftmax here
        # shape: (T, B, num_classes) is what CTCLoss expects, we transpose in encode()
        self.ctc_head = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # weight tying between target embedding and final linear projection
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(
        self,
        padded_sources: torch.Tensor,
        source_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict, dict]:
        '''
        Run the encoder on speech features.

        Args:
            padded_sources : (B, T, input_dim) - padded filterbank features
            source_lengths : (B,) - actual frame lengths before padding

        Returns:
            x_enc        : (B, T', d_model) - encoder output (T' is after time reduction)
            pad_mask_src : (B, T') - padding mask for the reduced-time encoder output
            running_att  : dict of encoder self-attention weights per layer
            ctc_inputs   : dict with keys 'log_probs' and 'lengths' for CTC loss
                           log_probs shape: (T', B, num_classes)
                           lengths shape: (B,)
        '''
        # speech embedding: applies time reduction + projects to d_model
        # source_lengths gets updated to reflect the reduced time dimension
        x_enc, handel_enc_lengths = self.source_embedding(padded_sources, source_lengths)

        # optionally add positional encoding to encoder input
        # can skip if the LSTM speech embedding already captures position well
        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)

        # dropout after embedding
        x_enc = self.dropout(x_enc)

        # padding mask for the encoder after time reduction
        # True where the position is padding (should be ignored in attention)
        pad_mask_src = PadMask(x_enc, handel_enc_lengths).to(x_enc.device)

        # run through encoder layers
        running_att = {}
        for i in range(self.num_encoder_layers):
            # LayerDrop: randomly skip layers during training
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue

            x_enc, henry_enc_attn = self.enc_layers[i](
                x=x_enc,
                key_padding_mask=pad_mask_src
            )
            if not self.training:
                running_att[f'layer{i+1}_enc_self'] = henry_enc_attn

        # final encoder normalization
        x_enc = self.encoder_norm(x_enc)

        # CTC head: compute log-probabilities over vocabulary from encoder output
        # CTCLoss expects shape (T, B, num_classes), so we permute (B, T, C) -> (T, B, C)
        ctc_logits = self.ctc_head(x_enc)                 # (B, T', num_classes)
        ctc_logits_tbv = ctc_logits.permute(1, 0, 2)       # (T', B, num_classes)

        ctc_inputs = {
            'log_probs': ctc_logits_tbv,          # (T', B, num_classes)
            'lengths': handel_enc_lengths.long()  # (B,) - actual frame counts after reduction
        }

        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
        self,
        padded_targets: torch.Tensor,
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Run the decoder on token sequences conditioned on encoder output.

        Args:
            padded_targets  : (B, T_dec) - shifted token ids (SOS-prepended)
            encoder_output  : (B, T_enc, d_model) - from encode()
            target_lengths  : (B,) - actual lengths of padded_targets (for padding mask)
            pad_mask_src    : (B, T_enc) - encoder padding mask (for cross-attention)

        Returns:
            seq_out     : (B, T_dec, num_classes) - logit predictions
            running_att : dict of self-attention and cross-attention weights per layer
        '''
        # decoder padding mask (True = padding position)
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths).to(padded_targets.device)

        if pad_mask_tgt is None and self.training:
            warnings.warn(
                "pad_mask_tgt is None during training. Provide target_lengths for correct masking."
            )

        # causal mask to prevent decoder from seeing future tokens
        # shape: (T_dec, T_dec), True in upper triangle = blocked
        # thierry_dec_len = padded_targets.size(1)
        # causal_mask = CausalMask(thierry_dec_len).to(padded_targets.device)
        causal_mask = CausalMask(padded_targets).to(padded_targets.device)

        # embed tokens and optionally add positional encoding
        x_dec = self.target_embedding(padded_targets)  # (B, T_dec, d_model)
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        # pass through cross-attention decoder layers
        running_att = {}
        for i in range(self.num_decoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue

            x_dec, handel_self_attn, handel_cross_attn = self.dec_layers[i](
                x=x_dec,
                enc_output=encoder_output,
                dec_key_padding_mask=pad_mask_tgt,
                enc_key_padding_mask=pad_mask_src,
                attn_mask=causal_mask
            )
            if not self.training:
                   running_att[f'layer{i+1}_dec_self']  = handel_self_attn
                   running_att[f'layer{i+1}_dec_cross'] = handel_cross_attn

        # final decoder normalization and projection to vocabulary
        seq_out = self.final_linear(self.decoder_norm(x_dec))

        return seq_out, running_att

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict, dict]:
        '''
        Full encoder-decoder forward pass. Used during training.

        Args:
            padded_sources : (B, T_src, input_dim) - filterbank features
            padded_targets : (B, T_tgt) - shifted (SOS-prepended) token ids
            source_lengths : (B,) - actual feature lengths before padding
            target_lengths : (B,) - actual transcript lengths before padding

        Returns:
            seq_out     : (B, T_tgt, num_classes) - vocabulary logits from decoder
            running_att : combined dict of all encoder and decoder attention weights
            ctc_inputs  : dict with 'log_probs' and 'lengths' for CTC loss computation
        '''
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")

        # encoder: speech -> hidden states
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(
            padded_sources=padded_sources,
            source_lengths=source_lengths
        )

        # decoder: token ids + encoder hidden states -> vocabulary logits
        seq_out, dec_running_att = self.decode(
            padded_targets=padded_targets,
            encoder_output=encoder_output,
            target_lengths=target_lengths,
            pad_mask_src=pad_mask_src
        )

        # merge attention dicts from encoder and decoder
        running_att = {**enc_running_att, **dec_running_att}

        return seq_out, running_att, ctc_inputs

    def score(
        self,
        batch_prompts: torch.Tensor,
        encoder_output: torch.Tensor,
        pad_mask_src: torch.Tensor
    ) -> torch.Tensor:
        '''
        Score function used during inference (greedy/beam search).
        Given encoder output and current decoder prompt, returns logits for the NEXT token.

        Args:
            batch_prompts   : (B, T) - current token sequence (no padding)
            encoder_output  : (B, T_enc, d_model) - from encode()
            pad_mask_src    : (B, T_enc) - encoder padding mask

        Returns:
            logits : (B, num_classes) - logits for the next token
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        # decode with no target lengths (no padding mask - prompts are not padded during inference)
        seq_out, _ = self.decode(
            padded_targets=batch_prompts,
            encoder_output=encoder_output,
            target_lengths=None,
            pad_mask_src=pad_mask_src
        )

        # return only the last position's logits for next-token prediction
        return seq_out[:, -1, :]

    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
    ) -> Tuple['EncoderDecoderTransformer', dict]:
        """
        Initialize an encoder-decoder transformer with decoder weights
        loaded from a pretrained decoder-only model (our HW4P1 checkpoint).

        This is the pre-trained decoder initialization strategy from the bootcamp.
        It loads the target_embedding, final_linear, decoder_norm, and the
        self_attn and ffn sublayers of each decoder layer from the P1 checkpoint.
        The cross_attn sublayers and encoder components are newly initialized.

        Args:
            decoder_checkpoint_path : path to the HW4P1 checkpoint .pth file
            config                  : dict of constructor arguments for this class

        Returns:
            model      : initialized EncoderDecoderTransformer
            param_info : dict with 'transferred' and 'new' parameter lists
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")

        # build the new model first
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # load P1 checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']

        transferred_params = []
        new_params = []

        def transfer_module_weights(target_module, prefix):
            # filter the checkpoint state dict to only include keys starting with prefix
            module_state_dict = {
                k.replace(prefix, ''): v
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} parameters)")
            target_module.load_state_dict(module_state_dict)
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        # transfer shared components
        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')

        # transfer decoder layers (self_attn and ffn sublayers only)
        # the cross_attn sublayers are new and get random initialization
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")

        for i in range(num_layers):
            print(f"\nLayer {i + 1}/{num_layers}:")
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )

        # collect params that were NOT transferred (encoder, cross_attn, ctc_head, etc.)
        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))

        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0

        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable

            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")

        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


## -------------------------------------------------------------------------------------------------
## Test helpers (kept for compatibility with test suite)
## -------------------------------------------------------------------------------------------------

def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def test_decoder_only(
    num_layers: int = 12,
    num_heads: int = 8,
    d_model: int = 512,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_len: int = 300,
    num_classes: int = 1000
):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])


if __name__ == "__main__":
    test_decoder_only()
