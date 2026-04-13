from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from ..decoding.sequence_generator import SequenceGenerator
from ..utils import create_scheduler, create_optimizer
from ..model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
# torchmetrics.text imported locally in _calculate_asr_metrics
from torch.utils.data import Subset
import pandas as pd


class ASRTrainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles training, validation, and recognition loops.

    This trainer implements:
    1. Training loop with gradient accumulation, mixed precision training, and optional CTC loss
    2. Validation loop for model evaluation
    3. Recognition capabilities with different decoding strategies (greedy, beam search)
    4. Language model shallow fusion during recognition

    Implementation Tasks:
    - TODO: Initialize CE and CTC loss in __init__
    - TODO: Implement key parts of the training loop in _train_epoch
    - TODO: Implement recognition functionality in recognize
    - TODO: Implement key parts of the validation loop in _validate_epoch
    - TODO: Implement key parts of the full training loop in train

    Implementation Notes:
    1. For __init__:
        - Initialize CrossEntropyLoss with appropriate padding index and label smoothing
        - Initialize CTCLoss if ctc_weight > 0
        
    2. For _train_epoch:
        - Unpack the batch (features, shifted targets, golden targets, lengths)
        - Get model predictions, attention weights and CTC inputs
        - Calculate CE loss and CTC loss if enabled
        - Backpropagate the loss
        
    3. For _validate_epoch:
        - Use recognize() to generate transcriptions
        - Extract references and hypotheses from recognition results
        
    4. For train:
        - Set maximum transcript length
        - Implement epoch loop with training and validation
        
    5. For recognize:
        - Run inference
        - Handle both greedy and optionally beam search decoding
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # cross-entropy loss for the decoder predictions
        # ignore_index=pad_id so we do not compute loss on padding tokens
        # label_smoothing helps prevent overconfident predictions (regularization)
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss'].get('label_smoothing', 0.0)
        )

        # CTC loss for the encoder output (auxiliary training signal)
        # zero_infinity=True prevents NaN when the CTC alignment is impossible
        # (can happen with very short speech or very long transcripts)
        self.ctc_criterion = None
        self.ctc_weight = self.config['loss'].get('ctc_weight', 0.0)
        if self.ctc_weight > 0:
            self.ctc_criterion = nn.CTCLoss(
                blank=self.tokenizer.pad_id,
                zero_infinity=True
            )


    def _train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, torch.Tensor]]: Training metrics and attention weights
        """
        # Initialize training variables
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training ASR]")
        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0
        running_att = {}

        # Only zero gradients when starting a new accumulation cycle
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            # unpack batch and move everything to GPU
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch
            feats              = feats.to(self.device)
            targets_shifted    = targets_shifted.to(self.device)
            targets_golden     = targets_golden.to(self.device)
            feat_lengths       = feat_lengths.to(self.device)
            transcript_lengths = transcript_lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # forward pass through the encoder-decoder model
                # returns logits (B, T_dec, num_classes), attention dicts, and ctc_inputs
                seq_out, curr_att, ctc_inputs = self.model(
                    padded_sources=feats,
                    padded_targets=targets_shifted,
                    source_lengths=feat_lengths,
                    target_lengths=transcript_lengths,
                    collect_attention=False
                )
                
                # CE loss: compare predictions to golden targets
                # seq_out is (B, T, num_classes), need (B, num_classes, T) for CrossEntropyLoss
                # targets_golden is (B, T) of token ids
                ce_loss = self.ce_criterion(
                    seq_out.permute(0, 2, 1),   # (B, num_classes, T)
                    targets_golden               # (B, T)
                )

                # CTC loss if enabled (auxiliary loss on encoder output)
                if self.ctc_weight > 0:
                    # ctc_inputs has 'log_probs' (T, B, num_classes) and 'lengths' (B,)
                    # golden targets for CTC: same text but without SOS/EOS convention issues
                    # we use targets_golden which has EOS appended - CTC handles this ok
                    # transcript_lengths here are the full lengths including SOS in shifted,
                    # but for CTC we need the actual transcript length (without EOS/SOS)
                    # golden transcript length = shifted length (both same size = len(tokens)+1)
                    ctc_loss = self.ctc_criterion(
                        ctc_inputs['log_probs'],          # (T_enc, B, num_classes)
                        targets_golden,                    # (B, T_dec) - the reference tokens
                        ctc_inputs['lengths'],            # (B,) - encoder output lengths
                        transcript_lengths                 # (B,) - transcript lengths
                    )
                    loss = ce_loss + self.ctc_weight * ctc_loss
                else:
                    ctc_loss = torch.tensor(0.0)
                    loss = ce_loss

            # Calculate metrics
            batch_tokens = transcript_lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += ce_loss.item() * batch_tokens
            if self.ctc_weight > 0:
                running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens
            
            # Normalize loss by accumulation steps
            loss = loss / self.config['training']['gradient_accumulation_steps']

            # scale the loss and backpropagate
            # GradScaler handles mixed precision gradient scaling
            self.scaler.scale(loss).backward()

            # Only update weights after accumulating enough gradients
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update progress bar
            avg_ce_loss = running_ce_loss / total_tokens
            avg_ctc_loss = running_ctc_loss / total_tokens
            avg_joint_loss = running_joint_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_ce_loss))
            
            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}", 
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Clean up
            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del seq_out, curr_att, ctc_inputs, loss
            torch.cuda.empty_cache()

        # Handle remaining gradients
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # Compute final metrics
        avg_ce_loss = running_ce_loss / total_tokens
        avg_ctc_loss = running_ctc_loss / total_tokens
        avg_joint_loss = running_joint_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()))
        batch_bar.close()

        return {
            'ce_loss': avg_ce_loss,
            'ctc_loss': avg_ctc_loss,
            'joint_loss': avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        
        Args:
            dataloader: DataLoader for validation data
        Returns:
            Tuple[Dict[str, float], List[Dict[str, Any]]]: Validation metrics and recognition results
        """
        # run recognition on the validation set using greedy decoding
        # recognize() handles the model.eval() and torch.inference_mode() internally
        val_recognition_config = {
            'num_batches': 1,     # keep train-time validation fast; use full eval separately
            'beam_width': 1,        # greedy decoding for speed
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': 0.0,
            'lm_model': None
        }

        # run inference and collect results
        val_results = self.recognize(
            dataloader=dataloader,
            recognition_config=val_recognition_config,
            config_name='greedy_val',
            max_length=self.text_max_len
        )

        # extract references and hypotheses from results
        # recognize() returns list of dicts with 'target' and 'generated' keys
        henry_refs  = [r['target']    for r in val_results if 'target' in r]
        henry_hyps  = [r['generated'] for r in val_results if 'target' in r]

        # compute ASR metrics (WER, CER, edit distance)
        if henry_refs and henry_hyps:
            val_metrics = self._calculate_asr_metrics(henry_refs, henry_hyps)
        else:
            val_metrics = {'word_dist': 0.0, 'wer': 0.0, 'cer': 0.0}

        return val_metrics, val_results
    
    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for ASR training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: int, number of epochs to train
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")
        
        # set max transcript length from the dataloaders
        # this is used by the sequence generator during inference
        self.text_max_len = max(
            val_dataloader.dataset.text_max_len,
            train_dataloader.dataset.text_max_len
        )

        # Training loop
        best_val_loss = float('inf')
        best_val_wer  = float('inf')
        best_val_cer  = float('inf')
        best_val_dist = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):

            # train for one epoch, get metrics and attention weights
            train_metrics, train_attn = self._train_epoch(train_dataloader)

            # validate using greedy decoding, get CER and recognition results
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])
            
            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Save attention plots
            train_attn_keys = list(train_attn.keys())
            if train_attn_keys: 
                # Get the first self-attention and cross-attention layers
                decoder_self_keys  = [k for k in train_attn_keys if 'dec_self' in k]
                decoder_cross_keys = [k for k in train_attn_keys if 'dec_cross' in k]
                
                if decoder_self_keys:
                    # Plot first layer (layer1) if available
                    first_self_key = decoder_self_keys[0]
                    if first_self_key in train_attn:
                        self._save_attention_plot(train_attn[first_self_key][0], epoch, "decoder_self")
                
                if decoder_cross_keys:
                    # Plot last layer if available
                    last_cross_key = decoder_cross_keys[-1]
                    if last_cross_key in train_attn:
                        self._save_attention_plot(train_attn[last_cross_key][0], epoch, "decoder_cross")
            
            # Save generated text
            self._save_generated_text(val_results, f'val_epoch_{epoch}')
            
            # Save checkpoints
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Check if this is the best model
            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = val_metrics['cer']
                self.save_checkpoint('checkpoint-best-metric-model.pth') 

            self.current_epoch += 1
                

    def evaluate(self, dataloader, max_length: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on the test set. Sequentially evaluates with each recognition config.
        
        Args:
            dataloader: DataLoader for test data
            max_length: Optional[int], maximum length of the generated sequence
        Returns:
            Dictionary containing recognition results for each recognition config
            Each result is a pandas DataFrame with columns 'id' and 'transcription'
        """

        # Get recognition configs
        recognition_configs = self._get_evaluation_recognition_configs()
        
        eval_results = {}
        # Evaluate with each recognition config
        for config_name, config in recognition_configs.items():
            try:
                print(f"Evaluating with {config_name} config")
                results = self.recognize(dataloader, config, config_name, max_length)     
                # Calculate metrics on full batch
                generated = [r['generated'] for r in results]
                results_df = pd.DataFrame(
                    {
                        'id': range(len(generated)),
                        'transcription': generated
                    }
                )
                eval_results[config_name] = results_df
                self._save_generated_text(results, f'test_{config_name}_results')
            except Exception as e:
                print(f"Error evaluating with {config_name} config: {e}")
                continue
        
        return eval_results

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None, config_name: Optional[str] = None, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating transcriptions from audio features.
        
        Args:
            dataloader: DataLoader containing the evaluation data
            recognition_config: Optional dictionary containing recognition parameters:
                - num_batches: int, number of batches to process
                - beam_width: int, beam search width
                - temperature: float, temperature for beam search
                - repeat_penalty: float, repeat penalty for beam search
                - lm_weight: float, language model interpolation weight
                - lm_model: Optional[DecoderOnlyTransformer], language model for shallow fusion
            max_length: Optional[int], maximum length of the generated sequence
        Returns:
            List of dictionaries containing recognition results with generated sequences and scores
            (targets included if available)
        """
        if max_length is None and not hasattr(self, 'text_max_len'):
            raise ValueError("text_max_len is not set. Please run training loop first or provide a max_length")

        if recognition_config is None:
            # default config: greedy search
            recognition_config = {
                'num_batches': 5,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = 'greedy'

        if recognition_config.get('lm_model') is not None:
            recognition_config['lm_model'].eval()
            recognition_config['lm_model'].to(self.device)

        # initialize sequence generator
        generator = SequenceGenerator(
            score_fn=None,   # will be set for each batch below
            tokenizer=self.tokenizer,
            max_length=max_length if max_length is not None else self.text_max_len,
            device=self.device
        )

        # inference mode
        self.model.eval()
        batch_bar = tqdm(
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc=f"[Recognizing ASR] : {config_name}"
        )
        results = []

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # unpack batch - test-clean has no transcripts so golden may be None
                feats, _, targets_golden, feat_lengths, _ = batch
                feats        = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)

                # encode speech features to get encoder hidden states
                # we do not need CTC inputs during inference, just encoder output and mask
                encoder_output, pad_mask_src, _, _ = self.model.encode(
                    padded_sources=feats,
                    source_lengths=feat_lengths
                )

                # define the scoring function for this batch
                # it captures encoder_output and pad_mask_src from the enclosing scope
                def get_score(x):
                    # get logits for next token from the ASR decoder
                    asr_logits = self.model.score(x, encoder_output, pad_mask_src)
                    # optional shallow fusion with a language model from HW4P1
                    if recognition_config.get('lm_model') is not None:
                        lm_logits = recognition_config['lm_model'].score(x)
                        return asr_logits + recognition_config['lm_weight'] * lm_logits
                    return asr_logits

                # wire the scoring function into the generator
                generator.score_fn = get_score

                # initialize prompts: one SOS token per sequence in the batch
                # shape (B, 1) - the decoder starts with just SOS and generates from there
                batch_size = feats.size(0)
                prompts = torch.full(
                    (batch_size, 1),
                    fill_value=self.tokenizer.sos_id,
                    dtype=torch.long,
                    device=self.device
                )

                # generate sequences using the chosen decoding strategy
                if recognition_config['beam_width'] > 1:
                    # beam search (optional but recommended for better CER)
                    seqs, scores = generator.generate_beam(
                        x=prompts,
                        beam_width=recognition_config['beam_width'],
                        temperature=recognition_config['temperature'],
                        repeat_penalty=recognition_config['repeat_penalty']
                    )
                    # pick the best beam (highest score, already sorted by generate_beam)
                    seqs   = seqs[:, 0, :]
                    scores = scores[:, 0]
                else:
                    # greedy decoding: always pick the highest probability token
                    seqs, scores = generator.generate_greedy(
                        x=prompts,
                        temperature=recognition_config['temperature'],
                        repeat_penalty=recognition_config['repeat_penalty']
                    )

                # clean up GPU tensors before post-processing
                del feats, feat_lengths, encoder_output, pad_mask_src, prompts
                torch.cuda.empty_cache()

                # post-process: strip padding and special tokens from generated sequences
                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)

                # store results as list of dicts
                if targets_golden is not None:
                    post_processed_targets = generator.post_process_sequence(
                        targets_golden, self.tokenizer
                    )
                    for j, (pred, target) in enumerate(zip(post_processed_preds, post_processed_targets)):
                        results.append({
                            'target':    self.tokenizer.decode(target.tolist(), skip_special_tokens=True),
                            'generated': self.tokenizer.decode(pred.tolist(),   skip_special_tokens=True),
                            'score':     scores[j].item()
                        })
                else:
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score':     scores[j].item()
                        })

                batch_bar.update()

                # optionally limit to num_batches (used during validation to save time)
                if recognition_config['num_batches'] is not None and i >= recognition_config['num_batches'] - 1:
                    break

        batch_bar.close()
        return results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None, lm_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of recognition configurations for seqential evaluation.
        
        Returns:
            Dictionary containing recognition configurations
        """

        common_config = {
            'num_batches': 2,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        greedy_config = common_config.copy()
        greedy_config.update({
            'beam_width': 1,
        })

        beam_10_config = common_config.copy()
        beam_10_config.update({
            'beam_width': 10,
        })
        
        beam_20_config = common_config.copy()
        beam_20_config.update({
            'beam_width': 20,
        })
        
        return {
            'greedy': greedy_config,
            'beam_10': beam_10_config,
            'beam_20': beam_20_config
        }
        
    def _calculate_asr_metrics(self, references: Union[str, List[str]], hypotheses: Union[str, List[str]]) -> Tuple[float, float, float]:
        """
        Calculate Levenshtein distance, WER, CER for strings or lists of strings.
        
        Args:
            references: Reference string(s)
            hypotheses: Hypothesis string(s)
        Returns:
            Tuple of (word_dist, wer, cer)
        """
        import torchmetrics.text as tmt

        # Initialize metrics
        wer_metric = tmt.WordErrorRate()
        word_edit_metric = tmt.EditDistance(reduction='mean')
        cer_metric = tmt.CharErrorRate()
        
        # Calculate metrics
        word_dist = word_edit_metric(hypotheses, references)
        wer = wer_metric(hypotheses, references)  # torchmetrics returns as decimal
        cer = cer_metric(hypotheses, references)  # torchmetrics returns as decimal

        return {
            'word_dist': word_dist.item(),
            'wer': wer.item() * 100,
            'cer': cer.item() * 100
        }
    
# -------------------------------------------------------------------------------------------------

class ProgressiveTrainer(ASRTrainer):
    """
    Progressive Trainer class that implements curriculum learning for ASR training.

    This trainer extends ASRTrainer to implement:
    1. Stage-based training with increasing model complexity
    2. Gradual unfreezing of model layers
    3. Dynamic data subsetting
    4. Smooth transition to full model training

    Implementation Tasks:
    - Store original model layers in __init__
    - Configure model for each stage in configure_stage
    - Implement progressive training loop in progressive_train
    - Handle transition to full training in transition_to_full_training
    - Create data subsets in get_subset_dataloader

    Implementation Notes:
    1. For __init__:
        - Store original encoder and decoder layers
        - Initialize stage counter
        
    2. For configure_stage:
        - Update dropout and label smoothing
        - Activate specified encoder and decoder layers
        - Handle layer freezing based on configuration
        - Print detailed configuration information
        
    3. For progressive_train:
        - Configure model for each stage
        - Create appropriate data subset
        - Train using parent class methods
        
    4. For transition_to_full_training:
        - Restore all model layers
        - Reset loss function parameters
        - Unfreeze all parameters
        - Reset best metrics
        
    5. For get_subset_dataloader:
        - Create subset while preserving dataset attributes
        - Maintain collate function and other dataloader settings

    # -------------------------------------------------------------------------------------------------
    ##### Stage Configuration

    Each stage is defined as a dictionary with the following parameters:
    ```python
    {
        'name': str,                        # Name of the training stage
        'epochs': int,                      # Number of epochs to train in this stage
        'encoder_active_layers': List[int], # Which encoder layers to use
        'decoder_active_layers': List[int], # Which decoder layers to use
        'encoder_freeze': List[bool],       # Whether to freeze each encoder layer
        'decoder_freeze': List[bool],       # Whether to freeze each decoder layer
        'dropout': float,                   # Dropout rate for this stage
        'label_smoothing': float,           # Label smoothing value
        'data_subset': float                # Fraction of training data to use (0.0-1.0)
    }
    ```
    #### Example
    It is best understood by an example. Here is a breakdown of the stages defined below for a model with 6 encoder and 6 decoder layers:

    stages = [
                {
                    # `Initial (1 layers)`:
                    # This stage starts with a model with only 1 encoder and 1 decoder layer.
                    # No freezing or regularization is applied.
                    # It uses 20% of the training data.
                    'name': 'Initial (1 Encoder + 1 Decoder layers)',
                    'epochs': 5,
                    'encoder_active_layers': list(range(1)),
                    'decoder_active_layers': list(range(1)),
                    'encoder_freeze': [False],
                    'decoder_freeze': [False],
                    'dropout': 0.0,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `2 layers`:
                    # This stage increases the number of layers to 2 for both the encoder and decoder.
                    # The previous layer (encoder layer 1 and decoder layer 1) are frozen.
                    # No regularization is applied.
                    # It uses 20% of the training data.
                    'name': '2 Encoder + 2 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(2)),
                    'decoder_active_layers': list(range(2)),
                    'encoder_freeze': [True, False],
                    'decoder_freeze': [True, False],
                    'dropout': 0.0,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `4 layers`:
                    # This stage increases the number of layers to 4 for both the encoder and decoder.
                    # The previous layers (encoder layers 1 and 2 and decoder layers 1 and 2) are frozen.
                    # Dropout is set to 0.05 and label smoothing is set to 0.0.
                    # It uses 20% of the training data.
                    'name': '4 Encoder + 4 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(4)),
                    'decoder_active_layers': list(range(4)),
                    'encoder_freeze': [True, True, False, False],
                    'decoder_freeze': [True, True, False, False],
                    'dropout': 0.05,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `All 6 layers`:
                    # This stage uses all 6 encoder and 6 decoder layers.
                    # The 4 previous layers are frozen and the last 2 layers are trained.
                    # Dropout is set to 0.1 and label smoothing is set to 0.0.
                    # It uses 20% of the training data.
                    'name': '6 Encoder + 6 Decoder layers',
                    'epochs': 5,
                    'encoder_active_layers': list(range(6)),
                    'decoder_active_layers': list(range(6)),
                    'encoder_freeze': [True, True, True, True, False, False],
                    'decoder_freeze': [True, True, True, True, False, False],
                    'dropout': 0.1,
                    'label_smoothing': 0.0,
                    'data_subset': 0.2
                },
                {
                    # `Final (with label smoothing)`:
                    # This stage uses all 6 encoder and 6 decoder layers.
                    # All layers are unfrozen and trained.
                    # Dropout is set to 0.1 and label smoothing is set to 0.1.
                    # It uses 20% of the training data.
                    'name': 'Final (with label smoothing)',
                    'epochs': 5,
                    'encoder_active_layers': list(range(6)),
                    'decoder_active_layers': list(range(6)),
                    'encoder_freeze': [False, False, False, False, False, False],
                    'decoder_freeze': [False, False, False, False, False, False],
                    'dropout': 0.1,
                    'label_smoothing': 0.1,
                    'data_subset': 0.2
                }
            ]    

    ##### Important Notes
    - Ensure `encoder_freeze` and `decoder_freeze` lists match the length of their respective `active_layers`
    - `data_subset` should be between 0 and 1
    - Stage transitions are handled automatically by the trainer
    - The same optimizer and scheduler are used for all stages so keep that in mind while setting the learning rates and other parameters
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.current_stage = 0
        # Store original layer states
        self.all_encoder_layers = list(self.model.enc_layers)
        self.all_decoder_layers = list(self.model.dec_layers)


    def configure_stage(self, stage_config):
        """Configure model for current training stage"""
        # Create a pretty header
        print("\n" + "="*80)
        print(f"Starting Stage: {stage_config['name']}".center(80))
        print("="*80)
        
        # Print key configuration details
        print(f"\nConfiguration Details:")
        print(f"├── Data Subset: {stage_config['data_subset']*100:.1f}% of training data")
        print(f"├── Training Epochs: {stage_config['epochs']}")
        print(f"├── Dropout: {stage_config['dropout']}")
        print(f"├── Label Smoothing: {stage_config['label_smoothing']}")
        
        # Update dropout and label smoothing
        self.model.dropout.p = stage_config['dropout']
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=stage_config['label_smoothing']
        )
        
        # Get freeze configurations
        encoder_freeze = stage_config.get('encoder_freeze', [])
        decoder_freeze = stage_config.get('decoder_freeze', [])
        
        # Activate and configure encoder layers
        encoder_active_layers = stage_config['encoder_active_layers']
        if encoder_freeze and len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError(f"Encoder freeze list length ({len(encoder_freeze)}) must match number of active encoder layers ({len(encoder_active_layers)})")
        
        # Set the active encoder layers of the model
        self.model.enc_layers = nn.ModuleList([
            self.all_encoder_layers[i] for i in encoder_active_layers
        ])
        self.model.num_encoder_layers = len(encoder_active_layers)
        
        # Activate and configure decoder layers
        decoder_active_layers = stage_config['decoder_active_layers']
        if decoder_freeze and len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError(f"Decoder freeze list length ({len(decoder_freeze)}) must match number of active decoder layers ({len(decoder_active_layers)})")
        
        # Set the active decoder layers of the model
        self.model.dec_layers = nn.ModuleList([
            self.all_decoder_layers[i] for i in decoder_active_layers
        ])
        self.model.num_decoder_layers = len(decoder_active_layers)

        # Handle layer freezing
        frozen_count = 0
        trainable_count = 0
        
        # Configure encoder layers freezing
        print("├── Encoder Layers:")
        for idx, layer in enumerate(self.model.enc_layers):
            should_freeze = encoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {encoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        # Configure decoder layers
        print("├── Decoder Layers:")
        for idx, layer in enumerate(self.model.dec_layers):
            should_freeze = decoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {decoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        print(f"├── Frozen Parameters: {frozen_count:,}")
        print(f"└── Trainable Parameters: {trainable_count:,}")
    

    def progressive_train(self, train_dataloader, val_dataloader, stages: List[Dict[str, Any]]):
        """
        Progressive training through stages
        Each stage configuration is defined as a dictionary with the following parameters:

        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            stages: List of dictionaries containing stage configuration
        """
        # Train through stages
        for stage_idx, stage_config in enumerate(stages):
            self.current_stage = stage_idx
            self.configure_stage(stage_config)
            # Get subset of train_dataloader
            subset_train_dataloader = self.get_subset_dataloader(train_dataloader, stage_config['data_subset'])
            super().train(subset_train_dataloader, val_dataloader, epochs=stage_config['epochs'])

    def transition_to_full_training(self):
        """Transition from progressive training to full training"""
        print("\n=== Transitioning to Full Training ===")
        
        # Restore all layers
        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)

        # Restore CrossEntropyLoss
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss']['label_smoothing']
        )
        
        # Unfreeze all parameters
        unfrozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            unfrozen_count += param.numel()
        print(f"├── Total Unfrozen Parameters: {unfrozen_count:,}")
        
        # Reset best metrics for new training phase
        self.best_metric = float('inf')

    
    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Run full training phase.
        It is recommended to set the optimizer and scheduler explicitly before calling this function.
        like this:
        cls.optimizer = create_optimizer(self.model, self.config['optimizer'])
        cls.scheduler = create_scheduler(cls.optimizer, cls.config['scheduler'], train_dataloader)
        cls.progressive_train(train_dataloader, val_dataloader, stages)
        """
        self.transition_to_full_training()
        super().train(train_dataloader, val_dataloader, epochs=epochs)


    def get_subset_dataloader(self, dataloader, subset_fraction):
        """
        Creates a new DataLoader with a subset of the original data while preserving dataset attributes.
        
        Args:
            dataloader: Original DataLoader
            subset_fraction: Float between 0 and 1 indicating what fraction of data to keep
        
        Returns:
            New DataLoader containing only the subset of data
        """
        # Calculate how many samples we want to keep
        dataset = dataloader.dataset
        total_samples = len(dataset)
        subset_size = int(total_samples * subset_fraction)
        
        # Create random indices for the subset
        indices = torch.randperm(total_samples)[:subset_size]
        
        # Create a Subset dataset
        subset_dataset = Subset(dataset, indices)
        
        # Add necessary attributes from original dataset to subset
        subset_dataset.text_max_len = dataset.text_max_len
        subset_dataset.feat_max_len = dataset.feat_max_len
        subset_dataset.get_avg_chars_per_token = dataset.get_avg_chars_per_token
        
        # Create new DataLoader with same configuration as original
        subset_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['NUM_WORKERS'],
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        
        return subset_loader
        
        
        
