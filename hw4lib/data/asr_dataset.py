from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

'''
ASRDataset for HW4P2.

Loads filterbank features and text transcripts from the LibriSpeech-style
directory structure. Handles normalization, spec augmentation, and batching.

Data structure expected:
    root/
        train-clean-100/
            fbank/    *.npy  (shape: num_feats x time, float32)
            text/     *.npy  (shape: (), a Python string stored as numpy scalar)
        dev-clean/
            fbank/
            text/
        test-clean/
            fbank/         (no text/ directory - we are generating transcriptions)

Feature shape: the .npy files store features as (num_feats, time).
We transpose to (num_feats, time) internally and then to (time, num_feats)
in the collate_fn so the model sees (B, T, F) as expected by SpeechEmbedding.

IMPORTANT: The transcript .npy files store a numpy string scalar.
Loading them requires: str(np.load(path, allow_pickle=True))
The text is already uppercase (LibriSpeech convention).
'''


class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        '''
        Args:
            partition        : which data split to load
            config           : dict containing dataset settings:
                               - root: path to hw4p2_data directory
                               - num_feats: number of filterbank features to use (e.g. 80)
                               - norm: 'global_mvn', 'cepstral', or 'none'
                               - subset: fraction of data to load (1.0 = full dataset)
                               - specaug: bool, whether to apply spec augment
                               - specaug_conf: dict with spec augment parameters
            tokenizer        : H4Tokenizer instance for encoding/decoding text
            isTrainPartition : whether this is the training partition
                               (controls whether spec augment is applied)
            global_stats     : (mean, std) tensors of shape (num_feats,) for global_mvn
                               Must be provided for dev/test when using global_mvn
                               Should be None for training (computed during loading)
        '''
        # store basic configuration
        self.config           = config
        self.partition        = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer        = tokenizer

        # get special token ids from tokenizer
        # these are the ids we prepend/append to transcripts
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # build paths to the feature and text directories
        # root is the path to hw4p2_data, partition is the subdirectory name
        self.fbank_dir = os.path.join(config['root'], partition, 'fbank')

        # get sorted list of all .npy feature files
        # sorting is critical so that fbank files and text files align correctly
        self.fbank_files = sorted([
            os.path.join(self.fbank_dir, f)
            for f in os.listdir(self.fbank_dir)
            if f.endswith('.npy')
        ])

        # optionally use a subset of the data (useful for debugging)
        subset_size = config.get('subset', 1.0)
        if subset_size < 1.0:
            thierry_subset_count = max(1, int(len(self.fbank_files) * subset_size))
            self.fbank_files = self.fbank_files[:thierry_subset_count]

        # total number of samples in this partition
        self.length = len(self.fbank_files)

        # test-clean has no transcripts - we are generating predictions for it
        # all other partitions (train and dev) have paired text files
        if self.partition != 'test-clean':
            self.text_dir = os.path.join(config['root'], partition, 'text')

            self.text_files = sorted([
                os.path.join(self.text_dir, f)
                for f in os.listdir(self.text_dir)
                if f.endswith('.npy')
            ])

            # apply same subset as features
            if subset_size < 1.0:
                self.text_files = self.text_files[:thierry_subset_count]

            # sanity check - feature and text files must be aligned
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError(
                    f"Feature and transcript file count mismatch: "
                    f"{len(self.fbank_files)} fbank vs {len(self.text_files)} text"
                )

        # storage for loaded data
        self.feats                = []
        self.transcripts_shifted  = []
        self.transcripts_golden   = []

        # counters for character and token statistics
        # DO NOT MODIFY - used by the trainer to compute perplexity
        self.total_chars  = 0
        self.total_tokens = 0

        # track max lengths for padding and sequence generator initialization
        # DO NOT MODIFY
        self.feat_max_len = 0
        self.text_max_len = 0

        # Welford online mean/variance computation for global_mvn normalization
        # this avoids loading everything twice - we compute stats during the loading loop
        # DO NOT MODIFY
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError(
                    "global_stats must be provided for non-training partitions "
                    "when using global_mvn normalization"
                )
            count = 0
            mean  = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2    = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # load filterbank features
            # stored as numpy array of shape (num_feats, time) - float32
            handel_raw_feat = np.load(self.fbank_files[i])   # (num_feats_full, time)

            # truncate to the configured number of features
            # this lets us use fewer than 80 features to save memory
            feat = handel_raw_feat[:self.config['num_feats'], :]   # (num_feats, time)

            # store as float32 tensor
            feat = torch.FloatTensor(feat)

            self.feats.append(feat)

            # track max time dimension (feat.shape[1] is the time axis)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # update Welford accumulators for global mean/variance if needed
            # DO NOT MODIFY
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor  = feat.double()          # (num_feats, time)
                batch_count  = feat_tensor.shape[1]   # number of time steps
                count       += batch_count

                delta  = feat_tensor - mean.unsqueeze(1)
                mean  += delta.mean(dim=1)
                delta2 = feat_tensor - mean.unsqueeze(1)
                M2    += (delta * delta2).sum(dim=1)

            # load and process transcripts for train and dev partitions
            if self.partition != 'test-clean':
                # IMPORTANT: the transcript is stored as a numpy scalar containing a string
                # we need allow_pickle=True and str() to correctly extract it
                # The text is already uppercase in LibriSpeech
                transcript = str(np.load(self.text_files[i], allow_pickle=True))

                # count raw characters before tokenization
                self.total_chars += len(transcript)

                # tokenize the transcript into subword/character token ids
                # tokenizer.encode returns a list of integer token ids
                tokenized = tokenizer.encode(transcript)

                # count tokens (before adding SOS/EOS)
                # DO NOT MODIFY
                self.total_tokens += len(tokenized)

                # track max token length (add 1 for the SOS or EOS we will add)
                # DO NOT MODIFY
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                # create shifted version (model input): SOS prepended
                # decoder receives: [SOS, t1, t2, ..., tN]
                # and tries to predict: [t1, t2, ..., tN, EOS]
                thierry_shifted = torch.LongTensor([self.sos_token] + tokenized)

                # create golden version (target): EOS appended
                # this is what the decoder should output
                thierry_golden  = torch.LongTensor(tokenized + [self.eos_token])

                self.transcripts_shifted.append(thierry_shifted)
                self.transcripts_golden.append(thierry_golden)

        # average characters per token - used for character-level perplexity
        # DO NOT MODIFY
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        )

        # verify alignment after loading
        if self.partition != 'test-clean':
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned after loading")

        # finalize global normalization statistics
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                # use provided stats (for dev/test)
                self.global_mean, self.global_std = global_stats
            else:
                # compute from Welford accumulators (for training)
                variance = M2 / (count - 1)
                self.global_std  = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # initialize SpecAugment transforms
        # DO NOT MODIFY - uses torchaudio transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        '''
        Return the average number of characters per token.
        Used by the trainer to compute character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        '''Return the total number of samples in this partition.'''
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Load a single sample and apply normalization.

        Args:
            idx : sample index

        Returns:
            feat                : (num_feats, time) FloatTensor - normalized features
            shifted_transcript  : (T,) LongTensor or None (None for test-clean)
            golden_transcript   : (T,) LongTensor or None (None for test-clean)
        '''
        # retrieve the pre-loaded feature tensor
        feat = self.feats[idx]   # (num_feats, time)

        # apply normalization
        if self.config['norm'] == 'global_mvn':
            # global mean/variance normalization using training set statistics
            # global_mean and global_std are both shape (num_feats,)
            # we unsqueeze to broadcast over the time dimension
            assert self.global_mean is not None and self.global_std is not None
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)

        elif self.config['norm'] == 'cepstral':
            # per-utterance mean and variance normalization (CMVN)
            # normalize each feature dimension independently across time
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)

        elif self.config['norm'] == 'none':
            # no normalization
            pass

        # get transcripts for non-test partitions
        shifted_transcript, golden_transcript = None, None
        if self.partition != 'test-clean':
            shifted_transcript = self.transcripts_shifted[idx]
            golden_transcript  = self.transcripts_golden[idx]

        return feat, shifted_transcript, golden_transcript

    def collate_fn(
        self,
        batch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Collate a list of samples into a padded batch.

        The features are stored as (num_feats, time) but the model expects (B, T, F)
        so we transpose each feature before padding. This is done by collecting
        feat.T (transposed) so pad_sequence pads along the time dimension.

        Args:
            batch : list of (feat, shifted, golden) tuples from __getitem__

        Returns:
            padded_feats        : (B, T_max, num_feats) - padded feature tensors
            padded_shifted      : (B, T_text_max) or None
            padded_golden       : (B, T_text_max) or None
            feat_lengths        : (B,) - actual time lengths of each utterance
            transcript_lengths  : (B,) - actual token lengths of each transcript (or None)
        '''
        # unzip the batch into separate lists
        batch_feats_raw, batch_shifted_raw, batch_golden_raw = zip(*batch)

        # transpose each feature from (num_feats, time) to (time, num_feats)
        # so that pad_sequence pads along the time dimension correctly
        # the model expects (B, T, F) not (B, F, T)
        handel_feats_list = [f.T for f in batch_feats_raw]   # list of (time_i, num_feats)

        # actual time lengths before padding
        feat_lengths = torch.LongTensor([f.shape[0] for f in handel_feats_list])

        # pad features to the longest utterance in the batch
        # pad_sequence returns (T_max, B, num_feats), then permute to (B, T_max, num_feats)
        padded_feats = pad_sequence(handel_feats_list, batch_first=True, padding_value=0.0)
        # pad_sequence with batch_first=True already gives (B, T_max, num_feats)

        # handle transcripts for non-test partitions
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != 'test-clean':
            # actual transcript lengths (length of shifted = length of golden = len(tokenized) + 1)
            transcript_lengths = torch.LongTensor([s.shape[0] for s in batch_shifted_raw])

            # pad shifted and golden transcripts
            # pad_sequence with batch_first=True gives (B, T_text_max)
            padded_shifted = pad_sequence(
                list(batch_shifted_raw),
                batch_first=True,
                padding_value=self.pad_token
            )
            padded_golden = pad_sequence(
                list(batch_golden_raw),
                batch_first=True,
                padding_value=self.pad_token
            )

        # apply SpecAugment only during training
        if self.config['specaug'] and self.isTrainPartition:
            # SpecAugment expects (B, F, T) so we permute from (B, T, F)
            padded_feats = padded_feats.permute(0, 2, 1)   # (B, num_feats, T_max)

            # frequency masking: mask random frequency bands
            if self.config['specaug_conf']['apply_freq_mask']:
                for _ in range(self.config['specaug_conf']['num_freq_mask']):
                    padded_feats = self.freq_mask(padded_feats)

            # time masking: mask random time steps
            if self.config['specaug_conf']['apply_time_mask']:
                for _ in range(self.config['specaug_conf']['num_time_mask']):
                    padded_feats = self.time_mask(padded_feats)

            # permute back to (B, T, F) for the model
            padded_feats = padded_feats.permute(0, 2, 1)   # (B, T_max, num_feats)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
