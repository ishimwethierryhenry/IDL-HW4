import os
import gc
import yaml
import torch
import shutil
from torch.utils.data import DataLoader

os.chdir("/local/IDL-HW4")

from hw4lib.data import H4Tokenizer, ASRDataset
from hw4lib.model import EncoderDecoderTransformer
from hw4lib.utils import create_scheduler, create_optimizer
from hw4lib.trainers import ASRTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config_str = """
Name: "hw4p2-henry-asr"

tokenization:
  token_type: "5k"
  token_map:
    char: 'hw4lib/data/tokenizer_jsons/tokenizer_char.json'
    1k:   'hw4lib/data/tokenizer_jsons/tokenizer_1000.json'
    5k:   'hw4lib/data/tokenizer_jsons/tokenizer_5000.json'
    10k:  'hw4lib/data/tokenizer_jsons/tokenizer_10000.json'

data:
  root: "/local/dataset/hw4p2_data"
  train_partition: "train-clean-100"
  val_partition: "dev-clean"
  test_partition: "test-clean"
  subset: 1.0
  batch_size: 8
  NUM_WORKERS: 4
  norm: global_mvn
  num_feats: 80
  specaug: True
  specaug_conf:
    apply_freq_mask: True
    freq_mask_width_range: 5
    num_freq_mask: 2
    apply_time_mask: True
    time_mask_width_range: 40
    num_time_mask: 2

model:
  input_dim: 80
  time_reduction: 4
  reduction_method: conv
  d_model: 256
  num_encoder_layers: 4
  num_decoder_layers: 4
  num_encoder_heads: 4
  num_decoder_heads: 4
  d_ff_encoder: 1024
  d_ff_decoder: 1024
  skip_encoder_pe: False
  skip_decoder_pe: False
  dropout: 0.1
  layer_drop_rate: 0.0
  weight_tying: False

training:
  use_wandb: True
  wandb_run_id: "none"
  resume: True
  gradient_accumulation_steps: 4
  wandb_project: "hw4p2-asr-henry"

loss:
  label_smoothing: 0.1
  ctc_weight: 0.2

optimizer:
  name: adamw
  lr: 1.0e-4
  weight_decay: 0.000001
  param_groups:
    - name: self_attn
      patterns: []
      lr: 1.0e-4
      layer_decay:
        enabled: False
        decay_rate: 0.8
    - name: ffn
      patterns: []
      lr: 1.0e-4
      layer_decay:
        enabled: False
        decay_rate: 0.8
  layer_decay:
    enabled: False
    decay_rate: 0.75
  sgd:
    momentum: 0.9
    nesterov: True
    dampening: 0
  adam:
    betas: [0.9, 0.999]
    eps: 1.0e-8
    amsgrad: False
  adamw:
    betas: [0.9, 0.999]
    eps: 1.0e-8
    amsgrad: False

scheduler:
  name: cosine_warm
  reduce_lr:
    mode: min
    factor: 0.5
    patience: 3
    threshold: 0.0001
    threshold_mode: rel
    cooldown: 0
    min_lr: 0.0000001
    eps: 1.0e-8
  cosine:
    T_max: 15
    eta_min: 0.0000001
    last_epoch: -1
  cosine_warm:
    T_0: 10
    T_mult: 2
    eta_min: 0.0000001
    last_epoch: -1
  warmup:
    enabled: True
    type: exponential
    epochs: 3
    start_factor: 0.1
    end_factor: 1.0
"""

with open("config.yaml", "w") as f:
    f.write(config_str)

config = yaml.safe_load(config_str)

print("Loading tokenizer...")
Tokenizer = H4Tokenizer(
    token_map=config['tokenization']['token_map'],
    token_type=config['tokenization']['token_type']
)

print("Loading datasets...")
train_dataset = ASRDataset(
    partition=config['data']['train_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=True,
    global_stats=None
)

global_stats = None
if config['data']['norm'] == 'global_mvn':
    global_stats = (train_dataset.global_mean, train_dataset.global_std)
    print("Global stats computed from training set.")

val_dataset = ASRDataset(
    partition=config['data']['val_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

test_dataset = ASRDataset(
    partition=config['data']['test_partition'],
    config=config['data'],
    tokenizer=Tokenizer,
    isTrainPartition=False,
    global_stats=global_stats
)

gc.collect()

print("Creating dataloaders...")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=config['data']['NUM_WORKERS'],
    pin_memory=True,
    collate_fn=train_dataset.collate_fn
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=False,
    num_workers=config['data']['NUM_WORKERS'],
    pin_memory=True,
    collate_fn=val_dataset.collate_fn
)

max_feat_len = max(train_dataset.feat_max_len, val_dataset.feat_max_len, test_dataset.feat_max_len)
max_transcript_len = max(train_dataset.text_max_len, val_dataset.text_max_len)
max_len = min(max(max_feat_len, max_transcript_len), 2500)
print(f"max_len: {max_len}")

print("Creating model...")
model_config = config['model'].copy()
model_config.update({
    'max_len': max_len,
    'num_classes': Tokenizer.vocab_size
})
model = EncoderDecoderTransformer(**model_config)

total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_param:,}")
assert total_param < 30_000_000, f"Too many parameters: {total_param}"

print("Creating trainer...")
trainer = ASRTrainer(
    model=model,
    tokenizer=Tokenizer,
    config=config,
    run_name="hw4p2-henry-cold-start",
    config_file="config.yaml",
    device=device
)

# resume from ocean checkpoint if it exists
ocean_ckpt = "/ocean/projects/cis250019p/thierryh/checkpoints/checkpoint-last-epoch-model.pth"
if os.path.exists(ocean_ckpt):
    print(f"Resuming from ocean checkpoint...")
    # create the local checkpoint directory first
    local_ckpt_dir = "expts/hw4p2-henry-cold-start/checkpoints"
    os.makedirs(local_ckpt_dir, exist_ok=True)
    shutil.copy(ocean_ckpt, f"{local_ckpt_dir}/checkpoint-last-epoch-model.pth")
    trainer.load_checkpoint("checkpoint-last-epoch-model.pth")
    print(f"Resuming from epoch {trainer.current_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

trainer.optimizer = create_optimizer(
    model=model,
    opt_config=config['optimizer']
)

trainer.scheduler = create_scheduler(
    optimizer=trainer.optimizer,
    scheduler_config=config['scheduler'],
    train_loader=train_loader,
    gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
)

print("Starting training...")
trainer.train(train_loader, val_loader, epochs=30)

# save checkpoints to ocean after training
os.makedirs("/ocean/projects/cis250019p/thierryh/checkpoints", exist_ok=True)
expt_dir = "expts/hw4p2-henry-cold-start/checkpoints"
if os.path.exists(expt_dir):
    shutil.copy(f"{expt_dir}/checkpoint-best-metric-model.pth",
                "/ocean/projects/cis250019p/thierryh/checkpoints/checkpoint-best-metric-model.pth")
    shutil.copy(f"{expt_dir}/checkpoint-last-epoch-model.pth",
                "/ocean/projects/cis250019p/thierryh/checkpoints/checkpoint-last-epoch-model.pth")
    print("Checkpoints saved to ocean.")

print("Done!")
