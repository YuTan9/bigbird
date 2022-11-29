#!/bin/bash

# TF_XLA_FLAGS=--tf_xla_auto_jit=2
python3 run_translation.py \
  --data_dir="./data/UM" \
  --output_dir="./result" \
  --attention_type=block_sparse \
  --couple_encoder_decoder=True \
  --max_encoder_length=3072 \
  --max_decoder_length=256 \
  --num_attention_heads=12 \
  --num_hidden_layers=12 \
  --hidden_size=768 \
  --intermediate_size=3072 \
  --block_size=64 \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --do_train=True \
  --do_eval=False \
  --use_tpu=False \