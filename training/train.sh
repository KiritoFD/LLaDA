#!/bin/bash

# LLaDA Training Script
# This script demonstrates how to train LLaDA models from scratch

set -e

# Configuration
MODEL_NAME="GSAI-ML/LLaDA-8B-Base"  # Starting model for fine-tuning
TRAIN_DATA_PATH="./data/train.jsonl"
EVAL_DATA_PATH="./data/eval.jsonl"
OUTPUT_DIR="./checkpoints"
MAX_LENGTH=4096
BATCH_SIZE=4
LEARNING_RATE=4e-4
MAX_STEPS=100000

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "=== LLaDA Training Pipeline ==="
echo "Model: $MODEL_NAME"
echo "Output Directory: $OUTPUT_DIR"
echo "Max Steps: $MAX_STEPS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"

# Pre-training (if starting from scratch)
echo "Starting Pre-training..."
python training/pretraining.py \
    --model_name_or_path $MODEL_NAME \
    --train_data_path $TRAIN_DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $OUTPUT_DIR/pretraining \
    --max_length $MAX_LENGTH \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --min_learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --log_interval 100 \
    --eval_interval 1000 \
    --save_interval 5000 \
    --num_workers 4 \
    --seed 42

echo "Pre-training completed!"

# SFT Training
echo "Starting SFT Training..."
SFT_MODEL_PATH="$OUTPUT_DIR/pretraining/best_model.pt"
SFT_TRAIN_DATA="./data/sft_train.jsonl"
SFT_EVAL_DATA="./data/sft_eval.jsonl"

python training/sft_training.py \
    --model_name_or_path $SFT_MODEL_PATH \
    --train_data_path $SFT_TRAIN_DATA \
    --eval_data_path $SFT_EVAL_DATA \
    --output_dir $OUTPUT_DIR/sft \
    --max_length $MAX_LENGTH \
    --max_steps 10000 \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --min_learning_rate 1e-6 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --log_interval 50 \
    --eval_interval 500 \
    --save_interval 1000 \
    --num_workers 4 \
    --seed 42

echo "SFT Training completed!"

# Test inference
echo "Testing inference..."
FINAL_MODEL="$OUTPUT_DIR/sft/best_sft_model.pt"

python training/inference.py \
    --model_name_or_path $FINAL_MODEL \
    --prompt "What is the capital of France?" \
    --method fixed_length \
    --gen_length 128 \
    --steps 128 \
    --temperature 0.0 \
    --cfg_scale 0.0 \
    --remasking low_confidence \
    --is_instruct

echo "=== Training Pipeline Completed ==="