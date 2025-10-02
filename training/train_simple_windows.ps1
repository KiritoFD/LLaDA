# LLaDA Simple Training Script for Windows
# This script trains LLaDA from scratch using online tokenizer only

# Configuration
$PYTHON = "python"
$OUTPUT_DIR = ".\outputs_simple"
$MAX_STEPS = 2000
$BATCH_SIZE = 2
$LEARNING_RATE = 0.0003
$MAX_LENGTH = 512
$LOG_INTERVAL = 20
$SAVE_INTERVAL = 200
$TOKENIZER_NAME = "gpt2"

Write-Host "=== LLaDA Windows Training (Simple) ===" -ForegroundColor Green
Write-Host "This script trains LLaDA from scratch with online tokenizer only" -ForegroundColor Green
Write-Host "Tokenizer: $TOKENIZER_NAME (downloaded)" -ForegroundColor Yellow
Write-Host "Model: Created from scratch (no download)" -ForegroundColor Yellow
Write-Host "Output Directory: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "Max Steps: $MAX_STEPS" -ForegroundColor Yellow
Write-Host "Batch Size: $BATCH_SIZE" -ForegroundColor Yellow

# Create directories
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "data\pretrain"

# Generate sample data first
Write-Host "Generating sample training data..." -ForegroundColor Cyan
python training\generate_sample_data.py

# Check if data files exist
$TRAIN_DATA_PATH = ".\data\pretrain\train.jsonl"
$EVAL_DATA_PATH = ".\data\pretrain\eval.jsonl"

if (-not (Test-Path $TRAIN_DATA_PATH)) {
    Write-Host "Error: Training data not found at $TRAIN_DATA_PATH" -ForegroundColor Red
    Write-Host "Please run: python training\generate_sample_data.py" -ForegroundColor Yellow
    exit 1
}

# Start training with simple trainer
Write-Host "Starting LLaDA training from scratch..." -ForegroundColor Cyan
Write-Host "Note: Only tokenizer will be downloaded, model is created from scratch" -ForegroundColor Yellow

python training\train_simple.py `
    --tokenizer_name $TOKENIZER_NAME `
    --train_data_path $TRAIN_DATA_PATH `
    --eval_data_path $EVAL_DATA_PATH `
    --output_dir $OUTPUT_DIR `
    --max_length $MAX_LENGTH `
    --max_steps $MAX_STEPS `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --min_learning_rate "1e-5" `
    --weight_decay "0.01" `
    --max_grad_norm "1.0" `
    --d_model 512 `
    --nhead 8 `
    --num_layers 6 `
    --log_interval $LOG_INTERVAL `
    --eval_interval 100 `
    --save_interval $SAVE_INTERVAL `
    --num_workers 0 `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Training Completed Successfully ===" -ForegroundColor Green
Write-Host "Checkpoints saved in: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "Model created from scratch with $TOKENIZER_NAME tokenizer" -ForegroundColor Yellow

# Test inference with trained model
$BEST_MODEL = "$OUTPUT_DIR\best_model.pt"
if (Test-Path $BEST_MODEL) {
    Write-Host "Testing inference with trained model..." -ForegroundColor Cyan
    
    # Create a simple inference test script
    $INFERENCE_SCRIPT = @"
import torch
import json
from transformers import GPT2Tokenizer

# Load checkpoint
checkpoint = torch.load('$BEST_MODEL', map_location='cpu')
print(f"Loaded checkpoint from step {checkpoint['global_step']}")
print(f"Best loss: {checkpoint['best_loss']:.4f}")
print(f"Vocabulary size: {checkpoint['vocab_size']}")
print(f"Model config: {checkpoint['model_config']}")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('$OUTPUT_DIR/best_tokenizer')
print(f"Tokenizer vocabulary size: {len(tokenizer)}")
print(f"Mask token ID: {checkpoint['mask_token_id']}")

print("Model training completed successfully!")
"@
    
    $INFERENCE_SCRIPT | Out-File -FilePath "test_model.py" -Encoding utf8
    python test_model.py
    Remove-Item "test_model.py"
} else {
    Write-Host "Best model not found, check latest checkpoint" -ForegroundColor Yellow
    $latest_checkpoint = Get-ChildItem "$OUTPUT_DIR\checkpoint_step_*.pt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest_checkpoint) {
        Write-Host "Latest checkpoint: $($latest_checkpoint.Name)" -ForegroundColor Yellow
    }
}