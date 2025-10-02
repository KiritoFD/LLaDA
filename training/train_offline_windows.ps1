# LLaDA Completely Offline Training Script for Windows
# This script creates and trains a LLaDA model completely offline (no downloads)

# Configuration
$TRAIN_DATA_PATH = ".\data\pretrain\train.jsonl"
$EVAL_DATA_PATH = ".\data\pretrain\eval.jsonl"
$OUTPUT_DIR = ".\checkpoints"
$MAX_LENGTH = 512     # Reduced for Windows compatibility
$BATCH_SIZE = 2       # Small batch size for limited GPU memory
$LEARNING_RATE = "3e-4"
$MAX_STEPS = 5000     # Shorter training for demonstration
$VOCAB_SIZE = 5000    # Smaller vocabulary

# Create directories
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "data\pretrain"
New-Item -ItemType Directory -Force -Path "data\sft"

Write-Host "=== LLaDA Completely Offline Training (Windows) ===" -ForegroundColor Green
Write-Host "No downloads required - everything created from scratch!" -ForegroundColor Yellow
Write-Host "Output Directory: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "Max Steps: $MAX_STEPS" -ForegroundColor Yellow
Write-Host "Batch Size: $BATCH_SIZE" -ForegroundColor Yellow
Write-Host "Learning Rate: $LEARNING_RATE" -ForegroundColor Yellow
Write-Host "Vocabulary Size: $VOCAB_SIZE" -ForegroundColor Yellow

# Generate sample data first
Write-Host "Generating sample training data..." -ForegroundColor Cyan
python training\generate_sample_data.py

# Check if data files exist
if (-not (Test-Path $TRAIN_DATA_PATH)) {
    Write-Host "Error: Training data not found at $TRAIN_DATA_PATH" -ForegroundColor Red
    Write-Host "Please run: python training\generate_sample_data.py" -ForegroundColor Yellow
    exit 1
}

# Pre-training completely offline (no downloads)
Write-Host "Starting completely offline pre-training..." -ForegroundColor Cyan
Write-Host "Note: This creates everything from scratch - no internet required!" -ForegroundColor Yellow

python training\train_completely_offline.py `
    --train_data_path $TRAIN_DATA_PATH `
    --eval_data_path $EVAL_DATA_PATH `
    --output_dir "$OUTPUT_DIR\pretraining" `
    --max_length $MAX_LENGTH `
    --vocab_size $VOCAB_SIZE `
    --max_steps $MAX_STEPS `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --min_learning_rate "1e-5" `
    --weight_decay "0.01" `
    --max_grad_norm "1.0" `
    --d_model 256 `
    --nhead 8 `
    --num_layers 6 `
    --log_interval 50 `
    --eval_interval 500 `
    --save_interval 1000 `
    --num_workers 0 `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pre-training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Pre-training completed!" -ForegroundColor Green

# Test the trained model
Write-Host "Testing the trained model..." -ForegroundColor Cyan

# Create a simple inference test script
$inferenceTest = @"
import torch
import sys
import os

# Load the trained model
checkpoint_path = 'checkpoints/pretraining/best_model.pt'
if os.path.exists(checkpoint_path):
    print('Loading trained model...')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f'Model trained for {checkpoint["global_step"]} steps')
    print(f'Best loss: {checkpoint["best_loss"]:.4f}')
    print(f'Vocabulary size: {checkpoint["vocab_size"]}')
    print('Model training successful!')
else:
    print('Model checkpoint not found!')
    sys.exit(1)
"@

$inferenceTest | Out-File -FilePath "test_model.py" -Encoding utf8

python test_model.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Training Pipeline Completed Successfully ===" -ForegroundColor Green
    Write-Host "Checkpoints saved in: $OUTPUT_DIR" -ForegroundColor Yellow
    Write-Host "Logs available in: logs\" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Your LLaDA model has been trained completely offline!" -ForegroundColor Green
    Write-Host "No internet connection was required for training." -ForegroundColor Green
} else {
    Write-Host "Model test failed!" -ForegroundColor Red
    exit 1
}

# Clean up test file
Remove-Item "test_model.py" -ErrorAction SilentlyContinue