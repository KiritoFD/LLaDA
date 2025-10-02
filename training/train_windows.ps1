# LLaDA From-Scratch Training Script # Pre-training from scratch (creating new model)
Write-Host "Starting Pre-training from scratch (creating new model)..." -ForegroundColor Cyan
Write-Host "Note: This will create a completely new model, not download any pre-trained weights" -ForegroundColor Yellow

python training\train_from_scratch.py `
    --base_tokenizer $BASE_TOKENIZER `
    --train_data_path $TRAIN_DATA_PATH `
    --eval_data_path $EVAL_DATA_PATH `
    --output_dir "$OUTPUT_DIR\pretraining" `
    --max_length $MAX_LENGTH `
    --max_steps $MAX_STEPS `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --min_learning_rate "1e-5" `
    --weight_decay "0.01" `
    --max_grad_norm "1.0" `
    --log_interval 50 `
    --eval_interval 500 `
    --save_interval 1000 `
    --num_workers 0 `
    --seed 42 script creates and trains a LLaDA model completely from scratch

# Configuration
$BASE_TOKENIZER = "gpt2"        # Only use tokenizer, not the model
$TRAIN_DATA_PATH = ".\data\pretrain\train.jsonl"
$EVAL_DATA_PATH = ".\data\pretrain\eval.jsonl"
$OUTPUT_DIR = ".\checkpoints"
$MAX_LENGTH = 1024   # Reduced for Windows compatibility
$BATCH_SIZE = 2      # Small batch size for limited GPU memory
$LEARNING_RATE = "3e-4"
$MAX_STEPS = 10000   # Reasonable number of steps for demonstration

# Create directories
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "data\pretrain"
New-Item -ItemType Directory -Force -Path "data\sft"

Write-Host "=== LLaDA From-Scratch Training (Windows) ===" -ForegroundColor Green
Write-Host "Base Tokenizer: $BASE_TOKENIZER (vocabulary only)" -ForegroundColor Yellow
Write-Host "Output Directory: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "Max Steps: $MAX_STEPS" -ForegroundColor Yellow
Write-Host "Batch Size: $BATCH_SIZE" -ForegroundColor Yellow
Write-Host "Learning Rate: $LEARNING_RATE" -ForegroundColor Yellow

# Generate sample data first
Write-Host "Generating sample training data..." -ForegroundColor Cyan
python training\generate_sample_data.py

# Check if data files exist
if (-not (Test-Path $TRAIN_DATA_PATH)) {
    Write-Host "Error: Training data not found at $TRAIN_DATA_PATH" -ForegroundColor Red
    Write-Host "Please run: python training\generate_sample_data.py" -ForegroundColor Yellow
    exit 1
}

# Pre-training from scratch
Write-Host "Starting Pre-training from scratch..." -ForegroundColor Cyan
python training\pretraining.py `
    --model_name_or_path $MODEL_NAME `
    --train_data_path $TRAIN_DATA_PATH `
    --eval_data_path $EVAL_DATA_PATH `
    --output_dir "$OUTPUT_DIR\pretraining" `
    --max_length $MAX_LENGTH `
    --max_steps $MAX_STEPS `
    --batch_size $BATCH_SIZE `
    --learning_rate $LEARNING_RATE `
    --min_learning_rate "1e-5" `
    --weight_decay "0.01" `
    --max_grad_norm "1.0" `
    --log_interval 50 `
    --eval_interval 500 `
    --save_interval 2000 `
    --num_workers 0 `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pre-training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Pre-training completed!" -ForegroundColor Green

# SFT Training
Write-Host "Starting SFT Training..." -ForegroundColor Cyan
$SFT_MODEL_PATH = "$OUTPUT_DIR\pretraining\best_model.pt"
$SFT_TRAIN_DATA = ".\data\sft\train.jsonl"
$SFT_EVAL_DATA = ".\data\sft\eval.jsonl"

# Check if SFT model exists, otherwise use the latest checkpoint
if (-not (Test-Path $SFT_MODEL_PATH)) {
    $checkpoints = Get-ChildItem "$OUTPUT_DIR\pretraining\checkpoint_step_*.pt" | Sort-Object LastWriteTime -Descending
    if ($checkpoints.Count -gt 0) {
        $SFT_MODEL_PATH = $checkpoints[0].FullName
        Write-Host "Using checkpoint: $SFT_MODEL_PATH" -ForegroundColor Yellow
    } else {
        Write-Host "No pre-training checkpoint found. Using base model." -ForegroundColor Yellow
        $SFT_MODEL_PATH = $MODEL_NAME
    }
}

python training\sft_training.py `
    --model_name_or_path $SFT_MODEL_PATH `
    --train_data_path $SFT_TRAIN_DATA `
    --eval_data_path $SFT_EVAL_DATA `
    --output_dir "$OUTPUT_DIR\sft" `
    --max_length $MAX_LENGTH `
    --max_steps 5000 `
    --batch_size $BATCH_SIZE `
    --learning_rate "2e-5" `
    --min_learning_rate "1e-6" `
    --weight_decay "0.01" `
    --max_grad_norm "1.0" `
    --log_interval 25 `
    --eval_interval 250 `
    --save_interval 500 `
    --num_workers 0 `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Host "SFT Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "SFT Training completed!" -ForegroundColor Green

# Test inference
Write-Host "Testing inference..." -ForegroundColor Cyan
$FINAL_MODEL = "$OUTPUT_DIR\sft\best_sft_model.pt"

# Check if final model exists, otherwise use the latest SFT checkpoint
if (-not (Test-Path $FINAL_MODEL)) {
    $sft_checkpoints = Get-ChildItem "$OUTPUT_DIR\sft\sft_checkpoint_step_*.pt" | Sort-Object LastWriteTime -Descending
    if ($sft_checkpoints.Count -gt 0) {
        $FINAL_MODEL = $sft_checkpoints[0].FullName
        Write-Host "Using SFT checkpoint: $FINAL_MODEL" -ForegroundColor Yellow
    } else {
        Write-Host "No SFT checkpoint found. Using pre-training model." -ForegroundColor Yellow
        $FINAL_MODEL = $SFT_MODEL_PATH
    }
}

python training\inference.py `
    --model_name_or_path $FINAL_MODEL `
    --prompt "What is the capital of France?" `
    --method fixed_length `
    --gen_length 64 `
    --steps 64 `
    --temperature 0.0 `
    --cfg_scale 0.0 `
    --remasking low_confidence

if ($LASTEXITCODE -ne 0) {
    Write-Host "Inference test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Training Pipeline Completed Successfully ===" -ForegroundColor Green
Write-Host "Checkpoints saved in: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "Logs available in: logs\" -ForegroundColor Yellow