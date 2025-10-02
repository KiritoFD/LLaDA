@echo off
setlocal enabledelayedexpansion

echo === LLaDA Training Setup for Windows ===
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

python -c "import transformers; print('Transformers version:', transformers.__version__)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing Transformers...
    pip install transformers
)

python -c "import yaml" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing PyYAML...
    pip install pyyaml
)

REM Create necessary directories
if not exist "data\pretrain" mkdir "data\pretrain"
if not exist "data\sft" mkdir "data\sft"
if not exist "checkpoints" mkdir "checkpoints"
if not exist "logs" mkdir "logs"

echo.
echo === Generating Sample Data ===
python training\generate_sample_data.py
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to generate sample data
    pause
    exit /b 1
)

echo.
echo === Starting Completely Offline Pre-training ===
echo This creates everything from scratch without any downloads.
echo No internet connection required!
echo.

REM Start completely offline training
python training\train_completely_offline.py ^
    --train_data_path "data\pretrain\train.jsonl" ^
    --eval_data_path "data\pretrain\eval.jsonl" ^
    --output_dir "checkpoints\pretraining" ^
    --max_length 512 ^
    --vocab_size 5000 ^
    --max_steps 5000 ^
    --batch_size 2 ^
    --learning_rate 3e-4 ^
    --min_learning_rate 1e-5 ^
    --weight_decay 0.01 ^
    --max_grad_norm 1.0 ^
    --d_model 256 ^
    --nhead 8 ^
    --num_layers 6 ^
    --log_interval 50 ^
    --eval_interval 500 ^
    --save_interval 1000 ^
    --num_workers 0 ^
    --seed 42

if %ERRORLEVEL% neq 0 (
    echo Error: Pre-training failed
    pause
    exit /b 1
)

echo.
echo === Pre-training Completed Successfully ===
echo Checkpoints saved in: checkpoints\pretraining
echo.

REM Optionally run SFT training
set /p run_sft="Do you want to run SFT training as well? (y/n): "
if /i "!run_sft!"=="y" (
    echo.
    echo === Starting SFT Training ===
    
    python training\sft_training.py ^
        --model_name_or_path "checkpoints\pretraining\best_model.pt" ^
        --train_data_path "data\sft\train.jsonl" ^
        --eval_data_path "data\sft\eval.jsonl" ^
        --output_dir "checkpoints\sft" ^
        --max_length 1024 ^
        --max_steps 2000 ^
        --batch_size 2 ^
        --learning_rate 2e-5 ^
        --min_learning_rate 1e-6 ^
        --weight_decay 0.01 ^
        --max_grad_norm 1.0 ^
        --log_interval 25 ^
        --eval_interval 200 ^
        --save_interval 400 ^
        --num_workers 0 ^
        --seed 42
    
    if !ERRORLEVEL! neq 0 (
        echo Warning: SFT training failed, but pre-training was successful
    ) else (
        echo.
        echo === SFT Training Completed Successfully ===
        echo Checkpoints saved in: checkpoints\sft
    )
)

echo.
echo === Training Pipeline Summary ===
echo Pre-training: Complete
if /i "!run_sft!"=="y" (
    echo SFT training: Complete
)
echo.
echo You can now test inference with:
echo python training\inference.py --model_name_or_path checkpoints\pretraining\best_model.pt --prompt "Hello, how are you?"
echo.
pause