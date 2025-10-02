@echo off
echo === LLaDA Simple Training for Windows ===
echo This trains LLaDA from scratch with online tokenizer only
echo.

cd /d "%~dp0\.."

echo Generating sample data...
python .\training\generate_sample_data.py

echo.
echo Starting LLaDA training from scratch...
echo Note: Only tokenizer will be downloaded, model is created from scratch
echo.

python .\training\train_simple.py ^
    --tokenizer_name gpt2 ^
    --train_data_path .\data\pretrain\train.jsonl ^
    --eval_data_path .\data\pretrain\eval.jsonl ^
    --output_dir .\outputs_simple ^
    --max_length 512 ^
    --max_steps 2000 ^
    --batch_size 2 ^
    --learning_rate 0.0003 ^
    --min_learning_rate 1e-5 ^
    --weight_decay 0.01 ^
    --max_grad_norm 1.0 ^
    --d_model 512 ^
    --nhead 8 ^
    --num_layers 6 ^
    --log_interval 20 ^
    --eval_interval 100 ^
    --save_interval 200 ^
    --num_workers 0 ^
    --seed 42

if %errorlevel% neq 0 (
    echo Training failed!
    pause
    exit /b 1
)

echo.
echo === Training Completed Successfully ===
echo Checkpoints saved in: outputs_simple\
echo Model created from scratch with gpt2 tokenizer

pause