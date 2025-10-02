@echo off
echo === LLaDA Model Evaluation ===
echo.

set MODEL_PATH=.\outputs_simple\best_model.pt
set EVAL_DATA=.\data\pretrain\eval.jsonl
set OUTPUT_DIR=.\eval_results

echo Checking for trained model...
if not exist %MODEL_PATH% (
    echo Model not found at %MODEL_PATH%
    echo Looking for latest checkpoint...
    
    for /f "delims=" %%i in ('dir /b /o-d ".\outputs_simple\checkpoint_step_*.pt" 2^>nul') do (
        set MODEL_PATH=.\outputs_simple\%%i
        goto :found
    )
    
    echo No model checkpoints found in outputs_simple\
    echo Please train a model first using train_simple.bat
    pause
    exit /b 1
)

:found
echo Using model: %MODEL_PATH%
echo Evaluation data: %EVAL_DATA%
echo Output directory: %OUTPUT_DIR%
echo.

echo Running comprehensive evaluation...
python .\training\evaluate_simple.py ^
    --model_path "%MODEL_PATH%" ^
    --eval_data_path "%EVAL_DATA%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --batch_size 4 ^
    --max_length 512 ^
    --test_generation

if %errorlevel% neq 0 (
    echo Evaluation failed!
    pause
    exit /b 1
)

echo.
echo === Evaluation Completed ===
echo Results saved in: %OUTPUT_DIR%
echo Check evaluation_results.json for detailed metrics
echo Check metrics_plot.png for visualization

pause