@echo off
echo === Fix LLaDA Checkpoints ===
echo This script fixes checkpoint compatibility issues with PyTorch 2.6+
echo.

echo Checking for checkpoint files in outputs_simple\...
if not exist "outputs_simple\" (
    echo Directory outputs_simple\ not found
    echo Please train a model first using train_simple.bat
    pause
    exit /b 1
)

echo.
echo Running checkpoint fix...
python .\training\fix_checkpoints.py --directory outputs_simple

if %errorlevel% neq 0 (
    echo Fix failed!
    pause
    exit /b 1
)

echo.
echo === Checkpoint Fix Completed ===
echo You can now run evaluate.bat to evaluate your model

pause