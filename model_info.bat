@echo off
echo === Quick Model Info ===
echo.

python .\training\quick_info.py

echo.
echo === Available Commands ===
echo 1. train_simple.bat     - Train a new LLaDA model
echo 2. evaluate.bat         - Evaluate trained model with accuracy metrics
echo 3. model_info.bat       - Show this information
echo.

pause