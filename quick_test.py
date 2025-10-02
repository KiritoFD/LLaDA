#!/usr/bin/env python3
"""
Quick test script for LLaDA training on Windows
This script runs a minimal training session to test if everything is working
"""

import os
import sys
import subprocess
import time
import json

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - will use CPU")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    try:
        import yaml
        print("✓ PyYAML available")
    except ImportError:
        print("✗ PyYAML not installed")
        return False
    
    return True

def create_minimal_data():
    """Create minimal test data"""
    print("Creating minimal test data...")
    
    os.makedirs("data/pretrain", exist_ok=True)
    os.makedirs("data/sft", exist_ok=True)
    
    # Minimal pre-training data
    pretrain_data = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning is a subset of artificial intelligence."},
        {"text": "The solar system has eight planets."},
        {"text": "Programming helps solve complex problems."},
        {"text": "Natural language processing enables human-computer interaction."}
    ]
    
    with open("data/pretrain/train.jsonl", "w") as f:
        for item in pretrain_data * 20:  # Repeat for more data
            f.write(json.dumps(item) + "\n")
    
    with open("data/pretrain/eval.jsonl", "w") as f:
        for item in pretrain_data[:3]:  # Small eval set
            f.write(json.dumps(item) + "\n")
    
    # Minimal SFT data
    sft_data = [
        {
            "conversations": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help you?"}
            ]
        }
    ]
    
    with open("data/sft/train.jsonl", "w") as f:
        for item in sft_data * 10:  # Repeat for more data
            f.write(json.dumps(item) + "\n")
    
    with open("data/sft/eval.jsonl", "w") as f:
        for item in sft_data:
            f.write(json.dumps(item) + "\n")
    
    print("✓ Test data created")

def run_quick_test():
    """Run a quick training test"""
    print("Running quick pre-training test (100 steps)...")
    
    cmd = [
        sys.executable, "training/pretraining_from_scratch.py",
        "--model_name_or_path", "microsoft/DialoGPT-small",
        "--train_data_path", "data/pretrain/train.jsonl",
        "--eval_data_path", "data/pretrain/eval.jsonl",
        "--output_dir", "test_checkpoints",
        "--max_length", "256",
        "--max_steps", "100",
        "--batch_size", "1",
        "--learning_rate", "3e-4",
        "--log_interval", "10",
        "--eval_interval", "50",
        "--save_interval", "50",
        "--num_workers", "0",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Quick test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ Quick test failed!")
        print("Error output:", e.stderr)
        return False

def main():
    print("=== LLaDA Windows Training Quick Test ===\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies:")
        print("pip install torch transformers pyyaml")
        return
    
    print()
    
    # Create minimal data
    create_minimal_data()
    print()
    
    # Run quick test
    start_time = time.time()
    success = run_quick_test()
    end_time = time.time()
    
    print(f"\nTest completed in {end_time - start_time:.1f} seconds")
    
    if success:
        print("\n🎉 Everything is working! You can now run the full training:")
        print("1. Run: train_windows.bat")
        print("2. Or use PowerShell: .\\training\\train_windows.ps1")
        print("3. Or Python pipeline: python training\\train_pipeline.py --config training\\config_windows.yaml")
    else:
        print("\n❌ There were issues. Please check the error messages above.")
        print("Common solutions:")
        print("- Make sure you have enough GPU memory (try batch_size=1)")
        print("- Check if all dependencies are correctly installed")
        print("- Try running on CPU by setting CUDA_VISIBLE_DEVICES=\"\"")

if __name__ == "__main__":
    main()