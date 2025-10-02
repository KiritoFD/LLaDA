#!/usr/bin/env python3
"""
Test script for LLaDA from-scratch training
This script runs a very short training session to verify everything works
"""

import os
import sys
import subprocess
import json

def test_from_scratch_training():
    """Test from-scratch training with minimal data"""
    print("=== Testing LLaDA From-Scratch Training ===\n")
    
    # Create minimal test data
    print("Creating minimal test data...")
    os.makedirs("test_data", exist_ok=True)
    
    test_data = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning is powerful."},
        {"text": "Python is a programming language."}
    ]
    
    with open("test_data/train.jsonl", "w") as f:
        for item in test_data * 10:  # Repeat for more samples
            f.write(json.dumps(item) + "\n")
    
    with open("test_data/eval.jsonl", "w") as f:
        for item in test_data[:2]:
            f.write(json.dumps(item) + "\n")
    
    print("✓ Test data created")
    
    # Run minimal training
    print("\nRunning minimal from-scratch training (50 steps)...")
    
    cmd = [
        sys.executable, "training/train_from_scratch.py",
        "--base_tokenizer", "gpt2",
        "--train_data_path", "test_data/train.jsonl",
        "--eval_data_path", "test_data/eval.jsonl", 
        "--output_dir", "test_output",
        "--max_length", "128",
        "--max_steps", "50",
        "--batch_size", "1",
        "--learning_rate", "3e-4",
        "--log_interval", "10",
        "--eval_interval", "25",
        "--save_interval", "25",
        "--num_workers", "0",
        "--seed", "42"
    ]
    
    try:
        print("Command:", " ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        print("✓ From-scratch training test completed successfully!")
        print("Output:", result.stdout[-500:])  # Show last 500 characters
        return True
    except subprocess.TimeoutExpired:
        print("✗ Training test timed out (>10 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        print("✗ Training test failed!")
        print("Error output:", e.stderr)
        print("stdout:", e.stdout)
        return False

def main():
    success = test_from_scratch_training()
    
    if success:
        print("\n🎉 From-scratch training test passed!")
        print("You can now run the full training with:")
        print("  train_windows.bat")
        print("  or")
        print("  powershell .\\training\\train_windows.ps1")
    else:
        print("\n❌ From-scratch training test failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()