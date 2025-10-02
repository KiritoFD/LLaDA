import torch
import os
import json
from pathlib import Path


def quick_model_info(checkpoint_path):
    """快速查看模型信息"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print(f"=== Model Info: {os.path.basename(checkpoint_path)} ===")
        print(f"Training Step: {checkpoint.get('global_step', 'Unknown')}")
        print(f"Best Loss: {checkpoint.get('best_loss', 'Unknown'):.4f}")
        print(f"Vocabulary Size: {checkpoint.get('vocab_size', 'Unknown')}")
        print(f"Mask Token ID: {checkpoint.get('mask_token_id', 'Unknown')}")
        
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            print(f"\nModel Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        # 计算模型参数量
        if 'model_state_dict' in checkpoint:
            total_params = 0
            for key, tensor in checkpoint['model_state_dict'].items():
                total_params += tensor.numel()
            print(f"\nTotal Parameters: {total_params:,}")
        
        print()
        return True
        
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return False


def find_models(directory="outputs_simple"):
    """查找所有模型文件"""
    if not os.path.exists(directory):
        print(f"Directory {directory} not found")
        return []
    
    models = []
    
    # 查找 best_model.pt
    best_model = os.path.join(directory, "best_model.pt")
    if os.path.exists(best_model):
        models.append(best_model)
    
    # 查找所有checkpoint
    for file in os.listdir(directory):
        if file.startswith("checkpoint_step_") and file.endswith(".pt"):
            models.append(os.path.join(directory, file))
    
    return sorted(models)


def main():
    print("=== LLaDA Model Quick Info ===\n")
    
    # 查找模型
    models = find_models()
    
    if not models:
        print("No trained models found in outputs_simple/")
        print("Please train a model first using train_simple.bat")
        return
    
    print(f"Found {len(models)} model(s):\n")
    
    # 显示所有模型信息
    for model_path in models:
        quick_model_info(model_path)
    
    # 检查是否有评估结果
    eval_results_path = "eval_results/evaluation_results.json"
    if os.path.exists(eval_results_path):
        print("=== Previous Evaluation Results ===")
        try:
            with open(eval_results_path, 'r') as f:
                results = json.load(f)
            
            print(f"Overall Accuracy: {results.get('overall_accuracy', 'N/A'):.4f}")
            print(f"Masked Token Accuracy: {results.get('masked_token_accuracy', 'N/A'):.4f}")
            print(f"Average Loss: {results.get('average_loss', 'N/A'):.4f}")
            
            if 'perplexity' in results:
                print(f"Perplexity: {results['perplexity']:.2f}")
            
            print(f"Total Tokens Evaluated: {results.get('total_tokens', 'N/A'):,}")
            
        except Exception as e:
            print(f"Error reading evaluation results: {e}")
    else:
        print("No evaluation results found. Run evaluate.bat to evaluate the model.")


if __name__ == '__main__':
    main()