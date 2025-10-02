import torch
import os
import argparse
import logging


def fix_checkpoint(checkpoint_path, output_path=None):
    """修复checkpoint文件，移除不兼容的对象"""
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # 使用weights_only=False加载
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 创建新的clean checkpoint
        clean_checkpoint = {}
        
        # 复制基本信息
        basic_keys = [
            'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
            'global_step', 'best_loss', 'mask_token_id', 'vocab_size', 'model_config'
        ]
        
        for key in basic_keys:
            if key in checkpoint:
                clean_checkpoint[key] = checkpoint[key]
        
        # 处理args - 转换为dict
        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, '__dict__'):  # 如果是对象
                args_dict = {
                    'learning_rate': getattr(args, 'learning_rate', 3e-4),
                    'weight_decay': getattr(args, 'weight_decay', 0.01),
                    'max_steps': getattr(args, 'max_steps', 10000),
                    'batch_size': getattr(args, 'batch_size', 2),
                    'max_length': getattr(args, 'max_length', 1024),
                    'min_learning_rate': getattr(args, 'min_learning_rate', 1e-5)
                }
                clean_checkpoint['args'] = args_dict
            else:  # 如果已经是dict
                clean_checkpoint['args'] = args
        
        # 保存修复后的checkpoint
        if output_path is None:
            output_path = checkpoint_path.replace('.pt', '_fixed.pt')
        
        torch.save(clean_checkpoint, output_path)
        print(f"Fixed checkpoint saved to: {output_path}")
        
        # 验证修复后的文件可以正常加载
        test_checkpoint = torch.load(output_path, map_location='cpu', weights_only=False)
        print("✓ Fixed checkpoint verified successfully")
        
        return output_path
        
    except Exception as e:
        print(f"Error fixing checkpoint: {e}")
        return None


def fix_all_checkpoints(directory="outputs_simple"):
    """修复目录中的所有checkpoint文件"""
    if not os.path.exists(directory):
        print(f"Directory {directory} not found")
        return
    
    print(f"Scanning {directory} for checkpoint files...")
    
    fixed_files = []
    
    # 查找所有.pt文件
    for file in os.listdir(directory):
        if file.endswith('.pt') and not file.endswith('_fixed.pt'):
            checkpoint_path = os.path.join(directory, file)
            print(f"\nProcessing: {file}")
            
            fixed_path = fix_checkpoint(checkpoint_path)
            if fixed_path:
                fixed_files.append(fixed_path)
    
    if fixed_files:
        print(f"\n✓ Successfully fixed {len(fixed_files)} checkpoint(s)")
        print("Fixed files:")
        for file in fixed_files:
            print(f"  - {os.path.basename(file)}")
        
        # 可选：替换原文件
        print(f"\nDo you want to replace the original files? (y/n): ", end="")
        try:
            response = input().lower()
            if response == 'y':
                for fixed_file in fixed_files:
                    original_file = fixed_file.replace('_fixed.pt', '.pt')
                    if os.path.exists(original_file):
                        os.remove(original_file)
                        os.rename(fixed_file, original_file)
                        print(f"Replaced: {os.path.basename(original_file)}")
                print("✓ All original files replaced with fixed versions")
        except:
            print("Skipping file replacement")
    else:
        print("No files were fixed")


def main():
    parser = argparse.ArgumentParser(description='Fix LLaDA checkpoint files for PyTorch 2.6+ compatibility')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint file to fix')
    parser.add_argument('--directory', type=str, default='outputs_simple', 
                       help='Directory containing checkpoint files')
    parser.add_argument('--output', type=str, help='Output path for fixed checkpoint')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # 修复单个文件
        if os.path.exists(args.checkpoint):
            fix_checkpoint(args.checkpoint, args.output)
        else:
            print(f"Checkpoint file not found: {args.checkpoint}")
    else:
        # 修复目录中的所有文件
        fix_all_checkpoints(args.directory)


if __name__ == '__main__':
    main()