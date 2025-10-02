import yaml
import argparse
import os
import subprocess
import sys
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_pretraining(config, args):
    """Run pre-training with the given configuration"""
    pretraining_config = config['pretraining']
    model_config = config['model']
    
    cmd = [
        sys.executable, 'training/pretraining.py',
        '--model_name_or_path', model_config['name_or_path'],
        '--train_data_path', pretraining_config['train_data_path'],
        '--output_dir', pretraining_config['output_dir'],
        '--max_length', str(model_config['max_length']),
        '--max_steps', str(pretraining_config['max_steps']),
        '--batch_size', str(pretraining_config['batch_size']),
        '--learning_rate', str(pretraining_config['learning_rate']),
        '--min_learning_rate', str(pretraining_config['min_learning_rate']),
        '--weight_decay', str(pretraining_config['weight_decay']),
        '--max_grad_norm', str(pretraining_config['max_grad_norm']),
        '--log_interval', str(pretraining_config['log_interval']),
        '--eval_interval', str(pretraining_config['eval_interval']),
        '--save_interval', str(pretraining_config['save_interval']),
        '--num_workers', str(pretraining_config['num_workers']),
        '--seed', str(pretraining_config['seed'])
    ]
    
    # Add optional parameters
    if pretraining_config.get('eval_data_path'):
        cmd.extend(['--eval_data_path', pretraining_config['eval_data_path']])
    
    if model_config.get('tokenizer_name'):
        cmd.extend(['--tokenizer_name', model_config['tokenizer_name']])
    
    if args.resume_from_checkpoint:
        cmd.extend(['--resume_from_checkpoint', args.resume_from_checkpoint])
    
    print("Running pre-training with command:")
    print(' '.join(cmd))
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_sft(config, args):
    """Run SFT training with the given configuration"""
    sft_config = config['sft']
    model_config = config['model']
    pretraining_config = config['pretraining']
    
    # Determine model path
    if args.sft_model_path:
        model_path = args.sft_model_path
    else:
        # Use best model from pre-training
        model_path = os.path.join(pretraining_config['output_dir'], 'best_model.pt')
    
    cmd = [
        sys.executable, 'training/sft_training.py',
        '--model_name_or_path', model_path,
        '--train_data_path', sft_config['train_data_path'],
        '--output_dir', sft_config['output_dir'],
        '--max_length', str(model_config['max_length']),
        '--max_steps', str(sft_config['max_steps']),
        '--batch_size', str(sft_config['batch_size']),
        '--learning_rate', str(sft_config['learning_rate']),
        '--min_learning_rate', str(sft_config['min_learning_rate']),
        '--weight_decay', str(sft_config['weight_decay']),
        '--max_grad_norm', str(sft_config['max_grad_norm']),
        '--log_interval', str(sft_config['log_interval']),
        '--eval_interval', str(sft_config['eval_interval']),
        '--save_interval', str(sft_config['save_interval']),
        '--num_workers', str(sft_config['num_workers']),
        '--seed', str(sft_config['seed'])
    ]
    
    # Add optional parameters
    if sft_config.get('eval_data_path'):
        cmd.extend(['--eval_data_path', sft_config['eval_data_path']])
    
    if model_config.get('tokenizer_name'):
        cmd.extend(['--tokenizer_name', model_config['tokenizer_name']])
    
    if args.resume_from_checkpoint:
        cmd.extend(['--resume_from_checkpoint', args.resume_from_checkpoint])
    
    print("Running SFT training with command:")
    print(' '.join(cmd))
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_inference(config, args):
    """Run inference with the given configuration"""
    inference_config = config['inference']
    sft_config = config['sft']
    
    # Determine model path
    if args.inference_model_path:
        model_path = args.inference_model_path
    else:
        # Use best model from SFT
        model_path = os.path.join(sft_config['output_dir'], 'best_sft_model.pt')
    
    cmd = [
        sys.executable, 'training/inference.py',
        '--model_name_or_path', model_path,
        '--prompt', args.prompt or "What is the capital of France?",
        '--method', inference_config['default_method'],
        '--gen_length', str(inference_config['gen_length']),
        '--temperature', str(inference_config['temperature']),
        '--cfg_scale', str(inference_config['cfg_scale']),
        '--remasking', inference_config['remasking']
    ]
    
    # Add optional parameters
    if inference_config.get('steps'):
        cmd.extend(['--steps', str(inference_config['steps'])])
    
    if inference_config['default_method'] == 'semi_autoregressive_padding':
        cmd.extend(['--block_length', str(inference_config['block_length'])])
    
    if args.is_instruct:
        cmd.append('--is_instruct')
    
    print("Running inference with command:")
    print(' '.join(cmd))
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='LLaDA Training Pipeline')
    
    # Configuration
    parser.add_argument('--config', type=str, default='training/config.yaml',
                       help='Path to configuration file')
    
    # Pipeline control
    parser.add_argument('--stage', type=str, 
                       choices=['pretraining', 'sft', 'inference', 'all'],
                       default='all',
                       help='Which stage to run')
    
    # Model paths
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--sft_model_path', type=str, default=None,
                       help='Model path for SFT (default: use pre-training output)')
    parser.add_argument('--inference_model_path', type=str, default=None,
                       help='Model path for inference (default: use SFT output)')
    
    # Inference parameters
    parser.add_argument('--prompt', type=str, default=None,
                       help='Prompt for inference')
    parser.add_argument('--is_instruct', action='store_true',
                       help='Use instruct model format for inference')
    
    # Dry run
    parser.add_argument('--dry_run', action='store_true',
                       help='Print commands without executing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=== LLaDA Training Pipeline ===")
    print(f"Configuration: {args.config}")
    print(f"Stage: {args.stage}")
    
    # Create output directories
    os.makedirs(config['pretraining']['output_dir'], exist_ok=True)
    os.makedirs(config['sft']['output_dir'], exist_ok=True)
    
    try:
        if args.stage in ['pretraining', 'all']:
            print("\n--- Pre-training Stage ---")
            if not args.dry_run:
                success = run_pretraining(config, args)
                if not success:
                    print("Pre-training failed!")
                    return 1
            else:
                print("Dry run: Pre-training command would be executed")
        
        if args.stage in ['sft', 'all']:
            print("\n--- SFT Stage ---")
            if not args.dry_run:
                success = run_sft(config, args)
                if not success:
                    print("SFT training failed!")
                    return 1
            else:
                print("Dry run: SFT command would be executed")
        
        if args.stage in ['inference', 'all']:
            print("\n--- Inference Stage ---")
            if not args.dry_run:
                success = run_inference(config, args)
                if not success:
                    print("Inference failed!")
                    return 1
            else:
                print("Dry run: Inference command would be executed")
        
        print("\n=== Pipeline Completed Successfully ===")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())