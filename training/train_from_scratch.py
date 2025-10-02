import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
import json
import random
import argparse
import os
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def create_model_from_scratch(vocab_size, max_length=1024):
    """创建一个全新的小型GPT模型"""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_length,
        n_embd=512,          # 较小的嵌入维度
        n_layer=8,           # 较少的层数
        n_head=8,            # 较少的注意力头
        n_inner=2048,        # 较小的前馈网络
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
    )
    
    model = GPT2LMHeadModel(config)
    
    # 初始化权重
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    logging.info(f"Created model from scratch with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def get_mask_token_id(tokenizer):
    """Get or add mask token to tokenizer"""
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        return tokenizer.mask_token_id
    
    # Add mask token if it doesn't exist
    special_tokens_dict = {'mask_token': '[MASK]'}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logging.info(f"Added {num_added} special tokens to tokenizer")
    return tokenizer.mask_token_id


def forward_process(input_ids, eps=1e-3, mask_token_id=None):
    """
    Forward process for LLaDA pre-training as described in the guidelines.
    """
    b, l = input_ids.shape
    
    # Sample random masking ratios for each sequence
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    # Create random mask based on probability
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    
    # Apply mask token to masked positions
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)
    
    return noisy_batch, masked_indices, p_mask


class PretrainingDataset(Dataset):
    """Dataset for LLaDA pre-training"""
    
    def __init__(self, data_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self.data.append(item['text'])
                    except:
                        continue
        elif data_path.endswith('.txt'):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = text.split('\n\n')
                self.data.extend([chunk.strip() for chunk in chunks if chunk.strip()])
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logging.info(f"Loaded {len(self.data)} training examples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


class LLaDAFromScratchTrainer:
    """LLaDA from-scratch training class"""
    
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.mask_token_id = get_mask_token_id(tokenizer)
        
        logging.info(f"Using mask token ID: {self.mask_token_id}")
        
        # Resize model embeddings if new tokens were added
        if len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            logging.info(f"Resized model embeddings to {len(tokenizer)}")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_steps,
            eta_min=args.min_learning_rate
        )
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def compute_loss(self, batch):
        """Compute pre-training loss as described in guidelines"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Handle random length sequences (1% of the time)
        if torch.rand(1) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            input_ids = input_ids[:, :random_length]
            attention_mask = attention_mask[:, :random_length]
        
        # Apply forward process to add noise
        noisy_batch, masked_indices, p_mask = forward_process(
            input_ids, 
            eps=1e-3, 
            mask_token_id=self.mask_token_id
        )
        
        # Get model predictions
        outputs = self.model(input_ids=noisy_batch, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute cross-entropy loss only on masked tokens
        if masked_indices.any():
            token_loss = F.cross_entropy(
                logits[masked_indices], 
                input_ids[masked_indices], 
                reduction='none'
            ) / p_mask[masked_indices]
            
            # Average loss per sequence
            loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        return loss
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            self.global_step += 1
            
            return loss.item()
        except Exception as e:
            logging.error(f"Error in training step: {e}")
            return float('nan')
    
    def evaluate(self, eval_dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                try:
                    # Move batch to device
                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    
                    loss = self.compute_loss(batch)
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        num_batches += 1
                except Exception as e:
                    logging.warning(f"Error in evaluation step: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def save_checkpoint(self, save_dir, is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.config.to_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'mask_token_id': self.mask_token_id,
            'vocab_size': len(self.tokenizer),
            'args': self.args
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{self.global_step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save the model separately
        self.model.save_pretrained(os.path.join(save_dir, f'model_step_{self.global_step}'))
        self.tokenizer.save_pretrained(os.path.join(save_dir, f'model_step_{self.global_step}'))
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            
            # Save best model
            self.model.save_pretrained(os.path.join(save_dir, 'best_model'))
            self.tokenizer.save_pretrained(os.path.join(save_dir, 'best_model'))
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        logging.info("Starting LLaDA from-scratch training...")
        logging.info(f"Total steps: {self.args.max_steps}")
        logging.info(f"Batch size: {self.args.batch_size}")
        logging.info(f"Learning rate: {self.args.learning_rate}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        running_loss = 0
        log_steps = 0
        
        while self.global_step < self.args.max_steps:
            for batch in train_dataloader:
                if self.global_step >= self.args.max_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Training step
                loss = self.train_step(batch)
                
                if not np.isnan(loss):
                    running_loss += loss
                    log_steps += 1
                
                # Logging
                if self.global_step % self.args.log_interval == 0 and log_steps > 0:
                    avg_loss = running_loss / log_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    logging.info(
                        f"Step: {self.global_step}, "
                        f"Loss: {avg_loss:.4f}, "
                        f"LR: {lr:.2e}"
                    )
                    
                    running_loss = 0
                    log_steps = 0
                
                # Evaluation
                if eval_dataloader and self.global_step % self.args.eval_interval == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    logging.info(f"Eval Loss: {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(self.args.output_dir, is_best=True)
                
                # Save checkpoint
                if self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(self.args.output_dir)
        
        logging.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='LLaDA From-Scratch Training')
    
    # Model arguments
    parser.add_argument('--base_tokenizer', type=str, default='gpt2',
                       help='Base tokenizer to use for vocabulary')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'medium'],
                       help='Model size (tiny/small/medium)')
    
    # Data arguments
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to evaluation data')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum number of training steps')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5,
                       help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    
    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=50,
                       help='Log every N steps')
    parser.add_argument('--eval_interval', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save checkpoint every N steps')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging(args.output_dir)
    
    logging.info("=== LLaDA From-Scratch Training ===")
    logging.info("Creating new model and tokenizer from scratch...")
    
    # Load base tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_tokenizer)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model from scratch
    model = create_model_from_scratch(
        vocab_size=len(tokenizer), 
        max_length=args.max_length
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logging.info(f"Model created on device: {device}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = PretrainingDataset(
        args.train_data_path, 
        tokenizer, 
        args.max_length
    )
    
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = PretrainingDataset(
            args.eval_data_path,
            tokenizer,
            args.max_length
        )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # Create trainer
    trainer = LLaDAFromScratchTrainer(model, tokenizer, args)
    
    # Start training
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()