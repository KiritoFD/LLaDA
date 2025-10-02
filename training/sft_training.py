import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
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
            logging.FileHandler(os.path.join(log_dir, 'sft_training.log')),
            logging.StreamHandler()
        ]
    )


def forward_process(input_ids, eps=1e-3, mask_token_id=126336):
    """
    Forward process for SFT training.
    Same as pre-training forward process.
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


class SFTDataset(Dataset):
    """Dataset for LLaDA SFT training"""
    
    def __init__(self, data_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load conversation data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def format_conversation(self, conversation):
        """
        Format conversation following the chat template format.
        Expected format in JSON:
        {
            "conversations": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."}
            ]
        }
        """
        if 'conversations' in conversation:
            messages = conversation['conversations']
        elif 'messages' in conversation:
            messages = conversation['messages']
        else:
            # Handle single turn format
            if 'instruction' in conversation and 'output' in conversation:
                messages = [
                    {"role": "user", "content": conversation['instruction']},
                    {"role": "assistant", "content": conversation['output']}
                ]
            else:
                raise ValueError("Unsupported conversation format")
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return formatted_text, messages
    
    def get_prompt_length(self, messages):
        """Calculate the length of the prompt (user messages + system + assistant prefix)"""
        # Create prompt without the assistant's response
        prompt_messages = []
        for msg in messages:
            if msg['role'] != 'assistant':
                prompt_messages.append(msg)
            else:
                # Add assistant role but without content
                prompt_messages.append({"role": "assistant", "content": ""})
                break
        
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        return len(prompt_tokens)
    
    def __getitem__(self, idx):
        conversation = self.data[idx]
        
        # Format conversation
        formatted_text, messages = self.format_conversation(conversation)
        
        # Tokenize the full conversation
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Calculate prompt length
        try:
            prompt_length = self.get_prompt_length(messages)
        except:
            # Fallback: assume first message is the prompt
            first_msg = messages[0]['content'] if messages else ""
            prompt_tokens = self.tokenizer(first_msg, add_special_tokens=False)['input_ids']
            prompt_length = len(prompt_tokens) + 10  # Add some buffer for special tokens
        
        # Ensure prompt_length doesn't exceed sequence length
        prompt_length = min(prompt_length, input_ids.shape[0] - 1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_length': prompt_length
        }


class LLaDASFTTrainer:
    """LLaDA SFT training class"""
    
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.mask_token_id = 126336
        
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
    
    def compute_sft_loss(self, batch):
        """
        Compute SFT loss as described in guidelines.
        Key difference from pre-training: do not add noise to the prompt.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompt_lengths = batch['prompt_length']
        
        # Apply forward process to add noise
        noisy_batch, _, p_mask = forward_process(
            input_ids,
            eps=1e-3,
            mask_token_id=self.mask_token_id
        )
        
        # Do not add noise to the prompt
        batch_size, seq_len = noisy_batch.shape
        token_positions = torch.arange(seq_len, device=noisy_batch.device).expand(batch_size, seq_len)
        
        # Create prompt mask
        prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
        
        # Restore original tokens in prompt positions
        noisy_batch[prompt_mask] = input_ids[prompt_mask]
        
        # Calculate the answer length (including padded tokens)
        prompt_mask_int = prompt_mask.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_mask_int), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, seq_len)
        
        # Get masked positions
        masked_indices = (noisy_batch == self.mask_token_id)
        
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
            
            # Normalize by answer length and batch size
            ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
        else:
            ce_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        return ce_loss
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_sft_loss(batch)
        loss.backward()
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def evaluate(self, eval_dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                loss = self.compute_sft_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def save_checkpoint(self, save_dir, is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'args': self.args
        }
        
        checkpoint_path = os.path.join(save_dir, f'sft_checkpoint_step_{self.global_step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_sft_model.pt')
            torch.save(checkpoint, best_path)
        
        logging.info(f"SFT Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logging.info(f"SFT Checkpoint loaded from {checkpoint_path}")
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main SFT training loop"""
        logging.info("Starting LLaDA SFT training...")
        logging.info(f"Total steps: {self.args.max_steps}")
        logging.info(f"Batch size: {self.args.batch_size}")
        logging.info(f"Learning rate: {self.args.learning_rate}")
        
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
                running_loss += loss
                log_steps += 1
                
                # Logging
                if self.global_step % self.args.log_interval == 0:
                    avg_loss = running_loss / log_steps if log_steps > 0 else 0
                    lr = self.scheduler.get_last_lr()[0]
                    
                    logging.info(
                        f"SFT Step: {self.global_step}, "
                        f"Loss: {avg_loss:.4f}, "
                        f"LR: {lr:.2e}"
                    )
                    
                    running_loss = 0
                    log_steps = 0
                
                # Evaluation
                if eval_dataloader and self.global_step % self.args.eval_interval == 0:
                    eval_loss = self.evaluate(eval_dataloader)
                    logging.info(f"SFT Eval Loss: {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(self.args.output_dir, is_best=True)
                
                # Save checkpoint
                if self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(self.args.output_dir)
        
        logging.info("SFT Training completed!")


def main():
    parser = argparse.ArgumentParser(description='LLaDA SFT Training')
    
    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, required=True,
                       help='Path to pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--tokenizer_name', type=str, default=None,
                       help='Tokenizer name, defaults to model_name_or_path')
    
    # Data arguments
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='Path to SFT training data (JSONL format)')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to SFT evaluation data (JSONL format)')
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum number of training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (typically lower than pre-training)')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6,
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
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging(args.output_dir)
    
    # Load tokenizer
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Load model
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logging.info(f"Model loaded on device: {device}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = SFTDataset(
        args.train_data_path,
        tokenizer,
        args.max_length
    )
    
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = SFTDataset(
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
        pin_memory=True
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create trainer
    trainer = LLaDASFTTrainer(model, tokenizer, args)
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()