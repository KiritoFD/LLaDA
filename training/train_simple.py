import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
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


class SimpleTransformer(nn.Module):
    """Simple Transformer model for LLaDA training from scratch"""
    
    def __init__(self, vocab_size, max_length=1024, d_model=512, nhead=8, num_layers=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer layers (without causal mask for LLaDA)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logging.info(f"Created Transformer with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        hidden_states = token_emb + pos_emb
        
        # Create attention mask for transformer (inverted for PyTorch)
        if attention_mask is not None:
            # Convert to mask that transformer expects (True = ignore)
            attention_mask = attention_mask == 0
        
        # Transformer
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return type('ModelOutput', (), {'logits': logits})()


def forward_process(input_ids, eps=1e-3, mask_token_id=None):
    """Forward process for LLaDA pre-training as described in the guidelines"""
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


class LLaDATrainer:
    """LLaDA training class"""
    
    def __init__(self, model, tokenizer, args, device):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        
        # Get mask token ID
        if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            # Add mask token if it doesn't exist
            special_tokens_dict = {'mask_token': '[MASK]'}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            self.mask_token_id = tokenizer.mask_token_id
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
                logging.info(f"Added {num_added} special tokens, resized embeddings to {len(tokenizer)}")
        
        logging.info(f"Using mask token ID: {self.mask_token_id}")
        
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
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
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
        
        # Create a clean args dict instead of storing the args object directly
        args_dict = {
            'learning_rate': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
            'max_steps': self.args.max_steps,
            'batch_size': self.args.batch_size,
            'max_length': self.args.max_length,
            'min_learning_rate': self.args.min_learning_rate
        }
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'mask_token_id': self.mask_token_id,
            'vocab_size': len(self.tokenizer),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'max_length': self.model.max_length,
                'd_model': self.model.d_model,
            },
            'args': args_dict  # Use clean dict instead of args object
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{self.global_step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, f'tokenizer_step_{self.global_step}'))
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            
            # Save best tokenizer
            self.tokenizer.save_pretrained(os.path.join(save_dir, 'best_tokenizer'))
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        logging.info("Starting LLaDA training from scratch...")
        logging.info(f"Total steps: {self.args.max_steps}")
        logging.info(f"Batch size: {self.args.batch_size}")
        logging.info(f"Learning rate: {self.args.learning_rate}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"Vocabulary size: {len(self.tokenizer)}")
        
        running_loss = 0
        log_steps = 0
        
        while self.global_step < self.args.max_steps:
            for batch in train_dataloader:
                if self.global_step >= self.args.max_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
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
    parser = argparse.ArgumentParser(description='LLaDA Training from Scratch (with online tokenizer)')
    
    # Model arguments
    parser.add_argument('--tokenizer_name', type=str, default='gpt2',
                       help='Tokenizer to download and use')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of transformer layers')
    
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
    
    logging.info("=== LLaDA Training from Scratch ===")
    logging.info("Downloading tokenizer (model weights will be created from scratch)...")
    
    # Load tokenizer (only download tokenizer, not model weights)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Tokenizer loaded: {len(tokenizer)} tokens")
    
    # Create model from scratch
    model = SimpleTransformer(
        vocab_size=len(tokenizer),
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
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
    trainer = LLaDATrainer(model, tokenizer, args, device)
    
    # Start training
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()