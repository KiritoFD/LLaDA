import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import json
import random
import argparse
import os
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
import re
from collections import Counter
from typing import Dict, Any, Optional
import datetime
import jieba


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


CONFIG_SECTION_MAP: Dict[str, Dict[str, str]] = {
    "data": {
        "train_data_path": "train_data_path",
        "eval_data_path": "eval_data_path",
    },
    "model": {
        "max_length": "max_length",
        "vocab_size": "vocab_size",
        "d_model": "d_model",
        "nhead": "nhead",
        "num_layers": "num_layers",
    },
    "training": {
        "max_steps": "max_steps",
        "batch_size": "batch_size",
        "eval_batch_size": "eval_batch_size",
        "learning_rate": "learning_rate",
        "min_learning_rate": "min_learning_rate",
        "weight_decay": "weight_decay",
        "max_grad_norm": "max_grad_norm",
        "log_interval": "log_interval",
        "eval_interval": "eval_interval",
        "save_interval": "save_interval",
        "resume_from": "resume_from",
        "eval_subset_ratio": "eval_subset_ratio",
        "seed": "seed",
    },
    "runtime": {
        "output_dir": "output_dir",
        "num_workers": "num_workers",
        "pin_memory": "pin_memory",
        "clear_cuda_cache": "clear_cuda_cache",
        "cuda_cache_interval": "cuda_cache_interval",
    },
}


def str_to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in {"true", "1", "yes", "y"}:
            return True
        if value_lower in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise argparse.ArgumentTypeError(f"Cannot interpret {value!r} as boolean")


def load_config_defaults(config_path: str) -> Dict[str, Any]:
    """Load configuration JSON and return parser default overrides."""
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open('r', encoding='utf-8') as handle:
        config_data = json.load(handle)

    defaults: Dict[str, Any] = {}

    for section, mapping in CONFIG_SECTION_MAP.items():
        section_data = config_data.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for key, dest in mapping.items():
            if key in section_data:
                defaults[dest] = section_data[key]

    # Allow top-level keys matching argument names
    for key, value in config_data.items():
        if key in defaults:
            # already set via section
            continue
        if any(key == mapping_key for mapping in CONFIG_SECTION_MAP.values() for mapping_key in mapping.values()):
            defaults[key] = value

    return defaults


def validate_required_args(args: argparse.Namespace, config_path: Optional[str] = None) -> None:
    missing_fields = [
        field for field in ("train_data_path", "output_dir")
        if getattr(args, field, None) in (None, "")
    ]

    if missing_fields:
        location = f" in config {config_path}" if config_path else ""
        raise ValueError(f"Missing required values{location}: {', '.join(missing_fields)}")


def extract_text_from_record(record: Any) -> Optional[str]:
    """Extract a usable text string from a JSONL record."""

    if not isinstance(record, dict):
        return None

    text_field = record.get('text')
    if isinstance(text_field, str) and text_field.strip():
        return text_field.strip()

    parts = []

    question = record.get('question')
    if isinstance(question, str) and question.strip():
        parts.append(f"问题: {question.strip()}")

    context = record.get('context')
    if isinstance(context, str) and context.strip():
        parts.append(context.strip())

    answer = record.get('answer')
    if isinstance(answer, str) and answer.strip():
        parts.append(f"答案: {answer.strip()}")

    if parts:
        return "\n".join(parts)

    return None


class SimpleTokenizer:
    """Simple tokenizer that creates vocabulary from training data with fixed mask token ID."""

    def __init__(self, vocab_size=10000, mask_token_id=126336):
        self.vocab_size = vocab_size
        self._mask_token_id = mask_token_id

        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }

        # Initialize special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

        # Register mask token (using reserved ID from guidelines)
        self.word_to_id['<MASK>'] = self._mask_token_id
        self.id_to_word[self._mask_token_id] = '<MASK>'

        self.next_id = len(self.special_tokens)
        self.vocab_size_actual = max(self.next_id, self._mask_token_id + 1)

    def _assign_id(self, word):
        """Assign the next available ID to a token, skipping the mask token ID."""
        idx = self.next_id
        if idx == self._mask_token_id:
            idx += 1

        self.word_to_id[word] = idx
        self.id_to_word[idx] = word
        self.next_id = idx + 1
        self.vocab_size_actual = max(self.vocab_size_actual, self.next_id, self._mask_token_id + 1)

    def build_vocab(self, texts):
        """Build vocabulary from training texts using jieba for Chinese segmentation"""
        print("Building vocabulary from training data using jieba...")

        word_counts = Counter()
        for text in texts:
            # Use jieba for Chinese word segmentation
            words = list(jieba.cut(text))
            word_counts.update(words)

        limit = self.vocab_size - len(self.special_tokens)
        most_common = word_counts.most_common(max(0, limit))

        for word, _ in most_common:
            if word not in self.word_to_id:
                self._assign_id(word)

        # Ensure embeddings cover the mask token ID
        self.vocab_size_actual = max(self.vocab_size_actual, self._mask_token_id + 1)
        print(f"Built vocabulary with {len(self.word_to_id)} tokens (embedding size {self.vocab_size_actual})")

    def encode(self, text, max_length=512, padding=True):
        """Encode text to token ids using jieba segmentation"""
        words = list(jieba.cut(text))
        token_ids = [self.special_tokens['<BOS>']]

        for word in words:
            token_ids.append(self.word_to_id.get(word, self.special_tokens['<UNK>']))

        token_ids.append(self.special_tokens['<EOS>'])

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length - 1] + [self.special_tokens['<EOS>']]

        attention_mask = [1] * len(token_ids)

        if padding and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids.extend([self.special_tokens['<PAD>']] * pad_len)
            attention_mask.extend([0] * pad_len)

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def decode(self, token_ids, skip_special_tokens=True):
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(int(token_id))
            if word is None:
                continue
            if skip_special_tokens and word in self.special_tokens:
                continue
            words.append(word)
        return ''.join(words)

    @property
    def pad_token_id(self):
        return self.special_tokens['<PAD>']

    @property
    def eos_token_id(self):
        return self.special_tokens['<EOS>']

    @property
    def mask_token_id(self):
        return self._mask_token_id

    def __len__(self):
        return self.vocab_size_actual

    def load_state(self, word_to_id, special_tokens, mask_token_id, embedding_vocab_size):
        self.word_to_id = {str(k) if isinstance(k, bytes) else k: int(v) for k, v in word_to_id.items()}
        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self.special_tokens = {token: int(idx) for token, idx in special_tokens.items()}
        self._mask_token_id = int(mask_token_id)

        max_index = max(self.word_to_id.values(), default=0)
        self.next_id = max_index + 1
        if self.next_id == self._mask_token_id:
            self.next_id += 1

        self.vocab_size = max(self.vocab_size, len(self.word_to_id))
        self.vocab_size_actual = max(int(embedding_vocab_size), self._mask_token_id + 1, self.next_id)


class SimpleTransformer(nn.Module):
    """Simple Transformer model for LLaDA"""
    
    def __init__(self, vocab_size, max_length=512, d_model=256, nhead=8, num_layers=6):
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


def forward_process(input_ids, eps=1e-3, mask_token_id=126336):
    """Forward process for LLaDA pre-training"""
    b, l = input_ids.shape
    
    # Sample random masking ratios for each sequence
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    # Create random mask based on probability
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    
    # Apply mask token to masked positions
    mask_tokens = torch.full_like(input_ids, mask_token_id)
    noisy_batch = torch.where(masked_indices, mask_tokens, input_ids)
    
    return noisy_batch, masked_indices, p_mask


class PretrainingDataset(Dataset):
    """Dataset for LLaDA pre-training"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = extract_text_from_record(item)
                        if text:
                            self.data.append(text)
                    except Exception as exc:
                        logging.debug(f"Skipping malformed record: {exc}")
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
        encoding = self.tokenizer.encode(text, max_length=self.max_length, padding=True)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }


class LLaDAOfflineTrainer:
    """LLaDA offline training class"""
    
    def __init__(self, model, tokenizer, args, device, train_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        self.train_dataset = train_dataset

        self.clear_cuda_cache = bool(getattr(args, 'clear_cuda_cache', False)) and torch.cuda.is_available()
        self.cuda_cache_interval = max(1, int(getattr(args, 'cuda_cache_interval', 100)))
        self.eval_subset_ratio = float(max(0.0, min(1.0, getattr(args, 'eval_subset_ratio', 0.0))))

        self.metrics_dir = Path(self.args.output_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.train_loss_path = self.metrics_dir / "train_loss.jsonl"
        self.eval_loss_path = self.metrics_dir / "eval_loss.jsonl"
        self._init_metric_files()
        
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
    
    def _prepare_eval_dataloader(self, eval_dataloader):
        if eval_dataloader is not None and self.eval_subset_ratio == 0:
            return eval_dataloader

        if self.eval_subset_ratio == 0:
            return eval_dataloader

        dataset_size = len(self.train_dataset)
        if dataset_size == 0:
            logging.warning("Evaluation skipped: training dataset is empty.")
            return None

        subset_size = int(dataset_size * self.eval_subset_ratio)
        subset_size = max(1, min(dataset_size, subset_size))

        indices = random.sample(range(dataset_size), subset_size)
        subset = Subset(self.train_dataset, indices)

        logging.info(
            f"Sampling {subset_size} / {dataset_size} training examples for evaluation (ratio={self.eval_subset_ratio:.2f})."
        )

        return DataLoader(
            subset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory
        )

    def _write_metric(self, path: Path, payload: Dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("run_id", self.run_id)
        record.setdefault("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    def _init_metric_files(self) -> None:
        for metric_path, metric_type in (
            (self.train_loss_path, "train"),
            (self.eval_loss_path, "eval"),
        ):
            if not metric_path.exists():
                metric_path.touch()
            self._write_metric(metric_path, {
                "type": metric_type,
                "event": "run_start"
            })

    def _record_train_loss(self, step: int, loss: float) -> None:
        if not np.isfinite(loss):
            return
        self._write_metric(self.train_loss_path, {
            "type": "train",
            "step": int(step),
            "loss": float(loss)
        })

    def _record_eval_loss(self, step: int, loss: float) -> None:
        if not np.isfinite(loss):
            return
        self._write_metric(self.eval_loss_path, {
            "type": "eval",
            "step": int(step),
            "loss": float(loss)
        })

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])

        # Ensure optimizer tensors are on correct device
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)

        self.global_step = int(state.get('global_step', 0))
        self.best_loss = float(state.get('best_loss', float('inf')))

        logging.info(
            f"Resumed training from checkpoint at step {self.global_step}"
        )

    def compute_loss(self, batch):
        """Compute pre-training loss"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Random sequence length for 1% of batches per guidelines
        if torch.rand(1, device=input_ids.device) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,), device=input_ids.device).item()
            input_ids = input_ids[:, :random_length]
            attention_mask = attention_mask[:, :random_length]

        # Apply forward process to add noise
        noisy_batch, masked_indices, p_mask = forward_process(
            input_ids, 
            eps=1e-3, 
            mask_token_id=self.tokenizer.mask_token_id
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
    
    def evaluate(self, eval_dataloader, step=None):
        """Evaluate model on validation set"""
        if step is None:
            step = self.global_step
        if self.clear_cuda_cache:
            torch.cuda.empty_cache()
        self.model.eval()
        total_loss = 0
        num_batches = 0

        dataloader = self._prepare_eval_dataloader(eval_dataloader)

        if dataloader is None:
            logging.warning("No evaluation dataloader available; skipping evaluation.")
            return float('inf')

        with torch.no_grad():
            for batch in dataloader:
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
        if self.clear_cuda_cache:
            torch.cuda.empty_cache()
        if np.isfinite(avg_loss):
            self._record_eval_loss(step, avg_loss)
        return avg_loss
    
    def save_checkpoint(self, save_dir, is_best=False):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        args_dict = {
            'train_data_path': self.args.train_data_path,
            'eval_data_path': self.args.eval_data_path,
            'max_length': self.args.max_length,
            'vocab_size': self.args.vocab_size,
            'd_model': self.args.d_model,
            'nhead': self.args.nhead,
            'num_layers': self.args.num_layers,
            'learning_rate': self.args.learning_rate,
            'min_learning_rate': self.args.min_learning_rate,
            'weight_decay': self.args.weight_decay,
            'max_steps': self.args.max_steps,
            'batch_size': self.args.batch_size,
            'eval_batch_size': self.args.eval_batch_size,
            'max_grad_norm': self.args.max_grad_norm,
            'log_interval': self.args.log_interval,
            'eval_interval': self.args.eval_interval,
            'save_interval': self.args.save_interval,
            'resume_from': getattr(self.args, 'resume_from', None),
            'seed': self.args.seed,
            'num_workers': self.args.num_workers,
            'pin_memory': bool(getattr(self.args, 'pin_memory', False)),
            'clear_cuda_cache': bool(getattr(self.args, 'clear_cuda_cache', False)),
            'cuda_cache_interval': int(getattr(self.args, 'cuda_cache_interval', 100)),
            'config_path': getattr(self.args, 'config_path', None),
        }

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'tokenizer_vocab': self.tokenizer.word_to_id,
            'tokenizer_special': self.tokenizer.special_tokens,
            'mask_token_id': self.tokenizer.mask_token_id,
            'embedding_vocab_size': len(self.tokenizer),
            'vocab_size': len(self.tokenizer.word_to_id),
            'args': args_dict
        }
        
        checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{self.global_step}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_dataloader, eval_dataloader=None):
        """Main training loop"""
        logging.info("Starting LLaDA offline training...")
        logging.info(f"Total steps: {self.args.max_steps}")
        logging.info(f"Batch size: {self.args.batch_size}")
        logging.info(f"Eval batch size: {self.args.eval_batch_size}")
        logging.info(f"Learning rate: {self.args.learning_rate}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"Vocabulary size: {len(self.tokenizer)}")
        if self.clear_cuda_cache:
            logging.info(f"CUDA cache clearing enabled every {self.cuda_cache_interval} steps")
        logging.info(f"DataLoader pin_memory: {bool(getattr(self.args, 'pin_memory', False))}")
        if self.eval_subset_ratio > 0:
            logging.info(
                f"Evaluation subset ratio enabled: sampling {self.eval_subset_ratio:.2f} of training data each eval step"
            )
        
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
                
                if np.isfinite(loss):
                    self._record_train_loss(self.global_step, loss)
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
                if (eval_dataloader is not None or self.eval_subset_ratio > 0) and self.global_step % self.args.eval_interval == 0:
                    eval_loss = self.evaluate(eval_dataloader, step=self.global_step)
                    logging.info(f"Eval Loss: {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(self.args.output_dir, is_best=True)
                
                # Save checkpoint
                if self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(self.args.output_dir)

                if self.clear_cuda_cache and self.global_step > 0 and self.global_step % self.cuda_cache_interval == 0:
                    torch.cuda.empty_cache()
                    logging.info(f"Cleared CUDA cache at step {self.global_step}")
        
        logging.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='LLaDA Offline Training (No Downloads)')

    # Config
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to JSON configuration file')

    # Data arguments
    parser.add_argument('--train_data_path', type=str, default=None,
                       help='Path to training data')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to evaluation data')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=8000,
                       help='Vocabulary size')

    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum number of training steps')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                       help='Evaluation batch size (defaults to training batch size)')
    parser.add_argument('--eval_subset_ratio', type=float, default=0.0,
                       help='When >0, sample this ratio of the training data for evaluation each time (requires no eval dataset).')
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
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a checkpoint to resume training from')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--pin_memory', type=str_to_bool, default=None,
                       help='Whether to pin memory in data loaders (defaults to true when CUDA available)')
    parser.add_argument('--clear_cuda_cache', action='store_true',
                       help='Clear CUDA cache periodically during training')
    parser.add_argument('--cuda_cache_interval', type=int, default=100,
                       help='Steps between CUDA cache clearing when enabled')

    preliminary_args, _ = parser.parse_known_args()
    config_path = preliminary_args.config_path
    if config_path:
        config_defaults = load_config_defaults(config_path)
        config_defaults['config_path'] = config_path
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    validate_required_args(args, args.config_path)
    if args.config_path:
        print(f"Loaded training configuration from {args.config_path}")

    if args.eval_batch_size is None or args.eval_batch_size <= 0:
        args.eval_batch_size = args.batch_size

    if args.pin_memory is None:
        args.pin_memory = torch.cuda.is_available()

    args.pin_memory = bool(args.pin_memory)
    
    # Setup
    set_seed(args.seed)
    setup_logging(args.output_dir)
    
    logging.info("=== LLaDA Offline Training (No Downloads) ===")
    logging.info("Creating tokenizer and model from scratch...")
    
    # Load training data to build vocabulary
    train_texts = []
    with open(args.train_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.debug(f"Skipping malformed JSON line: {exc}")
                continue

            text = extract_text_from_record(item)
            if text:
                train_texts.append(text)

    if not train_texts:
        raise RuntimeError(
            f"No usable text records were found in {args.train_data_path}. "
            "Ensure the dataset contains either a 'text' field or question/context/answer strings."
        )
    
    # Create tokenizer and build vocabulary
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    
    resume_state = None
    if getattr(args, 'resume_from', None):
        resume_path = Path(args.resume_from).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        checkpoint = torch.load(resume_path, map_location='cpu')

        if 'tokenizer_vocab' in checkpoint:
            tokenizer.load_state(
                checkpoint['tokenizer_vocab'],
                checkpoint.get('tokenizer_special', tokenizer.special_tokens),
                checkpoint.get('mask_token_id', tokenizer.mask_token_id),
                checkpoint.get('embedding_vocab_size', len(tokenizer))
            )

        resume_state = {
            'model_state_dict': checkpoint['model_state_dict'],
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'scheduler_state_dict': checkpoint['scheduler_state_dict'],
            'global_step': checkpoint.get('global_step', 0),
            'best_loss': checkpoint.get('best_loss', float('inf')),
        }

        args.vocab_size = max(args.vocab_size, len(tokenizer.word_to_id))

        logging.info(
            f"Loaded resume checkpoint from {resume_path} at step {resume_state['global_step']}"
        )

    # Create model
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
    logging.info(f"Vocabulary size: {len(tokenizer)}")
    logging.info(f"Mask token ID (reserved): {tokenizer.mask_token_id}")
    
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

    if getattr(args, 'eval_subset_ratio', 0.0) > 0:
        if eval_dataset is not None:
            logging.warning(
                "eval_subset_ratio specified; ignoring explicit eval dataset and sampling from training data instead."
            )
        eval_dataset = None
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
    
    # Create trainer
    trainer = LLaDAOfflineTrainer(model, tokenizer, args, device, train_dataset)

    if resume_state:
        trainer.load_checkpoint_state(resume_state)
    
    # Start training
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == '__main__':
    main()