import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import logging
from collections import defaultdict
import matplotlib.pyplot as plt


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class EvaluationDataset(Dataset):
    """Dataset for evaluation"""
    
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
                        self.data.append(item['text'])
                    except:
                        continue
        elif data_path.endswith('.txt'):
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = text.split('\n\n')
                self.data.extend([chunk.strip() for chunk in chunks if chunk.strip()])
        
        logging.info(f"Loaded {len(self.data)} evaluation examples from {data_path}")
    
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
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text
        }


def load_model_and_tokenizer(model_path, device):
    """Load trained model and tokenizer"""
    # Load checkpoint (disable weights_only for compatibility)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load tokenizer
    tokenizer_path = os.path.dirname(model_path)
    if os.path.exists(os.path.join(tokenizer_path, 'best_tokenizer')):
        tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(tokenizer_path, 'best_tokenizer'))
    else:
        # Fallback to gpt2
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_simple import SimpleTransformer
    model_config = checkpoint['model_config']
    model = SimpleTransformer(
        vocab_size=model_config['vocab_size'],
        max_length=model_config['max_length'],
        d_model=model_config['d_model']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    mask_token_id = checkpoint['mask_token_id']
    
    logging.info(f"Loaded model from {model_path}")
    logging.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    logging.info(f"Vocabulary size: {len(tokenizer)}")
    logging.info(f"Mask token ID: {mask_token_id}")
    
    return model, tokenizer, mask_token_id


def forward_process(input_ids, eps=1e-3, mask_token_id=None):
    """Forward process for evaluation (same as training)"""
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


def compute_accuracy_metrics(model, dataloader, mask_token_id, device):
    """Compute various accuracy metrics"""
    model.eval()
    
    total_tokens = 0
    correct_predictions = 0
    total_masked_tokens = 0
    correct_masked_predictions = 0
    
    # For different masking ratios
    mask_ratio_results = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    # For per-token analysis
    token_accuracies = []
    loss_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing accuracy"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Apply forward process
            noisy_batch, masked_indices, p_mask = forward_process(
                input_ids, eps=1e-3, mask_token_id=mask_token_id
            )
            
            # Get model predictions
            outputs = model(input_ids=noisy_batch, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Only consider non-padding tokens
            valid_mask = attention_mask.bool()
            
            # Overall accuracy (all tokens)
            correct_all = (predictions == input_ids) & valid_mask
            total_tokens += valid_mask.sum().item()
            correct_predictions += correct_all.sum().item()
            
            # Masked token accuracy
            masked_valid = masked_indices & valid_mask
            if masked_valid.any():
                correct_masked = (predictions == input_ids) & masked_valid
                total_masked_tokens += masked_valid.sum().item()
                correct_masked_predictions += correct_masked.sum().item()
                
                # Per masking ratio
                for i in range(input_ids.shape[0]):
                    seq_mask_ratio = masked_indices[i].float().mean().item()
                    ratio_bin = int(seq_mask_ratio * 10) / 10  # Bin to 0.1 intervals
                    
                    seq_masked_valid = masked_valid[i]
                    if seq_masked_valid.any():
                        seq_correct = correct_masked[i].sum().item()
                        seq_total = seq_masked_valid.sum().item()
                        
                        mask_ratio_results[ratio_bin]['total'] += seq_total
                        mask_ratio_results[ratio_bin]['correct'] += seq_correct
            
            # Compute loss for this batch
            if masked_indices.any():
                token_loss = F.cross_entropy(
                    logits[masked_indices], 
                    input_ids[masked_indices], 
                    reduction='none'
                ) / p_mask[masked_indices]
                
                # Average loss per sequence
                loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
                loss_values.append(loss.item())
            
            # Token-level accuracy for each position
            for i in range(logits.shape[1]):  # sequence length
                if valid_mask[:, i].any():
                    pos_correct = correct_all[:, i][valid_mask[:, i]]
                    if len(pos_correct) > 0:
                        token_accuracies.append(pos_correct.float().mean().item())
    
    # Calculate metrics
    metrics = {
        'overall_accuracy': correct_predictions / total_tokens if total_tokens > 0 else 0,
        'masked_token_accuracy': correct_masked_predictions / total_masked_tokens if total_masked_tokens > 0 else 0,
        'total_tokens': total_tokens,
        'total_masked_tokens': total_masked_tokens,
        'average_loss': np.mean(loss_values) if loss_values else float('inf'),
        'loss_std': np.std(loss_values) if loss_values else 0,
        'token_position_accuracy': np.mean(token_accuracies) if token_accuracies else 0,
        'mask_ratio_accuracies': {}
    }
    
    # Process mask ratio results
    for ratio, results in mask_ratio_results.items():
        if results['total'] > 0:
            metrics['mask_ratio_accuracies'][ratio] = results['correct'] / results['total']
    
    return metrics


def evaluate_perplexity(model, dataloader, tokenizer, device):
    """Compute perplexity on the dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # For perplexity, we use the original LM objective
            # Shift inputs for next token prediction
            if input_ids.shape[1] > 1:
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                mask = attention_mask[:, 1:]
                
                # Get model predictions
                outputs = model(input_ids=inputs, attention_mask=attention_mask[:, :-1])
                logits = outputs.logits
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction='none'
                )
                
                # Only count non-padding tokens
                valid_loss = loss * mask.reshape(-1)
                total_loss += valid_loss.sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def evaluate_text_generation(model, tokenizer, mask_token_id, device, prompts=None):
    """Evaluate text generation quality"""
    if prompts is None:
        prompts = [
            "The capital of France is",
            "Machine learning is",
            "In the future, artificial intelligence will",
            "The most important thing in life is",
            "Technology has changed the way we"
        ]
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors='pt', padding=True)
            input_ids = inputs['input_ids'].to(device)
            
            # Simple generation: mask some tokens and predict
            seq_len = input_ids.shape[1]
            
            # Mask the last few tokens for generation
            num_to_generate = min(10, seq_len // 2)
            if num_to_generate > 0:
                masked_input = input_ids.clone()
                masked_input[0, -num_to_generate:] = mask_token_id
                
                # Get predictions
                outputs = model(input_ids=masked_input)
                logits = outputs.logits
                
                # Get predicted tokens
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                
                results.append({
                    'prompt': prompt,
                    'generated': predicted_text,
                    'original_length': seq_len,
                    'generated_tokens': num_to_generate
                })
    
    return results


def plot_metrics(metrics, save_path=None):
    """Plot evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Overall metrics
    metric_names = ['Overall Accuracy', 'Masked Token Accuracy']
    metric_values = [metrics['overall_accuracy'], metrics['masked_token_accuracy']]
    
    axes[0, 0].bar(metric_names, metric_values)
    axes[0, 0].set_title('Accuracy Metrics')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Mask ratio vs accuracy
    if metrics['mask_ratio_accuracies']:
        ratios = sorted(metrics['mask_ratio_accuracies'].keys())
        accuracies = [metrics['mask_ratio_accuracies'][r] for r in ratios]
        
        axes[0, 1].plot(ratios, accuracies, 'o-')
        axes[0, 1].set_title('Accuracy vs Mask Ratio')
        axes[0, 1].set_xlabel('Mask Ratio')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Token counts
    token_data = ['Total Tokens', 'Masked Tokens']
    token_counts = [metrics['total_tokens'], metrics['total_masked_tokens']]
    
    axes[1, 0].bar(token_data, token_counts)
    axes[1, 0].set_title('Token Counts')
    axes[1, 0].set_ylabel('Count')
    
    # Plot 4: Loss information
    loss_data = ['Average Loss']
    loss_values = [metrics['average_loss']]
    
    axes[1, 1].bar(loss_data, loss_values)
    axes[1, 1].set_title('Loss Metrics')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Metrics plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLaDA Model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--eval_data_path', type=str, required=True,
                       help='Path to evaluation data')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Evaluation batch size')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--compute_perplexity', action='store_true',
                       help='Compute perplexity metric')
    parser.add_argument('--generate_text', action='store_true',
                       help='Evaluate text generation')
    parser.add_argument('--plot_metrics', action='store_true',
                       help='Generate metric plots')
    
    args = parser.parse_args()
    
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer, mask_token_id = load_model_and_tokenizer(args.model_path, device)
    
    # Load evaluation dataset
    eval_dataset = EvaluationDataset(args.eval_data_path, tokenizer, args.max_length)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Compute accuracy metrics
    logging.info("Computing accuracy metrics...")
    metrics = compute_accuracy_metrics(model, eval_dataloader, mask_token_id, device)
    
    # Compute perplexity if requested
    if args.compute_perplexity:
        logging.info("Computing perplexity...")
        perplexity, avg_loss = evaluate_perplexity(model, eval_dataloader, tokenizer, device)
        metrics['perplexity'] = perplexity
        metrics['lm_loss'] = avg_loss
    
    # Text generation evaluation
    if args.generate_text:
        logging.info("Evaluating text generation...")
        generation_results = evaluate_text_generation(model, tokenizer, mask_token_id, device)
        metrics['generation_examples'] = generation_results
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Masked Token Accuracy: {metrics['masked_token_accuracy']:.4f}")
    print(f"Average Loss: {metrics['average_loss']:.4f}")
    print(f"Total Tokens Evaluated: {metrics['total_tokens']:,}")
    print(f"Total Masked Tokens: {metrics['total_masked_tokens']:,}")
    
    if 'perplexity' in metrics:
        print(f"Perplexity: {metrics['perplexity']:.2f}")
    
    print(f"\nAccuracy by Mask Ratio:")
    for ratio in sorted(metrics['mask_ratio_accuracies'].keys()):
        acc = metrics['mask_ratio_accuracies'][ratio]
        print(f"  {ratio:.1f}: {acc:.4f}")
    
    if 'generation_examples' in metrics:
        print(f"\nText Generation Examples:")
        for i, example in enumerate(metrics['generation_examples'][:3]):
            print(f"  {i+1}. Prompt: {example['prompt']}")
            print(f"     Generated: {example['generated']}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        # Convert numpy types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Results saved to {results_path}")
    
    # Plot metrics if requested
    if args.plot_metrics:
        plot_path = os.path.join(args.output_dir, 'metrics_plot.png')
        plot_metrics(metrics, plot_path)
    
    print("="*50)


if __name__ == '__main__':
    main()