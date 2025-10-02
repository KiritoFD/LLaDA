import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
from tqdm import tqdm
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
import logging

# Add training directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model from train_simple
try:
    from train_simple import SimpleTransformer, forward_process
except ImportError:
    print("Error: Cannot import SimpleTransformer from train_simple.py")
    print("Please make sure train_simple.py is in the same directory")
    sys.exit(1)


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
    try:
        # Load checkpoint (disable weights_only for compatibility)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Load tokenizer
        tokenizer_dir = os.path.dirname(model_path)
        best_tokenizer_path = os.path.join(tokenizer_dir, 'best_tokenizer')
        
        if os.path.exists(best_tokenizer_path):
            tokenizer = GPT2Tokenizer.from_pretrained(best_tokenizer_path)
        else:
            # Fallback to gpt2
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Handle different checkpoint formats
        if 'model_config' in checkpoint:
            # New format
            model_config = checkpoint['model_config']
            vocab_size = model_config['vocab_size']
            max_length = model_config['max_length']
            d_model = model_config['d_model']
        else:
            # Old format - infer from checkpoint
            logging.warning("Old checkpoint format detected, inferring model configuration")
            vocab_size = len(tokenizer)
            max_length = 1024  # Default
            d_model = 512      # Default
            
            # Try to infer from model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                if 'token_embedding.weight' in state_dict:
                    vocab_size = state_dict['token_embedding.weight'].shape[0]
                    d_model = state_dict['token_embedding.weight'].shape[1]
                if 'position_embedding.weight' in state_dict:
                    max_length = state_dict['position_embedding.weight'].shape[0]
        
        # Create model
        model = SimpleTransformer(
            vocab_size=vocab_size,
            max_length=max_length,
            d_model=d_model
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Get mask token ID
        if 'mask_token_id' in checkpoint:
            mask_token_id = checkpoint['mask_token_id']
        else:
            # Fallback: add mask token if needed
            if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
                mask_token_id = tokenizer.mask_token_id
            else:
                # Add mask token
                special_tokens_dict = {'mask_token': '[MASK]'}
                tokenizer.add_special_tokens(special_tokens_dict)
                mask_token_id = tokenizer.mask_token_id
                logging.warning(f"Added mask token, ID: {mask_token_id}")
        
        logging.info(f"Loaded model from {model_path}")
        logging.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        logging.info(f"Vocabulary size: {vocab_size}")
        logging.info(f"Model dimension: {d_model}")
        logging.info(f"Max length: {max_length}")
        logging.info(f"Mask token ID: {mask_token_id}")
        
        return model, tokenizer, mask_token_id
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def compute_accuracy_metrics(model, dataloader, mask_token_id, device):
    """Compute accuracy metrics"""
    model.eval()
    
    total_tokens = 0
    correct_predictions = 0
    total_masked_tokens = 0
    correct_masked_predictions = 0
    loss_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing accuracy"):
            try:
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
            
            except Exception as e:
                logging.warning(f"Error in batch processing: {e}")
                continue
    
    # Calculate metrics
    metrics = {
        'overall_accuracy': correct_predictions / total_tokens if total_tokens > 0 else 0,
        'masked_token_accuracy': correct_masked_predictions / total_masked_tokens if total_masked_tokens > 0 else 0,
        'total_tokens': total_tokens,
        'total_masked_tokens': total_masked_tokens,
        'average_loss': np.mean(loss_values) if loss_values else float('inf'),
        'loss_std': np.std(loss_values) if loss_values else 0,
    }
    
    return metrics


def evaluate_simple_generation(model, tokenizer, mask_token_id, device):
    """Simple generation test"""
    prompts = [
        "The capital of France is",
        "Machine learning is a field of",
        "In the future, technology will",
        "The most important thing is",
    ]
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                # Tokenize prompt
                inputs = tokenizer(prompt, return_tensors='pt', padding=True)
                input_ids = inputs['input_ids'].to(device)
                
                # Simple completion: extend the sequence
                seq_len = input_ids.shape[1]
                max_new_tokens = 10
                
                # Create extended sequence with mask tokens
                extended_ids = torch.cat([
                    input_ids,
                    torch.full((1, max_new_tokens), mask_token_id, device=device)
                ], dim=1)
                
                # Get predictions for the extended sequence
                outputs = model(input_ids=extended_ids)
                logits = outputs.logits
                
                # Get predicted tokens for the new positions
                predicted_ids = torch.argmax(logits[0, seq_len:], dim=-1)
                
                # Decode the completion
                completion = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                full_text = prompt + " " + completion
                
                results.append({
                    'prompt': prompt,
                    'completion': completion,
                    'full_text': full_text
                })
                
            except Exception as e:
                logging.warning(f"Error in generation for prompt '{prompt}': {e}")
                results.append({
                    'prompt': prompt,
                    'completion': "[Error]",
                    'full_text': prompt + " [Error]"
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Simple LLaDA Model Evaluation')
    
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
    parser.add_argument('--test_generation', action='store_true',
                       help='Test simple text generation')
    
    args = parser.parse_args()
    
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
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
        
        # Test generation if requested
        if args.test_generation:
            logging.info("Testing text generation...")
            generation_results = evaluate_simple_generation(model, tokenizer, mask_token_id, device)
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
        
        if 'generation_examples' in metrics:
            print(f"\nText Generation Examples:")
            for i, example in enumerate(metrics['generation_examples']):
                print(f"  {i+1}. {example['full_text']}")
        
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
        print("="*50)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())