# LLaDA Training Implementation

This directory contains a complete implementation of LLaDA (Large Language Diffusion with mAsking) training code based on the guidelines provided in `GUIDELINES.md`.

## Overview

LLaDA is a diffusion-based language model that uses masked token prediction instead of autoregressive generation. The implementation includes:

1. **Pre-training**: Training the mask predictor from scratch or from an existing model
2. **Supervised Fine-tuning (SFT)**: Fine-tuning on instruction-following data
3. **Inference**: Multiple sampling strategies for text generation

## Files Structure

```
training/
├── pretraining.py          # Pre-training implementation
├── sft_training.py         # SFT training implementation  
├── inference.py            # Inference with multiple sampling methods
├── train_pipeline.py       # Complete training pipeline
├── config.yaml            # Configuration file
├── train.sh               # Bash training script
└── README.md              # This file
```

## Key Features

### Pre-training (`pretraining.py`)
- Implements the forward process with random masking ratios
- Handles random length sequences (1% of training data)
- Cross-entropy loss normalized by masking probability
- Supports gradient checkpointing and mixed precision training

### SFT Training (`sft_training.py`)
- Preserves prompt tokens without adding noise
- Proper calculation of answer lengths for loss normalization
- Support for various conversation formats
- Compatible with chat templates

### Inference (`inference.py`)
- **Fixed-length sampling**: Generate exactly N tokens in fixed steps
- **Semi-autoregressive-origin**: Gradually increase sequence length
- **Semi-autoregressive-padding**: Generate in blocks with padding
- Both low-confidence and random remasking strategies
- Classifier-free guidance support

## Quick Start

### 1. Environment Setup

```bash
pip install torch transformers datasets pyyaml tqdm numpy
```

### 2. Data Preparation

#### Pre-training Data Format
```jsonl
{"text": "This is a sample text for pre-training."}
{"text": "Another piece of text for language modeling."}
```

#### SFT Data Format
```jsonl
{
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."}
  ]
}
```

### 3. Using the Training Pipeline

#### Option A: Use the Pipeline Script (Recommended)
```bash
# Configure training in config.yaml, then run:
python training/train_pipeline.py --config training/config.yaml --stage all

# Or run individual stages:
python training/train_pipeline.py --stage pretraining
python training/train_pipeline.py --stage sft
python training/train_pipeline.py --stage inference --prompt "Hello, how are you?"
```

#### Option B: Run Individual Scripts
```bash
# Pre-training
python training/pretraining.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --train_data_path "./data/train.jsonl" \
    --output_dir "./checkpoints/pretraining" \
    --max_steps 100000 \
    --batch_size 4 \
    --learning_rate 4e-4

# SFT Training
python training/sft_training.py \
    --model_name_or_path "./checkpoints/pretraining/best_model.pt" \
    --train_data_path "./data/sft_train.jsonl" \
    --output_dir "./checkpoints/sft" \
    --max_steps 10000 \
    --batch_size 4 \
    --learning_rate 2e-5

# Inference
python training/inference.py \
    --model_name_or_path "./checkpoints/sft/best_sft_model.pt" \
    --prompt "What is the capital of France?" \
    --method fixed_length \
    --gen_length 128 \
    --steps 128 \
    --remasking low_confidence \
    --is_instruct
```

## Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  name_or_path: "GSAI-ML/LLaDA-8B-Base"
  max_length: 4096
  
pretraining:
  max_steps: 100000
  batch_size: 4
  learning_rate: 4e-4
  
sft:
  max_steps: 10000
  learning_rate: 2e-5
  
inference:
  default_method: "fixed_length"
  gen_length: 128
  remasking: "low_confidence"
```

## Sampling Methods

### 1. Fixed-length Sampling
- Generates exactly `gen_length` tokens
- Uses a fixed number of denoising steps
- Best for consistent output lengths

```python
python training/inference.py \
    --method fixed_length \
    --gen_length 128 \
    --steps 128
```

### 2. Semi-autoregressive Origin
- Gradually increases sequence length
- Starts from the prompt and extends
- Good for variable-length generation

```python
python training/inference.py \
    --method semi_autoregressive_origin \
    --gen_length 128
```

### 3. Semi-autoregressive Padding
- Generates in fixed-size blocks
- Uses padding for efficiency
- Balances speed and quality

```python
python training/inference.py \
    --method semi_autoregressive_padding \
    --gen_length 128 \
    --block_length 32
```

## Remasking Strategies

### Low-confidence Remasking
- Keeps tokens with high prediction confidence
- Generally produces better quality
- Recommended for most use cases

### Random Remasking
- Randomly selects tokens to keep
- More diverse but potentially lower quality
- Useful for avoiding repetition

## Hardware Requirements

### Minimum Requirements
- GPU: RTX 3090 (24GB VRAM)
- RAM: 64GB
- CPU: 16 cores

### Recommended Requirements
- GPU: A100 80GB
- RAM: 256GB
- CPU: 32+ cores

## Implementation Details

### Forward Process
```python
def forward_process(input_ids, eps=1e-3, mask_token_id=126336):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)
    
    return noisy_batch, masked_indices, p_mask
```

### Loss Calculation
```python
# Pre-training loss
token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

# SFT loss (don't mask prompt)
noisy_batch[prompt_mask] = input_ids[prompt_mask]
token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
```

## Tips and Best Practices

1. **Memory Management**: Use gradient checkpointing and mixed precision
2. **Batch Size**: Start small and increase based on available GPU memory
3. **Learning Rate**: Use lower learning rates for SFT than pre-training
4. **Evaluation**: Monitor both training and validation loss
5. **Checkpointing**: Save checkpoints frequently, especially for long training runs
6. **Data Quality**: Ensure high-quality, diverse training data

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **NaN Loss**: Reduce learning rate or check data quality
3. **Slow Training**: Increase batch size or use multiple GPUs
4. **Poor Generation Quality**: Adjust remasking strategy or increase sampling steps

### Debug Mode

Run with smaller settings for testing:
```bash
python training/train_pipeline.py \
    --config training/config.yaml \
    --stage all \
    --dry_run  # Print commands without executing
```

## References

- [Original LLaDA Paper](https://arxiv.org/abs/2502.09992)
- [GUIDELINES.md](../GUIDELINES.md) - Detailed implementation guidelines
- [Hugging Face Models](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)

## License

This implementation follows the same license as the original LLaDA project.