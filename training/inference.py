import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import argparse
import logging


def add_gumbel_noise(logits, temperature):
    """
    Add Gumbel noise for sampling.
    Uses float64 for better precision as mentioned in the guidelines.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_num, steps):
    """
    Calculate the number of tokens to transfer at each step.
    """
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(steps, device=mask_num.device, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    
    return num_transfer_tokens


class LLaDAInference:
    """LLaDA inference class with multiple sampling strategies"""
    
    def __init__(self, model, tokenizer, mask_id=126336):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.device = model.device
    
    def predict_tokens(self, x, prompt_mask=None, cfg_scale=0.):
        """
        Get model predictions with optional classifier-free guidance.
        """
        if cfg_scale > 0. and prompt_mask is not None:
            # Create unconditional input by masking the prompt
            un_x = x.clone()
            un_x[prompt_mask] = self.mask_id
            
            # Concatenate conditional and unconditional inputs
            x_combined = torch.cat([x, un_x], dim=0)
            logits = self.model(x_combined).logits
            
            # Split and apply guidance
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
        else:
            logits = self.model(x).logits
        
        return logits
    
    def apply_remasking(self, x0, x, mask_index, num_transfer, remasking='low_confidence', logits=None):
        """
        Apply remasking strategy to determine which tokens to keep.
        
        Args:
            x0: Predicted tokens
            x: Current sequence
            mask_index: Current mask positions
            num_transfer: Number of tokens to transfer
            remasking: Strategy ('low_confidence' or 'random')
            logits: Model logits (required for low_confidence)
        """
        if remasking == 'low_confidence':
            if logits is None:
                raise ValueError("Logits required for low_confidence remasking")
            
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand_like(x0, dtype=torch.float, device=x0.device)
        else:
            raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
        
        # Only consider masked positions for confidence calculation
        confidence = torch.where(mask_index, x0_p, -np.inf)
        
        # Select top-k tokens to keep based on confidence
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        if num_transfer > 0:
            _, select_index = torch.topk(confidence, k=min(num_transfer, mask_index.sum().item()))
            transfer_index[select_index] = True
        
        # Update sequence
        new_x = x.clone()
        new_x[transfer_index] = x0[transfer_index]
        
        return new_x
    
    @torch.no_grad()
    def fixed_length_sampling(self, prompt, gen_length=128, steps=None, temperature=0., 
                             cfg_scale=0., remasking='low_confidence'):
        """
        Fixed-length sampling strategy.
        Generate exactly gen_length tokens in a fixed number of steps.
        """
        if steps is None:
            steps = gen_length
            
        prompt_len = prompt.shape[1]
        total_len = prompt_len + gen_length
        
        # Initialize sequence with prompt + masked tokens
        x = torch.full((1, total_len), self.mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt.clone()
        
        prompt_mask = torch.zeros(total_len, dtype=torch.bool, device=self.device)
        prompt_mask[:prompt_len] = True
        
        # Calculate number of tokens to unmask at each step
        mask_num = gen_length
        num_transfer_tokens = get_num_transfer_tokens(torch.tensor(mask_num), steps)
        
        for i in range(steps):
            mask_index = (x == self.mask_id)
            
            # Get predictions
            logits = self.predict_tokens(x, prompt_mask, cfg_scale)
            
            # Sample tokens
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Apply remasking
            x = self.apply_remasking(
                x0.squeeze(0), x.squeeze(0), mask_index.squeeze(0), 
                num_transfer_tokens[i].item(), remasking, logits.squeeze(0)
            ).unsqueeze(0)
        
        return x
    
    @torch.no_grad()
    def semi_autoregressive_origin_sampling(self, prompt, max_length=512, steps_per_length=4, 
                                          temperature=0., cfg_scale=0., remasking='low_confidence'):
        """
        Semi-autoregressive sampling starting from origin.
        Gradually increase the sequence length.
        """
        prompt_len = prompt.shape[1]
        x = prompt.clone()
        
        current_length = prompt_len
        while current_length < max_length:
            # Determine next length increment
            length_increment = min(32, max_length - current_length)
            new_length = current_length + length_increment
            
            # Extend sequence with mask tokens
            mask_tokens = torch.full((1, length_increment), self.mask_id, 
                                   dtype=torch.long, device=self.device)
            x = torch.cat([x, mask_tokens], dim=1)
            
            prompt_mask = torch.zeros(new_length, dtype=torch.bool, device=self.device)
            prompt_mask[:current_length] = True
            
            # Generate new tokens
            num_transfer_tokens = get_num_transfer_tokens(torch.tensor(length_increment), steps_per_length)
            
            for i in range(steps_per_length):
                mask_index = (x == self.mask_id)
                
                if not mask_index.any():
                    break
                
                # Get predictions
                logits = self.predict_tokens(x, prompt_mask, cfg_scale)
                
                # Sample tokens
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                # Apply remasking
                x = self.apply_remasking(
                    x0.squeeze(0), x.squeeze(0), mask_index.squeeze(0),
                    num_transfer_tokens[i].item(), remasking, logits.squeeze(0)
                ).unsqueeze(0)
            
            current_length = new_length
            
            # Check for early stopping (if EOS token is generated)
            if self.tokenizer.eos_token_id in x[0, prompt_len:]:
                break
        
        return x
    
    @torch.no_grad()
    def semi_autoregressive_padding_sampling(self, prompt, gen_length=128, block_length=32, 
                                           steps=None, temperature=0., cfg_scale=0., 
                                           remasking='low_confidence'):
        """
        Semi-autoregressive sampling with padding.
        Generate in blocks of fixed size.
        """
        if steps is None:
            steps = gen_length
            
        prompt_len = prompt.shape[1]
        total_len = prompt_len + gen_length
        
        # Initialize sequence with prompt + masked tokens
        x = torch.full((1, total_len), self.mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt.clone()
        
        prompt_mask = torch.zeros(total_len, dtype=torch.bool, device=self.device)
        prompt_mask[:prompt_len] = True
        
        # Process in blocks
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
        steps_per_block = steps // num_blocks
        
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            
            # Create block mask
            block_mask_positions = torch.arange(block_start, block_end, device=self.device)
            
            # Calculate tokens to transfer per step for this block
            num_transfer_tokens = get_num_transfer_tokens(torch.tensor(block_length), steps_per_block)
            
            for step in range(steps_per_block):
                # Find current mask positions in the block
                block_mask_index = (x[:, block_start:block_end] == self.mask_id)
                
                if not block_mask_index.any():
                    break
                
                # Get predictions for entire sequence
                logits = self.predict_tokens(x, prompt_mask, cfg_scale)
                
                # Sample tokens
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                # Apply remasking only to current block
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == 'random':
                    x0_p = torch.rand_like(x0, dtype=torch.float)
                else:
                    raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
                
                # Set confidence to -inf outside current block
                confidence = x0_p.clone()
                confidence[:, :block_start] = -np.inf
                confidence[:, block_end:] = -np.inf
                
                # Only consider masked positions
                full_mask_index = (x == self.mask_id)
                confidence = torch.where(full_mask_index, confidence, -np.inf)
                
                # Select tokens to transfer
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                if num_transfer_tokens[step] > 0:
                    for batch_idx in range(x.shape[0]):
                        valid_positions = confidence[batch_idx] > -np.inf
                        if valid_positions.any():
                            num_valid = valid_positions.sum().item()
                            k = min(num_transfer_tokens[step].item(), num_valid)
                            if k > 0:
                                _, select_idx = torch.topk(confidence[batch_idx], k=k)
                                transfer_index[batch_idx, select_idx] = True
                
                # Update sequence
                x[transfer_index] = x0[transfer_index]
        
        return x
    
    def generate(self, prompt, method='fixed_length', **kwargs):
        """
        Main generation interface.
        
        Args:
            prompt: Input prompt tensor of shape (1, prompt_length)
            method: Sampling method ('fixed_length', 'semi_autoregressive_origin', 'semi_autoregressive_padding')
            **kwargs: Additional arguments for the specific method
        """
        if method == 'fixed_length':
            return self.fixed_length_sampling(prompt, **kwargs)
        elif method == 'semi_autoregressive_origin':
            return self.semi_autoregressive_origin_sampling(prompt, **kwargs)
        elif method == 'semi_autoregressive_padding':
            return self.semi_autoregressive_padding_sampling(prompt, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


def main():
    parser = argparse.ArgumentParser(description='LLaDA Inference')
    
    # Model arguments
    parser.add_argument('--model_name_or_path', type=str, required=True,
                       help='Path to model')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Input prompt')
    
    # Generation arguments
    parser.add_argument('--method', type=str, default='fixed_length',
                       choices=['fixed_length', 'semi_autoregressive_origin', 'semi_autoregressive_padding'],
                       help='Sampling method')
    parser.add_argument('--gen_length', type=int, default=128,
                       help='Generation length')
    parser.add_argument('--block_length', type=int, default=32,
                       help='Block length for semi_autoregressive_padding')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of sampling steps')
    parser.add_argument('--temperature', type=float, default=0.,
                       help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.,
                       help='Classifier-free guidance scale')
    parser.add_argument('--remasking', type=str, default='low_confidence',
                       choices=['low_confidence', 'random'],
                       help='Remasking strategy')
    parser.add_argument('--is_instruct', action='store_true',
                       help='Use instruct model format')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Prepare prompt
    if args.is_instruct:
        messages = [{"role": "user", "content": args.prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    else:
        formatted_prompt = args.prompt
    
    # Tokenize prompt
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # Create inference object
    inference = LLaDAInference(model, tokenizer)
    
    # Generate
    print(f"Using {args.method} sampling...")
    print(f"Prompt: {formatted_prompt}")
    print("Generating...")
    
    if args.method == 'fixed_length':
        output = inference.generate(
            input_ids,
            method=args.method,
            gen_length=args.gen_length,
            steps=args.steps or args.gen_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking
        )
    elif args.method == 'semi_autoregressive_origin':
        output = inference.generate(
            input_ids,
            method=args.method,
            max_length=input_ids.shape[1] + args.gen_length,
            steps_per_length=4,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking
        )
    elif args.method == 'semi_autoregressive_padding':
        output = inference.generate(
            input_ids,
            method=args.method,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps or args.gen_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking
        )
    
    # Decode output
    generated_tokens = output[:, input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print(f"Generated text: {generated_text}")


if __name__ == '__main__':
    main()