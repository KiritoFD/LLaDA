import torch
import json

def inspect_checkpoint(checkpoint_path):
    """详细检查checkpoint内容"""
    print(f"=== Inspecting {checkpoint_path} ===")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}: {type(checkpoint[key])}")
        
        print("\nDetailed content:")
        for key, value in checkpoint.items():
            if key == 'model_state_dict':
                print(f"  {key}: StateDict with {len(value)} entries")
                print("    First few keys:")
                for i, state_key in enumerate(list(value.keys())[:5]):
                    print(f"      {state_key}: {value[state_key].shape}")
                if len(value) > 5:
                    print(f"      ... and {len(value) - 5} more")
            elif isinstance(value, dict):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # 尝试从state_dict推断模型配置
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\nInferred model configuration:")
            
            if 'token_embedding.weight' in state_dict:
                vocab_size = state_dict['token_embedding.weight'].shape[0]
                d_model = state_dict['token_embedding.weight'].shape[1]
                print(f"  vocab_size: {vocab_size}")
                print(f"  d_model: {d_model}")
            
            if 'position_embedding.weight' in state_dict:
                max_length = state_dict['position_embedding.weight'].shape[0]
                print(f"  max_length: {max_length}")
            
            # 计算transformer层数
            transformer_layers = 0
            for key in state_dict.keys():
                if 'transformer.layers.' in key:
                    layer_num = int(key.split('.')[2])
                    transformer_layers = max(transformer_layers, layer_num + 1)
            if transformer_layers > 0:
                print(f"  num_layers: {transformer_layers}")
            
            # 查找attention头数
            for key in state_dict.keys():
                if 'self_attn.in_proj_weight' in key:
                    in_proj_weight = state_dict[key]
                    if in_proj_weight.shape[0] % d_model == 0:
                        nhead = in_proj_weight.shape[0] // (3 * d_model)  # Q, K, V
                        print(f"  nhead: {nhead}")
                    break
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "outputs_simple/best_model.pt"
    
    inspect_checkpoint(checkpoint_path)