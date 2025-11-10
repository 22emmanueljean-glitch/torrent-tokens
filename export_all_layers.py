import torch
import numpy as np
from transformers import GPT2Model

model_name = "distilgpt2"
print(f"Loading {model_name}...")
model = GPT2Model.from_pretrained(model_name)

# Export all 6 layers (DistilGPT-2 has 6, not 12!)
for layer_idx in range(6):
    print(f"\nExporting layer {layer_idx}...")
    layer = model.h[layer_idx]  # h = hidden layers
    
    # Attention weights
    qkv_weight = layer.attn.c_attn.weight.data.T.numpy().astype(np.float32)
    qkv_bias = layer.attn.c_attn.bias.data.numpy().astype(np.float32)
    o_weight = layer.attn.c_proj.weight.data.T.numpy().astype(np.float32)
    o_bias = layer.attn.c_proj.bias.data.numpy().astype(np.float32)
    
    # FFN weights
    ff1_weight = layer.mlp.c_fc.weight.data.T.numpy().astype(np.float32)
    ff1_bias = layer.mlp.c_fc.bias.data.numpy().astype(np.float32)
    ff2_weight = layer.mlp.c_proj.weight.data.T.numpy().astype(np.float32)
    ff2_bias = layer.mlp.c_proj.bias.data.numpy().astype(np.float32)
    
    # LayerNorm weights
    ln1_g = layer.ln_1.weight.data.numpy().astype(np.float32)
    ln1_b = layer.ln_1.bias.data.numpy().astype(np.float32)
    ln2_g = layer.ln_2.weight.data.numpy().astype(np.float32)
    ln2_b = layer.ln_2.bias.data.numpy().astype(np.float32)
    
    # Save
    qkv_weight.tofile(f"assets/weights/layer{layer_idx}_qkv.bin")
    qkv_bias.tofile(f"assets/weights/layer{layer_idx}_qkv_b.bin")
    o_weight.tofile(f"assets/weights/layer{layer_idx}_o.bin")
    o_bias.tofile(f"assets/weights/layer{layer_idx}_o_b.bin")
    ff1_weight.tofile(f"assets/weights/layer{layer_idx}_ff1.bin")
    ff1_bias.tofile(f"assets/weights/layer{layer_idx}_ff1_b.bin")
    ff2_weight.tofile(f"assets/weights/layer{layer_idx}_ff2.bin")
    ff2_bias.tofile(f"assets/weights/layer{layer_idx}_ff2_b.bin")
    ln1_g.tofile(f"assets/weights/layer{layer_idx}_ln1_g.bin")
    ln1_b.tofile(f"assets/weights/layer{layer_idx}_ln1_b.bin")
    ln2_g.tofile(f"assets/weights/layer{layer_idx}_ln2_g.bin")
    ln2_b.tofile(f"assets/weights/layer{layer_idx}_ln2_b.bin")
    
    print(f"✅ Layer {layer_idx} exported")

print("\n🎉 All 6 layers exported!")