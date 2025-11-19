import torch
from transformers import GPT2LMHeadModel
import numpy as np
import json
import os

print("🔄 Loading full GPT-2 (124M params, 12 layers)...")
model = GPT2LMHeadModel.from_pretrained("gpt2")  # NOT distilgpt2!
model.eval()

output_dir = "./assets/weights"
os.makedirs(output_dir, exist_ok=True)

print("📦 Exporting embeddings...")
# WTE (50257 vocab × 768 dim)
wte = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float32)
wte.tofile(f"{output_dir}/wte.bin")
print(f"✅ WTE: {wte.shape} -> {wte.nbytes/1024/1024:.1f}MB")

# WPE (1024 positions × 768 dim)
wpe = model.transformer.wpe.weight.detach().cpu().numpy().astype(np.float32)
wpe.tofile(f"{output_dir}/wpe.bin")
print(f"✅ WPE: {wpe.shape} -> {wpe.nbytes/1024/1024:.1f}MB")

# Final layer norm
ln_f_g = model.transformer.ln_f.weight.detach().cpu().numpy().astype(np.float32)
ln_f_b = model.transformer.ln_f.bias.detach().cpu().numpy().astype(np.float32)
ln_f_g.tofile(f"{output_dir}/ln_f_g.bin")
ln_f_b.tofile(f"{output_dir}/ln_f_b.bin")
print(f"✅ Final LayerNorm")

# Export all 12 layers
for i in range(12):
    print(f"\n📥 Exporting layer {i}/12...")
    layer = model.transformer.h[i]
    
    qkv_weight = layer.attn.c_attn.weight.detach().cpu().numpy().astype(np.float32).T  # ADD .T
    qkv_bias = layer.attn.c_attn.bias.detach().cpu().numpy().astype(np.float32)
    
    o_weight = layer.attn.c_proj.weight.detach().cpu().numpy().astype(np.float32).T  # ADD .T
    o_bias = layer.attn.c_proj.bias.detach().cpu().numpy().astype(np.float32)
    
    ff1_weight = layer.mlp.c_fc.weight.detach().cpu().numpy().astype(np.float32).T  # ADD .T
    ff1_bias = layer.mlp.c_fc.bias.detach().cpu().numpy().astype(np.float32)
    ff2_weight = layer.mlp.c_proj.weight.detach().cpu().numpy().astype(np.float32).T  # ADD .T
    ff2_bias = layer.mlp.c_proj.bias.detach().cpu().numpy().astype(np.float32)
    
    # Layer norms
    ln1_g = layer.ln_1.weight.detach().cpu().numpy().astype(np.float32)
    ln1_b = layer.ln_1.bias.detach().cpu().numpy().astype(np.float32)
    ln2_g = layer.ln_2.weight.detach().cpu().numpy().astype(np.float32)
    ln2_b = layer.ln_2.bias.detach().cpu().numpy().astype(np.float32)
    
    # Save to files
    qkv_weight.tofile(f"{output_dir}/layer{i}_qkv.bin")
    qkv_bias.tofile(f"{output_dir}/layer{i}_qkv_b.bin")
    o_weight.tofile(f"{output_dir}/layer{i}_o.bin")
    o_bias.tofile(f"{output_dir}/layer{i}_o_b.bin")
    ff1_weight.tofile(f"{output_dir}/layer{i}_ff1.bin")
    ff1_bias.tofile(f"{output_dir}/layer{i}_ff1_b.bin")
    ff2_weight.tofile(f"{output_dir}/layer{i}_ff2.bin")
    ff2_bias.tofile(f"{output_dir}/layer{i}_ff2_b.bin")
    ln1_g.tofile(f"{output_dir}/layer{i}_ln1_g.bin")
    ln1_b.tofile(f"{output_dir}/layer{i}_ln1_b.bin")
    ln2_g.tofile(f"{output_dir}/layer{i}_ln2_g.bin")
    ln2_b.tofile(f"{output_dir}/layer{i}_ln2_b.bin")
    
    print(f"  ✅ Layer {i} exported")

print("\n🎉 ALL 12 LAYERS EXPORTED!")
print(f"📁 Files saved to {output_dir}/")
