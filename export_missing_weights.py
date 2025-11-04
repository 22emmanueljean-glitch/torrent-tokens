#!/usr/bin/env python3
"""
export_missing_weights.py - Export only the missing bias and layernorm weights
"""
import torch
from transformers import GPT2LMHeadModel
import numpy as np
import os

print("🔽 Downloading DistilGPT-2 from HuggingFace...")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
print("✅ Model loaded")

# Get first layer
layer0 = model.transformer.h[0]

print("\n📦 Exporting missing weights...")

# Create output directory if needed
os.makedirs("./assets/weights", exist_ok=True)

# WPE (positional embeddings) - optional but helps
wpe = model.transformer.wpe.weight.detach().numpy().astype(np.float32)
wpe.tofile("./assets/weights/wpe.bin")
print(f"✅ wpe.bin - {wpe.shape} ({wpe.nbytes:,} bytes)")

# Layer norm 1
ln1_g = layer0.ln_1.weight.detach().numpy().astype(np.float32)
ln1_b = layer0.ln_1.bias.detach().numpy().astype(np.float32)
ln1_g.tofile("./assets/weights/ln1_g.bin")
ln1_b.tofile("./assets/weights/ln1_b.bin")
print(f"✅ ln1_g.bin - {ln1_g.shape} ({ln1_g.nbytes:,} bytes)")
print(f"✅ ln1_b.bin - {ln1_b.shape} ({ln1_b.nbytes:,} bytes)")

# Attention biases
qkv_bias = layer0.attn.c_attn.bias.detach().numpy().astype(np.float32)
o_bias = layer0.attn.c_proj.bias.detach().numpy().astype(np.float32)
qkv_bias.tofile("./assets/weights/qkv_b.bin")
o_bias.tofile("./assets/weights/o_b.bin")
print(f"✅ qkv_b.bin - {qkv_bias.shape} ({qkv_bias.nbytes:,} bytes)")
print(f"✅ o_b.bin - {o_bias.shape} ({o_bias.nbytes:,} bytes)")

# Layer norm 2
ln2_g = layer0.ln_2.weight.detach().numpy().astype(np.float32)
ln2_b = layer0.ln_2.bias.detach().numpy().astype(np.float32)
ln2_g.tofile("./assets/weights/ln2_g.bin")
ln2_b.tofile("./assets/weights/ln2_b.bin")
print(f"✅ ln2_g.bin - {ln2_g.shape} ({ln2_g.nbytes:,} bytes)")
print(f"✅ ln2_b.bin - {ln2_b.shape} ({ln2_b.nbytes:,} bytes)")

# MLP biases
ff1_bias = layer0.mlp.c_fc.bias.detach().numpy().astype(np.float32)
ff2_bias = layer0.mlp.c_proj.bias.detach().numpy().astype(np.float32)
ff1_bias.tofile("./assets/weights/ff1_b.bin")
ff2_bias.tofile("./assets/weights/ff2_b.bin")
print(f"✅ ff1_b.bin - {ff1_bias.shape} ({ff1_bias.nbytes:,} bytes)")
print(f"✅ ff2_b.bin - {ff2_bias.shape} ({ff2_bias.nbytes:,} bytes)")

print("\n✅ All missing weights exported!")
print("\nRun 'python3 verify_weights.py' to confirm everything is OK.")