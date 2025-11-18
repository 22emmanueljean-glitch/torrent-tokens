import torch
import numpy as np
from transformers import GPT2Model

model = GPT2Model.from_pretrained("distilgpt2")

# Export final layer norm
ln_f_g = model.ln_f.weight.data.numpy().astype(np.float32)
ln_f_b = model.ln_f.bias.data.numpy().astype(np.float32)

ln_f_g.tofile("assets/weights/ln_f_g.bin")
ln_f_b.tofile("assets/weights/ln_f_b.bin")

print("✅ Final layer norm exported!")
print(f"   ln_f_g: {ln_f_g.shape}")
print(f"   ln_f_b: {ln_f_b.shape}")