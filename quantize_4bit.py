"""
MIT  quantize TinyStories-33M → 4-bit tiles for MVP
pip install torch transformers
"""
import torch, struct, os
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("karpathy/TinyStories-33M")
state = model.state_dict()

os.makedirs("tiles", exist_ok=True)
tile_id = 0
for name, param in state.items():
    if param.dtype != torch.float32:
        continue
    w = param.flatten()
    w_scale = w.abs().max()
    w_q = torch.clamp((w / w_scale * 7), -8, 7).round().to(torch.int8)
    # pack two 4-bit weights into one byte
    n = w_q.numel()
    pad = (n % 2)
    if pad:
        w_q = torch.cat([w_q, torch.zeros(1, dtype=torch.int8)])
    packed = (w_q[0::2] & 0x0F) | ((w_q[1::2] & 0x0F) << 4)
    tile_bytes = packed.cpu().numpy().tobytes()
    meta = struct.pack(">ff", w_scale, 0.0)
    with open(f"tiles/{tile_id}.tile", "wb") as f:
        f.write(meta + tile_bytes)
    print(f"tile {tile_id}: {name} → {len(tile_bytes)} B")
    tile_id += 1
print("8 MB total → tiles/ folder ready")