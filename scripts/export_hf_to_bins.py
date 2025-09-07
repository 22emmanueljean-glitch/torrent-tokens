#!/usr/bin/env python3
# MIT License
#
# Export one GPT-2 style transformer block and tokenizer into the
# 4 projection matrices our runtime expects (+ token embeddings wte).
#
#   qkv.bin : [d_model, 3*d_model]   (packed Q|K|V, row-major, f32)
#   o.bin   : [d_model, d_model]
#   ff1.bin : [d_model, mlp_hidden]
#   ff2.bin : [mlp_hidden, d_model]
#   wte.bin : [vocab,   d_model]     (token embeddings for logits)
#
# Also writes assets/weights/manifest.json with shapes/paths and
# saves tokenizer to assets/tokenizer/tokenizer.json.
#
# Usage:
#   python scripts/export_hf_to_bins.py --model distilgpt2 --layer 0
#
# You can also try --model gpt2 (larger), but phones may be slower.

import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_f32_rowmajor(path, tensor):
    """Save a torch tensor as row-major float32 binary."""
    arr = tensor.contiguous().to(torch.float32).cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))
    nbytes = arr.nbytes
    print(f"wrote {path:<40} shape={tuple(tensor.shape)}  size={nbytes/1e6:.1f} MB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2",
                    help="Hugging Face repo id (e.g., distilgpt2, gpt2)")
    ap.add_argument("--layer", type=int, default=0,
                    help="which block to export (0-based index)")
    ap.add_argument("--outdir", default="assets",
                    help="output base dir; writes to assets/weights and assets/tokenizer")
    args = ap.parse_args()

    print(f"[load] model={args.model} layer={args.layer}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tok   = AutoTokenizer.from_pretrained(args.model)

    cfg = model.config
    d_model   = int(cfg.hidden_size)
    n_heads   = int(cfg.num_attention_heads)
    d_head    = d_model // n_heads
    mlp_hidden = int(getattr(cfg, "n_inner", 4 * d_model))
    vocab     = int(getattr(cfg, "vocab_size", 50257))

    # ---- get a GPT-2 style block ----
    try:
        block = model.transformer.h[args.layer]  # GPT-2 family
    except Exception as e:
        raise SystemExit(
            "This exporter currently supports GPT-2 style blocks at model.transformer.h[i]. "
            "Use distilgpt2 or gpt2.\n"
            f"Details: {e}"
        )

    weights_dir = os.path.join(args.outdir, "weights")
    tok_dir     = os.path.join(args.outdir, "tokenizer")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    # ---- Attention projections ----
    # GPT-2 packs Q,K,V together in c_attn: out_dim = 3 * d_model
    W_c_attn = block.attn.c_attn.weight.data   # [d_model, 3*d_model]
    # Bias exists too (b_c_attn), but your runtime does not add bias; skip for now.
    # Split not required for the runtime, we keep packed QKV as one matrix.
    W_qkv = W_c_attn  # shape [d_model, 3*d_model]
    save_f32_rowmajor(os.path.join(weights_dir, "qkv.bin"), W_qkv)

    # Output projection
    Wo = block.attn.c_proj.weight.data         # [d_model, d_model]
    save_f32_rowmajor(os.path.join(weights_dir, "o.bin"), Wo)

    # ---- MLP (GELU/SiLU path) ----
    # GPT-2 MLP uses c_fc (up) and c_proj (down)
    Wff1 = block.mlp.c_fc.weight.data          # [d_model, mlp_hidden]
    Wff2 = block.mlp.c_proj.weight.data        # [mlp_hidden, d_model]
    save_f32_rowmajor(os.path.join(weights_dir, "ff1.bin"), Wff1)
    save_f32_rowmajor(os.path.join(weights_dir, "ff2.bin"), Wff2)

    # ---- Token embeddings for logits ----
    # Coordinator will do logits = y @ wte.T
    try:
        Wte = model.transformer.wte.weight.data    # [vocab, d_model]
    except Exception as e:
        raise SystemExit(f"Could not locate token embeddings (wte) on this model: {e}")
    save_f32_rowmajor(os.path.join(weights_dir, "wte.bin"), Wte)

    # ---- Write manifest.json that your coordinator/workers read ----
    manifest = {
        "modelId": f"{args.model}-layer{args.layer}",
        "dtype": "f32",
        "layout": "row_major",
        "dims": {
            "dModel":    d_model,
            "nHeads":    n_heads,
            "dHead":     d_head,
            "mlpHidden": mlp_hidden,
            "nLayers":   1,
            "vocab":     vocab
        },
        "tensors": {
            "qkv": "./assets/weights/qkv.bin",
            "o":   "./assets/weights/o.bin",
            "ff1": "./assets/weights/ff1.bin",
            "ff2": "./assets/weights/ff2.bin",
            "wte": "./assets/weights/wte.bin"
        }
    }
    with open(os.path.join(weights_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote assets/weights/manifest.json")

    # ---- Save tokenizer (needed by coordinator UI) ----
    tok.save_pretrained(tok_dir)
    print("wrote tokenizer to assets/tokenizer/")

    # Final summary
    print("\n[done]")
    print(f"dModel={d_model}  nHeads={n_heads}  dHead={d_head}  mlpHidden={mlp_hidden}  vocab={vocab}")
    print("Place these on your server so they’re reachable at /assets/... (no 404s).")

if __name__ == "__main__":
    main()
