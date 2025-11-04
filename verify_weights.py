#!/usr/bin/env python3
"""
verify_weights.py - Check if weight files are correct format and size
"""
import os
import numpy as np
from pathlib import Path

def check_file(path, expected_shape=None, expected_bytes=None):
    """Check if a weight file exists and has correct size"""
    if not os.path.exists(path):
        return f"❌ MISSING: {path}"
    
    size = os.path.getsize(path)
    
    # Try to load as float32 array
    try:
        arr = np.fromfile(path, dtype=np.float32)
        
        # Check for common issues
        if np.all(arr == 0):
            return f"⚠️  ALL ZEROS: {path} ({size:,} bytes)"
        if np.any(np.isnan(arr)):
            return f"❌ CONTAINS NaN: {path}"
        if np.any(np.isinf(arr)):
            return f"❌ CONTAINS INF: {path}"
        
        # Check expected size
        if expected_bytes and size != expected_bytes:
            return f"⚠️  WRONG SIZE: {path} ({size:,} bytes, expected {expected_bytes:,})"
        
        # Check expected shape
        if expected_shape:
            expected_len = np.prod(expected_shape)
            if len(arr) != expected_len:
                return f"⚠️  WRONG SHAPE: {path} (got {len(arr)} floats, expected {expected_len})"
        
        # Check value ranges (should be reasonable for neural net weights)
        vmin, vmax = arr.min(), arr.max()
        vmean, vstd = arr.mean(), arr.std()
        
        if abs(vmin) > 100 or abs(vmax) > 100:
            return f"⚠️  SUSPICIOUS RANGE: {path} (min={vmin:.2f}, max={vmax:.2f})"
        
        return f"✅ OK: {path} ({size:,} bytes, {len(arr)} floats, mean={vmean:.4f}, std={vstd:.4f})"
        
    except Exception as e:
        return f"❌ ERROR: {path} - {e}"

def main():
    print("🔍 Torrent-Tokens Weight Verification")
    print("=" * 60)
    print()
    
    # Expected dimensions for DistilGPT-2
    dims = {
        "dModel": 768,
        "nHeads": 12,
        "dHead": 64,
        "mlpHidden": 3072,
        "vocab": 50257,
        "maxSeq": 1024
    }
    
    D = dims["dModel"]
    H = dims["mlpHidden"]
    V = dims["vocab"]
    L = dims["maxSeq"]
    
    # Define expected sizes
    weight_files = {
        "wte.bin": (V * D * 4, (V, D)),              # 50257 × 768
        "wpe.bin": (L * D * 4, (L, D)),              # 1024 × 768 (optional)
        "ln1_g.bin": (D * 4, (D,)),                  # 768
        "ln1_b.bin": (D * 4, (D,)),                  # 768
        "qkv.bin": (D * 3*D * 4, (D, 3*D)),          # 768 × 2304
        "qkv_b.bin": (3*D * 4, (3*D,)),              # 2304
        "o.bin": (D * D * 4, (D, D)),                # 768 × 768
        "o_b.bin": (D * 4, (D,)),                    # 768
        "ln2_g.bin": (D * 4, (D,)),                  # 768
        "ln2_b.bin": (D * 4, (D,)),                  # 768
        "ff1.bin": (D * H * 4, (D, H)),              # 768 × 3072
        "ff1_b.bin": (H * 4, (H,)),                  # 3072
        "ff2.bin": (H * D * 4, (H, D)),              # 3072 × 768
        "ff2_b.bin": (D * 4, (D,)),                  # 768
    }
    
    base_path = Path("./assets/weights")
    
    if not base_path.exists():
        print(f"❌ ERROR: Directory not found: {base_path}")
        print("   Make sure you're running this from the project root.")
        return
    
    print("Checking weight files:")
    print()
    
    total_size = 0
    errors = 0
    warnings = 0
    
    for filename, (expected_bytes, expected_shape) in weight_files.items():
        filepath = base_path / filename
        result = check_file(filepath, expected_shape, expected_bytes)
        print(result)
        
        if "❌" in result:
            errors += 1
        elif "⚠️" in result:
            warnings += 1
        
        if os.path.exists(filepath):
            total_size += os.path.getsize(filepath)
    
    print()
    print("=" * 60)
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
    print()
    
    # Check tokenizer files
    print("Checking tokenizer files:")
    print()
    
    tokenizer_path = Path("./assets/tokenizer")
    if not tokenizer_path.exists():
        print(f"❌ ERROR: Tokenizer directory not found: {tokenizer_path}")
    else:
        vocab_path = tokenizer_path / "vocab.json"
        merges_path = tokenizer_path / "merges.txt"
        
        if vocab_path.exists():
            size = os.path.getsize(vocab_path)
            print(f"✅ vocab.json ({size:,} bytes)")
            
            # Try to parse it
            try:
                import json
                with open(vocab_path) as f:
                    vocab = json.load(f)
                print(f"   → {len(vocab)} tokens")
                if len(vocab) != 50257:
                    print(f"   ⚠️  Expected 50257 tokens for GPT-2")
            except Exception as e:
                print(f"   ❌ Failed to parse: {e}")
        else:
            print(f"❌ vocab.json not found")
            errors += 1
        
        if merges_path.exists():
            size = os.path.getsize(merges_path)
            with open(merges_path) as f:
                lines = f.readlines()
            print(f"✅ merges.txt ({size:,} bytes, {len(lines)} lines)")
        else:
            print(f"❌ merges.txt not found")
            errors += 1
    
    print()
    print("=" * 60)
    
    if errors == 0 and warnings == 0:
        print("✅ All checks passed! You're ready to run the model.")
    elif errors == 0:
        print(f"⚠️  {warnings} warning(s) found. Model might work but check above.")
    else:
        print(f"❌ {errors} error(s) found. Fix these before running the model.")
    
    print()
    print("Next steps:")
    print("1. If all OK: Run 'npm start' and open http://localhost:8080/test_local.html")
    print("2. If errors: Re-export weights using 'python scripts/export_hf_to_bins.py'")

if __name__ == "__main__":
    main()