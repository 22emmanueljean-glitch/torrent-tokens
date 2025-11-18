import json
import os

base_url = "https://torrent-tokens.onrender.com/assets/weights"

# Main manifest
manifest = {
    "dims": {
        "dModel": 768,
        "nHeads": 12,
        "dHead": 64,
        "mlpHidden": 3072,
        "nLayers": 12,  # Changed from 6 to 12!
        "vocab": 50257,
        "maxSeq": 1024
    },
    "tensors": {
        "wte": f"{base_url}/wte.bin",
        "wpe": f"{base_url}/wpe.bin",
        "ln_f_g": f"{base_url}/ln_f_g.bin",
        "ln_f_b": f"{base_url}/ln_f_b.bin"
    },
    "tokenizer": {
        "vocab": "./assets/tokenizer/vocab.json",
        "merges": "./assets/tokenizer/merges.txt"
    }
}

with open("./assets/weights/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("✅ Created main manifest.json")

# Create manifests for all 12 layers
for i in range(12):
    layer_manifest = {
        "tensors": {
            "qkv": f"{base_url}/layer{i}_qkv.bin",
            "qkv_b": f"{base_url}/layer{i}_qkv_b.bin",
            "o": f"{base_url}/layer{i}_o.bin",
            "o_b": f"{base_url}/layer{i}_o_b.bin",
            "ff1": f"{base_url}/layer{i}_ff1.bin",
            "ff1_b": f"{base_url}/layer{i}_ff1_b.bin",
            "ff2": f"{base_url}/layer{i}_ff2.bin",
            "ff2_b": f"{base_url}/layer{i}_ff2_b.bin",
            "ln1_g": f"{base_url}/layer{i}_ln1_g.bin",
            "ln1_b": f"{base_url}/layer{i}_ln1_b.bin",
            "ln2_g": f"{base_url}/layer{i}_ln2_g.bin",
            "ln2_b": f"{base_url}/layer{i}_ln2_b.bin"
        }
    }
    
    with open(f"./assets/weights/manifest_layer{i}.json", "w") as f:
        json.dump(layer_manifest, f, indent=2)
    print(f"✅ Created manifest_layer{i}.json")

print("\n🎉 All 12 layer manifests created!")
