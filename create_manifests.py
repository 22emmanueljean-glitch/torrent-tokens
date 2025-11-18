import json

base_url = "https://pub-bf0385368be044e3b557e08813923f95.r2.dev/weights"

for layer_idx in range(6):
    manifest = {
        "modelId": f"distilgpt2-layer{layer_idx}",
        "dtype": "f32",
        "layout": "row_major",
        "dims": {
            "dModel": 768,
            "nHeads": 12,
            "dHead": 64,
            "mlpHidden": 3072,
            "nLayers": 1,
            "vocab": 50257,
            "maxSeq": 1024
        },
        "tensors": {
            "qkv": f"{base_url}/layer{layer_idx}_qkv.bin",
            "qkv_b": f"{base_url}/layer{layer_idx}_qkv_b.bin",
            "o": f"{base_url}/layer{layer_idx}_o.bin",
            "o_b": f"{base_url}/layer{layer_idx}_o_b.bin",
            "ff1": f"{base_url}/layer{layer_idx}_ff1.bin",
            "ff1_b": f"{base_url}/layer{layer_idx}_ff1_b.bin",
            "ff2": f"{base_url}/layer{layer_idx}_ff2.bin",
            "ff2_b": f"{base_url}/layer{layer_idx}_ff2_b.bin",
            "ln1_g": f"{base_url}/layer{layer_idx}_ln1_g.bin",
            "ln1_b": f"{base_url}/layer{layer_idx}_ln1_b.bin",
            "ln2_g": f"{base_url}/layer{layer_idx}_ln2_g.bin",
            "ln2_b": f"{base_url}/layer{layer_idx}_ln2_b.bin"
        }
    }
    
    with open(f"assets/weights/manifest_layer{layer_idx}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Created manifest_layer{layer_idx}.json")

print("\n🎉 All manifests created!")