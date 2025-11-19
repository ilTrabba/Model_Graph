import os
import json
import numpy as np
from safetensors.torch import load_file
from sklearn.random_projection import GaussianRandomProjection
import blake3

# ---------------------------
# CONFIGURAZIONE
# ---------------------------
RANDOM_VECTOR_DIM = 256  # dimensione del global vector finale
LAYER_FEATURES = ["l2norm", "mean", "std", "min", "max", "hash_bucket"]


# ---------------------------
# FUNZIONI AUSILIARIE
# ---------------------------

def canonical_name(layer_name: str) -> str:
    """
    Normalizza il nome del layer.
    Esempio: 
    encoder.layer.0.attention.self.query.weight -> L0.ATTN_Q.W
    encoder.layer.0.attn.q_proj.weight -> L0.ATTN_Q.W
    """
    name = layer_name.lower()
    
    # Layer index
    import re
    idx_match = re.search(r"layer\.(\d+)", name)
    idx = idx_match.group(1) if idx_match else "0"
    
    # Rilevamento tipo layer
    if "query" in name or "q_proj" in name:
        suffix = "ATTN_Q"
    elif "key" in name or "k_proj" in name:
        suffix = "ATTN_K"
    elif "value" in name or "v_proj" in name:
        suffix = "ATTN_V"
    elif "out_proj" in name or "attn.out" in name:
        suffix = "ATTN_OUT"
    elif "ln" in name:
        suffix = "LN"
    elif "bias" in name:
        suffix = "B"
    elif "weight" in name:
        suffix = "W"
    elif "up_proj" in name:
        suffix = "MLP_UP"
    elif "down_proj" in name:
        suffix = "MLP_DOWN"
    else:
        suffix = "OTHER"
    
    return f"L{idx}.{suffix}"


def layer_stats(tensor: np.ndarray) -> dict:
    """Calcola statistiche base di un tensore e digest hash bucket"""
    l2 = float(np.linalg.norm(tensor))
    mean = float(np.mean(tensor))
    std = float(np.std(tensor))
    min_val = float(np.min(tensor))
    max_val = float(np.max(tensor))
    
    # Hash bucketizzato
    h = blake3.blake3(tensor.tobytes()).digest()
    bucket = int.from_bytes(h[:2], "big") / 65535.0
    
    return {
        "l2norm": l2,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "hash_bucket": bucket
    }


# ---------------------------
# FUNZIONE PRINCIPALE
# ---------------------------

def generate_fingerprint(safetensors_path: str, output_json: str):
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"{safetensors_path} non esiste.")
    
    # Carica tensori in modalità lazy
    tensors = load_file(safetensors_path)
    
    fingerprint = {
        "metadata": {
            "model_format": "safetensors",
            "total_parameters": sum(t.numel() for t in tensors.values()),
            "total_layers": len(tensors)
        },
        "canonical_layer_map": {},
        "layer_stats": {},
        "global_vector": []
    }
    
    giant_vector = []
    
    for orig_name, tensor in tensors.items():
        # converti in numpy
        t_np = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
        
        # canonical name
        cname = canonical_name(orig_name)
        fingerprint["canonical_layer_map"][orig_name] = {
            "canonical_name": cname,
            "shape": list(t_np.shape),
            "dtype": str(t_np.dtype)
        }
        
        # layer statistics
        stats = layer_stats(t_np)
        fingerprint["layer_stats"][cname] = stats
        
        # costruzione giant vector
        giant_vector.extend([stats[f] for f in LAYER_FEATURES])
    
    # Conversione in numpy
    giant_vector = np.array(giant_vector).reshape(1, -1)
    
    # Random projection per global vector compatto
    transformer = GaussianRandomProjection(n_components=RANDOM_VECTOR_DIM, random_state=42)
    global_vector = transformer.fit_transform(giant_vector)
    
    fingerprint["global_vector"] = global_vector.flatten().tolist()
    
    # Scrittura su JSON
    with open(output_json, "w") as f:
        json.dump(fingerprint, f, indent=2)
    
    print(f"Fingerprint generata: {output_json}")


# ---------------------------
# ESEMPIO DI USO
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera fingerprint da un file safetensors.")
    parser.add_argument("input_file", help="Percorso al file .safetensors")
    parser.add_argument("output_file", help="Percorso al file .json di output")
    args = parser.parse_args()
    
    generate_fingerprint(args.input_file, args.output_file)
