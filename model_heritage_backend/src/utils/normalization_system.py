import logging
import re
import os
import json

from typing import Dict, List, Set
from datetime import datetime, timezone
from typing import Dict, Any
from src.log_handler import logHandler

logger = logging.getLogger(__name__)

MAPPING_FOLDER = os.path.join(os.path.dirname(__file__), '../../weights/fingerprints')

def count_structural_layers(normalized_keys: List[str]) -> Dict:
    """
    Conta i layer strutturali effettivi analizzando i nomi normalizzati.
    
    Identifica i layer logici (es. 12 layer per BERT) piuttosto che il numero
    totale di tensori. Rileva automaticamente architetture multi-componente
    (encoder-decoder, vision-text, ecc.) e fornisce un breakdown dettagliato.
    
    Args:
        normalized_keys: Lista dei nomi layer normalizzati
        
    Returns:
        Dict con struttura:
        {
            "total_layers": int,
            "breakdown": Optional[Dict[str, int]]  # solo per architetture multi-componente
        }
        
    Examples:
        BERT (encoder-only):
        >>> count_structural_layers(["layers.0.attn.weight", "layers.11.mlp.bias"])
        {"total_layers": 12}
        
        T5 (encoder-decoder):
        >>> count_structural_layers(["encoder.layers.0.weight", "decoder.layers.0.weight"])
        {"total_layers": 24, "breakdown": {"encoder": 12, "decoder": 12}}
    """
    # Pattern per identificare indici di layer strutturali
    layer_patterns = [
        r'encoder\.layers?\.(\d+)',
        r'decoder\.layers?\.(\d+)',
        r'vision_model\.encoder\.layers?\.(\d+)',
        r'text_model\.encoder\.layers?\.(\d+)',
        r'visual\.encoder\.blocks?\.(\d+)',
        r'transformer\.layers?\.(\d+)',
        r'transformer\.h\.(\d+)',
        r'model\.layers?\.(\d+)',
        r'(?:^|\.)layers?\.(\d+)',
        r'(?:^|\.)blocks?\.(\d+)',
        r'(?:^|\.)h\.(\d+)',
    ]
    
    # Dizionario per tracciare layer per componente
    component_layers: Dict[str, Set[int]] = {}
    all_layer_indices: Set[int] = set()
    
    for key in normalized_keys:
        for pattern in layer_patterns:
            match = re.search(pattern, key)
            if match:
                layer_idx = int(match.group(1))
                all_layer_indices.add(layer_idx)
                
                # Identifica il componente (encoder, decoder, vision, text, etc.)
                if 'encoder.layers' in key or 'encoder.layer' in key:
                    component_layers.setdefault('encoder', set()).add(layer_idx)
                elif 'decoder.layers' in key or 'decoder.layer' in key:
                    component_layers.setdefault('decoder', set()).add(layer_idx)
                elif 'vision_model.encoder' in key or 'visual.encoder' in key:
                    component_layers.setdefault('vision_encoder', set()).add(layer_idx)
                elif 'text_model.encoder' in key or 'text_encoder' in key:
                    component_layers.setdefault('text_encoder', set()).add(layer_idx)
                
                break  # Trovato pattern, passa al prossimo key
    
    # Calcola total_layers
    total_layers = len(all_layer_indices)
    
    # Costruisci il risultato
    result = {"total_layers": total_layers}
    
    # Aggiungi breakdown solo se ci sono multiple componenti
    if len(component_layers) > 1:
        result["breakdown"] = {
            component: len(indices) 
            for component, indices in component_layers.items()
        }
    
    # Logging
    if "breakdown" in result:
        breakdown_str = ", ".join(
            f"{comp}={count}" for comp, count in result["breakdown"].items()
        )
        logger.info(
            f"Layer strutturali rilevati: {total_layers} totali ({breakdown_str})"
        )
    else:
        logger.info(f"Layer strutturali rilevati: {total_layers}")
    
    return result

def save_layer_mapping_json(
    original_keys: List[str],
    normalized_keys: List[str],
    model_id: str,
    original_filename: str
) -> int:
    """
    Crea e salva un file JSON contenente il mapping tra nomi layer originali e normalizzati.
    
    Questo fingerprint permette di ricostruire il file safetensors originale a partire
    dalla versione normalizzata, mantenendo traccia della corrispondenza nome-per-nome.
    
    Args:
        original_keys: Lista ordinata dei nomi layer originali (grezzi)
        normalized_keys: Lista ordinata dei nomi layer normalizzati
        model_id: ID univoco del modello
        original_filename: Nome del file safetensors originale
        
    Note:
        - Le due liste devono avere la stessa lunghezza
        - L'ordine è significativo: original_keys[i] corrisponde a normalized_keys[i]
        - Il file viene salvato in MAPPING_FOLDER con nome {model_id}_mapping.json
        - In caso di errore logga warning senza sollevare eccezioni
    """
    if len(original_keys) != len(normalized_keys):
        logger.error(
            f"Errore salvataggio mapping per model_id={model_id}: "
            f"lunghezza liste non corrispondente "
            f"(original={len(original_keys)}, normalized={len(normalized_keys)})"
        )
        return
    
    # Crea la struttura del fingerprint
    fingerprint = {
        "model_id": model_id,
        "original_filename": original_filename,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer_count": count_structural_layers(normalized_keys),
        "metadata": {},
        "mapping": {
            normalized: original 
            for normalized, original in zip(normalized_keys, original_keys)
        }
    }
    
    # Prepara il path di salvataggio
    os.makedirs(MAPPING_FOLDER, exist_ok=True)
    mapping_filename = f"{model_id}_mapping.json"
    mapping_path = os.path.join(MAPPING_FOLDER, mapping_filename)
    
    # Salva il file JSON
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Mapping salvato con successo: {mapping_path} "
            f"({len(original_keys)} layer mappati)"
        )

        return fingerprint["layer_count"].get("total_layers", 0)
    except Exception as e:
        return logHandler.error_handler(e, "save_layer_mapping_json", f"Errore salvataggio mapping per model_id={model_id}"  )

# PATTERNS compilati a livello di modulo (una sola volta all'import)
STRUCTURAL_PATTERNS = [
    re.compile(r'(encoder\.layer\.\d+)'),
    re.compile(r'(decoder\.layer\.\d+)'),
    re.compile(r'(encoder\.blocks\.\d+)'),
    re.compile(r'(decoder\.blocks\.\d+)'),
    re.compile(r'(encoder_block\.\d+)'),
    re.compile(r'(decoder_block\.\d+)'),
    re.compile(r'(transformer\.h\.\d+)'),
    re.compile(r'(transformer\.layers\.\d+)'),
    re.compile(r'(transformer\.blocks\.\d+)'),
    re.compile(r'(transformer_block\.\d+)'),
    re.compile(r'(model\.layers\.\d+)'),
    re.compile(r'(model\.decoder\.layers\.\d+)'),
    re.compile(r'(gpt_neox\.layers\.\d+)'),
    re.compile(r'(layers\.\d+)'),
    re.compile(r'(blocks\.\d+)'),
    re.compile(r'(layer\.\d+)'),
    re.compile(r'(block\.\d+)'),
    re.compile(r'(h\.\d+)'),
    re.compile(r'(vision_model\.encoder\.layers\.\d+)'),
    re.compile(r'(visual\. encoder\.blocks\.\d+)'),
    re.compile(r'(vision_tower\.blocks\.\d+)'),
    re.compile(r'(backbone\.stages\.\d+)'),
    re.compile(r'(modules\.\d+)'),
    re.compile(r'(block_list\.\d+)'),
]

CLEANUP_PATTERN = re.compile(r'^(?:model|base_model|transformer)\.')

KNOWN_PREFIXES = (
    'backbone.', 'base_model.', 'decoder.', 'discriminator.', 'encoder.',
    'generator.', 'head.', 'model.', 'proj.', 'student.', 'teacher.',
    'transformer.', 'unet.', 'vae.',
    'bart.', 'bert.', 'bloom.', 'deberta.', 'distilbert.', 'falcon.',
    'gpt_neox.', 'gptj.', 'llama.', 'marian.', 'mistral.', 'mt5.',
    'opt.', 'roberta.', 't5.',
    'beit.', 'clip.', 'clip_model.', 'clip_vision_model.', 'convnext.',
    'dino.', 'dinov2.', 'open_clip.', 'sam.', 'swin.', 'vit.',
    'image_encoder.', 'speech_encoder_decoder.', 'text_encoder.',
    'text_model.', 'vision_model.', 'vision_tower.', 'visual.', 'whisper.',
)

ABBREVIATION_MAP = {
    '.ln_f.': '.layernorm_final.',
    '.ln_1.': '.layernorm_before.',
    '.ln_2.': '.layernorm_after.',
    '.ln.': '.layernorm.',
    '.attn.': '.attention.',
    '.mlp.':  '.feedforward.',
    '.c_attn.': '.attention.combined.',
    '.c_proj.': '.attention.projection.',
    '.c_fc.': '.feedforward.fc.',
}

def normalize_single_name(name: str) -> str:
    """Normalizza un singolo nome di layer."""
    # Step 1: TensorFlow → PyTorch LayerNorm
    name = name.replace('.beta', '.bias').replace('.gamma', '.weight')

    # Step 2: Pattern strutturali
    for pattern in STRUCTURAL_PATTERNS:
        match = pattern.search(name)
        if match:
            name = name[match.start():]
            break
    else:
        # Step 3: Prefissi noti (solo se nessun pattern trovato)
        if name.startswith(KNOWN_PREFIXES):
            for prefix in KNOWN_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break

    # Step 4: Abbreviazioni
    for abbrev, expanded in ABBREVIATION_MAP.items():
        if abbrev in name:
            name = name.replace(abbrev, expanded)

    # Step 5: Pulizia finale
    return CLEANUP_PATTERN.sub('', name)

def normalize_safetensors_layers(weights:  Dict[str, Any]) -> Dict[str, Any]: 
    """
    Normalizza i nomi dei layer IN-PLACE (zero overhead memoria per tensori).
    
    I tensori non vengono mai copiati né modificati - solo le chiavi stringa
    vengono rinominate.  Reversibile tramite fingerprint/mapping.
    """
    if not weights:
        logger.warning("Input weights vuoto")
        return weights

    original_count = len(weights)
    
    # Calcola tutte le rinominazioni prima di modificare
    renames = []  # (old_key, new_key)
    seen = {}     # new_key -> old_key (per collisioni)
    collisions = []
    
    for old_name in list(weights.keys()):
        new_name = normalize_single_name(old_name)
        
        if new_name in seen:
            collisions.append((old_name, new_name, seen[new_name]))
            del weights[old_name]  # Scarta duplicato
        elif new_name != old_name: 
            seen[new_name] = old_name
            renames.append((old_name, new_name))
        else:
            seen[new_name] = old_name
    
    # Applica rinominazioni
    for old_key, new_key in renames: 
        weights[new_key] = weights.pop(old_key)
    
    # Logging
    for old, new, first in collisions:
        logger.warning(f"Collisione: '{old}' → '{new}' (già usato da '{first}')")
    
    if collisions:
        logger.warning(f"Normalizzazione:  {original_count} → {len(weights)} layer ({len(collisions)} collisioni)")
    else:
        logger.info(f"Normalizzazione: {original_count} → {len(weights)} layer")
    
    return weights