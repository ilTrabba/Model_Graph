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
    mapping: Dict[str, str], 
    model_id: str, 
    original_filename: str
) -> int:
    """
    Salva il fingerprint JSON usando il mapping garantito.
    """
    # Estraiamo i nomi normalizzati per il conteggio strutturale
    normalized_keys = list(mapping.keys())
    
    fingerprint = {
        "model_id": model_id,
        "original_filename": original_filename,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer_count": count_structural_layers(normalized_keys),
        "metadata": {},
        "mapping": mapping  # Usiamo il mapping passato, senza fare zip()
    }
    
    os.makedirs(MAPPING_FOLDER, exist_ok=True)
    mapping_path = os.path.join(MAPPING_FOLDER, f"{model_id}_mapping.json")
    
    try:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(fingerprint, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Mapping salvato con successo: {mapping_path}")
        return fingerprint["layer_count"].get("total_layers", 0)
    except Exception as e:
        # Assumendo che logHandler sia disponibile nel tuo scope
        return logHandler.error_handler(e, "save_layer_mapping_json", f"Errore salvataggio mapping {model_id}")

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

def normalize_safetensors_layers(weights: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]: 
    """
    Normalizza i nomi dei layer e restituisce (pesi_aggiornati, mapping_nomi).
    """
    if not weights:
        logger.warning("Input weights vuoto")
        return weights, {}

    original_count = len(weights)
    mapping = {}  # Struttura: { "nome_normalizzato": "nome_originale" }
    
    # Creiamo una lista statica delle chiavi per evitare errori durante la mutazione del dict
    original_keys = list(weights.keys())
    
    for old_name in original_keys:
        # Calcoliamo il nuovo nome
        new_name = normalize_single_name(old_name)
        
        # Gestione Collisioni
        if new_name in mapping:
            first_original = mapping[new_name]
            logger.warning(f"Collisione rilevata: '{old_name}' verrebbe normalizzato in '{new_name}', "
                           f"ma questo nome è già occupato da '{first_original}'. "
                           f"Il layer '{old_name}' verrà rimosso.")
            weights.pop(old_name)
            continue
        
        # Registriamo il mapping atomico
        mapping[new_name] = old_name
        
        # Se il nome è cambiato, aggiorniamo il dizionario dei pesi
        if new_name != old_name:
            weights[new_name] = weights.pop(old_name)
    
    logger.info(f"Normalizzazione: {original_count} -> {len(weights)} layer (Collisioni: {original_count - len(weights)})")
    
    return weights, mapping