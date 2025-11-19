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

def normalize_safetensors_layers(weights: Dict[str, Any]) -> Dict[str, Any]:
    
        """
        Normalizza i nomi dei layer di un singolo file safetensors.

        Gestisce molteplici architetture Hugging Face (ViT, BERT, GPT, LLaMA, T5, 
        Whisper, CLIP, Swin, ecc.) per garantire compatibilità cross-framework.

        Operazioni applicate:
        1. LayerNorm TensorFlow → PyTorch (.beta → .bias, .gamma → .weight)
        2. Estrazione core strutturale (layer.X, block.X, h.X, etc.)
        3. Rimozione prefissi di wrapping del modello
        4. Normalizzazione abbreviazioni comuni
        5. Pulizia finale
        
        Args:
            weights: Dizionario {layer_name: tensor}

        Returns:
            Dizionario normalizzato {normalized_layer_name: tensor}
            
        Raises:
            Nessuna eccezione; logga warning per collisioni
        """
        if not weights:
            logger.warning("Input weights vuoto")
            return {}

        normalized_weights = {}

        # =========================================================================
        # PATTERN STRUTTURALI (ordinati per priorità/specificità)
        # =========================================================================
        structural_patterns = [
            # Encoder-Decoder specifici (più specifici prima)
            r'(encoder\.layer\.\d+)',
            r'(decoder\.layer\.\d+)',
            r'(encoder\.blocks\.\d+)',
            r'(decoder\.blocks\.\d+)',
            r'(encoder_block\.\d+)',
            r'(decoder_block\.\d+)',
            
            # Transformer generici
            r'(transformer\.h\.\d+)',
            r'(transformer\.layers\.\d+)',
            r'(transformer\.blocks\.\d+)',
            r'(transformer_block\.\d+)',
            
            # Layer/Block generici (LLaMA, GPT, Mistral, etc.)
            r'(model\.layers\.\d+)',
            r'(model\.decoder\.layers\.\d+)',
            r'(gpt_neox\.layers\.\d+)',
            r'(layers\.\d+)',
            r'(blocks\.\d+)',
            r'(layer\.\d+)',
            r'(block\.\d+)',
            r'(h\.\d+)',
            
            # Vision Models (CLIP, ViT, Swin, SAM, etc.)
            r'(vision_model\.encoder\.layers\.\d+)',
            r'(visual\.encoder\.blocks\.\d+)',
            r'(vision_tower\.blocks\.\d+)',
            r'(backbone\.stages\.\d+)',
            
            # Altri pattern
            r'(modules\.\d+)',
            r'(block_list\.\d+)',
        ]

        # =========================================================================
        # PREFISSI DA RIMUOVERE (ordinati alfabeticamente per manutenibilità)
        # =========================================================================
        known_prefixes = [
            # Generici
            'backbone.', 'base_model.', 'decoder.', 'discriminator.', 'encoder.',
            'generator.', 'head.', 'model.', 'proj.', 'student.', 'teacher.', 
            'transformer.', 'unet.', 'vae.',
            
            # Text Models
            'bart.', 'bert.', 'bloom.', 'deberta.', 'distilbert.', 'falcon.', 
            'gpt_neox.', 'gptj.', 'llama.', 'marian.', 'mistral.', 'mt5.', 
            'opt.', 'roberta.', 't5.',
            
            # Vision Models
            'beit.', 'clip.', 'clip_model.', 'clip_vision_model.', 'convnext.', 
            'dino.', 'dinov2.', 'open_clip.', 'sam.', 'swin.', 'vit.',
            
            # Multimodal
            'clip.', 'image_encoder.', 'speech_encoder_decoder.', 'text_encoder.', 
            'text_model.', 'vision_model.', 'vision_tower.', 'visual.', 'whisper.',
        ]

        # =========================================================================
        # ABBREVIAZIONI DA NORMALIZZARE
        # =========================================================================
        abbreviation_map = {
            '.ln_f.': '.layernorm_final.',
            '.ln_1.': '.layernorm_before.',
            '.ln_2.': '.layernorm_after.',
            '.ln.': '.layernorm.',
            '.attn.': '.attention.',
            '.mlp.': '.feedforward.',
            '.c_attn.': '.attention.combined.',  
            '.c_proj.': '.attention.projection.',
            '.c_fc.': '.feedforward.fc.',
        }

        # =========================================================================
        # NORMALIZZAZIONE
        # =========================================================================
        collision_count = 0
        
        for original_name, tensor in weights.items():
            normalized_name = original_name

            # Step 1: TensorFlow → PyTorch LayerNorm
            normalized_name = normalized_name.replace('.beta', '.bias')
            normalized_name = normalized_name.replace('.gamma', '.weight')

            # Step 2: Cerca pattern strutturali (dal più specifico al generico)
            core_name_found = False
            for pattern in structural_patterns:
                match = re.search(pattern, normalized_name)
                if match:
                    start_idx = match.start()
                    normalized_name = normalized_name[start_idx:]
                    core_name_found = True
                    break

            # Step 3: Se non trovato pattern, rimuovi prefissi noti
            if not core_name_found:
                for prefix in known_prefixes:
                    if normalized_name.startswith(prefix):
                        normalized_name = normalized_name[len(prefix):]
                        break

            # Step 4: Normalizza abbreviazioni comuni
            for abbrev, expanded in abbreviation_map.items():
                normalized_name = normalized_name.replace(abbrev, expanded)

            # Step 5: Pulizia finale - rimuovi prefissi residui
            # (potrebbe esserci dopo Step 2 se pattern non era all'inizio)
            normalized_name = re.sub(
                r'^(?:model|base_model|transformer)\.',
                '',
                normalized_name
            )

            # Step 6: Gestisci collisioni
            if normalized_name in normalized_weights:
                collision_count += 1
                logger.warning(
                    f"Collisione durante la normalizzazione #{collision_count}: "
                    f"'{original_name}' → '{normalized_name}' (già esistente). "
                    f"Layer in collisione trovato!."
                )
                logHandler.error_handler(None, "normalize_safetensors_layers", "Collision occured between layer's names: original name → " + f"{original_name}" + ", normalized name → " + f"{normalized_name}")

            # Step 7: Aggiungi al risultato
            normalized_weights[normalized_name] = tensor

        # =========================================================================
        # LOGGING FINALE
        # =========================================================================
        original_count = len(weights)
        normalized_count = len(normalized_weights)
        
        if collision_count > 0:
            logger.warning(
                f"Normalizzazione completata con {collision_count} collisioni: "
                f"{original_count} layer originali → {normalized_count} layer normalizzati "
                f"({original_count - normalized_count} scartati)"
            )
        else:
            logger.info(
                f"Normalizzazione completata: "
                f"{original_count} layer originali → {normalized_count} layer normalizzati"
            )

        return normalized_weights