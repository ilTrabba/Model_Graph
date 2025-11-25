class FilteringPatterns:
    """Predefined layer exclusion patterns for distance calculations"""
    # Lista completa di pattern per layer da ESCLUDERE dal calcolo della distanza

    #utilizzando un pattern, vengono esclusi tutti i layer tranne quelli del nome del pattern

    FULL_MODEL = frozenset([
        # Solo normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
    ])

    BACKBONE_ONLY = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Embedding layers
        'embed', 'embedding', 'embeddings',
        'position_embed', 'positional_embed', 'pos_embed',
        'token_embed', 'word_embed',
        'patch_embed',
        'wte', 'wpe',  # GPT-style embeddings
        
        # Head/Output layers
        'lm_head', 'language_model_head',
        'classifier', 'classification_head',
        'head', 
        'output_projection', 'output_proj',
        'score', 'scorer',
        'cls', 'pooler',
        'prediction_head', 'pred_head',
        'qa_outputs', 
        'seq_relationship',
        
        # Additional output-specific patterns
        'logits',
        'final_layer',
        'output_layer'
    ])

    BACKBONE_EMBEDDING = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Head/Output layers
        'lm_head', 'language_model_head',
        'classifier', 'classification_head',
        'head', 
        'output_projection', 'output_proj',
        'score', 'scorer',
        'cls', 'pooler',
        'prediction_head', 'pred_head',
        'qa_outputs', 
        'seq_relationship',
        'logits',
        'final_layer',
        'output_layer',
    ])

    BACKBONE_HEAD = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Embedding layers
        'embed', 'embedding', 'embeddings',
        'position_embed', 'positional_embed', 'pos_embed',
        'token_embed', 'word_embed',
        'patch_embed',
        'wte', 'wpe',
    ])

    EMBEDDING_ONLY = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Head/Output layers
        'lm_head', 'language_model_head',
        'classifier', 'classification_head',
        'head', 
        'output_projection', 'output_proj',
        'score', 'scorer',
        'cls', 'pooler',
        'prediction_head', 'pred_head',
        'qa_outputs', 
        'seq_relationship',
        'logits',
        'final_layer',
        'output_layer',
        
        # Backbone/Attention layers
        'attention', 'attn', 'self_attn', 'cross_attn',
        'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
        'dense', 'intermediate', 'output',
        'mlp', 'ffn', 'feed_forward',
        'conv', 'convolution',
        'linear',
        'layer.', 'layers.',
    ])

    HEAD_ONLY = frozenset([
        # Normalization layers
        'layernorm', 'layer_norm', 'ln_', '.ln.', 
        'batchnorm', 'batch_norm', 'bn', '.bn.',
        'groupnorm', 'group_norm', 'gn',
        'instancenorm', 'instance_norm',
        'rmsnorm', 'rms_norm',
        
        # Embedding layers
        'embed', 'embedding', 'embeddings',
        'position_embed', 'positional_embed', 'pos_embed',
        'token_embed', 'word_embed',
        'patch_embed',
        'wte', 'wpe',
        
        # Backbone/Attention layers
        'attention', 'attn', 'self_attn', 'cross_attn',
        'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
        'dense', 'intermediate', 'output',
        'mlp', 'ffn', 'feed_forward',
        'conv', 'convolution',
        'linear',
        'layer.', 'layers.',  # generic layer indexing
    ])