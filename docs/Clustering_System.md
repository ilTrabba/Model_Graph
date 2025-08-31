# Model Heritage Clustering System Documentation

This document describes the comprehensive clustering system implemented for the Model Heritage Backend.

## Overview

The clustering system provides automatic model family assignment and genealogical tree construction using sophisticated distance calculation, clustering algorithms, and the MoTHer (Model Tree Heritage Recovery) algorithm.

## Architecture

### Core Components

1. **ModelDistanceCalculator** (`src/clustering/distance_calculator.py`)
   - Calculates distances between AI model weights
   - Supports multiple distance metrics (L2, matrix rank, cosine similarity)
   - Auto-detects optimal metric based on model type (full fine-tuned vs LoRA)
   - Filters relevant layers (attention, dense, linear layers)

2. **FamilyClusteringSystem** (`src/clustering/family_clustering.py`)
   - Assigns models to families based on weight similarities
   - Maintains family centroids and statistics
   - Supports multiple clustering algorithms (DBSCAN, K-means, threshold-based)
   - Uses configurable distance thresholds

3. **MoTHerTreeBuilder** (`src/clustering/tree_builder.py`)
   - Builds genealogical trees within model families
   - Uses kurtosis-based direction determination
   - Applies Edmonds' algorithm for minimum directed spanning tree
   - Supports different tree building methods

4. **ModelManagementSystem** (`src/clustering/model_management.py`)
   - Coordinates all clustering components
   - Manages complete pipeline from upload to heritage detection
   - Provides comprehensive genealogy queries

## Integration with Existing System

### Database Models

The clustering system integrates with existing database models:
- `Model`: Enhanced with family_id, parent_id, confidence_score
- `Family`: Tracks family statistics and metadata

### MoTHer Algorithm Compatibility

The system leverages the existing MoTHer algorithm implementation:
- Reuses `load_model_weights`, `calc_ku`, `calculate_l2_distance` from `mother_utils.py`
- Uses existing `build_tree` function for tree construction
- Maintains compatibility with current parent-finding logic

### API Integration

New endpoints added to `src/routes/models.py`:

#### Clustering Operations
- `POST /api/clustering/recluster` - Recluster all models
- `POST /api/families/{family_id}/tree/rebuild` - Rebuild family tree
- `POST /api/models/{model_id}/reprocess` - Reprocess single model

#### Genealogy Queries
- `GET /api/families/{family_id}/genealogy` - Get complete family genealogy
- `GET /api/models/{model_id}/lineage` - Get model lineage (ancestors/descendants)
- `GET /api/clustering/statistics` - Get system statistics

## Configuration Options

### Distance Metrics
```python
DistanceMetric.L2_DISTANCE      # Standard L2 norm (default for full models)
DistanceMetric.MATRIX_RANK      # Matrix rank analysis (optimal for LoRA)
DistanceMetric.COSINE_SIMILARITY # Cosine similarity
DistanceMetric.AUTO             # Auto-select based on model type
```

### Clustering Methods
```python
ClusteringMethod.DBSCAN      # Density-based clustering
ClusteringMethod.KMEANS      # K-means clustering
ClusteringMethod.THRESHOLD   # Simple threshold-based
ClusteringMethod.AUTO        # Auto-select based on data size
```

### Tree Building Methods
```python
TreeBuildingMethod.MOTHER        # Full MoTHer algorithm (kurtosis + distance)
TreeBuildingMethod.DISTANCE_ONLY # Distance-based MST only
TreeBuildingMethod.KURTOSIS_ONLY # Kurtosis-based ordering only
```

## Usage Examples

### Initialize the Management System
```python
from src.clustering.model_management import ModelManagementSystem

# Default configuration
mgmt = ModelManagementSystem()

# Custom configuration
mgmt = ModelManagementSystem(
    distance_metric=DistanceMetric.MATRIX_RANK,
    family_threshold=0.3,
    clustering_method=ClusteringMethod.DBSCAN,
    tree_method=TreeBuildingMethod.MOTHER,
    lambda_param=0.7
)
```

### Process a New Model
```python
# Complete processing pipeline
result = mgmt.process_new_model(model)

if result['status'] == 'success':
    family_id = result['family_id']
    parent_id = result['parent_id']
    confidence = result['parent_confidence']
```

### Rebuild Family Tree
```python
result = mgmt.rebuild_family_tree(family_id)

if result['status'] == 'success':
    updated_count = result['models_updated']
    tree_stats = result['tree_statistics']
```

### Get Family Genealogy
```python
genealogy = mgmt.get_family_genealogy(family_id)

family_info = genealogy['family']
models = genealogy['models']
tree = genealogy['tree']
statistics = genealogy['statistics']
```

## API Usage Examples

### Trigger Complete Reclustering
```bash
curl -X POST http://localhost:5001/api/clustering/recluster
```

### Get Model Lineage
```bash
curl http://localhost:5001/api/models/{model_id}/lineage
```

### Rebuild Family Tree
```bash
curl -X POST http://localhost:5001/api/families/{family_id}/tree/rebuild
```

### Get System Statistics
```bash
curl http://localhost:5001/api/clustering/statistics
```

## Error Handling and Fallbacks

The system includes comprehensive error handling:

1. **Graceful Degradation**: Falls back to simple parameter-based clustering if advanced algorithms fail
2. **Dependency Safety**: Handles missing dependencies gracefully
3. **Database Safety**: Validates all database operations
4. **Logging**: Comprehensive logging for debugging and monitoring

### Fallback Behavior

If the clustering system fails:
1. Falls back to parameter-count-based family assignment
2. Uses original MoTHer algorithm for parent finding
3. Provides basic parameter similarity as final fallback

## Performance Considerations

### Memory Usage
- Distance matrices scale O(n²) with number of models
- For large families (>100 models), consider chunked processing
- Tree building is optimized for families up to ~50 models

### Processing Time
- L2 distance calculation: ~100ms per model pair
- Matrix rank calculation: ~200ms per model pair (for LoRA)
- Tree building: ~500ms for 10-model family

### Optimization Tips
1. Use `DistanceMetric.AUTO` for optimal performance
2. Set appropriate `family_threshold` to avoid oversized families
3. Consider `ClusteringMethod.THRESHOLD` for faster processing
4. Use batch operations for multiple models

## Monitoring and Debugging

### Key Metrics to Monitor
- Family size distribution
- Processing success rate
- Average confidence scores
- Tree validity metrics

### Debugging Tools
```python
# Get comprehensive system statistics
stats = mgmt.get_system_statistics()

# Validate tree structure
tree, _ = tree_builder.build_family_tree(family_id)
is_valid, issues = tree_builder.validate_tree(tree)

# Get tree statistics
tree_stats = tree_builder.get_tree_statistics(tree)
```

### Log Analysis
Key log messages to monitor:
- `"Clustering system failed, using fallback"` - Algorithm failures
- `"Tree validation failed"` - Invalid genealogical structures
- `"Distance calculation error"` - Model weight loading issues

## Future Enhancements

### Planned Features
1. **Incremental Clustering**: Update families without full recalculation
2. **Advanced LoRA Detection**: Better automatic LoRA model identification
3. **Multi-Modal Support**: Support for different model architectures
4. **Clustering Validation**: Automatic quality assessment of clusters
5. **Performance Optimization**: Faster distance calculations for large datasets

### Extension Points
The modular architecture allows easy extension:
- Add new distance metrics by extending `ModelDistanceCalculator`
- Implement new clustering algorithms in `FamilyClusteringSystem`
- Add tree building methods to `MoTHerTreeBuilder`
- Extend pipeline functionality in `ModelManagementSystem`

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'sklearn'**
```bash
pip install scikit-learn>=1.0.0
```

**2. Model weights loading fails**
- Check file permissions and paths
- Verify model file format (safetensors, .pt, .bin supported)
- Check available disk space

**3. Tree building produces invalid trees**
- Verify model weights are properly loaded
- Check for corrupted distance matrices
- Ensure minimum 2 models per family

**4. Poor clustering quality**
- Adjust `family_threshold` parameter
- Try different distance metrics
- Check for outlier models affecting centroids

### Recovery Procedures

**Reset all clustering data:**
```python
# Clear all family assignments
Model.query.update({'family_id': None, 'parent_id': None})
Family.query.delete()
db.session.commit()

# Trigger complete reclustering
mgmt.recluster_all_models()
```

**Rebuild specific family:**
```python
mgmt.rebuild_family_tree(family_id)
```

## Dependencies

### Required Packages
- `numpy>=1.21.0` - Numerical computations
- `scipy>=1.7.0` - Scientific computing
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `networkx>=2.6` - Graph operations
- `torch>=1.9.0` - PyTorch tensors
- `pandas>=1.3.0` - Data manipulation

### Optional Packages
- `safetensors>=0.3.0` - Safe tensor serialization
- `transformers>=4.20.0` - Hugging Face models support

## License and Attribution

This clustering system builds upon the MoTHer algorithm implementation and integrates with the existing Model Heritage Backend architecture. It maintains compatibility with all existing functionality while providing enhanced clustering capabilities.