
def normalize_layer_with_stats(tensor, layer_stats):
    """
    Min-max normalization with pre-computed stats
    """
    if layer_stats["range"] > 1e-12:  # Avoid division by zero
        return (tensor - layer_stats["min"]) / layer_stats["range"]
    else:
        return tensor

def compute_lineage_confidence(edges, distance_matrix, family_id):
    """
    Confidence scores for parent-child relationships
    """
    family_max_distance = np.max(distance_matrix)
    confidences = {}
    
    for parent_idx, child_idx, distance in edges:
        # Confidence = 1 - (distance / max_distance_family)
        confidence = 1.0 - (distance / family_max_distance)
        confidence = max(0.0, min(1.0, confidence))  # Clamp [0,1]
        
        confidences[(parent_idx, child_idx)] = confidence
    
    return confidences
```

### **PHASE 4: Neo4j Persistence**

```python
def update_neo4j_family_lineage(family_id, parent_child_edges, confidence_scores, family_members):
    """
    Atomic family lineage graph update
    """
    with neo4j_driver.session() as session:
        with session.begin_transaction() as tx:
            # 1. Remove old PARENT_OF for family
            tx.run("""
                MATCH (m1:Model)-[r:PARENT_OF]->(m2:Model)
                WHERE m1.family_id = $family_id AND m2.family_id = $family_id
                DELETE r
            """, family_id=family_id)
            
            # 2. Insert new parent-child relationships
            for parent_idx, child_idx, distance in parent_child_edges:
                parent_model = family_members[parent_idx]
                child_model = family_members[child_idx]
                confidence = confidence_scores.get((parent_idx, child_idx), 0.0)
                
                tx.run("""
                    MATCH (parent:Model {id: $parent_id})
                    MATCH (child:Model {id: $child_id})
                    CREATE (parent)-[:PARENT_OF {
                        weight_distance: $distance,
                        confidence: $confidence,
                        method: "mother_weight_pure",
                        computed_at: datetime()
                    }]->(child)
                """, 
                parent_id=parent_model["model_id"],
                child_id=child_model["model_id"],
                distance=distance,
                confidence=confidence)
            
            # 3. Update family stats
            avg_distance = np.mean([d for _, _, d in parent_child_edges])
            max_distance = np.max([d for _, _, d in parent_child_edges])
            
            tx.run("""
                MATCH (f:Family {family_id: $family_id})
                SET f.avg_intra_distance = $avg_distance,
                    f.diameter = $max_distance,
                    f.last_centroid_update = datetime()
            """, 
            family_id=family_id,
            avg_distance=avg_distance,
            max_distance=max_distance)
```

---

## 🔌 API Endpoints

```python
@app.post("/models")
async def ingest_model(file: UploadFile):
    """
    Weight-only model ingestion with automatic clustering
    """
    model_id = generate_model_id()
    
    # Validation and SafeTensors conversion
    model_path = await save_uploaded_model(file, model_id)
    
    # Weight signature extraction
    signature = extract_weight_signature_from_file(model_path)
    
    # Enqueue clustering job
    job = assign_model_job.delay(model_id, signature)
    
    return {
        "model_id": model_id,
        "status": "processing",
        "job_id": job.id,
        "signature_preview": {
            "total_parameters": signature["total_parameters"],
            "layer_count": len(signature["layer_shapes"]),
            "structural_hash": signature["structural_hash"]
        }
    }

@app.get("/models/{model_id}/lineage")
async def get_model_lineage(model_id: str):
    """
    Complete model lineage (parent + children + siblings)
    """
    with neo4j_driver.session() as session:
        # Parent
        parent = session.run("""
            MATCH (parent:Model)-[r:PARENT_OF]->(model:Model {id: $model_id})
            RETURN parent, r.confidence as confidence, r.weight_distance as distance
        """, model_id=model_id).single()
        
        # Children
        children = session.run("""
            MATCH (model:Model {id: $model_id})-[r:PARENT_OF]->(child:Model)
            RETURN child, r.confidence as confidence, r.weight_distance as distance
        """, model_id=model_id).data()
        
        # Siblings (same parent)
        siblings = session.run("""
            MATCH (parent:Model)-[:PARENT_OF]->(model:Model {id: $model_id})
            MATCH (parent)-[:PARENT_OF]->(sibling:Model)
            WHERE sibling.id <> $model_id
            RETURN sibling
        """, model_id=model_id).data()
    
    return {
        "model_id": model_id,
        "parent": parent,
        "children": children,
        "siblings": siblings
    }

@app.get("/families/{family_id}/tree")
async def get_family_tree(family_id: str):
    """
    Family tree structure with nodes and edges
    """
    with neo4j_driver.session() as session:
        # All models in family
        models = session.run("""
            MATCH (m:Model)-[:IN_FAMILY]->(f:Family {family_id: $family_id})
            RETURN m
        """, family_id=family_id).data()
        
        # All parent-child relationships in family
        edges = session.run("""
            MATCH (p:Model)-[r:PARENT_OF]->(c:Model)
            WHERE p.family_id = $family_id AND c.family_id = $family_id
            RETURN p.id as parent, c.id as child, r.confidence as confidence, r.weight_distance as distance
        """, family_id=family_id).data()
    
    return {
        "family_id": family_id,
        "nodes": models,
        "edges": edges
    }

@app.post("/families/{family_id}/rebuild-lineage")
async def rebuild_family_lineage(family_id: str):
    """
    Rebuild MoTHer lineage for specific family
    """
    job = mother_lineage_job.delay(family_id)
    
    return {
        "family_id": family_id,
        "job_id": job.id,
        "status": "queued"
    }

@app.post("/clustering/full-rebuild")
async def full_system_rebuild():
    """
    Complete rebuild: clustering + centroids + lineage for all families
    """
    job = full_rebuild_job.delay()
    
    return {
        "job_id": job.id,
        "status": "queued",
        "estimated_duration": "30-60 minutes"
    }

@app.get("/families")
async def list_families():
    """
    List all families with statistics
    """
    with neo4j_driver.session() as session:
        families = session.run("""
            MATCH (f:Family)
            OPTIONAL MATCH (m:Model)-[:IN_FAMILY]->(f)
            RETURN f, count(m) as model_count
        """).data()
    
    return {"families": families}

@app.get("/system/stats")
async def get_system_stats():
    """
    System-wide statistics and health metrics
    """
    with neo4j_driver.session() as session:
        stats = session.run("""
            MATCH (f:Family)
            MATCH (m:Model)
            MATCH (e:Model)-[:PARENT_OF]->(:Model)
            RETURN 
                count(DISTINCT f) as total_families,
                count(DISTINCT m) as total_models,
                count(e) as total_lineage_edges,
                avg(f.avg_intra_distance) as avg_family_cohesion
        """).single()
    
    return {"system_stats": stats}
```

---

## ⚙️ Async Worker Jobs

```python
@job('default', timeout=3600)
def assign_model_job(model_id, signature):
    """
    Model assignment job with 4-step strategy
    """
    try:
        # Load existing families
        existing_families = load_all_families()
        
        # 4-step clustering
        family_id, assignment_confidence = assign_model_to_family(
            model_id, 
            signature, 
            existing_families
        )
        
        # Update Neo4j
        update_model_family_assignment(model_id, family_id, assignment_confidence)
        
        # Trigger lineage rebuild for family
        if get_family_size(family_id) >= 2:
            mother_lineage_job.delay(family_id)
        
        log_clustering_success(model_id, family_id, assignment_confidence)
        
    except Exception as e:
        log_clustering_error(model_id, str(e))
        update_model_status(model_id, "error", str(e))

@job('lineage', timeout=1800)
def mother_lineage_job(family_id):
    """
    Family lineage reconstruction job with MoTHer
    """
    try:
        edges, confidences = mother_lineage_reconstruction(family_id)
        
        # Atomic Neo4j update
        family_members = get_family_members(family_id)
        update_neo4j_family_lineage(family_id, edges, confidences, family_members)
        
        log_lineage_success(family_id, len(edges))
        
    except Exception as e:
        log_lineage_error(family_id, str(e))

@job('maintenance', timeout=7200)
def full_rebuild_job():
    """
    Complete system rebuild: clustering + lineage
    """
    try:
        # 1. Global re-clustering
        all_models = get_all_models()
        new_families = global_weight_clustering(all_models)
        
        # 2. Update family assignments
        update_all_family_assignments(new_families)
        
        # 3. Recompute centroids
        for family_id in new_families.keys():
            recompute_family_centroid(family_id)
        
        # 4. Rebuild lineage for each family
        for family_id in new_families.keys():
            if get_family_size(family_id) >= 2:
                mother_lineage_reconstruction(family_id)
        
        log_full_rebuild_success(len(new_families))
        
    except Exception as e:
        log_full_rebuild_error(str(e))

@job('maintenance', timeout=3600)
def cleanup_cache_job():
    """
    Cache cleanup and storage optimization
    """
    try:
        # Clean old distance cache entries
        cleanup_distance_cache(max_age_days=30)
        
        # Compress and archive old logs
        archive_old_logs(max_age_days=90)
        
        # Vacuum Neo4j database
        optimize_neo4j_database()
        
        log_cleanup_success()
        
    except Exception as e:
        log_cleanup_error(str(e))
```

---

## 🔧 Configuration

```yaml
# config.yaml
storage:
  root_path: "./data"
  models_path: "./data/models"
  families_path: "./data/families"
  cache_path: "./data/cache"
  force_safetensors: true
  max_storage_gb: 200

clustering:
  structural_threshold: 0.80        # Structural compatibility threshold
  uncertainty_threshold: 0.30       # Validation step threshold
  validation_factor: 1.2            # Validation vs diameter factor
  max_family_size: 50               # Model limit per family

weight_analysis:
  layer_sampling_ratio: 1.0         # Full layers for lineage
  normalization_epsilon: 1e-12      # Numerical stability
  distance_metric: "L2"             # Distance metric
  enable_layer_weighting: false     # Uniform layer weights

mother:
  min_family_size: 2                # Minimum for lineage
  confidence_threshold: 0.1         # Minimum confidence threshold
  arborescence_algorithm: "edmonds" # Spanning tree algorithm

performance:
  worker_concurrency: 1             # Single worker to avoid races
  batch_size_layers: 10             # Layer processing batch size
  memory_limit_gb: 12               # Worker RAM limit
  cache_distances: true             # Cache pairwise distances
  mmap_models: true                 # Memory mapping SafeTensors

neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
  database: "neo4j"
  pool_size: 10

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: null

logging:
  level: "INFO"
  format: "json"
  file: "./data/logs/model_genealogy.log"
  max_size_mb: 100
  backup_count: 5

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: false
  log_level: "info"
```

---

## 🐳 Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    container_name: modelgenealogy_api
    ports: ["8000:8000"]
    volumes: 
      - "./data:/app/data"
      - "./config.yaml:/app/config.yaml"
    environment:
      - PYTHONPATH=/app
      - CONFIG_PATH=/app/config.yaml
    depends_on: 
      - redis
      - neo4j
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    
  worker:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    container_name: modelgenealogy_worker
    command: ["python", "-m", "rq", "worker", "default", "lineage", "maintenance"]
    volumes:
      - "./data:/app/data"
      - "./config.yaml:/app/config.yaml"
    environment:
      - PYTHONPATH=/app
      - CONFIG_PATH=/app/config.yaml
    depends_on: 
      - redis
      - neo4j
    deploy:
      replicas: 1  # Single worker to prevent race conditions
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    container_name: modelgenealogy_redis
    ports: ["6379:6379"]
    volumes: ["redis_data:/data"]
    command: ["redis-server", "--appendonly", "yes"]
    restart: unless-stopped
    
  neo4j:
    image: neo4j:5-community
    container_name: modelgenealogy_neo4j
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"
      NEO4J_dbms_memory_heap_initial_size: "2G"
      NEO4J_dbms_memory_heap_max_size: "4G"
    ports: ["7474:7474", "7687:7687"]
    volumes: 
      - "./data/neo4j:/data"
      - "./data/neo4j/logs:/logs"
    restart: unless-stopped
    
  frontend:
    build: ./frontend
    container_name: modelgenealogy_frontend
    ports: ["3000:3000"]
    environment:
      - REACT_APP_API_URL=http://localhost:8000
      - REACT_APP_NEO4J_URL=http://localhost:7474
    depends_on: 
      - api
    restart: unless-stopped

volumes:
  redis_data:
    driver: local

networks:
  default:
    name: modelgenealogy_network
```

---

## 📊 Performance Metrics

```python
# Performance Monitoring
PERFORMANCE_METRICS = {
    "ingest": {
        "signature_extraction_seconds": float,
        "structural_analysis_seconds": float,
        "file_conversion_seconds": float,
        "total_parameters_extracted": int,
        "file_size_mb": float
