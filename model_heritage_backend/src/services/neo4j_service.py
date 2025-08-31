from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging
from ..config import Config

logger = logging.getLogger(__name__)

class Neo4jService:
    """Service class for Neo4j graph operations"""
    
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                Config.NEO4J_URI,
                auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
            )
            # Test connection
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
    
    def is_connected(self) -> bool:
        """Check if Neo4j is connected and available"""
        if not self.driver:
            return False
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False
    
    def create_constraints(self):
        """Create Neo4j constraints and indexes"""
        if not self.driver:
            return False
        
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Family) REQUIRE f.id IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:FamilyCentroid) REQUIRE c.id IS UNIQUE"
        ]
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                for constraint in constraints:
                    session.run(constraint)
            logger.info("Neo4j constraints created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Neo4j constraints: {e}")
            return False
    
    def clear_all_data(self):
        """Clear all data from Neo4j (for testing/reset)"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j database cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Neo4j data: {e}")
            return False
    
    def create_or_update_model(self, model_data: Dict[str, Any], family_color: str) -> bool:
        """Create or update a Model node"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (m:Model {id: $id})
                SET m.name = $name,
                    m.weights_size_MB = $weights_size_MB,
                    m.upload_date = $upload_date,
                    m.embedding = $embedding,
                    m.total_parameters = $total_parameters,
                    m.layer_count = $layer_count,
                    m.structural_hash = $structural_hash,
                    m.status = $status,
                    m.color = $color,
                    m.family_id = $family_id,
                    m.weights_uri=$weights_uri
                RETURN m
                """
                
                # Calculate file size in MB (rough estimate)
                weights_size_mb = (model_data.get('total_parameters', 0) * 4) / (1024 * 1024)  # float32 assumption
                
                result = session.run(query, {
                    'id': model_data['id'],
                    'name': model_data['name'],
                    'weights_size_MB': weights_size_mb,
                    'upload_date': model_data.get('created_at', ''),
                    'embedding': [0.0],  # Placeholder embedding
                    'total_parameters': model_data.get('total_parameters', 0),
                    'layer_count': model_data.get('layer_count', 0),
                    'structural_hash': model_data.get('structural_hash', ''),
                    'status': model_data.get('status', 'processing'),
                    'color': family_color,
                    'family_id': model_data.get('family_id'),
                    'weights_uri': model_data.get('weights_uri')
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create/update model node: {e}")
            return False
    
    def create_or_update_family(self, family_data: Dict[str, Any]) -> bool:
        """Create or update a Family node (always black color)"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (f:Family {id: $id})
                SET f.name = $name,
                    f.created_at = $created_at,
                    f.member_count = $member_count,
                    f.color = 'black'
                RETURN f
                """
                
                session.run(query, {
                    'id': family_data['id'],
                    'name': family_data['id'],  # Use ID as name for now
                    'created_at': family_data.get('created_at', ''),
                    'member_count': family_data.get('member_count', 0)
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create/update family node: {e}")
            return False
    
    def create_or_update_family_centroid(self, family_id: str) -> bool:
        """Create or update a FamilyCentroid node (always white color)"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (c:FamilyCentroid {id: $id})
                SET c.family_id = $family_id,
                    c.embedding = $embedding,
                    c.color = 'white'
                RETURN c
                """
                
                centroid_id = f"centroid_{family_id}"
                session.run(query, {
                    'id': centroid_id,
                    'family_id': family_id,
                    'embedding': [0.0]  # Placeholder embedding
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create/update family centroid: {e}")
            return False
    
    def create_belongs_to_relationship(self, model_id: str, family_id: str) -> bool:
        """Create BELONGS_TO relationship between Model and Family"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $model_id})
                MATCH (f:Family {id: $family_id})
                MERGE (m)-[:BELONGS_TO]->(f)
                """
                
                session.run(query, {
                    'model_id': model_id,
                    'family_id': family_id
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create BELONGS_TO relationship: {e}")
            return False
    
    def create_has_centroid_relationship(self, family_id: str) -> bool:
        """Create HAS_CENTROID relationship between Family and FamilyCentroid"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family {id: $family_id})
                MATCH (c:FamilyCentroid {family_id: $family_id})
                MERGE (f)-[:HAS_CENTROID]->(c)
                """
                
                session.run(query, {'family_id': family_id})
                
            return True
        except Exception as e:
            logger.error(f"Failed to create HAS_CENTROID relationship: {e}")
            return False
    
    def create_parent_child_relationship(self, parent_id: str, child_id: str, confidence: float = 0.0) -> bool:
        """Create IS_CHILD_OF relationship between Models"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (parent:Model {id: $parent_id})
                MATCH (child:Model {id: $child_id})
                MERGE (child)-[:IS_CHILD_OF {confidence: $confidence}]->(parent)
                """
                
                session.run(query, {
                    'parent_id': parent_id,
                    'child_id': child_id,
                    'confidence': confidence
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create IS_CHILD_OF relationship: {e}")
            return False
    
    def get_full_graph(self) -> Dict[str, Any]:
        """Get complete graph data for visualization"""
        if not self.driver:
            return {'nodes': [], 'edges': [], 'error': 'Neo4j not connected'}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get all nodes
                nodes_query = """
                MATCH (n)
                RETURN 
                    id(n) as neo_id,
                    labels(n) as labels,
                    properties(n) as props
                """
                
                # Get all relationships
                edges_query = """
                MATCH (a)-[r]->(b)
                RETURN 
                    id(a) as source_neo_id,
                    id(b) as target_neo_id,
                    type(r) as relationship_type,
                    properties(r) as props,
                    a.id as source_id,
                    b.id as target_id
                """
                
                nodes_result = session.run(nodes_query)
                edges_result = session.run(edges_query)
                
                nodes = []
                for record in nodes_result:
                    node = {
                        'id': record['props'].get('id', f"node_{record['neo_id']}"),
                        'neo_id': record['neo_id'],
                        'label': record['labels'][0] if record['labels'] else 'Unknown',
                        'color': record['props'].get('color', 'gray'),
                        **record['props']
                    }
                    nodes.append(node)
                
                edges = []
                for record in edges_result:
                    edge = {
                        'source': record['source_id'],
                        'target': record['target_id'],
                        'type': record['relationship_type'],
                        **record['props']
                    }
                    edges.append(edge)
                
                return {
                    'nodes': nodes,
                    'edges': edges,
                    'node_count': len(nodes),
                    'edge_count': len(edges)
                }
                
        except Exception as e:
            logger.error(f"Failed to get full graph: {e}")
            return {'nodes': [], 'edges': [], 'error': str(e)}
    
    def get_family_subgraph(self, family_id: str) -> Dict[str, Any]:
        """Get subgraph for a specific family"""
        if not self.driver:
            return {'nodes': [], 'edges': [], 'error': 'Neo4j not connected'}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get family, its models, and centroid
                query = """
                MATCH (f:Family {id: $family_id})
                OPTIONAL MATCH (m:Model)-[:BELONGS_TO]->(f)
                OPTIONAL MATCH (f)-[:HAS_CENTROID]->(c:FamilyCentroid)
                OPTIONAL MATCH (m1:Model)-[r:IS_CHILD_OF]->(m2:Model)
                WHERE m1.family_id = $family_id AND m2.family_id = $family_id
                
                WITH f, collect(DISTINCT m) as models, c, collect(DISTINCT {source: m1, target: m2, type: type(r), props: properties(r)}) as relationships
                
                RETURN f, models, c, relationships
                """
                
                result = session.run(query, {'family_id': family_id})
                record = result.single()
                
                if not record:
                    return {'nodes': [], 'edges': [], 'error': 'Family not found'}
                
                nodes = []
                edges = []
                
                # Add family node
                family = record['f']
                if family:
                    nodes.append({
                        'id': family['id'],
                        'label': 'Family',
                        'color': 'black',
                        **dict(family)
                    })
                
                # Add centroid node
                centroid = record['c']
                if centroid:
                    nodes.append({
                        'id': centroid['id'],
                        'label': 'FamilyCentroid',
                        'color': 'white',
                        **dict(centroid)
                    })
                    
                    # Add family -> centroid edge
                    edges.append({
                        'source': family_id,
                        'target': centroid['id'],
                        'type': 'HAS_CENTROID'
                    })
                
                # Add model nodes and BELONGS_TO edges
                for model in record['models'] or []:
                    if model:
                        nodes.append({
                            'id': model['id'],
                            'label': 'Model',
                            **dict(model)
                        })
                        
                        # Add model -> family edge
                        edges.append({
                            'source': model['id'],
                            'target': family_id,
                            'type': 'BELONGS_TO'
                        })
                
                # Add parent-child relationships
                for rel in record['relationships'] or []:
                    if rel['source'] and rel['target']:
                        edges.append({
                            'source': rel['source']['id'],
                            'target': rel['target']['id'],
                            'type': rel['type'],
                            **rel['props']
                        })
                
                return {
                    'family_id': family_id,
                    'nodes': nodes,
                    'edges': edges,
                    'node_count': len(nodes),
                    'edge_count': len(edges)
                }
                
        except Exception as e:
            logger.error(f"Failed to get family subgraph: {e}")
            return {'nodes': [], 'edges': [], 'error': str(e)}

    # ===========================================
    # CRUD Operations for Model-only Architecture
    # ===========================================
    
    def create_model(self, model_data: Dict[str, Any]) -> bool:
        """Create a new Model node with all required properties"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                CREATE (m:Model {
                    id: $id,
                    name: $name,
                    description: $description,
                    file_path: $file_path,
                    checksum: $checksum,
                    weights_uri: $weights_uri,
                    total_parameters: $total_parameters,
                    layer_count: $layer_count,
                    structural_hash: $structural_hash,
                    family_id: $family_id,
                    parent_id: $parent_id,
                    confidence_score: $confidence_score,
                    status: $status,
                    created_at: $created_at,
                    processed_at: $processed_at,
                    weights_size_MB: $weights_size_MB,
                    embedding: $embedding,
                    color: $color
                })
                RETURN m
                """
                
                # Calculate file size in MB (rough estimate)
                weights_size_mb = (model_data.get('total_parameters', 0) * 4) / (1024 * 1024)
                
                session.run(query, {
                    'id': model_data['id'],
                    'name': model_data['name'],
                    'description': model_data.get('description', ''),
                    'file_path': model_data['file_path'],
                    'checksum': model_data['checksum'],
                    'weights_uri': model_data.get('weights_uri', ''),
                    'total_parameters': model_data.get('total_parameters', 0),
                    'layer_count': model_data.get('layer_count', 0),
                    'structural_hash': model_data.get('structural_hash', ''),
                    'family_id': model_data.get('family_id'),
                    'parent_id': model_data.get('parent_id'),
                    'confidence_score': model_data.get('confidence_score'),
                    'status': model_data.get('status', 'processing'),
                    'created_at': model_data.get('created_at', ''),
                    'processed_at': model_data.get('processed_at'),
                    'weights_size_MB': weights_size_mb,
                    'embedding': [0.0],  # Placeholder
                    'color': model_data.get('color', 'gray')
                })
                
                # Create relationships if family_id is provided
                if model_data.get('family_id'):
                    self.create_belongs_to_relationship(model_data['id'], model_data['family_id'])
                
                # Create parent-child relationship if parent_id is provided
                if model_data.get('parent_id'):
                    self.create_parent_child_relationship(
                        model_data['parent_id'], 
                        model_data['id'], 
                        model_data.get('confidence_score', 0.0)
                    )
                
            return True
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return False
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a model by its ID"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $model_id})
                RETURN properties(m) as model
                """
                
                result = session.run(query, {'model_id': model_id})
                record = result.single()
                
                if record:
                    return dict(record['model'])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    def get_model_by_checksum(self, checksum: str) -> Optional[Dict[str, Any]]:
        """Get a model by its checksum (for duplicate detection)"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {checksum: $checksum})
                RETURN properties(m) as model
                """
                
                result = session.run(query, {'checksum': checksum})
                record = result.single()
                
                if record:
                    return dict(record['model'])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model by checksum: {e}")
            return None
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update a model with the provided data"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Build dynamic SET clause
                set_clauses = []
                params = {'model_id': model_id}
                
                for key, value in updates.items():
                    if key != 'id':  # Don't allow ID updates
                        set_clauses.append(f"m.{key} = ${key}")
                        params[key] = value
                
                if not set_clauses:
                    return True  # Nothing to update
                
                query = f"""
                MATCH (m:Model {{id: $model_id}})
                SET {', '.join(set_clauses)}
                RETURN m
                """
                
                result = session.run(query, params)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to update model {model_id}: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its relationships"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $model_id})
                DETACH DELETE m
                """
                
                session.run(query, {'model_id': model_id})
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def get_all_models(self, search: str = None) -> List[Dict[str, Any]]:
        """Get all models with optional search filter"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                if search:
                    query = """
                    MATCH (m:Model)
                    WHERE toLower(m.name) CONTAINS toLower($search)
                    RETURN properties(m) as model
                    ORDER BY m.name
                    """
                    params = {'search': search}
                else:
                    query = """
                    MATCH (m:Model)
                    RETURN properties(m) as model
                    ORDER BY m.name
                    """
                    params = {}
                
                result = session.run(query, params)
                return [dict(record['model']) for record in result]
                
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get parent and children for a model"""
        if not self.driver:
            return {'parent': None, 'children': []}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $model_id})
                OPTIONAL MATCH (m)-[:IS_CHILD_OF]->(parent:Model)
                OPTIONAL MATCH (child:Model)-[:IS_CHILD_OF]->(m)
                RETURN 
                    properties(parent) as parent,
                    collect(properties(child)) as children
                """
                
                result = session.run(query, {'model_id': model_id})
                record = result.single()
                
                if record:
                    return {
                        'parent': dict(record['parent']) if record['parent'] else None,
                        'children': [dict(child) for child in record['children'] if child]
                    }
                return {'parent': None, 'children': []}
                
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {'parent': None, 'children': []}
    
    def create_family(self, family_data: Dict[str, Any]) -> bool:
        """Create a new Family node"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                CREATE (f:Family {
                    id: $id,
                    structural_pattern_hash: $structural_pattern_hash,
                    member_count: $member_count,
                    avg_intra_distance: $avg_intra_distance,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    color: 'black'
                })
                RETURN f
                """
                
                session.run(query, {
                    'id': family_data['id'],
                    'structural_pattern_hash': family_data.get('structural_pattern_hash', ''),
                    'member_count': family_data.get('member_count', 0),
                    'avg_intra_distance': family_data.get('avg_intra_distance'),
                    'created_at': family_data.get('created_at', ''),
                    'updated_at': family_data.get('updated_at', '')
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to create family: {e}")
            return False
    
    def get_family_by_id(self, family_id: str) -> Optional[Dict[str, Any]]:
        """Get a family by its ID"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family {id: $family_id})
                RETURN properties(f) as family
                """
                
                result = session.run(query, {'family_id': family_id})
                record = result.single()
                
                if record:
                    return dict(record['family'])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get family {family_id}: {e}")
            return None
    
    def get_all_families(self) -> List[Dict[str, Any]]:
        """Get all families"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family)
                RETURN properties(f) as family
                ORDER BY f.created_at DESC
                """
                
                result = session.run(query)
                return [dict(record['family']) for record in result]
                
        except Exception as e:
            logger.error(f"Failed to get families: {e}")
            return []
    
    def update_family(self, family_id: str, updates: Dict[str, Any]) -> bool:
        """Update a family with the provided data"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Build dynamic SET clause
                set_clauses = []
                params = {'family_id': family_id}
                
                for key, value in updates.items():
                    if key != 'id':  # Don't allow ID updates
                        set_clauses.append(f"f.{key} = ${key}")
                        params[key] = value
                
                if not set_clauses:
                    return True  # Nothing to update
                
                query = f"""
                MATCH (f:Family {{id: $family_id}})
                SET {', '.join(set_clauses)}
                RETURN f
                """
                
                result = session.run(query, params)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to update family {family_id}: {e}")
            return False
    
    def get_family_models(self, family_id: str) -> List[Dict[str, Any]]:
        """Get all models in a family"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model)-[:BELONGS_TO]->(f:Family {id: $family_id})
                RETURN properties(m) as model
                ORDER BY m.created_at
                """
                
                result = session.run(query, {'family_id': family_id})
                return [dict(record['model']) for record in result]
                
        except Exception as e:
            logger.error(f"Failed to get family models: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.driver:
            return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                OPTIONAL MATCH (m:Model)
                WITH count(m) as total_models, 
                     sum(CASE WHEN m.status = 'processing' THEN 1 ELSE 0 END) as processing_models
                OPTIONAL MATCH (f:Family)
                RETURN total_models, count(f) as total_families, processing_models
                """
                
                result = session.run(query)
                record = result.single()
                
                if record:
                    return {
                        'total_models': record['total_models'] or 0,
                        'total_families': record['total_families'] or 0,
                        'processing_models': record['processing_models'] or 0
                    }
                return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
                
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
    
    def get_models_by_family_and_status(self, family_id: str, status: str) -> List[Dict[str, Any]]:
        """Get models by family ID and status (for similarity matching)"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model)
                WHERE m.family_id = $family_id AND m.status = $status
                RETURN properties(m) as model
                """
                
                result = session.run(query, {'family_id': family_id, 'status': status})
                return [dict(record['model']) for record in result]
                
        except Exception as e:
            logger.error(f"Failed to get models by family and status: {e}")
            return []
    
    def get_graph_status(self) -> Dict[str, Any]:
        """Get graph database status with node and edge counts"""
        if not self.driver:
            return {
                'neo4j_connected': False,
                'neo4j_nodes': 0,
                'neo4j_edges': 0,
                'error': 'Neo4j not connected'
            }
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get node count
                nodes_query = "MATCH (n) RETURN count(n) as node_count"
                nodes_result = session.run(nodes_query)
                node_count = nodes_result.single()['node_count']
                
                # Get edge count
                edges_query = "MATCH ()-[r]->() RETURN count(r) as edge_count"
                edges_result = session.run(edges_query)
                edge_count = edges_result.single()['edge_count']
                
                return {
                    'neo4j_connected': True,
                    'neo4j_nodes': node_count,
                    'neo4j_edges': edge_count
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph status: {e}")
            return {
                'neo4j_connected': False,
                'neo4j_nodes': 0,
                'neo4j_edges': 0,
                'error': str(e)
            }


# Global instance
neo4j_service = Neo4jService()