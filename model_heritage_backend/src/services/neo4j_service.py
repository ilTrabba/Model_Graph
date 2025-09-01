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
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:FamilyCentroid) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Model) REQUIRE m.checksum IS UNIQUE"
        ]
        
        indexes = [
            "CREATE INDEX model_family_idx IF NOT EXISTS FOR (m:Model) ON (m.family_id)",
            "CREATE INDEX model_status_idx IF NOT EXISTS FOR (m:Model) ON (m.status)",
            "CREATE INDEX family_pattern_idx IF NOT EXISTS FOR (f:Family) ON (f.structural_pattern_hash)"
        ]
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                for constraint in constraints:
                    session.run(constraint)
                for index in indexes:
                    session.run(index)
            logger.info("Neo4j constraints and indexes created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create Neo4j constraints and indexes: {e}")
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
    
    def upsert_model(self, model_data: Dict[str, Any]) -> bool:
        """Create or update a Model node using MERGE (unified method)"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (m:Model {id: $id})
                SET m.name = $name,
                    m.description = $description,
                    m.file_path = $file_path,
                    m.checksum = $checksum,
                    m.weights_uri = $weights_uri,
                    m.weights_size_MB = $weights_size_MB,
                    m.upload_date = $upload_date,
                    m.embedding = $embedding,
                    m.total_parameters = $total_parameters,
                    m.layer_count = $layer_count,
                    m.structural_hash = $structural_hash,
                    m.status = $status,
                    m.family_id = $family_id,
                    m.parent_id = $parent_id,
                    m.confidence_score = $confidence_score,
                    m.created_at = $created_at,
                    m.processed_at = $processed_at,
                    m.color = $color
                RETURN m
                """
                
                # Calculate file size in MB (rough estimate)
                weights_size_mb = (model_data.get('total_parameters', 0) * 4) / (1024 * 1024)  # float32 assumption
                
                session.run(query, {
                    'id': model_data['id'],
                    'name': model_data.get('name', ''),
                    'description': model_data.get('description', ''),
                    'file_path': model_data.get('file_path', ''),
                    'checksum': model_data.get('checksum', ''),
                    'weights_uri': model_data.get('weights_uri', ''),
                    'weights_size_MB': weights_size_mb,
                    'upload_date': model_data.get('created_at', ''),
                    'embedding': [0.0],  # Placeholder embedding
                    'total_parameters': model_data.get('total_parameters', 0),
                    'layer_count': model_data.get('layer_count', 0),
                    'structural_hash': model_data.get('structural_hash', ''),
                    'status': model_data.get('status', 'processing'),
                    'family_id': model_data.get('family_id'),
                    'parent_id': model_data.get('parent_id'),
                    'confidence_score': model_data.get('confidence_score', 0.0),
                    'created_at': model_data.get('created_at', ''),
                    'processed_at': model_data.get('processed_at', ''),
                    'color': model_data.get('color', 'gray')
                })
                
            return True
        except Exception as e:
            logger.error(f"Failed to upsert model node: {e}")
            return False
    
    def create_or_update_model(self, model_data: Dict[str, Any], family_color: str) -> bool:
        """Create or update a Model node (wrapper around upsert_model for compatibility)"""
        model_data = dict(model_data)  # copy to avoid modifying original
        model_data['color'] = family_color
        return self.upsert_model(model_data)
    
    def create_model(self, model_data: Dict[str, Any]) -> bool:
        """Create a Model node (wrapper around upsert_model)"""
        return self.upsert_model(model_data)
    
    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields of a Model node"""
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Build dynamic SET clause
                set_clauses = []
                params = {'id': model_id}
                
                for key, value in updates.items():
                    set_clauses.append(f"m.{key} = ${key}")
                    params[key] = value
                
                if not set_clauses:
                    return True  # Nothing to update
                
                query = f"""
                MATCH (m:Model {{id: $id}})
                SET {', '.join(set_clauses)}
                RETURN m
                """
                
                result = session.run(query, params)
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to update model {model_id}: {e}")
            return False
    
    def get_all_models(self, search: str = None) -> List[Dict[str, Any]]:
        """Get all models with optional search"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                if search:
                    query = """
                    MATCH (m:Model)
                    WHERE toLower(m.name) CONTAINS toLower($search)
                    RETURN m
                    ORDER BY m.name
                    """
                    result = session.run(query, {'search': search})
                else:
                    query = """
                    MATCH (m:Model)
                    RETURN m
                    ORDER BY m.name
                    """
                    result = session.run(query)
                
                models = []
                for record in result:
                    model_props = dict(record['m'])
                    models.append(model_props)
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to get all models: {e}")
            return []
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a single model by ID"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {id: $id})
                RETURN m
                """
                result = session.run(query, {'id': model_id})
                record = result.single()
                
                if record:
                    return dict(record['m'])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    def get_model_by_checksum(self, checksum: str) -> Optional[Dict[str, Any]]:
        """Get a model by checksum (for duplicate detection)"""
        if not self.driver:
            return None
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {checksum: $checksum})
                RETURN m
                """
                result = session.run(query, {'checksum': checksum})
                record = result.single()
                
                if record:
                    return dict(record['m'])
                return None
                
        except Exception as e:
            logger.error(f"Failed to get model by checksum {checksum}: {e}")
            return None
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete lineage (ancestors and descendants) for a model"""
        if not self.driver:
            return {'parent': None, 'children': []}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                # Get parent
                parent_query = """
                MATCH (child:Model {id: $id})-[:IS_CHILD_OF]->(parent:Model)
                RETURN parent
                """
                parent_result = session.run(parent_query, {'id': model_id})
                parent_record = parent_result.single()
                parent = dict(parent_record['parent']) if parent_record else None
                
                # Get children
                children_query = """
                MATCH (parent:Model {id: $id})<-[:IS_CHILD_OF]-(child:Model)
                RETURN child
                ORDER BY child.name
                """
                children_result = session.run(children_query, {'id': model_id})
                children = [dict(record['child']) for record in children_result]
                
                return {
                    'parent': parent,
                    'children': children
                }
                
        except Exception as e:
            logger.error(f"Failed to get lineage for model {model_id}: {e}")
            return {'parent': None, 'children': []}
    
    def create_family(self, family_data: Dict[str, Any]) -> bool:
        """Create a new family"""
        return self.create_or_update_family(family_data)
    
    def create_or_update_family(self, family_data: Dict[str, Any]) -> bool:
        if not self.driver:
            return False
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MERGE (f:Family {id: $id})
                SET f.name = $name,
                    f.created_at = $created_at,
                    f.updated_at = $updated_at,
                    f.member_count = $member_count,
                    f.structural_pattern_hash = $structural_pattern_hash,
                    f.avg_intra_distance = $avg_intra_distance,
                    f.color = 'black'
                RETURN f
                """
                
                session.run(query, {
                    'id': family_data['id'],
                    'name': family_data.get('name', family_data['id']),  # Use ID as name for now
                    'created_at': family_data.get('created_at', ''),
                    'updated_at': family_data.get('updated_at', ''),
                    'member_count': family_data.get('member_count', 0),
                    'structural_pattern_hash': family_data.get('structural_pattern_hash', ''),
                    'avg_intra_distance': family_data.get('avg_intra_distance', 0.0)
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
    
    def get_all_families(self) -> List[Dict[str, Any]]:
        """Get all families"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (f:Family)
                RETURN f
                ORDER BY f.created_at DESC
                """
                result = session.run(query)
                
                families = []
                for record in result:
                    family_props = dict(record['f'])
                    families.append(family_props)
                
                return families
                
        except Exception as e:
            logger.error(f"Failed to get all families: {e}")
            return []
    
    def get_family_models(self, family_id: str) -> List[Dict[str, Any]]:
        """Get all models in a specific family"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                query = """
                MATCH (m:Model {family_id: $family_id})
                RETURN m
                ORDER BY m.created_at
                """
                result = session.run(query, {'family_id': family_id})
                
                models = []
                for record in result:
                    model_props = dict(record['m'])
                    models.append(model_props)
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to get models for family {family_id}: {e}")
            return []
    
    def get_stats(self) -> Dict[str, int]:
        """Get system statistics"""
        if not self.driver:
            return {'total_models': 0, 'total_families': 0, 'processing_models': 0}
        
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                stats_query = """
                MATCH (m:Model)
                OPTIONAL MATCH (f:Family)
                RETURN 
                    count(DISTINCT m) as total_models,
                    count(DISTINCT f) as total_families,
                    count(DISTINCT CASE WHEN m.status = 'processing' THEN m END) as processing_models
                """
                result = session.run(stats_query)
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


# Global instance
neo4j_service = Neo4jService()