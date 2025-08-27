"""Synchronization service between SQLite and Neo4j"""

from .neo4j_service import neo4j_service
from .color_manager import color_palette
from ..models.model import Model, Family
import logging

logger = logging.getLogger(__name__)

class SyncService:
    """Service for synchronizing data between SQLite and Neo4j"""
    
    def __init__(self):
        self.neo4j = neo4j_service
    
    def sync_all_data(self) -> dict:
        """Synchronize all data from SQLite to Neo4j"""
        if not self.neo4j.is_connected():
            return {
                'success': False,
                'error': 'Neo4j not connected',
                'synced_models': 0,
                'synced_families': 0
            }
        
        try:
            # Ensure constraints exist
            self.neo4j.create_constraints()
            
            synced_families = 0
            synced_models = 0
            
            # Sync families first
            families = Family.query.all()
            for family in families:
                if self.sync_family(family):
                    synced_families += 1
            
            # Sync models
            models = Model.query.all()
            for model in models:
                if self.sync_model(model):
                    synced_models += 1
            
            # Sync relationships
            self.sync_all_relationships()
            
            logger.info(f"Synchronized {synced_models} models and {synced_families} families to Neo4j")
            
            return {
                'success': True,
                'synced_models': synced_models,
                'synced_families': synced_families,
                'message': f'Successfully synchronized {synced_models} models and {synced_families} families'
            }
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'synced_models': 0,
                'synced_families': 0
            }
    
    def sync_family(self, family: Family) -> bool:
        """Sync a single family to Neo4j"""
        try:
            family_data = family.to_dict()
            
            # Create/update family node
            if not self.neo4j.create_or_update_family(family_data):
                return False
            
            # Create/update family centroid
            if not self.neo4j.create_or_update_family_centroid(family.id):
                return False
            
            # Create HAS_CENTROID relationship
            if not self.neo4j.create_has_centroid_relationship(family.id):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync family {family.id}: {e}")
            return False
    
    def sync_model(self, model: Model) -> bool:
        """Sync a single model to Neo4j"""
        try:
            model_data = model.to_dict()
            
            # Get family color
            family_color = color_palette.get_family_color(model.family_id) if model.family_id else '#808080'
            
            # Create/update model node
            if not self.neo4j.create_or_update_model(model_data, family_color):
                return False
            
            # Create BELONGS_TO relationship if model has family
            if model.family_id:
                if not self.neo4j.create_belongs_to_relationship(model.id, model.family_id):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync model {model.id}: {e}")
            return False
    
    def sync_all_relationships(self):
        """Sync all parent-child relationships"""
        try:
            # Get all models with parents
            models_with_parents = Model.query.filter(Model.parent_id.isnot(None)).all()
            
            for model in models_with_parents:
                confidence = model.confidence_score or 0.0
                self.neo4j.create_parent_child_relationship(
                    model.parent_id, 
                    model.id, 
                    confidence
                )
            
        except Exception as e:
            logger.error(f"Failed to sync relationships: {e}")
    
    def sync_single_model(self, model_id: str) -> dict:
        """Sync a single model by ID"""
        try:
            model = Model.query.get(model_id)
            if not model:
                return {'success': False, 'error': 'Model not found'}
            
            # Sync model's family first if needed
            if model.family_id:
                family = Family.query.get(model.family_id)
                if family:
                    self.sync_family(family)
            
            # Sync the model
            if self.sync_model(model):
                # Sync parent-child relationship if exists
                if model.parent_id:
                    confidence = model.confidence_score or 0.0
                    self.neo4j.create_parent_child_relationship(
                        model.parent_id, 
                        model.id, 
                        confidence
                    )
                
                return {'success': True, 'message': f'Model {model_id} synchronized'}
            else:
                return {'success': False, 'error': 'Failed to sync model'}
                
        except Exception as e:
            logger.error(f"Failed to sync single model {model_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_sync_status(self) -> dict:
        """Get synchronization status"""
        try:
            neo4j_connected = self.neo4j.is_connected()
            
            if not neo4j_connected:
                return {
                    'neo4j_connected': False,
                    'sqlite_models': Model.query.count(),
                    'sqlite_families': Family.query.count(),
                    'neo4j_nodes': 0,
                    'neo4j_edges': 0
                }
            
            # Get Neo4j stats
            graph_data = self.neo4j.get_full_graph()
            
            return {
                'neo4j_connected': True,
                'sqlite_models': Model.query.count(),
                'sqlite_families': Family.query.count(),
                'neo4j_nodes': graph_data.get('node_count', 0),
                'neo4j_edges': graph_data.get('edge_count', 0),
                'last_sync': 'Not implemented'  # Could store in database
            }
            
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {
                'neo4j_connected': False,
                'error': str(e),
                'sqlite_models': Model.query.count(),
                'sqlite_families': Family.query.count(),
                'neo4j_nodes': 0,
                'neo4j_edges': 0
            }


# Global instance
sync_service = SyncService()