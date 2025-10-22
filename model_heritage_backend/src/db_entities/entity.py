"""
Model and Family proxy classes for Neo4j-only architecture.
These classes provide the same interface as the previous SQLAlchemy models
but use Neo4j as the backend through neo4j_service.
"""
import logging

from typing import List, Optional
from ..services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)

class Model:
    """Proxy class for Model operations using Neo4j backend"""
    
    def __init__(self, **kwargs):
        """Initialize model proxy with data"""
        self.data = kwargs
        
    def __getattr__(self, name):
        """Allow attribute access to model properties"""
        return self.data.get(name)
    
    @property
    def id(self):
        return self.data.get('id')
    
    @property
    def name(self):
        return self.data.get('name')
    
    @property
    def family_id(self):
        return self.data.get('family_id')
    
    @property
    def parent_id(self):
        return self.data.get('parent_id')
    
    @property
    def confidence_score(self):
        return self.data.get('confidence_score')
    
    @property
    def status(self):
        return self.data.get('status')
    
    @property
    def checksum(self):
        return self.data.get('checksum')
    
    @property
    def total_parameters(self):
        return self.data.get('total_parameters')
    
    @property
    def layer_count(self):
        return self.data.get('layer_count')
    
    @property
    def structural_hash(self):
        return self.data.get('structural_hash')
    
    @property
    def weights_uri(self):
        return self.data.get('weights_uri')
    
    @property 
    def file_path(self):
        return self.data.get('file_path')
    
    @property
    def description(self):
        return self.data.get('description')
    
    @property
    def created_at(self):
        return self.data.get('created_at')
    
    @property
    def processed_at(self):
        return self.data.get('processed_at')
    
    def to_dict(self):
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'total_parameters': self.total_parameters,
            'layer_count': self.layer_count,
            'structural_hash': self.structural_hash,
            'family_id': self.family_id,
            'parent_id': self.parent_id,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'weights_uri': self.weights_uri,
            'checksum': self.checksum,
            'file_path': self.file_path,
            'created_at': self.created_at,
            'processed_at': self.processed_at
        }
    
    def get_lineage(self):
        """Get parent and children for lineage display"""
        lineage = neo4j_service.get_model_lineage(self.id)
        
        # Convert to proxy objects for compatibility
        if lineage.get('parent'):
            lineage['parent'] = Model(**lineage['parent']).to_dict()
        
        if lineage.get('children'):
            lineage['children'] = [Model(**child).to_dict() for child in lineage['children']]
        
        return lineage

class Family:
    """Proxy class for Family operations using Neo4j backend"""
    
    def __init__(self, **kwargs):
        """Initialize family proxy with data"""
        self.data = kwargs
    
    def __getattr__(self, name):
        """Allow attribute access to family properties"""
        return self.data.get(name)
    
    @property
    def id(self):
        return self.data.get('id')
    
    @property
    def structural_pattern_hash(self):
        return self.data.get('structural_pattern_hash')
    
    @property
    def member_count(self):
        return self.data.get('member_count')
    
    @property
    def avg_intra_distance(self):
        return self.data.get('avg_intra_distance')
    
    @property
    def created_at(self):
        return self.data.get('created_at')
    
    @property
    def updated_at(self):
        return self.data.get('updated_at')
    
    def to_dict(self):
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'structural_pattern_hash': self.structural_pattern_hash,
            'member_count': self.member_count,
            'avg_intra_distance': self.avg_intra_distance,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

class ModelManager:
    """Manager class that provides SQLAlchemy-like query interface for models"""
    
    @staticmethod
    def query():
        """Return a query-like object for models"""
        return ModelQuery()
    
    @staticmethod
    def get(model_id: str) -> Optional[Model]:
        """Get a model by ID"""
        model_data = neo4j_service.get_model_by_id(model_id)
        if model_data:
            return Model(**model_data)
        return None
    
    @staticmethod
    def get_or_404(model_id: str) -> Model:
        """Get a model by ID or raise 404"""
        model = ModelManager.get(model_id)
        if not model:
            from flask import abort
            abort(404)
        return model

class ModelQuery:
    """Query-like interface for models to maintain compatibility"""
    
    def __init__(self):
        self._filters = {}
        self._search = None
    
    def filter_by(self, **kwargs):
        """Add filters (for compatibility)"""
        self._filters.update(kwargs)
        return self
    
    def filter(self, condition):
        """Add filter condition (basic support)"""
        # For simple contains searches
        if hasattr(condition, 'contains'):
            # This is a rough approximation for Model.name.contains(search)
            self._search = condition
        return self
    
    def order_by(self, field):
        """Order by field (handled in execution)"""
        return self
    
    def all(self) -> List[Model]:
        """Get all models matching filters"""
        if 'checksum' in self._filters:
            # Special case for checksum lookup
            model_data = neo4j_service.get_model_by_checksum(self._filters['checksum'])
            return [Model(**model_data)] if model_data else []
        
        if 'family_id' in self._filters:
            # Get models in specific family
            models_data = neo4j_service.get_family_models(self._filters['family_id'])
            models = [Model(**data) for data in models_data]
            
            # Apply additional filters
            if 'status' in self._filters:
                models = [m for m in models if m.status == self._filters['status']]
            
            return models
        
        # Get all models
        search_term = None
        if hasattr(self._search, 'right') and hasattr(self._search.right, 'value'):
            search_term = self._search.right.value
        
        models_data = neo4j_service.get_all_models(search=search_term)
        models = [Model(**data) for data in models_data]
        
        # Apply filters
        if 'status' in self._filters:
            models = [m for m in models if m.status == self._filters['status']]
        
        return models
    
    def first(self) -> Optional[Model]:
        """Get first model matching filters"""
        results = self.all()
        return results[0] if results else None
    
    def count(self) -> int:
        """Count models matching filters"""
        return len(self.all())

class FamilyManager:
    """Manager class that provides SQLAlchemy-like query interface for families"""
    
    @staticmethod
    def query():
        """Return a query-like object for families"""
        return FamilyQuery()
    
    @staticmethod
    def get(family_id: str) -> Optional[Family]:
        """Get a family by ID"""
        families = neo4j_service.get_all_families()
        for family_data in families:
            if family_data.get('id') == family_id:
                return Family(**family_data)
        return None
    
    @staticmethod
    def get_or_404(family_id: str) -> Family:
        """Get a family by ID or raise 404"""
        family = FamilyManager.get(family_id)
        if not family:
            from flask import abort
            abort(404)
        return family

class FamilyQuery:
    """Query-like interface for families to maintain compatibility"""
    
    def __init__(self):
        self._filters = {}
    
    def filter_by(self, **kwargs):
        """Add filters"""
        self._filters.update(kwargs)
        return self
    
    def order_by(self, field):
        """Order by field"""
        return self
    
    def get(self, family_id: str) -> Optional[Family]:
        """Get a single family by its ID."""
        return self.filter_by(id=family_id).first()
    
    def all(self) -> List[Family]:
        """Get all families"""
        families_data = neo4j_service.get_all_families()
        families = [Family(**data) for data in families_data]
        
        # Apply filters, filtro che se verrà usato in futuro, è ampiamente da rivedere
        #if 'id' in self._filters:
            #families = [f for f in families if f.id == self._filters['id']]
        
        return families
    
    def first(self) -> Optional[Family]:
        """Get first family matching filters"""
        results = self.all()
        return results[0] if results else None
    
    def count(self) -> int:
        """Count families"""
        return len(self.all())

