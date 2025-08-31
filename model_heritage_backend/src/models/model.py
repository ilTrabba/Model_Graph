"""
Model classes for Neo4j-only architecture.
These classes provide the same interface as the previous SQLAlchemy models
but work directly with Neo4j through the neo4j_service.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
import json


class ModelProxy:
    """Proxy class that provides SQLAlchemy-like interface for Neo4j models"""
    
    def __init__(self, model_data: Dict[str, Any]):
        """Initialize from Neo4j model data"""
        for key, value in model_data.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result
    
    def get_lineage(self) -> Dict[str, Any]:
        """Get parent and children for lineage display"""
        from ..services.neo4j_service import neo4j_service
        return neo4j_service.get_model_lineage(self.id)


class FamilyProxy:
    """Proxy class that provides SQLAlchemy-like interface for Neo4j families"""
    
    def __init__(self, family_data: Dict[str, Any]):
        """Initialize from Neo4j family data"""
        for key, value in family_data.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result


class ModelManager:
    """Manager class that provides SQLAlchemy-like query interface for models"""
    
    def __init__(self):
        from ..services.neo4j_service import neo4j_service
        self.neo4j_service = neo4j_service
    
    def get(self, model_id: str) -> Optional[ModelProxy]:
        """Get model by ID, return None if not found"""
        data = self.neo4j_service.get_model_by_id(model_id)
        if data:
            return ModelProxy(data)
        return None
    
    def get_or_404(self, model_id: str) -> ModelProxy:
        """Get model by ID, raise 404 if not found"""
        model = self.get(model_id)
        if model is None:
            from flask import abort
            abort(404)
        return model
    
    def filter_by(self, **kwargs) -> 'ModelQuery':
        """Filter models by attributes"""
        return ModelQuery(self.neo4j_service, **kwargs)
    
    def filter(self, condition) -> 'ModelQuery':
        """Filter models by condition"""
        return ModelQuery(self.neo4j_service, condition=condition)
    
    def order_by(self, field) -> 'ModelQuery':
        """Order models by field"""
        return ModelQuery(self.neo4j_service, order_by=field)
    
    def count(self) -> int:
        """Get total count of models"""
        stats = self.neo4j_service.get_stats()
        return stats.get('total_models', 0)
    
    def all(self) -> List[ModelProxy]:
        """Get all models"""
        data_list = self.neo4j_service.get_all_models()
        return [ModelProxy(data) for data in data_list]


class ModelQuery:
    """Query builder class that mimics SQLAlchemy query interface"""
    
    def __init__(self, neo4j_service, **filters):
        self.neo4j_service = neo4j_service
        self.filters = filters
        self.condition = filters.pop('condition', None)
        self.order_field = filters.pop('order_by', None)
    
    def filter_by(self, **kwargs) -> 'ModelQuery':
        """Add additional filters"""
        self.filters.update(kwargs)
        return self
    
    def filter(self, condition) -> 'ModelQuery':
        """Add filter condition"""
        self.condition = condition
        return self
    
    def order_by(self, field) -> 'ModelQuery':
        """Set order field"""
        self.order_field = field
        return self
    
    def first(self) -> Optional[ModelProxy]:
        """Get first result"""
        results = self._execute()
        return results[0] if results else None
    
    def all(self) -> List[ModelProxy]:
        """Get all results"""
        return self._execute()
    
    def count(self) -> int:
        """Get count of results"""
        return len(self._execute())
    
    def _execute(self) -> List[ModelProxy]:
        """Execute the query and return results"""
        # Handle special cases based on filters
        if 'checksum' in self.filters:
            data = self.neo4j_service.get_model_by_checksum(self.filters['checksum'])
            return [ModelProxy(data)] if data else []
        
        if 'family_id' in self.filters and 'status' in self.filters:
            data_list = self.neo4j_service.get_models_by_family_and_status(
                self.filters['family_id'], 
                self.filters['status']
            )
            return [ModelProxy(data) for data in data_list]
        
        if 'family_id' in self.filters:
            data_list = self.neo4j_service.get_family_models(self.filters['family_id'])
            return [ModelProxy(data) for data in data_list]
        
        if 'status' in self.filters:
            # Filter all models by status
            all_models = self.neo4j_service.get_all_models()
            filtered = [data for data in all_models if data.get('status') == self.filters['status']]
            return [ModelProxy(data) for data in filtered]
        
        # Handle search condition for name contains
        search_query = None
        if self.condition:
            # Try to extract search term from condition
            # This is a simplified approach for Model.name.contains(search)
            if hasattr(self.condition, '__str__'):
                condition_str = str(self.condition)
                if 'contains' in condition_str.lower():
                    # Extract the search term (this is a hack for the specific case)
                    import re
                    match = re.search(r"contains\('([^']+)'\)", condition_str)
                    if match:
                        search_query = match.group(1)
        
        # Get all models with optional search
        data_list = self.neo4j_service.get_all_models(search=search_query)
        return [ModelProxy(data) for data in data_list]


class FamilyManager:
    """Manager class that provides SQLAlchemy-like query interface for families"""
    
    def __init__(self):
        from ..services.neo4j_service import neo4j_service
        self.neo4j_service = neo4j_service
    
    def get(self, family_id: str) -> Optional[FamilyProxy]:
        """Get family by ID, return None if not found"""
        data = self.neo4j_service.get_family_by_id(family_id)
        if data:
            return FamilyProxy(data)
        return None
    
    def get_or_404(self, family_id: str) -> FamilyProxy:
        """Get family by ID, raise 404 if not found"""
        family = self.get(family_id)
        if family is None:
            from flask import abort
            abort(404)
        return family
    
    def filter_by(self, **kwargs) -> 'FamilyQuery':
        """Filter families by attributes"""
        return FamilyQuery(self.neo4j_service, **kwargs)
    
    def order_by(self, field) -> 'FamilyQuery':
        """Order families by field"""
        return FamilyQuery(self.neo4j_service, order_by=field)
    
    def count(self) -> int:
        """Get total count of families"""
        stats = self.neo4j_service.get_stats()
        return stats.get('total_families', 0)
    
    def all(self) -> List[FamilyProxy]:
        """Get all families"""
        data_list = self.neo4j_service.get_all_families()
        return [FamilyProxy(data) for data in data_list]


class FamilyQuery:
    """Query builder class for families"""
    
    def __init__(self, neo4j_service, **filters):
        self.neo4j_service = neo4j_service
        self.filters = filters
        self.order_field = filters.pop('order_by', None)
    
    def filter_by(self, **kwargs) -> 'FamilyQuery':
        """Add additional filters"""
        self.filters.update(kwargs)
        return self
    
    def order_by(self, field) -> 'FamilyQuery':
        """Set order field"""
        self.order_field = field
        return self
    
    def first(self) -> Optional[FamilyProxy]:
        """Get first result"""
        results = self._execute()
        return results[0] if results else None
    
    def all(self) -> List[FamilyProxy]:
        """Get all results"""
        return self._execute()
    
    def _execute(self) -> List[FamilyProxy]:
        """Execute the query and return results"""
        if 'id' in self.filters:
            data = self.neo4j_service.get_family_by_id(self.filters['id'])
            return [FamilyProxy(data)] if data else []
        
        # Default: get all families
        data_list = self.neo4j_service.get_all_families()
        return [FamilyProxy(data) for data in data_list]


# Create manager instances that mimic SQLAlchemy's Model.query interface
Model = type('Model', (), {
    'query': ModelManager()
})

Family = type('Family', (), {
    'query': FamilyManager()
})