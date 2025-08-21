from src.models.user import db
from datetime import datetime
import json

class Model(db.Model):
    __tablename__ = 'models'
    
    id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(500), nullable=False)
    checksum = db.Column(db.String(64), nullable=False)  # SHA-256
    
    # Weight signature fields
    total_parameters = db.Column(db.Integer)
    layer_count = db.Column(db.Integer)
    structural_hash = db.Column(db.String(64))
    
    # Family and lineage
    family_id = db.Column(db.String(50))
    parent_id = db.Column(db.String(50), db.ForeignKey('models.id'))
    confidence_score = db.Column(db.Float)
    
    # Status and timestamps
    status = db.Column(db.String(20), default='processing')  # processing, ok, error
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    
    # Relationships
    children = db.relationship('Model', backref=db.backref('parent', remote_side=[id]))
    
    def to_dict(self):
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None
        }
    
    def get_lineage(self):
        """Get parent and children for lineage display"""
        lineage = {
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children]
        }
        return lineage

class Family(db.Model):
    __tablename__ = 'families'
    
    id = db.Column(db.String(50), primary_key=True)
    structural_pattern_hash = db.Column(db.String(64))
    member_count = db.Column(db.Integer, default=0)
    avg_intra_distance = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'structural_pattern_hash': self.structural_pattern_hash,
            'member_count': self.member_count,
            'avg_intra_distance': self.avg_intra_distance,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

