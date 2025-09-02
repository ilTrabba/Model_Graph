"""
ModelManagementSystem

This module provides complete pipeline integration for model clustering and heritage detection.
It coordinates all clustering components and manages the complete workflow from model addition
to family assignment and tree reconstruction.
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

from src.models.model import Model, Family, db
from .distance_calculator import ModelDistanceCalculator, DistanceMetric
from .family_clustering import FamilyClusteringSystem, ClusteringMethod
from .tree_builder import MoTHerTreeBuilder, TreeBuildingMethod

logger = logging.getLogger(__name__)

class ModelManagementSystem:
    """
    Complete pipeline integration for model clustering and heritage detection.
    
    Features:
    - Coordinate all clustering components
    - Manage model and family databases  
    - Provide genealogy queries
    - Handle the complete workflow from upload to heritage detection
    """
    
    def __init__(self,
                 distance_metric: DistanceMetric = DistanceMetric.AUTO,
                 family_threshold: float = 0.5,
                 clustering_method: ClusteringMethod = ClusteringMethod.THRESHOLD,
                 tree_method: TreeBuildingMethod = TreeBuildingMethod.MOTHER,
                 lambda_param: float = 0.5):
        """
        Initialize the model management system.
        
        Args:
            distance_metric: Default distance metric for calculations
            family_threshold: Threshold for family assignment
            clustering_method: Method for family clustering
            tree_method: Method for tree building
            lambda_param: Balance parameter for MoTHer algorithm
        """
        # Initialize components
        self.distance_calculator = ModelDistanceCalculator(default_metric=distance_metric)
        self.family_clustering = FamilyClusteringSystem(
            distance_calculator=self.distance_calculator,
            family_threshold=family_threshold,
            clustering_method=clustering_method
        )
        self.tree_builder = MoTHerTreeBuilder(
            distance_calculator=self.distance_calculator,
            lambda_param=lambda_param,
            method=tree_method
        )
        
        logger.info("Initialized ModelManagementSystem with components")
    
    def process_new_model(self, model: Model) -> Dict[str, Any]:
        """
        Complete processing pipeline for a new model.
        
        This includes:
        1. Family assignment (existing or new)
        2. Parent finding within family
        3. Tree reconstruction for family
        4. Database updates
        
        Args:
            model: Model to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting complete processing for model {model.id}")
            
            # Step 1: Assign to family
            family_id, family_confidence = self.family_clustering.assign_model_to_family(model)
            
            # Update model with family assignment
            model.family_id = family_id
            db.session.commit()
            
            logger.info(f"Assigned model {model.id} to family {family_id} with confidence {family_confidence:.3f}")
            
            # Step 2: Find parent within family
            parent_id, parent_confidence = self.find_model_parent(model, family_id)
            
            # Update model with parent assignment
            if parent_id:
                model.parent_id = parent_id
                model.confidence_score = parent_confidence
                db.session.commit()
                logger.info(f"Found parent {parent_id} for model {model.id} with confidence {parent_confidence:.3f}")
            else:
                model.parent_id = None
                model.confidence_score = 0.0
                db.session.commit()
                logger.info(f"Model {model.id} assigned as root in family {family_id}")
            
            # Step 3: Update family statistics
            self.family_clustering.update_family_statistics(family_id)
            
            # Step 4: Mark as processed
            model.status = 'ok'
            model.processed_at = datetime.now(timezone.utc)
            db.session.commit()
            
            # Step 5: Get family tree for context
            family_tree, tree_confidence = self.tree_builder.build_family_tree(family_id)
            
            return {
                'model_id': model.id,
                'family_id': family_id,
                'family_confidence': family_confidence,
                'parent_id': parent_id,
                'parent_confidence': parent_confidence,
                'tree_nodes': family_tree.number_of_nodes(),
                'tree_edges': family_tree.number_of_edges(),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing model {model.id}: {e}")
            
            # Mark model as error state
            model.status = 'error'
            model.processed_at = datetime.now(timezone.utc)
            db.session.commit()
            
            return {
                'model_id': model.id,
                'status': 'error',
                'error': str(e)
            }
    
    def find_model_parent(self, 
                         model: Model, 
                         family_id: Optional[str] = None) -> Tuple[Optional[str], float]:
        """
        Find the most likely parent for a model within its family.
        
        Args:
            model: Model to find parent for
            family_id: Family ID (will use model's family if None)
            
        Returns:
            Tuple of (parent_model_id, confidence_score)
        """
        try:
            target_family_id = family_id or model.family_id
            if not target_family_id:
                logger.warning(f"Model {model.id} has no family assigned")
                return None, 0.0
            
            # Get other models in the family
            family_models = Model.query.filter(
                Model.family_id == target_family_id,
                Model.id != model.id,
                Model.status == 'ok'
            ).all()
            
            if not family_models:
                logger.info(f"Model {model.id} is the only model in family {target_family_id}")
                return None, 0.0
            
            # Use mother algorithm directly for parent finding
            from src.algorithms.mother_algorithm import find_model_parent_mother
            parent_id, confidence = find_model_parent_mother(model, target_family_id)
            
            return parent_id, confidence
            
        except Exception as e:
            logger.error(f"Error finding parent for model {model.id}: {e}")
            return None, 0.0
    
    def rebuild_family_tree(self, family_id: str) -> Dict[str, Any]:
        """
        Rebuild the genealogical tree for a family and update all parent relationships.
        
        Args:
            family_id: ID of the family to rebuild
            
        Returns:
            Dictionary with rebuild results
        """
        try:
            logger.info(f"Rebuilding tree for family {family_id}")
            
            # Get family models
            family_models = Model.query.filter_by(
                family_id=family_id,
                status='ok'
            ).all()
            
            if len(family_models) < 2:
                logger.warning(f"Family {family_id} has insufficient models for tree building")
                return {'status': 'skipped', 'reason': 'insufficient_models'}
            
            # Build new tree
            tree, confidence_scores = self.tree_builder.build_family_tree(family_id, family_models)
            
            if tree.number_of_nodes() == 0:
                logger.error(f"Failed to build tree for family {family_id}")
                return {'status': 'error', 'reason': 'tree_build_failed'}
            
            # Update parent relationships based on tree
            updated_count = 0
            for model in family_models:
                predecessors = list(tree.predecessors(model.id))
                
                if predecessors:
                    # Model has a parent
                    new_parent_id = predecessors[0]
                    new_confidence = confidence_scores.get(model.id, 0.0)
                    
                    if model.parent_id != new_parent_id or abs((model.confidence_score or 0.0) - new_confidence) > 0.01:
                        model.parent_id = new_parent_id
                        model.confidence_score = new_confidence
                        updated_count += 1
                        logger.debug(f"Updated parent for {model.id}: {new_parent_id} (confidence: {new_confidence:.3f})")
                else:
                    # Model is a root
                    if model.parent_id is not None:
                        model.parent_id = None
                        model.confidence_score = 0.0
                        updated_count += 1
                        logger.debug(f"Set {model.id} as root model")
            
            db.session.commit()
            
            # Update family statistics
            self.family_clustering.update_family_statistics(family_id)
            
            # Validate tree
            is_valid, issues = self.tree_builder.validate_tree(tree)
            
            # Get tree statistics
            tree_stats = self.tree_builder.get_tree_statistics(tree)
            
            logger.info(f"Rebuilt tree for family {family_id}: {updated_count} relationships updated")
            
            return {
                'status': 'success',
                'family_id': family_id,
                'models_updated': updated_count,
                'tree_valid': is_valid,
                'tree_issues': issues,
                'tree_statistics': tree_stats
            }
            
        except Exception as e:
            logger.error(f"Error rebuilding tree for family {family_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def recluster_all_models(self) -> Dict[str, Any]:
        """
        Perform complete reclustering of all models in the system.
        
        This will:
        1. Recluster all models into families
        2. Rebuild all family trees
        3. Update all relationships
        
        Returns:
            Dictionary with reclustering results
        """
        try:
            logger.info("Starting complete system reclustering")
            
            # Step 1: Recluster families
            new_families = self.family_clustering.recluster_all_families()
            
            if not new_families:
                logger.warning("Family reclustering returned no results")
                return {'status': 'error', 'reason': 'family_clustering_failed'}
            
            family_count = len(new_families)
            logger.info(f"Reclustering created {family_count} families")
            
            # Step 2: Rebuild trees for all families
            tree_results = []
            for family_id in new_families.keys():
                tree_result = self.rebuild_family_tree(family_id)
                tree_results.append(tree_result)
            
            # Count successes
            successful_trees = sum(1 for result in tree_results if result.get('status') == 'success')
            total_updates = sum(result.get('models_updated', 0) for result in tree_results)
            
            logger.info(f"Reclustering complete: {family_count} families, {successful_trees} trees rebuilt, {total_updates} relationships updated")
            
            return {
                'status': 'success',
                'families_created': family_count,
                'trees_rebuilt': successful_trees,
                'total_relationship_updates': total_updates,
                'family_details': new_families,
                'tree_results': tree_results
            }
            
        except Exception as e:
            logger.error(f"Error during complete reclustering: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_family_genealogy(self, family_id: str) -> Dict[str, Any]:
        """
        Get complete genealogy information for a family.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary with genealogy data
        """
        try:
            # Get family info
            family = Family.query.get(family_id)
            if not family:
                return {'error': 'Family not found'}
            
            # Get family models
            family_models = Model.query.filter_by(
                family_id=family_id,
                status='ok'
            ).all()
            
            if not family_models:
                return {
                    'family': family.to_dict(),
                    'models': [],
                    'tree': {'nodes': [], 'edges': []},
                    'statistics': {}
                }
            
            # Build current tree
            tree, confidence_scores = self.tree_builder.build_family_tree(family_id, family_models)
            
            # Convert tree to serializable format
            tree_data = {
                'nodes': [
                    {
                        'id': node,
                        'confidence': confidence_scores.get(node, 0.0)
                    }
                    for node in tree.nodes()
                ],
                'edges': [
                    {
                        'source': source,
                        'target': target,
                        'data': tree.get_edge_data(source, target) or {}
                    }
                    for source, target in tree.edges()
                ]
            }
            
            # Get tree statistics
            tree_stats = self.tree_builder.get_tree_statistics(tree)
            
            # Validate tree
            is_valid, issues = self.tree_builder.validate_tree(tree)
            
            return {
                'family': family.to_dict(),
                'models': [model.to_dict() for model in family_models],
                'tree': tree_data,
                'statistics': tree_stats,
                'validation': {
                    'is_valid': is_valid,
                    'issues': issues
                },
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error getting family genealogy for {family_id}: {e}")
            return {'error': str(e)}
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get complete lineage information for a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with lineage data
        """
        try:
            # Get model
            model = Model.query.get(model_id)
            if not model:
                return {'error': 'Model not found'}
            
            # Get family genealogy
            if model.family_id:
                family_genealogy = self.get_family_genealogy(model.family_id)
                
                if 'error' not in family_genealogy:
                    # Extract model-specific lineage from family tree
                    tree_data = family_genealogy['tree']
                    
                    # Find ancestors
                    ancestors = []
                    current_id = model_id
                    visited = set()
                    
                    while current_id and current_id not in visited:
                        visited.add(current_id)
                        
                        # Find parent in tree
                        parent_edges = [
                            edge for edge in tree_data['edges'] 
                            if edge['target'] == current_id
                        ]
                        
                        if parent_edges:
                            parent_id = parent_edges[0]['source']
                            parent_model = Model.query.get(parent_id)
                            if parent_model:
                                ancestors.append(parent_model.to_dict())
                            current_id = parent_id
                        else:
                            break
                    
                    # Find descendants
                    descendants = []
                    
                    def find_children(node_id):
                        child_edges = [
                            edge for edge in tree_data['edges']
                            if edge['source'] == node_id
                        ]
                        children = []
                        for edge in child_edges:
                            child_id = edge['target']
                            child_model = Model.query.get(child_id)
                            if child_model:
                                child_dict = child_model.to_dict()
                                child_dict['children'] = find_children(child_id)
                                children.append(child_dict)
                        return children
                    
                    descendants = find_children(model_id)
                    
                    return {
                        'model': model.to_dict(),
                        'family': family_genealogy['family'],
                        'ancestors': ancestors,
                        'descendants': descendants,
                        'confidence_score': family_genealogy['confidence_scores'].get(model_id, 0.0)
                    }
                else:
                    return {
                        'model': model.to_dict(),
                        'family': None,
                        'ancestors': [],
                        'descendants': [],
                        'error': family_genealogy.get('error')
                    }
            else:
                return {
                    'model': model.to_dict(),
                    'family': None,
                    'ancestors': [],
                    'descendants': [],
                    'confidence_score': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error getting model lineage for {model_id}: {e}")
            return {'error': str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the clustering system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Basic counts
            total_models = Model.query.count()
            processed_models = Model.query.filter_by(status='ok').count()
            processing_models = Model.query.filter_by(status='processing').count()
            error_models = Model.query.filter_by(status='error').count()
            total_families = Family.query.count()
            
            # Family statistics
            family_sizes = []
            families = Family.query.all()
            
            for family in families:
                family_model_count = Model.query.filter_by(
                    family_id=family.id,
                    status='ok'
                ).count()
                family_sizes.append(family_model_count)
            
            # Calculate averages
            avg_family_size = np.mean(family_sizes) if family_sizes else 0.0
            max_family_size = max(family_sizes) if family_sizes else 0
            min_family_size = min(family_sizes) if family_sizes else 0
            
            # Parent/child statistics
            models_with_parents = Model.query.filter(
                Model.parent_id.isnot(None),
                Model.status == 'ok'
            ).count()
            
            root_models = Model.query.filter(
                Model.parent_id.is_(None),
                Model.status == 'ok'
            ).count()
            
            return {
                'models': {
                    'total': total_models,
                    'processed': processed_models,
                    'processing': processing_models,
                    'error': error_models,
                    'with_parents': models_with_parents,
                    'roots': root_models
                },
                'families': {
                    'total': total_families,
                    'avg_size': float(avg_family_size),
                    'max_size': max_family_size,
                    'min_size': min_family_size,
                    'size_distribution': family_sizes
                },
                'clustering': {
                    'distance_metric': str(self.distance_calculator.default_metric),
                    'family_threshold': self.family_clustering.family_threshold,
                    'clustering_method': str(self.family_clustering.clustering_method),
                    'tree_method': str(self.tree_builder.method),
                    'lambda_param': self.tree_builder.lambda_param
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {e}")
            return {'error': str(e)}