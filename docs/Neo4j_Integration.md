# Neo4j Integration for Model Heritage Visualization

This document describes the Neo4j integration implementation for the Model Heritage project.

## Overview

The Neo4j integration provides graph database capabilities for visualizing model heritage and family relationships. The implementation includes:

- Neo4j driver integration with graceful error handling
- Graph data models (Model, Family, FamilyCentroid nodes)
- Automatic synchronization between SQLite and Neo4j
- Color-coded visualization support
- REST API endpoints for graph operations
- Frontend integration with status monitoring

## Architecture

### Services

#### Neo4jService (`src/services/neo4j_service.py`)
- Manages Neo4j database connection and operations
- Creates and manages graph nodes and relationships
- Provides graph data retrieval methods
- Handles connection failures gracefully

#### SyncService (`src/services/sync_service.py`)
- Synchronizes data between SQLite and Neo4j
- Handles incremental and full synchronization
- Provides sync status monitoring

#### ColorManager (`src/services/color_manager.py`)
- Manages color assignments for families
- Ensures unique colors per family (excluding black/white)
- Provides predefined color palette

### Graph Data Model

#### Nodes
- **Model**: Properties include id, name, weights_size_MB, upload_date, embedding, color (family-specific)
- **Family**: Properties include id, name, created_at, color (always black)
- **FamilyCentroid**: Properties include id, family_id, embedding, color (always white)

#### Relationships
- **BELONGS_TO**: Model → Family
- **HAS_CENTROID**: Family → FamilyCentroid  
- **IS_CHILD_OF**: Model → Model (parent-child relationships)

### API Endpoints

#### Graph Status
- `GET /api/graph/status` - Returns Neo4j connection status and sync statistics

#### Graph Data
- `GET /api/graph/full` - Returns complete graph data for visualization
- `GET /api/graph/family/{family_id}` - Returns subgraph for specific family

#### Synchronization
- `POST /api/graph/sync` - Manual synchronization from SQLite to Neo4j
- `POST /api/graph/model/{model_id}/sync` - Sync individual model

#### Utilities
- `POST /api/graph/clear` - Clear all Neo4j data (for testing)

## Configuration

### Environment Variables
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

### Dependencies
```
neo4j==5.24.0
```

## Frontend Integration

### Graph Page (`src/pages/GraphPage.jsx`)
- Displays Neo4j connection status
- Shows graph statistics (nodes/edges count)
- Provides sync controls
- Displays basic graph visualization
- Handles error states gracefully

### Navigation
- Added "Graph" link to main navigation
- Network icon for graph-related features

## Error Handling

The implementation includes comprehensive error handling:

1. **Connection Failures**: Graceful degradation when Neo4j is unavailable
2. **API Errors**: Proper HTTP status codes and error messages
3. **Frontend Errors**: User-friendly error displays and status indicators
4. **Sync Failures**: Detailed error reporting for troubleshooting

## Usage

### Starting Neo4j (Docker)
```bash
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5-community
```

### Synchronization
1. Upload models through the regular interface
2. Navigate to Graph page
3. Click "Sync Data" to populate Neo4j
4. View graph visualization

### Programmatic Usage
```python
from src.services.neo4j_service import neo4j_service
from src.services.sync_service import sync_service

# Check connection
if neo4j_service.is_connected():
    # Sync all data
    result = sync_service.sync_all_data()
    
    # Get graph data
    graph = neo4j_service.get_full_graph()
```

## Color Coding

- **Model nodes**: Each family gets a unique color from the predefined palette
- **Family nodes**: Always black (#000000)
- **FamilyCentroid nodes**: Always white (#FFFFFF)

The color palette includes 20 distinct colors excluding black and white:
- Red, Green, Blue, Purple, Orange, Yellow, Cyan, Magenta, etc.

## Future Enhancements

1. **Interactive Visualization**: Replace basic node list with D3.js or vis.js graph
2. **Real-time Updates**: WebSocket integration for live graph updates
3. **Advanced Queries**: Cypher query interface for complex graph analysis
4. **Performance Optimization**: Caching and pagination for large graphs
5. **Export Features**: Graph export in various formats (GraphML, JSON, etc.)

## Testing

The implementation includes basic tests in `/tmp/test_neo4j_integration.py` that verify:
- Service imports and initialization
- Color management functionality
- Graceful handling of disconnected Neo4j
- Flask app integration

## Deployment

For production deployment:
1. Set up Neo4j cluster or cloud instance
2. Configure environment variables
3. Run initial data sync
4. Monitor connection status and sync health