"""
Graph classes with optional Neo4j persistence
Extends EntityRelationshipGraph and VisualSpatialGraph with Neo4j backend support
"""

import logging
from typing import Any, Dict, List, Optional

from .entity_relationship import Entity, EntityRelationshipGraph, Relationship
from .neo4j_provider import (
    Neo4jConnection,
    Neo4jDualLayerBackend,
    Neo4jEntityRelationshipBackend,
    Neo4jVisualSpatialBackend,
)
from .visual_spatial import VisualSpatialGraph

logger = logging.getLogger(__name__)


class EntityRelationshipGraphWithNeo4j(EntityRelationshipGraph):
    """
    EntityRelationshipGraph with optional Neo4j persistence
    Syncs entity and relationship data to Neo4j in real-time
    """

    def __init__(self, neo4j_backend: Optional[Neo4jEntityRelationshipBackend] = None):
        """
        Initialize with optional Neo4j backend

        Args:
            neo4j_backend: Neo4jEntityRelationshipBackend instance (optional)
        """
        super().__init__()
        self.neo4j_backend = neo4j_backend

    def add_entity(self, entity: Entity):
        """
        Add entity to graph and optionally persist to Neo4j

        Args:
            entity: Entity object to add
        """
        # Add to NetworkX graph
        self.entities[entity.entity_id] = entity
        self.entity_name_index[entity.name.lower().strip()] = entity.entity_id
        self.entity_type_index[entity.entity_type].append(entity.entity_id)
        self.graph.add_node(entity.entity_id, **entity.to_dict())

        # Sync to Neo4j
        if self.neo4j_backend:
            try:
                self.neo4j_backend.create_entity_node(entity.entity_id, entity.to_dict())
                logger.debug(f"Synced entity {entity.entity_id} to Neo4j")
            except Exception as e:
                logger.warning(f"Failed to sync entity to Neo4j: {e}")

    def add_relationship(self, relationship: Relationship):
        """
        Add relationship to graph and optionally persist to Neo4j

        Args:
            relationship: Relationship object to add
        """
        # Add to NetworkX graph
        self.relationships.append(relationship)
        edge_data = relationship.to_dict()
        edge_data.pop("source", None)
        edge_data.pop("target", None)
        self.graph.add_edge(
            relationship.source_entity,
            relationship.target_entity,
            **edge_data,
        )

        # Sync to Neo4j
        if self.neo4j_backend:
            try:
                self.neo4j_backend.create_relationship(
                    relationship.source_entity,
                    relationship.target_entity,
                    relationship.relationship_type,
                    relationship.to_dict(),
                )
                logger.debug(
                    f"Synced relationship {relationship.source_entity} -[{relationship.relationship_type}]-> {relationship.target_entity} to Neo4j"
                )
            except Exception as e:
                logger.warning(f"Failed to sync relationship to Neo4j: {e}")

    def search_entities_hybrid(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        use_neo4j: bool = True,
    ) -> List[Entity]:
        """
        Search entities using Neo4j if available, otherwise use NetworkX

        Args:
            query: Search query
            entity_types: Optional entity type filter
            top_k: Number of results
            use_neo4j: Whether to use Neo4j if available

        Returns:
            List of matching entities
        """
        if use_neo4j and self.neo4j_backend:
            try:
                neo4j_results = self.neo4j_backend.search_entities(query, entity_types, top_k)
                # Convert Neo4j results to Entity objects
                entities = []
                for result in neo4j_results:
                    entity_data = result.get("e", {}) if "e" in result else result
                    entity_id = entity_data.get("entity_id")
                    if entity_id in self.entities:
                        entities.append(self.entities[entity_id])
                return entities
            except Exception as e:
                logger.warning(f"Neo4j search failed, falling back to NetworkX: {e}")

        # Fallback to NetworkX search
        return self.search_entities(query, entity_types, top_k)

    def get_entity_neighborhood_hybrid(
        self, entity_id: str, hop_distance: int = 1, use_neo4j: bool = True
    ) -> Dict:
        """
        Get entity neighborhood using Neo4j if available

        Args:
            entity_id: Entity ID
            hop_distance: Number of hops
            use_neo4j: Whether to use Neo4j if available

        Returns:
            Dictionary with entity context
        """
        if use_neo4j and self.neo4j_backend:
            try:
                neo4j_neighborhood = self.neo4j_backend.get_entity_neighborhood(
                    entity_id, hop_distance
                )
                return neo4j_neighborhood
            except Exception as e:
                logger.warning(f"Neo4j neighborhood query failed: {e}")

        # Fallback to NetworkX
        return self.get_entity_context(entity_id, max_depth=hop_distance)


class VisualSpatialGraphWithNeo4j(VisualSpatialGraph):
    """
    VisualSpatialGraph with optional Neo4j persistence
    Syncs visual nodes and spatial edges to Neo4j in real-time
    """

    def __init__(self, neo4j_backend: Optional[Neo4jVisualSpatialBackend] = None):
        """
        Initialize with optional Neo4j backend

        Args:
            neo4j_backend: Neo4jVisualSpatialBackend instance (optional)
        """
        super().__init__()
        self.neo4j_backend = neo4j_backend

    def add_visual_node(self, node):
        """
        Add visual node to graph and optionally persist to Neo4j

        Args:
            node: VisualNode object to add
        """
        # Add to NetworkX graph
        self.nodes[node.node_id] = node
        if node.page_num not in self.page_nodes:
            self.page_nodes[node.page_num] = []
        self.page_nodes[node.page_num].append(node.node_id)
        self.graph.add_node(node.node_id, **node.to_dict())

        # Sync to Neo4j
        if self.neo4j_backend:
            try:
                self.neo4j_backend.create_visual_node(node.node_id, node.to_dict())
                logger.debug(f"Synced visual node {node.node_id} to Neo4j")
            except Exception as e:
                logger.warning(f"Failed to sync visual node to Neo4j: {e}")

    def add_spatial_edge(self, edge):
        """
        Add spatial edge to graph and optionally persist to Neo4j

        Args:
            edge: SpatialEdge object to add
        """
        # Add to NetworkX graph
        self.edges.append(edge)
        self.graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())

        # Sync to Neo4j
        if self.neo4j_backend:
            try:
                self.neo4j_backend.create_spatial_edge(
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    edge.to_dict(),
                )
                logger.debug(f"Synced spatial edge to Neo4j")
            except Exception as e:
                logger.warning(f"Failed to sync spatial edge to Neo4j: {e}")

    def get_neighbors_hybrid(
        self, node_id: str, hop_distance: int = 1, use_neo4j: bool = True
    ) -> List[str]:
        """
        Get neighbors using Neo4j if available

        Args:
            node_id: Node ID
            hop_distance: Number of hops
            use_neo4j: Whether to use Neo4j if available

        Returns:
            List of neighboring node IDs
        """
        if use_neo4j and self.neo4j_backend:
            try:
                neo4j_neighbors = self.neo4j_backend.get_neighbors(node_id)
                return [
                    neighbor.get("neighbor", {}).get("node_id")
                    for neighbor in neo4j_neighbors
                    if "neighbor" in neighbor
                ]
            except Exception as e:
                logger.warning(f"Neo4j neighbor query failed: {e}")

        # Fallback to NetworkX
        return self.get_neighbors(node_id, hop_distance)


class DualLayerGraphWithNeo4j:
    """
    Dual-Layer Graph with full Neo4j persistence
    Combines EntityRelationshipGraph and VisualSpatialGraph with Neo4j backend
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
    ):
        """
        Initialize dual-layer graph with optional Neo4j persistence

        Args:
            neo4j_uri: Neo4j server URI (e.g., 'bolt://localhost:7687')
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Database name
        """
        # Initialize Neo4j connection if credentials provided
        self.neo4j_connection = None
        self.neo4j_dual_backend = None

        if neo4j_uri and neo4j_username and neo4j_password:
            try:
                self.neo4j_connection = Neo4jConnection(
                    neo4j_uri, neo4j_username, neo4j_password, neo4j_database
                )
                self.neo4j_connection.connect()

                # Create backends
                entity_backend = Neo4jEntityRelationshipBackend(self.neo4j_connection)
                visual_backend = Neo4jVisualSpatialBackend(self.neo4j_connection)
                self.neo4j_dual_backend = Neo4jDualLayerBackend(self.neo4j_connection)

                logger.info("Neo4j backends initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j: {e}")
                logger.info("Continuing with NetworkX-only mode")
        else:
            entity_backend = None
            visual_backend = None

        # Create graph layers with Neo4j backends
        self.entity_relationship = EntityRelationshipGraphWithNeo4j(
            entity_backend if self.neo4j_connection else None
        )
        self.visual_spatial = VisualSpatialGraphWithNeo4j(
            visual_backend if self.neo4j_connection else None
        )

        # Cross-layer mappings
        self.entity_to_regions: Dict[str, List[str]] = {}
        self.region_to_entities: Dict[str, List[str]] = {}
        self.figure_caption_links: Dict[str, str] = {}

    def link_entity_to_region(self, entity_id: str, region_id: str, weight: float = 1.0):
        """
        Link entity to visual region and optionally persist to Neo4j

        Args:
            entity_id: Entity ID
            region_id: Visual region node ID
            weight: Link weight
        """
        # Add to in-memory mappings
        if entity_id not in self.entity_to_regions:
            self.entity_to_regions[entity_id] = []
        if region_id not in self.entity_to_regions[entity_id]:
            self.entity_to_regions[entity_id].append(region_id)

        if region_id not in self.region_to_entities:
            self.region_to_entities[region_id] = []
        if entity_id not in self.region_to_entities[region_id]:
            self.region_to_entities[region_id].append(entity_id)

        # Sync to Neo4j
        if self.neo4j_dual_backend:
            try:
                self.neo4j_dual_backend.link_entity_to_region(entity_id, region_id, weight)
                logger.debug(f"Synced cross-layer link to Neo4j")
            except Exception as e:
                logger.warning(f"Failed to sync cross-layer link to Neo4j: {e}")

    def hybrid_search(
        self, entity_query: str, region_query: Optional[str] = None, use_neo4j: bool = True
    ) -> Dict:
        """
        Search across both layers

        Args:
            entity_query: Entity search query
            region_query: Optional visual region search query
            use_neo4j: Whether to use Neo4j if available

        Returns:
            Dictionary with results from both layers
        """
        if use_neo4j and self.neo4j_dual_backend:
            try:
                return self.neo4j_dual_backend.hybrid_search(entity_query, region_query)
            except Exception as e:
                logger.warning(f"Neo4j hybrid search failed: {e}")

        # Fallback to NetworkX-only search
        entity_results = self.entity_relationship.search_entities(entity_query, top_k=10)
        return {"entities": entity_results, "regions": [], "cross_layer_links": []}

    def get_statistics(self) -> Dict:
        """Get graph statistics from both layers"""
        stats = {"entity_relationship": {}, "visual_spatial": {}, "neo4j": {}}

        # Get NetworkX statistics
        stats["entity_relationship"] = self.entity_relationship.get_statistics()
        stats["visual_spatial"] = self.visual_spatial.get_statistics()

        # Get Neo4j statistics if available
        if self.neo4j_dual_backend:
            try:
                stats["neo4j"]["entity"] = self.neo4j_dual_backend.entity_backend.get_statistics()
                stats["neo4j"]["visual"] = self.neo4j_dual_backend.visual_backend.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get Neo4j statistics: {e}")

        return stats

    def close(self):
        """Close Neo4j connection"""
        if self.neo4j_connection:
            self.neo4j_connection.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
