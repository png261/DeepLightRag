"""
Neo4j Graph Database Provider
Provides Neo4j backend for Entity-Relationship and Visual-Spatial graphs
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """Manages Neo4j database connection"""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j server URI (e.g., 'bolt://localhost:7687')
            username: Database username
            password: Database password
            database: Database name (default: 'neo4j')
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.session = None

    def connect(self):
        """Establish connection to Neo4j"""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package not installed. Install with: pip install neo4j")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class Neo4jEntityRelationshipBackend:
    """Neo4j backend for EntityRelationshipGraph"""

    def __init__(self, connection: Neo4jConnection):
        """
        Initialize Neo4j backend for entity-relationship graph

        Args:
            connection: Neo4jConnection instance
        """
        self.connection = connection

    def create_entity_node(self, entity_id: str, entity_data: Dict[str, Any]):
        """
        Create an entity node in Neo4j

        Args:
            entity_id: Unique entity ID
            entity_data: Entity attributes
        """
        query = """
        MERGE (e:Entity {entity_id: $entity_id})
        SET e += $properties
        """
        properties = {
            "name": entity_data.get("name", ""),
            "entity_type": entity_data.get("entity_type", ""),
            "description": entity_data.get("description", ""),
            "confidence": entity_data.get("confidence", 1.0),
            "mention_count": entity_data.get("mention_count", 1),
            "value": str(entity_data.get("value", "")),
        }

        self.connection.execute_query(query, {"entity_id": entity_id, "properties": properties})
        logger.debug(f"Created entity node: {entity_id}")

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        rel_data: Dict[str, Any],
    ):
        """
        Create a relationship between entities

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type
            rel_data: Relationship attributes
        """
        query = f"""
        MATCH (source:Entity {{entity_id: $source_id}})
        MATCH (target:Entity {{entity_id: $target_id}})
        MERGE (source)-[r:{rel_type} {{type: $rel_type}}]->(target)
        SET r += $properties
        """
        properties = {
            "weight": rel_data.get("weight", 1.0),
            "description": rel_data.get("description", ""),
            "spatial_cooccurrence": rel_data.get("spatial_cooccurrence", False),
            "layout_aware_type": rel_data.get("layout_aware_type", ""),
        }

        self.connection.execute_query(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
                "properties": properties,
            },
        )
        logger.debug(f"Created relationship: {source_id} -[{rel_type}]-> {target_id}")

    def search_entities(self, query_text: str, entity_type: Optional[str] = None, limit: int = 10):
        """
        Search for entities by name or description

        Args:
            query_text: Search query
            entity_type: Optional entity type filter
            limit: Maximum results to return

        Returns:
            List of matching entities
        """
        cypher_query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $query_text OR e.description CONTAINS $query_text
        """
        if entity_type:
            cypher_query += " AND e.entity_type = $entity_type"

        cypher_query += " RETURN e ORDER BY e.mention_count DESC LIMIT $limit"

        params = {"query_text": query_text, "limit": limit}
        if entity_type:
            params["entity_type"] = entity_type

        results = self.connection.execute_query(cypher_query, params)
        return results

    def get_entity_neighborhood(self, entity_id: str, hop_distance: int = 1) -> Dict:
        """
        Get entity and its neighbors up to hop_distance

        Args:
            entity_id: Entity ID to start from
            hop_distance: Number of hops

        Returns:
            Dictionary with entity and related entities/relationships
        """
        query = f"""
        MATCH (e:Entity {{entity_id: $entity_id}})
        CALL apoc.path.expandConfig({{
            startNode: e,
            relationshipFilter: '',
            minLevel: 1,
            maxLevel: $hop_distance,
            limit: 100
        }})
        YIELD path
        RETURN DISTINCT nodes(path) as nodes, relationships(path) as rels
        LIMIT 1
        """

        try:
            results = self.connection.execute_query(
                query, {"entity_id": entity_id, "hop_distance": hop_distance}
            )
            return results[0] if results else {"nodes": [], "rels": []}
        except Exception as e:
            logger.warning(f"APOC not available, using basic neighborhood query: {e}")
            # Fallback to basic 1-hop query
            basic_query = """
            MATCH (e:Entity {entity_id: $entity_id})
            MATCH (e)-[r]-(related:Entity)
            RETURN e, r, related
            """
            results = self.connection.execute_query(basic_query, {"entity_id": entity_id})
            return results

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        queries = {
            "total_entities": "MATCH (e:Entity) RETURN count(e) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_types": "MATCH (e:Entity) RETURN e.entity_type as type, count(e) as count ORDER BY count DESC",
            "relationship_types": "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC",
        }

        stats = {}
        for key, query in queries.items():
            try:
                results = self.connection.execute_query(query)
                stats[key] = results
            except Exception as e:
                logger.warning(f"Failed to get {key}: {e}")
                stats[key] = []

        return stats

    def clear_graph(self):
        """Clear all nodes and relationships (DANGEROUS - USE WITH CAUTION)"""
        query = "MATCH (n) DETACH DELETE n"
        self.connection.execute_query(query)
        logger.info("Cleared Neo4j graph")


class Neo4jVisualSpatialBackend:
    """Neo4j backend for VisualSpatialGraph"""

    def __init__(self, connection: Neo4jConnection):
        """
        Initialize Neo4j backend for visual-spatial graph

        Args:
            connection: Neo4jConnection instance
        """
        self.connection = connection

    def create_visual_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Create a visual region node in Neo4j

        Args:
            node_id: Unique node ID
            node_data: Node attributes
        """
        query = """
        MERGE (v:VisualNode {node_id: $node_id})
        SET v += $properties,
            v.page_num = toInteger(v.page_num),
            v.area = toFloat(v.area)
        """
        properties = {
            "page_num": node_data.get("page_num", 0),
            "block_type": node_data.get("block_type", ""),
            "position": str(node_data.get("position", "")),
            "area": node_data.get("area", 0.0),
            "text_content": node_data.get("text_content", ""),
            "token_count": node_data.get("token_count", 0),
        }

        self.connection.execute_query(query, {"node_id": node_id, "properties": properties})
        logger.debug(f"Created visual node: {node_id}")

    def create_spatial_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        edge_data: Dict[str, Any],
    ):
        """
        Create a spatial edge between visual nodes

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type (adjacent, reading_order, semantic, hierarchical)
            edge_data: Edge attributes
        """
        query = f"""
        MATCH (source:VisualNode {{node_id: $source_id}})
        MATCH (target:VisualNode {{node_id: $target_id}})
        MERGE (source)-[r:{edge_type} {{type: $edge_type}}]->(target)
        SET r.weight = toFloat($weight)
        """
        self.connection.execute_query(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "weight": edge_data.get("weight", 1.0),
            },
        )
        logger.debug(f"Created spatial edge: {source_id} -[{edge_type}]-> {target_id}")

    def get_page_regions(self, page_num: int) -> List[Dict]:
        """
        Get all visual regions on a page

        Args:
            page_num: Page number

        Returns:
            List of visual nodes
        """
        query = "MATCH (v:VisualNode {page_num: $page_num}) RETURN v ORDER BY v.position"
        results = self.connection.execute_query(query, {"page_num": page_num})
        return results

    def get_neighbors(self, node_id: str, edge_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Get neighboring nodes

        Args:
            node_id: Node ID
            edge_types: Optional list of edge types to filter

        Returns:
            List of neighboring nodes
        """
        if edge_types:
            edge_filter = "|".join(edge_types)
            query = f"""
            MATCH (v:VisualNode {{node_id: $node_id}})-[:{edge_filter}]->(neighbor:VisualNode)
            RETURN neighbor, type(relationships) as edge_type
            """
        else:
            query = """
            MATCH (v:VisualNode {node_id: $node_id})-[]->(neighbor:VisualNode)
            RETURN neighbor
            """

        results = self.connection.execute_query(query, {"node_id": node_id})
        return results

    def get_statistics(self) -> Dict:
        """Get visual-spatial graph statistics"""
        queries = {
            "total_nodes": "MATCH (v:VisualNode) RETURN count(v) as count",
            "total_edges": "MATCH ()-[r]->() WHERE NOT r:Entity RETURN count(r) as count",
            "total_pages": "MATCH (v:VisualNode) RETURN max(v.page_num) as max_page",
            "block_types": "MATCH (v:VisualNode) RETURN v.block_type as type, count(v) as count ORDER BY count DESC",
        }

        stats = {}
        for key, query in queries.items():
            try:
                results = self.connection.execute_query(query)
                stats[key] = results
            except Exception as e:
                logger.warning(f"Failed to get {key}: {e}")
                stats[key] = []

        return stats


class Neo4jDualLayerBackend:
    """Neo4j backend for Dual-Layer Graph with cross-layer connections"""

    def __init__(self, connection: Neo4jConnection):
        """
        Initialize Neo4j backend for dual-layer graph

        Args:
            connection: Neo4jConnection instance
        """
        self.connection = connection
        self.entity_backend = Neo4jEntityRelationshipBackend(connection)
        self.visual_backend = Neo4jVisualSpatialBackend(connection)

    def link_entity_to_region(self, entity_id: str, region_id: str, weight: float = 1.0):
        """
        Create cross-layer link between entity and visual region

        Args:
            entity_id: Entity ID
            region_id: Visual region node ID
            weight: Link weight
        """
        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        MATCH (v:VisualNode {node_id: $region_id})
        MERGE (e)-[r:GROUNDED_IN {weight: $weight}]->(v)
        """
        self.connection.execute_query(
            query, {"entity_id": entity_id, "region_id": region_id, "weight": weight}
        )
        logger.debug(f"Linked entity {entity_id} to region {region_id}")

    def get_entity_visual_context(self, entity_id: str) -> Dict:
        """
        Get visual context for an entity

        Args:
            entity_id: Entity ID

        Returns:
            Dictionary with entity and related visual regions
        """
        query = """
        MATCH (e:Entity {entity_id: $entity_id})
        OPTIONAL MATCH (e)-[r:GROUNDED_IN]->(v:VisualNode)
        RETURN e, COLLECT({region: v, weight: r.weight}) as visual_regions
        """
        results = self.connection.execute_query(query, {"entity_id": entity_id})
        return results[0] if results else {}

    def hybrid_search(self, entity_query: str, region_query: Optional[str] = None) -> Dict:
        """
        Search across both layers

        Args:
            entity_query: Entity search query
            region_query: Optional region search query

        Returns:
            Combined results from both layers
        """
        # Entity layer search
        entity_query_str = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $entity_query OR e.description CONTAINS $entity_query
        RETURN DISTINCT e
        LIMIT 20
        """
        entity_results = self.connection.execute_query(
            entity_query_str, {"entity_query": entity_query}
        )

        # Visual layer search
        region_results = []
        if region_query:
            region_query_str = """
            MATCH (v:VisualNode)
            WHERE v.text_content CONTAINS $region_query
            RETURN DISTINCT v
            LIMIT 20
            """
            region_results = self.connection.execute_query(
                region_query_str, {"region_query": region_query}
            )

        # Cross-layer connections
        cross_layer_query = """
        MATCH (e:Entity)-[r:GROUNDED_IN]->(v:VisualNode)
        WHERE e.entity_id IN $entity_ids AND v.node_id IN $region_ids
        RETURN e, r, v
        """

        entity_ids = [e.get("e", {}).get("entity_id") for e in entity_results if "e" in e]
        region_ids = [v.get("v", {}).get("node_id") for v in region_results if "v" in v]

        cross_layer_results = []
        if entity_ids and region_ids:
            cross_layer_results = self.connection.execute_query(
                cross_layer_query, {"entity_ids": entity_ids, "region_ids": region_ids}
            )

        return {
            "entities": entity_results,
            "regions": region_results,
            "cross_layer_links": cross_layer_results,
        }

    def export_to_cypher(self, output_file: str):
        """
        Export graph as Cypher script

        Args:
            output_file: Output file path
        """
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(m)
        WITH n, COLLECT(DISTINCT {rel: r, node: m}) as rels
        RETURN apoc.export.cypher.all($file, {useTypes: true})
        """
        try:
            self.connection.execute_query(query, {"file": output_file})
            logger.info(f"Exported graph to {output_file}")
        except Exception as e:
            logger.warning(f"Export failed (APOC may not be available): {e}")
