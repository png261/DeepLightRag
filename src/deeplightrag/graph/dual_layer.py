"""
Dual-Layer Graph Architecture
Combines Visual-Spatial + Entity-Relationship with Cross-Layer Connections
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ocr.deepseek_ocr import PageOCRResult
from .entity_relationship import Entity, EntityRelationshipGraph
from .visual_spatial import VisualNode, VisualSpatialGraph


class DualLayerGraph:
    """
    Dual-Layer Graph Architecture
    Layer 1: Visual-Spatial (WHERE things are)
    Layer 2: Entity-Relationship (WHAT things mean)
    Cross-Layer: Connections bridging spatial ↔ semantic
    """

    def __init__(
        self,
        device: str = "cpu",
        enable_gpu_acceleration: bool = False,
        ner_config: Optional[Dict] = None,
        re_config: Optional[Dict] = None,
        llm=None,
    ):
        self.visual_spatial = VisualSpatialGraph()
        self.entity_relationship = EntityRelationshipGraph(
            device=device, ner_config=ner_config, re_config=re_config, llm=llm
        )

        # Cross-layer mappings
        self.entity_to_regions: Dict[str, List[str]] = {}  # Entity → Visual Regions
        self.region_to_entities: Dict[str, List[str]] = {}  # Visual Region → Entities
        self.figure_caption_links: Dict[str, str] = {}  # Figure ↔ Caption

    def build_from_ocr_results(self, ocr_results: List[PageOCRResult]):
        """
        Build complete dual-layer graph from OCR results

        Args:
            ocr_results: List of PageOCRResult from DeepSeek-OCR
        """
        print("=" * 50)
        print("Building Dual-Layer Graph Architecture")
        print("=" * 50)

        # Build Layer 1: Visual-Spatial Graph
        print("\n[Layer 1] Visual-Spatial Graph")
        self.visual_spatial.build_from_ocr_results(ocr_results)

        # Build Layer 2: Entity-Relationship Graph
        print("\n[Layer 2] Entity-Relationship Graph")
        self.entity_relationship.extract_entities_from_ocr(ocr_results)
        self.entity_relationship.extract_relationships(ocr_results)

        # Build Cross-Layer Connections
        print("\n[Cross-Layer] Building Connections")
        self._build_cross_layer_connections()

        # Print statistics
        self._print_statistics()

    def _build_cross_layer_connections(self):
        """Build connections between visual and semantic layers"""

        # 1. Entity → Visual Region mappings
        print("  - Building Entity → Visual Region mappings...")
        valid_region_ids = set(self.visual_spatial.nodes.keys())
        entities_with_missing_regions = 0

        for entity_id, entity in self.entity_relationship.entities.items():
            # Filter to only valid regions
            valid_regions = [rid for rid in entity.source_regions if rid in valid_region_ids]

            if not valid_regions and entity.source_regions:
                entities_with_missing_regions += 1
                # Use first source region if available, fallback to empty list
                valid_regions = []

            self.entity_to_regions[entity_id] = valid_regions

            # Update reverse mapping only for valid regions
            for region_id in valid_regions:
                if region_id not in self.region_to_entities:
                    self.region_to_entities[region_id] = []
                if entity_id not in self.region_to_entities[region_id]:
                    self.region_to_entities[region_id].append(entity_id)

        if entities_with_missing_regions > 0:
            print(
                f"  - Warning: {entities_with_missing_regions} entities had missing region references"
            )

        # 2. Figure ↔ Caption links
        print("  - Linking Figures with Captions...")
        try:
            self._link_figures_captions()
        except Exception as e:
            print(f"  - Warning: Failed to link figures with captions: {e}")

        print(f"  - Created {len(self.entity_to_regions)} entity→region mappings")
        print(f"  - Created {len(self.region_to_entities)} region→entity mappings")
        print(f"  - Created {len(self.figure_caption_links)} figure↔caption links")

    def _link_figures_captions(self):
        """Link figure nodes with their captions"""
        for page_num, node_ids in self.visual_spatial.page_nodes.items():
            figures = []
            captions = []

            for node_id in node_ids:
                node = self.visual_spatial.nodes[node_id]
                if node.region.block_type == "figure":
                    figures.append(node)
                elif node.region.block_type == "caption":
                    captions.append(node)

            # Match figures with nearby captions
            for figure in figures:
                best_caption = None
                min_distance = float("inf")

                for caption in captions:
                    # Caption should be below or beside figure
                    y_distance = caption.position[1] - figure.position[1]
                    x_distance = abs(caption.position[0] - figure.position[0])

                    # Caption typically appears below figure
                    if 0 < y_distance < 0.2 and x_distance < 0.3:
                        distance = np.sqrt(y_distance**2 + x_distance**2)
                        if distance < min_distance:
                            min_distance = distance
                            best_caption = caption

                if best_caption:
                    self.figure_caption_links[figure.node_id] = best_caption.node_id

    def get_visual_regions_for_entity(self, entity_id: str) -> List[VisualNode]:
        """Get visual regions where an entity appears"""
        region_ids = self.entity_to_regions.get(entity_id, [])
        return [
            self.visual_spatial.nodes[rid] for rid in region_ids if rid in self.visual_spatial.nodes
        ]

    def get_entities_in_region(self, region_id: str) -> List[Entity]:
        """Get entities found in a visual region"""
        entity_ids = self.region_to_entities.get(region_id, [])
        return [
            self.entity_relationship.entities[eid]
            for eid in entity_ids
            if eid in self.entity_relationship.entities
        ]

    def get_caption_for_figure(self, figure_node_id: str) -> Optional[VisualNode]:
        """Get caption node for a figure"""
        caption_id = self.figure_caption_links.get(figure_node_id)
        if caption_id:
            return self.visual_spatial.nodes.get(caption_id)
        return None

    def query_hybrid(
        self, query_entities: List[str], query_regions: List[str], hop_distance: int = 2
    ) -> Dict[str, Any]:
        """
        Hybrid query using both layers

        Args:
            query_entities: List of entity IDs to start from
            query_regions: List of region IDs to start from
            hop_distance: Number of hops in graph traversal

        Returns:
            Combined results from both layers
        """
        results = {"entities": set(), "regions": set(), "relationships": [], "spatial_edges": []}

        # Expand from entities
        for entity_id in query_entities:
            # Get related entities
            related = self.entity_relationship.get_related_entities(
                entity_id, hop_distance=hop_distance
            )
            results["entities"].update(related)
            results["entities"].add(entity_id)

            # Cross to visual layer
            for eid in list(results["entities"]):
                regions = self.entity_to_regions.get(eid, [])
                results["regions"].update(regions)

        # Expand from regions
        for region_id in query_regions:
            # Get adjacent regions
            neighbors = self.visual_spatial.get_neighbors(region_id, hop_distance=hop_distance)
            results["regions"].update(neighbors)
            results["regions"].add(region_id)

            # Cross to entity layer
            for rid in list(results["regions"]):
                entities = self.region_to_entities.get(rid, [])
                results["entities"].update(entities)

        # Get relationships between found entities
        for eid in results["entities"]:
            for rel in self.entity_relationship.relationships:
                if rel.source_entity == eid or rel.target_entity == eid:
                    if (
                        rel.source_entity in results["entities"]
                        and rel.target_entity in results["entities"]
                    ):
                        results["relationships"].append(rel.to_dict())

        # Get spatial edges between found regions
        for rid in results["regions"]:
            for edge in self.visual_spatial.edges:
                if edge.source_id == rid or edge.target_id == rid:
                    if (
                        edge.source_id in results["regions"]
                        and edge.target_id in results["regions"]
                    ):
                        results["spatial_edges"].append(edge.to_dict())

        # Convert sets to lists
        results["entities"] = list(results["entities"])
        results["regions"] = list(results["regions"])

        return results

    def get_context_for_query(self, query: str, max_tokens: int = 6000) -> str:
        """
        Get context for a query by searching both layers

        Args:
            query: User query
            max_tokens: Maximum tokens for context

        Returns:
            Context string with structured information
        """
        # Search entities
        relevant_entities = self.entity_relationship.search_entities(query, top_k=10)

        # Build context
        context_parts = []
        current_tokens = 0

        # Add entity information
        context_parts.append("## Relevant Entities\n")
        for entity in relevant_entities:
            entity_text = f"- **{entity.name}** ({entity.entity_type}): {entity.description}\n"
            context_parts.append(entity_text)

            # Add related entities
            related = self.entity_relationship.get_related_entities(
                entity.entity_id, hop_distance=1
            )
            for related_id in related[:3]:
                related_entity = self.entity_relationship.get_entity(related_id)
                if related_entity:
                    context_parts.append(f"  - Related: {related_entity.name}\n")

        # Add visual region context
        context_parts.append("\n## Source Regions\n")
        seen_regions = set()
        for entity in relevant_entities[:5]:
            regions = self.get_visual_regions_for_entity(entity.entity_id)
            for region in regions[:2]:
                if region.node_id not in seen_regions:
                    seen_regions.add(region.node_id)
                    context_parts.append(
                        f"**[{region.region.block_type.upper()}] Page {region.page_num}:**\n"
                        f"{region.region.markdown_content}\n"
                    )

        # Estimate tokens (rough: 4 chars per token)
        context = "".join(context_parts)
        estimated_tokens = len(context) / 4

        # Truncate if needed
        if estimated_tokens > max_tokens:
            max_chars = int(max_tokens * 4)
            context = context[:max_chars] + "\n... [truncated]"

        return context

    def _print_statistics(self):
        """Print graph statistics"""
        print("\n" + "=" * 50)
        print("Dual-Layer Graph Statistics")
        print("=" * 50)

        vs_stats = self.visual_spatial.get_statistics()
        er_stats = self.entity_relationship.get_statistics()

        print(f"\nLayer 1 (Visual-Spatial):")
        print(f"  - Nodes: {vs_stats['num_nodes']}")
        print(f"  - Edges: {vs_stats['num_edges']}")
        print(f"  - Pages: {vs_stats['num_pages']}")
        print(f"  - Avg Degree: {vs_stats['avg_degree']:.2f}")

        print(f"\nLayer 2 (Entity-Relationship):")
        print(f"  - Entities: {er_stats['num_entities']}")
        print(f"  - Relationships: {er_stats['num_relationships']}")
        print(f"  - Avg Degree: {er_stats['avg_entity_degree']:.2f}")

        print(f"\nCross-Layer Connections:")
        print(f"  - Entity→Region: {len(self.entity_to_regions)}")
        print(f"  - Region→Entity: {len(self.region_to_entities)}")
        print(f"  - Figure↔Caption: {len(self.figure_caption_links)}")

        # Token summary
        total_tokens = sum(node.region.token_count for node in self.visual_spatial.nodes.values())
        print(f"\nToken Summary:")
        print(f"  - Total Compressed Tokens: {total_tokens}")
        print(f"  - Estimated Original (2500/page): {vs_stats['num_pages'] * 2500}")
        compression = (vs_stats["num_pages"] * 2500) / total_tokens if total_tokens > 0 else 0
        print(f"  - Compression Ratio: {compression:.1f}x")

    def save(self, directory: str):
        """Save dual-layer graph to directory"""
        import os

        os.makedirs(directory, exist_ok=True)

        # Save Layer 1
        self.visual_spatial.save(os.path.join(directory, "visual_spatial.json"))

        # Save Layer 2
        self.entity_relationship.save(os.path.join(directory, "entity_relationship.json"))

        # Save cross-layer connections
        cross_layer_data = {
            "entity_to_regions": self.entity_to_regions,
            "region_to_entities": self.region_to_entities,
            "figure_caption_links": self.figure_caption_links,
        }
        with open(os.path.join(directory, "cross_layer.json"), "w") as f:
            json.dump(cross_layer_data, f, indent=2)

        print(f"Dual-Layer Graph saved to {directory}")

    def load(self, directory: str):
        """Load dual-layer graph from directory"""
        import os

        # Load Layer 1: Visual-Spatial Graph
        visual_spatial_path = os.path.join(directory, "visual_spatial.json")
        if os.path.exists(visual_spatial_path):
            self.visual_spatial.load(visual_spatial_path)
        else:
            print(f"Warning: visual_spatial.json not found in {directory}")

        # Load Layer 2: Entity-Relationship Graph
        entity_relationship_path = os.path.join(directory, "entity_relationship.json")
        if os.path.exists(entity_relationship_path):
            self.entity_relationship.load(entity_relationship_path)
        else:
            print(f"Warning: entity_relationship.json not found in {directory}")

        # Load cross-layer connections
        cross_layer_path = os.path.join(directory, "cross_layer.json")
        if os.path.exists(cross_layer_path):
            with open(cross_layer_path, "r") as f:
                cross_layer_data = json.load(f)

            self.entity_to_regions = cross_layer_data["entity_to_regions"]
            self.region_to_entities = cross_layer_data["region_to_entities"]
            self.figure_caption_links = cross_layer_data["figure_caption_links"]
        else:
            print(f"Warning: cross_layer.json not found in {directory}")

        print(f"Dual-Layer Graph loaded from {directory}")

    def to_dict(self) -> Dict:
        """Serialize complete graph"""
        return {
            "visual_spatial": self.visual_spatial.to_dict(),
            "entity_relationship": self.entity_relationship.to_dict(),
            "cross_layer": {
                "entity_to_regions": self.entity_to_regions,
                "region_to_entities": self.region_to_entities,
                "figure_caption_links": self.figure_caption_links,
            },
        }
