"""
Visual-Spatial Graph (Layer 1)
Captures WHERE things are in the document
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..ocr.deepseek_ocr import BoundingBox, PageOCRResult, VisualRegion


@dataclass
class VisualNode:
    """Node in the Visual-Spatial Graph"""

    node_id: str
    region: VisualRegion
    page_num: int
    position: Tuple[float, float]  # Center position (normalized)
    area: float
    embedding: Optional[np.ndarray] = None
    entity_ids: List[str] = None  # Cross-layer: entities in this region

    def to_dict(self) -> Dict:
        result = {
            "node_id": self.node_id,
            "region_id": self.region.region_id,
            "page_num": self.page_num,
            "node_type": self.region.block_type,  # Classified type
            "block_type": self.region.block_type,  # Keep for compatibility
            "bbox": self.region.bbox.to_list(),
            "position": self.position,
            "area": self.area,
            "text_content": self.region.text_content,
            "token_count": self.region.token_count,
            # Add image information
            "image_path": self.region.image_path,
            "image_format": self.region.image_format,
            "image_size": self.region.image_size,
            "extracted_image": self.region.extracted_image,
        }
        if self.entity_ids:
            result["entity_ids"] = self.entity_ids
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        # Include image embedding if available
        if self.region.image_embedding is not None:
            result["image_embedding"] = self.region.image_embedding.tolist()
        return result


@dataclass
class SpatialEdge:
    """Edge in the Visual-Spatial Graph"""

    source_id: str
    target_id: str
    edge_type: str  # adjacent, reading_order, semantic, hierarchical
    weight: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type,
            "weight": self.weight,
            "metadata": self.metadata,
        }


class VisualSpatialGraph:
    """
    Visual-Spatial Graph Layer
    Captures spatial relationships between visual regions
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, VisualNode] = {}
        self.edges: List[SpatialEdge] = []
        self.page_nodes: Dict[int, List[str]] = {}  # Page -> Node IDs

    def build_from_ocr_results(self, ocr_results: List[PageOCRResult]):
        """
        Build visual-spatial graph from OCR results

        Args:
            ocr_results: List of PageOCRResult from DeepSeek-OCR
        """
        print("Building Visual-Spatial Graph (Layer 1)...")

        if not ocr_results:
            print("  Warning: No OCR results provided")
            return

        # Create nodes from visual regions, deduplicating by text and block type
        unique_regions = set()
        skipped_empty = 0
        skipped_duplicates = 0
        self.total_pages = 0

        for page_result in ocr_results:
            self.total_pages = max(self.total_pages, page_result.page_num + 1)
            page_num = page_result.page_num
            self.page_nodes[page_num] = []

            if not page_result.visual_regions:
                continue

            for region in page_result.visual_regions:
                text = region.text_content.strip()
                
                # Skip empty or very short text (increased from 3 to 10)
                if not text or len(text) < 10:
                    skipped_empty += 1
                    continue
                
                # Assess text quality
                if not self._is_quality_text(text):
                    skipped_empty += 1
                    continue
                
                # Check for duplicates (use similarity instead of exact match)
                is_dup = False
                for existing_key in unique_regions:
                    if self._is_similar_text(text, existing_key[0]):
                        is_dup = True
                        break
                
                if is_dup:
                    skipped_duplicates += 1
                    continue
                
                key = (text, region.block_type)
                unique_regions.add(key)
                try:
                    node = self._create_node(region, page_num)
                    self.nodes[node.node_id] = node
                    self.page_nodes[page_num].append(node.node_id)
                    self.graph.add_node(node.node_id, **node.to_dict())
                except Exception as e:
                    print(f"  Warning: Failed to create node for region {region.region_id}: {e}")
                    continue

        if skipped_empty > 0:
            print(f"  - Skipped {skipped_empty} empty regions")
        if skipped_duplicates > 0:
            print(f"  - Skipped {skipped_duplicates} duplicate regions")

        # Create edges
        try:
            self._create_spatial_edges()
            self._create_reading_order_edges()
            self._create_semantic_edges()
            self._create_hierarchical_edges()
        except Exception as e:
            print(f"  Warning: Edge creation encountered errors: {e}")

        print(f"Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def _is_quality_text(self, text: str) -> bool:
        """
        Assess if text is of sufficient quality
        Returns: True if quality is acceptable
        """
        if not text or len(text) < 3:
            return False
        
        # Check for excessive repetition (like "\uff09" repeated)
        unique_chars = len(set(text))
        if unique_chars < 3:  # Less than 3 unique characters = garbage
            return False
        
        # For very short text, be more lenient
        if len(text) < 20:
            return True  # Short text is OK if it passes basic checks
        
        # For longer text, check character diversity
        char_diversity = unique_chars / len(text)
        if char_diversity < 0.05:  # Less than 5% diversity = repetitive garbage
            return False
        
        return True  # Accept all other text
    
    def _is_similar_text(self, text1: str, text2: str, threshold: float = 0.9) -> bool:
        """
        Check if two texts are similar (for deduplication)
        
        Args:
            text1, text2: Texts to compare
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if texts are similar
        """
        from difflib import SequenceMatcher
        
        # Quick length check
        if abs(len(text1) - len(text2)) > max(len(text1), len(text2)) * 0.3:
            return False
        
        # Compute similarity
        sim = SequenceMatcher(None, text1, text2).ratio()
        return sim > threshold
    
    def _classify_node_type(self, region: VisualRegion) -> str:
        """
        Classify visual node type based on content and structure
        Returns: heading, paragraph, list, table, caption, code, or text
        """
        text = region.text_content.strip()
        block_type = region.block_type
        
        if not text:
            return "empty"
        
        # Use existing block_type if meaningful
        if block_type and block_type not in ["text", "unknown", ""]:
            return block_type
        
        # Rule-based classification
        # 1. Heading detection
        if len(text) < 100:
            if text.isupper() and len(text) > 5:
                return "heading"
            if text.endswith(':') and len(text) < 50:
                return "heading"
        
        # 2. List detection
        list_markers = ['•', '◦', '▪', '▫', '‣', '⁃', '-', '*', '→', '➤']
        if any(text.startswith(marker) for marker in list_markers):
            return "list"
        if text.startswith(tuple(f"{i}." for i in range(1, 100))):
            return "list"
        
        # 3. Code detection
        code_indicators = ['{', '}', '()', '=>', 'function', 'def ', 'class ', 'import ', 'const ']
        if any(indicator in text for indicator in code_indicators):
            if len([c for c in text if c in '{}();']) > len(text) * 0.1:
                return "code"
        
        # 4. Caption detection
        caption_starters = ['Figure', 'Table', 'Fig.', 'Image', 'Chart', 'Diagram']
        if any(text.startswith(starter) for starter in caption_starters):
            if len(text) < 200:
                return "caption"
        
        # 5. Paragraph (long text)
        if len(text) > 200:
            return "paragraph"
        
        # 6. Default
        return "text"
    
    def _create_node(self, region: VisualRegion, page_num: int) -> VisualNode:
        """Create a visual node from a region"""
        bbox = region.bbox

        # Calculate center position
        center_x = (bbox.x1 + bbox.x2) / 2
        center_y = (bbox.y1 + bbox.y2) / 2

        # Calculate area
        area = bbox.area()

        # Generate node embedding (combine visual tokens)
        if region.compressed_tokens:
            embeddings = [t.embedding for t in region.compressed_tokens]
            node_embedding = np.mean(embeddings, axis=0)
        else:
            node_embedding = None
        
        # Classify node type
        node_type = self._classify_node_type(region)
        # Update region block_type if it was generic
        if region.block_type in ["text", "unknown", "", None]:
            region.block_type = node_type

        return VisualNode(
            node_id=region.region_id,
            region=region,
            page_num=page_num,
            position=(center_x, center_y),
            area=area,
            embedding=node_embedding,
        )

    def _create_spatial_edges(self):
        """Create edges based on spatial adjacency"""
        for page_num, node_ids in self.page_nodes.items():
            nodes = [self.nodes[nid] for nid in node_ids]

            # Sort by position (top to bottom, left to right)
            nodes.sort(key=lambda n: (n.position[1], n.position[0]))

            # Connect adjacent nodes
            for i in range(len(nodes) - 1):
                current = nodes[i]
                next_node = nodes[i + 1]

                # Check if vertically adjacent
                if self._is_vertically_adjacent(current, next_node):
                    edge = SpatialEdge(
                        source_id=current.node_id,
                        target_id=next_node.node_id,
                        edge_type="adjacent",
                        weight=self._calculate_adjacency_weight(current, next_node),
                        metadata={"direction": "vertical"},
                    )
                    self._add_edge(edge)

            # Check horizontal adjacency
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self._is_horizontally_adjacent(nodes[i], nodes[j]):
                        edge = SpatialEdge(
                            source_id=nodes[i].node_id,
                            target_id=nodes[j].node_id,
                            edge_type="adjacent",
                            weight=self._calculate_adjacency_weight(nodes[i], nodes[j]),
                            metadata={"direction": "horizontal"},
                        )
                        self._add_edge(edge)

    def _is_vertically_adjacent(self, node1: VisualNode, node2: VisualNode) -> bool:
        """Check if two nodes are vertically adjacent"""
        bbox1 = node1.region.bbox
        bbox2 = node2.region.bbox

        # Check vertical overlap
        vertical_gap = bbox2.y1 - bbox1.y2
        horizontal_overlap = min(bbox1.x2, bbox2.x2) - max(bbox1.x1, bbox2.x1)

        return 0 <= vertical_gap < 0.1 and horizontal_overlap > 0.3

    def _is_horizontally_adjacent(self, node1: VisualNode, node2: VisualNode) -> bool:
        """Check if two nodes are horizontally adjacent"""
        bbox1 = node1.region.bbox
        bbox2 = node2.region.bbox

        # Check horizontal gap
        horizontal_gap = bbox2.x1 - bbox1.x2
        vertical_overlap = min(bbox1.y2, bbox2.y2) - max(bbox1.y1, bbox2.y1)

        return 0 <= horizontal_gap < 0.1 and vertical_overlap > 0.3

    def _calculate_adjacency_weight(self, node1: VisualNode, node2: VisualNode) -> float:
        """Calculate edge weight based on spatial distance"""
        # Euclidean distance between centers
        dist = np.sqrt(
            (node1.position[0] - node2.position[0]) ** 2
            + (node1.position[1] - node2.position[1]) ** 2
        )
        # Closer = higher weight
        return 1.0 / (1.0 + dist)

    def _create_reading_order_edges(self):
        """Create edges based on reading order"""
        for page_num, node_ids in self.page_nodes.items():
            nodes = [self.nodes[nid] for nid in node_ids]

            # Sort by spatial order metadata
            nodes.sort(key=lambda n: n.region.metadata.get("spatial_order", 0))

            # Connect in reading order
            for i in range(len(nodes) - 1):
                edge = SpatialEdge(
                    source_id=nodes[i].node_id,
                    target_id=nodes[i + 1].node_id,
                    edge_type="reading_order",
                    weight=1.0,
                    metadata={"order": i},
                )
                self._add_edge(edge)

    def _create_semantic_edges(self):
        """Create edges based on semantic similarity"""
        all_nodes = list(self.nodes.values())

        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                node1, node2 = all_nodes[i], all_nodes[j]

                # Only connect within same page for now
                if node1.page_num != node2.page_num:
                    continue

                # Check semantic relationship based on block types
                if self._has_semantic_relationship(node1, node2):
                    similarity = self._calculate_semantic_similarity(node1, node2)
                    if similarity > 0.5:
                        edge = SpatialEdge(
                            source_id=node1.node_id,
                            target_id=node2.node_id,
                            edge_type="semantic",
                            weight=similarity,
                            metadata={"similarity_score": similarity},
                        )
                        self._add_edge(edge)

    def _has_semantic_relationship(self, node1: VisualNode, node2: VisualNode) -> bool:
        """Check if two nodes have potential semantic relationship"""
        # Figure and caption
        if (node1.region.block_type == "figure" and node2.region.block_type == "caption") or (
            node1.region.block_type == "caption" and node2.region.block_type == "figure"
        ):
            return True

        # Table and caption/paragraph
        if node1.region.block_type == "table" or node2.region.block_type == "table":
            return True

        # Same block types often related
        if node1.region.block_type == node2.region.block_type:
            return True

        return False

    def _calculate_semantic_similarity(self, node1: VisualNode, node2: VisualNode) -> float:
        """Calculate semantic similarity between nodes"""
        if node1.embedding is not None and node2.embedding is not None:
            # Cosine similarity
            dot_product = np.dot(node1.embedding, node2.embedding)
            norm1 = np.linalg.norm(node1.embedding)
            norm2 = np.linalg.norm(node2.embedding)
            if norm1 > 0 and norm2 > 0:
                return float(dot_product / (norm1 * norm2))

        # Fallback: text-based similarity
        # Simple word overlap
        words1 = set(node1.region.text_content.lower().split())
        words2 = set(node2.region.text_content.lower().split())
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0

    def _create_hierarchical_edges(self):
        """Create edges based on hierarchical structure"""
        for page_num, node_ids in self.page_nodes.items():
            # Find header nodes
            headers = [
                self.nodes[nid] for nid in node_ids if self.nodes[nid].region.block_type == "header"
            ]

            # Connect headers to content they govern
            for header in headers:
                header_y = header.position[1]

                # Find nodes below this header
                subordinates = [
                    self.nodes[nid]
                    for nid in node_ids
                    if self.nodes[nid].position[1] > header_y
                    and self.nodes[nid].region.block_type != "header"
                ]

                # Connect header to immediate subordinates
                for sub in subordinates[:3]:  # Limit connections
                    edge = SpatialEdge(
                        source_id=header.node_id,
                        target_id=sub.node_id,
                        edge_type="hierarchical",
                        weight=0.8,
                        metadata={"relationship": "governs"},
                    )
                    self._add_edge(edge)

    def _add_edge(self, edge: SpatialEdge):
        """Add edge to graph"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source_id, edge.target_id, type=edge.edge_type, weight=edge.weight, **edge.metadata
        )

    def get_neighbors(
        self, node_id: str, edge_types: Optional[List[str]] = None, hop_distance: int = 1
    ) -> List[str]:
        """
        Get neighboring nodes

        Args:
            node_id: Starting node ID
            edge_types: Filter by edge types
            hop_distance: Number of hops

        Returns:
            List of neighbor node IDs
        """
        if hop_distance == 1:
            neighbors = list(self.graph.successors(node_id))
            neighbors.extend(list(self.graph.predecessors(node_id)))

            if edge_types:
                filtered = []
                for neighbor in neighbors:
                    if self.graph.has_edge(node_id, neighbor):
                        edge_data = self.graph.get_edge_data(node_id, neighbor)
                        if edge_data.get("type") in edge_types:
                            filtered.append(neighbor)
                    elif self.graph.has_edge(neighbor, node_id):
                        edge_data = self.graph.get_edge_data(neighbor, node_id)
                        if edge_data.get("type") in edge_types:
                            filtered.append(neighbor)
                return list(set(filtered))

            return list(set(neighbors))
        else:
            # Multi-hop traversal
            current_neighbors = set([node_id])
            all_neighbors = set()

            for _ in range(hop_distance):
                next_neighbors = set()
                for n in current_neighbors:
                    next_neighbors.update(self.get_neighbors(n, edge_types, 1))
                all_neighbors.update(next_neighbors)
                current_neighbors = next_neighbors

            all_neighbors.discard(node_id)
            return list(all_neighbors)

    def get_page_subgraph(self, page_num: int) -> nx.DiGraph:
        """Get subgraph for a specific page"""
        if page_num not in self.page_nodes:
            return nx.DiGraph()

        node_ids = self.page_nodes[page_num]
        return self.graph.subgraph(node_ids).copy()

    def to_dict(self) -> Dict:
        """Serialize graph to dictionary"""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "page_nodes": self.page_nodes,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
        }

    def save(self, path: str):
        """Save graph to JSON file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Visual-Spatial Graph saved to {path}")

    def load(self, path: str):
        """Load graph from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)

        # Clear existing data
        self.nodes = {}
        self.edges = []
        self.graph = nx.DiGraph()
        self.page_nodes = {}

        # Reconstruct nodes
        for node_data in data.get("nodes", []):
            # Create BoundingBox
            bbox = BoundingBox(*node_data["bbox"])

            # Create VisualRegion
            region = VisualRegion(
                region_id=node_data["region_id"],
                page_num=node_data["page_num"],
                block_type=node_data["block_type"],
                bbox=bbox,
                compressed_tokens=[],  # Empty list since we don't serialize tokens
                text_content=node_data["text_content"],
                markdown_content=node_data.get("markdown_content", node_data["text_content"]),
                token_count=node_data["token_count"],
                confidence=node_data.get("confidence", 1.0),
                metadata=node_data.get("metadata", {}),
                # Add image-related fields
                image_path=node_data.get("image_path"),
                image_format=node_data.get("image_format", "png"),
                image_size=tuple(node_data["image_size"]) if node_data.get("image_size") else None,
                extracted_image=node_data.get("extracted_image", False),
            )

            # Load image embedding if available
            if "image_embedding" in node_data and node_data["image_embedding"]:
                region.image_embedding = np.array(node_data["image_embedding"])

            # Create VisualNode
            embedding = None
            if "embedding" in node_data and node_data["embedding"]:
                embedding = np.array(node_data["embedding"])
            
            node = VisualNode(
                node_id=node_data["node_id"],
                region=region,
                page_num=node_data["page_num"],
                position=tuple(node_data["position"]),
                area=node_data["area"],
                embedding=embedding,
            )

            self.nodes[node.node_id] = node
            self.graph.add_node(node.node_id)

            # Update page_nodes
            if node.page_num not in self.page_nodes:
                self.page_nodes[node.page_num] = []
            self.page_nodes[node.page_num].append(node.node_id)

        # Reconstruct edges
        for edge_data in data.get("edges", []):
            edge = SpatialEdge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=edge_data["type"],
                weight=edge_data["weight"],
                metadata=edge_data.get("metadata", {}),
            )
            self.edges.append(edge)
            self.graph.add_edge(
                edge.source_id,
                edge.target_id,
                type=edge.edge_type,
                weight=edge.weight,
                **edge.metadata,
            )

        print(f"Visual-Spatial Graph loaded from {path}")

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        edge_type_counts = {}
        for edge in self.edges:
            edge_type_counts[edge.edge_type] = edge_type_counts.get(edge.edge_type, 0) + 1

        block_type_counts = {}
        for node in self.nodes.values():
            bt = node.region.block_type
            block_type_counts[bt] = block_type_counts.get(bt, 0) + 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_pages": len(self.page_nodes),
            "edge_types": edge_type_counts,
            "block_types": block_type_counts,
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0
            ),
        }
