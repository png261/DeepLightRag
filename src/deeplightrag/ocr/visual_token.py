"""
Visual Token representation for OCR
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
import numpy as np

from .geometry import BoundingBox

@dataclass
class VisualToken:
    """Compressed visual token representation with enhanced embedding support"""

    token_id: int
    embedding: np.ndarray  # Dense visual embedding (768 or compressed dims)
    confidence: float
    region_type: str = "general"  # semantic type of visual content
    spatial_position: Tuple[float, float] = (0.0, 0.0)  # relative position in region
    compression_method: str = "none"  # pca, quantize, sparse, none
    original_dims: int = 768  # original embedding dimension before compression

    def get_embedding_size_kb(self) -> float:
        """Calculate memory usage of embedding in KB"""
        return self.embedding.nbytes / 1024

    def compress(self, method: str = "pca", target_dim: int = 256) -> "VisualToken":
        """Return compressed version of this token"""
        # This would use the VisualEmbeddingExtractor's compression methods
        compressed_embedding = self.embedding[:target_dim]  # Simple truncation for now

        return VisualToken(
            token_id=self.token_id,
            embedding=compressed_embedding,
            confidence=self.confidence,
            region_type=self.region_type,
            spatial_position=self.spatial_position,
            compression_method=method,
            original_dims=len(self.embedding),
        )


@dataclass
class VisualRegion:
    """Visual region with comprehensive visual features and embeddings"""

    region_id: str
    page_num: int
    block_type: str  # header, paragraph, table, figure, caption, list, formula
    bbox: BoundingBox
    compressed_tokens: List[VisualToken]
    text_content: str
    markdown_content: str
    token_count: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced visual embedding fields
    region_embedding: Optional[np.ndarray] = None  # Overall region embedding
    embedding_confidence: float = 0.0
    visual_complexity: float = 0.0  # 0-1 score of visual complexity
    text_to_visual_ratio: float = 1.0  # How much text vs visual content

    # NEW: Comprehensive visual features
    global_page_embedding: Optional[np.ndarray] = None  # Page-level context
    local_context_embedding: Optional[np.ndarray] = None  # Local neighborhood
    spatial_features: Optional[Dict[str, float]] = None  # Spatial position features
    layout_features: Optional[Dict[str, Any]] = None  # Layout structure info
    quality_metrics: Optional[Dict[str, float]] = None  # Visual quality scores
    readability_metrics: Optional[Dict[str, float]] = None  # Text readability

    # Content-specific features
    table_structure: Optional[Dict[str, Any]] = None  # For table regions
    figure_analysis: Optional[Dict[str, Any]] = None  # For figure regions
    formula_features: Optional[Dict[str, Any]] = None  # For formula regions

    # Relationship features
    spatial_neighbors: List[str] = field(default_factory=list)  # Neighbor region IDs
    visual_hierarchy: Optional[str] = None  # Position in document hierarchy
    content_flow: Optional[List[str]] = None  # Content flow connections

    # Advanced embedding variants
    multi_scale_embeddings: Optional[Dict[str, np.ndarray]] = None  # Different scales
    semantic_embedding: Optional[np.ndarray] = None  # Semantic meaning
    structural_embedding: Optional[np.ndarray] = None  # Structure patterns

    # Image extraction fields
    image_path: Optional[str] = None  # Path to extracted image file
    extracted_image: bool = False  # Whether image was successfully extracted (backward compatibility)
    image_embedding: Optional[np.ndarray] = None  # Embedding of the image itself
    image_format: str = "png"  # Format of extracted image
    image_size: Optional[Tuple[int, int]] = None  # (width, height) of extracted image

    def should_use_visual_mode(self) -> bool:
        """Determine if this region is best represented visually"""
        return self.block_type in {"table", "figure", "formula", "chart"}
