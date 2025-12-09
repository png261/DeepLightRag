__version__ = "1.0.0"
__author__ = "DeepLightRAG Team"

from .core import DeepLightRAG

from .graph.dual_layer import DualLayerGraph
from .graph.entity_relationship import Entity, EntityRelationshipGraph, Relationship
from .graph.visual_spatial import SpatialEdge, VisualNode, VisualSpatialGraph

from .ocr.deepseek_ocr import (
    BoundingBox,
    DeepSeekOCR,
    PageOCRResult,
    VisualRegion,
    VisualToken,
)

from .ocr.processor import PDFProcessor
from .retrieval.adaptive_retriever import AdaptiveRetriever

from .retrieval.query_classifier import QueryClassifier, QueryLevel

__all__ = [
    "__version__",
    "__author__",
    # Main system
    "DeepLightRAG",
    # OCR
    "DeepSeekOCR",
    "PageOCRResult",
    "VisualRegion",
    "BoundingBox",
    "VisualToken",
    "PDFProcessor",
    # Graph
    "DualLayerGraph",
    "VisualSpatialGraph",
    "VisualNode",
    "SpatialEdge",
    "EntityRelationshipGraph",
    "Entity",
    "Relationship",
    # Retrieval
    "QueryClassifier",
    "QueryLevel",
    "AdaptiveRetriever",
]
