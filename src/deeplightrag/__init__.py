"""
DeepLightRAG: Document Indexing and Retrieval System

Focus: High-performance indexing and retrieval (NO generation included)

Features:
- 9-10x Vision-Text Compression using DeepSeek-OCR
- Dual-Layer Graph Architecture (Visual-Spatial + Entity-Relationship)  
- Adaptive Token Budgeting (2K-12K tokens vs 30K fixed)
- GPU acceleration (CUDA, MPS, CPU)
- Use with ANY LLM of your choice for generation

This package handles ONLY:
1. Document indexing (PDF → Knowledge Graph)
2. Context retrieval (Query → Relevant context)

You provide your own LLM for generation.
"""

__version__ = "1.0.0"
__author__ = "DeepLightRAG Team"

# Core system
from .core import DeepLightRAG, GraphWrapper, RetrieverWrapper

# Graph components
from .graph.dual_layer import DualLayerGraph
from .graph.entity_relationship import Entity, EntityRelationshipGraph, Relationship
from .graph.visual_spatial import SpatialEdge, VisualNode, VisualSpatialGraph

# OCR components
from .ocr.deepseek_ocr import (
    BoundingBox,
    DeepSeekOCR,
    PageOCRResult,
    VisualRegion,
    VisualToken,
)

# PDF processing
from .ocr.processor import PDFProcessor
from .retrieval.adaptive_retriever import AdaptiveRetriever

# Retrieval components
from .retrieval.query_classifier import QueryClassifier, QueryLevel

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Main system
    "DeepLightRAG",
    "GraphWrapper",
    "RetrieverWrapper",
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
