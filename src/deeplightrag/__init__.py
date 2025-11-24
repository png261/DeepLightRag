"""
DeepLightRAG: Efficient Document-based RAG with Vision-Text Compression

A production-ready RAG system featuring:
- 9-10x Vision-Text Compression using DeepSeek-OCR
- Dual-Layer Graph Architecture (Visual-Spatial + Entity-Relationship)
- Adaptive Token Budgeting (2K-12K tokens vs 30K fixed)
- 60-80% cost savings compared to traditional RAG systems
"""

__version__ = "1.0.0"
__author__ = "DeepLightRAG Team"

# Core system
from .core import DeepLightRAG, GraphWrapper, RetrieverWrapper

# Graph components
from .graph.dual_layer import DualLayerGraph
from .graph.entity_relationship import Entity, EntityRelationshipGraph, Relationship
from .graph.visual_spatial import SpatialEdge, VisualNode, VisualSpatialGraph

# LLM components
from .llm.deepseek_r1 import DeepSeekR1

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
    # LLM
    "DeepSeekR1",
]
