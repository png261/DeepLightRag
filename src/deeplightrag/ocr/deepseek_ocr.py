"""
DeepSeek-OCR Integration with 4-bit MLX Quantization
Vision-Text Compression for RAG - Refactored Facade
"""

# Re-export components for backward compatibility
from .geometry import BoundingBox
from .visual_token import VisualToken, VisualRegion
from .deepseek_model import DeepSeekOCR, DeepSeekOCR as DeepSeekOcrModel, PageOCRResult

__all__ = ["DeepSeekOCR", "VisualToken", "VisualRegion", "BoundingBox", "PageOCRResult"]
