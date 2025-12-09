"""
Component Interfaces for DeepLightRAG
Defines abstract base classes for swappable components (OCR, Formatter, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from .ocr.visual_token import VisualRegion

class BaseOCRProcessor(ABC):
    """Abstract base class for OCR/VLM processors"""
    
    @abstractmethod
    def process_image(self, image: Image.Image, page_num: int = 0) -> List[VisualRegion]:
        """
        Process a single image page and return structured regions.
        
        Args:
            image: PIL Image of the page
            page_num: Page number
            
        Returns:
            List of VisualRegion objects containing text and bounding boxes
        """
        pass
    
    @abstractmethod
    def batch_process(self, images: List[Image.Image], start_page: int = 0, show_progress: bool = False) -> List[Any]:
        """Process a batch of images."""
        pass

    @abstractmethod
    def get_compression_stats(self, results: List) -> Dict[str, Any]:
        """
        Calculate compression statistics for processed pages.
        
        Args:
            results: List of PageOCRResult objects
            
        Returns:
            Dictionary with compression stats
        """
        pass

class BaseFormatter(ABC):
    """Abstract base class for retrieval result formatting"""
    
    @abstractmethod
    def format_retrieval_result(self, result: Dict[str, Any]) -> str:
        """
        Format full retrieval result for LLM context.
        
        Args:
            result: Retrieval result dictionary
            
        Returns:
            Formatted string
        """
        pass
    
    @abstractmethod
    def format_simple_context(self, result: Dict[str, Any]) -> str:
        """
        Format simplified context (content-focused).
        
        Args:
            result: Retrieval result dictionary
            
        Returns:
            Formatted string
        """
        pass
