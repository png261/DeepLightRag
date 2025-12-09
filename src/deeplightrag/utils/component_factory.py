"""
Component Factory
Creates instances of various components based on configuration.
Enables swapping technologies (e.g., changing OCR engines or Formatters) easily.
"""

import logging
from typing import Dict, Any, Optional

from ..interfaces import BaseOCRProcessor, BaseFormatter
from ..utils.config_manager import config_manager

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory for creating DeepLightRAG components"""
    
    @staticmethod
    def create_ocr_processor(config: Dict[str, Any]) -> BaseOCRProcessor:
        """
        Create an OCR processor based on configuration.
        
        Args:
            config: Full system configuration or OCR-specific config
        
        Returns:
            Instance of BaseOCRProcessor
        """
        # Extract OCR config if nested
        ocr_config = config.get("ocr", config)
        model_name = ocr_config.get("model_name", "")
        
        # Determine implementation based on model name or explicit 'type' field
        # For now, we default to DeepSeekOCR, but this is where we'd add logic for 
        # Tesseract, EasyOCR, GotOCR, etc.
        
        # Example agnostic logic:
        # if ocr_config.get("type") == "tesseract":
        #     from ..ocr.tesseract_ocr import TesseractOCR
        #     return TesseractOCR(**ocr_config)
            
        # Default: DeepSeek-OCR
        from ..ocr.deepseek_ocr import DeepSeekOCR
        
        # Prepare init kwargs
        init_kwargs = {
            "model_name": model_name,
            "quantization": ocr_config.get("quantization", "none"),
            "resolution": ocr_config.get("resolution", "base"),
        }
        
        # Add optional params if present in config
        optional_params = [
            "device", "torch_dtype", "batch_size", 
            "enable_visual_embeddings", "embedding_compression", 
            "target_embedding_dim", "use_mlx"
        ]
        
        for param in optional_params:
            if param in ocr_config:
                init_kwargs[param] = ocr_config[param]
                
        # Handle torch_dtype conversion if needed (moved from core.py)
        if "torch_dtype" in init_kwargs and isinstance(init_kwargs["torch_dtype"], str):
             import torch
             init_kwargs["torch_dtype"] = (
                 torch.float16 if init_kwargs["torch_dtype"] == "float16" else torch.float32
             )
        
        # Add image extraction config
        if "image_extraction" in config:
            img_conf = config["image_extraction"]
            if img_conf.get("enabled", True):
                init_kwargs.update({
                    "extract_images": True,
                    "image_output_dir": img_conf.get("output_dir", "./extracted_images"),
                    "image_formats": img_conf.get("formats", ["figure", "table", "chart"]),
                    "min_image_size": img_conf.get("min_size", (100, 100)),
                    "max_image_size": img_conf.get("max_size", (2000, 2000)),
                    "image_quality": img_conf.get("quality", 95),
                })
                
        logger.info(f"🏭 Factory creating OCR processor: DeepSeekOCR ({model_name})")
        return DeepSeekOCR(**init_kwargs)

    @staticmethod
    def create_formatter(config: Dict[str, Any]) -> BaseFormatter:
        """
        Create a Retrieval Formatter based on configuration.
        
        Args:
            config: Full system configuration
            
        Returns:
            Instance of BaseFormatter
        """
        retrieval_config = config.get("retrieval", {})
        formatter_type = retrieval_config.get("formatter_type", "toon")
        
        if formatter_type == "json":
            # Placeholder for JSON formatter
            # return JsonFormatter()
            raise NotImplementedError("JSON formatter not yet implemented")
            
        elif formatter_type == "markdown":
            # Placeholder
            raise NotImplementedError("Markdown formatter not yet implemented")
            
        else: # Default: TOON
            from ..retrieval.toon_formatter import ToonFormatter
            use_tabs = retrieval_config.get("use_tabs", False)
            logger.info("🏭 Factory creating Formatter: ToonFormatter")
            return ToonFormatter(use_tabs=use_tabs)

component_factory = ComponentFactory()
