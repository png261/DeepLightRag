"""
Configuration Management Utility
Handles default configuration and hardware acceleration setup
"""

import logging
import platform
from typing import Dict, Any, Optional
from pathlib import Path

from .device import device_manager

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages DeepLightRAG configuration"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file and merge with defaults"""
        import yaml
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
            
        # Get defaults
        default_config = ConfigManager.get_default_config()
        
        # Deep merge
        return ConfigManager._deep_merge(default_config, user_config)

    @staticmethod
    def _deep_merge(base: Dict, update: Dict) -> Dict:
        """Recursive dictionary merge"""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Generate default configuration optimized for detected hardware"""
        import torch
        
        is_macos = platform.system() == "Darwin"
        device_info = device_manager.device_info
        
        # Use defaults based on hardware detection from device_manager
        if device_info["cuda_available"]:
            # GPU configuration - better models
            logger.info("  🎮 Applying CUDA-optimized default configuration")
            ocr_config = {
                "model_name": "deepseek-ai/deepseek-ocr",
                "quantization": "none",
                "resolution": "large",
                "device": "cuda",
                "torch_dtype": "float16",
                "batch_size": 4,
                "enable_visual_embeddings": True,
                "embedding_compression": "pca",
                "target_embedding_dim": 512,
            }
            ner_config = {
                "model_name": "fastino/gliner2-base-v1",
                "device": "cuda",
            }
            re_config = {
                "model_name": "jackboyla/glirel-large-v0",
                "device": "cuda",
                "confidence_threshold": 0.3,
                "max_length": 512,
                "top_k": 3,
            }
        else:
            # CPU configuration - use MLX on macOS, transformers elsewhere
            if is_macos and device_info["mps_available"]:
                # macOS: Use MLX 4-bit quantized model
                logger.info("  🚀 Applying Apple Silicon (MLX) default configuration")
                ocr_config = {
                    "model_name": "mlx-community/DeepSeek-OCR-4bit",
                    "quantization": "none",
                    "resolution": "base",
                    "device": "cpu", # MLX handles device internally
                    "batch_size": 2,
                    "enable_visual_embeddings": True,
                    "embedding_compression": "pca",
                    "target_embedding_dim": 256,
                }
            else:
                # Other platforms: Use transformers CPU mode
                logger.info("  ⚠️ Applying CPU-optimized default configuration")
                ocr_config = {
                    "model_name": "deepseek-ai/deepseek-ocr",
                    "quantization": "8bit",  # CPU-friendly quantization
                    "resolution": "base",
                    "device": "cpu",
                    "batch_size": 2,
                    "enable_visual_embeddings": True,
                    "embedding_compression": "pca",
                    "target_embedding_dim": 256,
                }
            ner_config = {
                "model_name": "urchade/gliner_small-v2.1",
                "device": "cpu",
            }
            re_config = {
                "model_name": "urchade/gliner_small-v2.1",
                "device": "cpu",
                "confidence_threshold": 0.35,
                "max_length": 512,
                "top_k": 2,
            }

        return {
            "ocr": ocr_config,
            "ner": ner_config,
            "relation_extraction": re_config,
            "image_extraction": {
                "enabled": True,
                "output_dir": "./extracted_images",
                "formats": ["figure", "table", "chart"],
                "min_size": (100, 100),
                "max_size": (2000, 2000),
                "quality": 95,
            },
            "retrieval": {
                "enable_adaptive": True,
                "default_level": 2,
                "visual_weight": 0.3,
                "enable_visual_fallback": True,
                "include_images": True,
            },
            "models": {
                "query_classifier": {
                    "save_path": "query_classifier_model"
                }
            },
            "storage": {
                "base_dir": "./deep_light_rag_data",
                "model_dir": "models",
            }
        }

    @staticmethod
    def setup_gpu_optimization(config: Dict[str, Any], rag_instance: Any = None) -> str:
        """
        Auto-detect and configure hardware acceleration.
        Returns the primary device string.
        """
        device = device_manager.get_torch_device()
        
        # Log status
        if device_manager.device_info["mps_available"]:
            print(f"  🚀 Apple Silicon Acceleration (MPS) Enabled")
        elif device_manager.device_info["cuda_available"]:
            print(f"  🎮 GPU Acceleration (CUDA) Enabled")
        else:
            print(f"  ⚠️ Running on CPU")

        # Update config with detected device unless CPU is explicitly requested
        if "ocr" in config and config["ocr"].get("device") != "cpu":
            config["ocr"]["device"] = device
        
        if "ner" in config and config["ner"].get("device") != "cpu":
            config["ner"]["device"] = device
            
        if "relation_extraction" in config and config["relation_extraction"].get("device") != "cpu":
            config["relation_extraction"]["device"] = device
            
        # If components are forced to cpu, return cpu
        if config.get("ocr", {}).get("device") == "cpu" and config.get("ner", {}).get("device") == "cpu":
            return "cpu"
            
        return device

config_manager = ConfigManager()
