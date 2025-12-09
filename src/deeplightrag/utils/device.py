"""
Device Management Utility
Handles detection and configuration of compute devices (MPS, CUDA, CPU)
"""

import logging
import platform
import os

logger = logging.getLogger(__name__)

class DeviceManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._detect_devices()
        return cls._instance
    
    def _detect_devices(self):
        """Detect available compute devices"""
        self.device_info = {
            "torch_device": "cpu",
            "mlx_device": "cpu",
            "mps_available": False,
            "cuda_available": False,
            "platform": platform.system()
        }
        
        # 1. Detect PyTorch Device
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device_info["torch_device"] = "mps"
                self.device_info["mps_available"] = True
                logger.info("🚀 PyTorch using MPS (Apple Silicon) acceleration")
            elif torch.cuda.is_available():
                self.device_info["torch_device"] = "cuda"
                self.device_info["cuda_available"] = True
                logger.info(f"🚀 PyTorch using CUDA acceleration ({torch.cuda.get_device_name(0)})")
            else:
                logger.info("⚠️ PyTorch using CPU (No acceleration detected)")
        except ImportError:
            logger.warning("PyTorch not installed or import failed")

        # 2. Detect MLX Device (Apple Silicon only)
        try:
            import mlx.core as mx
            # MLX defaults to GPU on Apple Silicon, but we can check default device
            default_device = mx.default_device()
            self.device_info["mlx_device"] = str(default_device)
            # MLX only runs on Apple Silicon effectively for now
            if "gpu" in str(default_device):
                logger.info(f"🚀 MLX using GPU acceleration")
            else:
                logger.info(f"MLX using device: {default_device}")
        except ImportError:
            logger.debug("MLX not installed (Expected if not on macOS or using standard torch env)")

    def get_torch_device(self):
        """Get the optimal PyTorch device"""
        return self.device_info["torch_device"]

    def get_mlx_device_type(self):
        """Get MLX device type string"""
        return self.device_info["mlx_device"]
    
    def log_device_status(self):
        """Log current device configuration"""
        logger.info(f"Device Configuration: Torch={self.device_info['torch_device']}, MLX={self.device_info['mlx_device']}")

# Global instance
device_manager = DeviceManager()
