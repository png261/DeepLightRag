"""
Visual Embedding Extractor for DeepSeek OCR
Extracts dense visual representations from VLM hidden states
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import tempfile
import os
import platform

# MLX is only available on macOS
IS_MACOS = platform.system() == "Darwin"

if IS_MACOS:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        HAS_MLX = True
    except ImportError:
        HAS_MLX = False
else:
    HAS_MLX = False


class VisualEmbeddingExtractor:
    """
    Extracts visual embeddings from VLM for regions
    Provides different compression strategies and pooling methods
    """

    def __init__(
        self,
        model=None,
        processor=None,
        config=None,
        device="cpu",
        compression_method="pca",
        target_dim=256,
    ):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.compression_method = compression_method
        self.target_dim = target_dim
        self.embedding_cache = {}

    def extract_region_embedding(
        self, image: Image.Image, bbox: List[float], region_type: str = "general"
    ) -> Tuple[np.ndarray, float]:
        """
        Extract visual embedding for a specific region

        Args:
            image: PIL Image
            bbox: Normalized bounding box [x1, y1, x2, y2]
            region_type: Type of region (header, paragraph, table, etc.)

        Returns:
            Tuple of (embedding_vector, confidence_score)
        """
        # Crop region
        w, h = image.size
        left = int(bbox[0] * w)
        top = int(bbox[1] * h)
        right = int(bbox[2] * w)
        bottom = int(bbox[3] * h)

        # Ensure valid dimensions
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))

        region_img = image.crop((left, top, right, bottom))

        # Extract embedding using VLM
        return self._extract_with_vlm(region_img, region_type)

    def _extract_with_vlm(self, image: Image.Image, region_type: str) -> Tuple[np.ndarray, float]:
        """Extract embedding using VLM hidden states"""
        if not HAS_MLX or self.model is None:
            return self._mock_embedding(image, region_type)

        # Save temp image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        try:
            # Create embedding-focused prompt
            prompt = self._create_embedding_prompt(region_type)

            formatted_prompt = apply_chat_template(
                self.processor, self.config, prompt, num_images=1
            )

            # Extract hidden states during generation
            embedding, confidence = self._extract_hidden_states(formatted_prompt, image_path)

            return embedding, confidence

        finally:
            if os.path.exists(image_path):
                os.unlink(image_path)

    def _extract_hidden_states(self, prompt: str, image_path: str) -> Tuple[np.ndarray, float]:
        """Extract and pool hidden states from VLM"""
        try:
            # This is a simplified version - in practice, you'd need to:
            # 1. Hook into the model's forward pass
            # 2. Extract hidden states from vision encoder
            # 3. Pool/compress the representations

            # For now, use the generation output as a proxy
            output = generate(
                self.model, self.processor, prompt, [image_path], max_tokens=50, temp=0.1
            )

            # Extract embedding from model's internal representation
            # This is a placeholder - real implementation would access hidden states
            embedding = self._simulate_embedding_extraction()
            confidence = 0.85  # Placeholder confidence

            return embedding, confidence

        except Exception as e:
            print(f"VLM embedding extraction failed: {e}")
            return self._mock_embedding_from_error()

    def _simulate_embedding_extraction(self) -> np.ndarray:
        """Simulate embedding extraction - replace with real hidden state access"""
        # In real implementation, this would:
        # 1. Access vision transformer's last hidden state
        # 2. Apply pooling (mean, attention, CLS token)
        # 3. Apply optional compression

        # For now, return a mock 768-dimensional embedding
        np.random.seed(42)  # For reproducibility
        return np.random.randn(768).astype(np.float32)

    def _mock_embedding(self, image: Image.Image, region_type: str) -> Tuple[np.ndarray, float]:
        """Mock embedding for testing"""
        # Create deterministic embedding based on image properties
        w, h = image.size
        pixel_mean = np.array(image).mean()

        # Create a simple feature vector
        features = [
            w / 1000.0,  # normalized width
            h / 1000.0,  # normalized height
            pixel_mean / 255.0,  # normalized pixel intensity
            len(region_type) / 10.0,  # region type encoding
        ]

        # Pad to 768 dimensions with noise
        np.random.seed(int(pixel_mean))
        embedding = np.concatenate([features, np.random.randn(764) * 0.1]).astype(np.float32)

        confidence = 0.7
        return embedding, confidence

    def _mock_embedding_from_error(self) -> Tuple[np.ndarray, float]:
        """Fallback embedding when extraction fails"""
        return np.zeros(768, dtype=np.float32), 0.1

    def _create_embedding_prompt(self, region_type: str) -> str:
        """Create prompt optimized for embedding extraction"""
        prompts = {
            "header": "Analyze this header region and understand its visual structure and semantic meaning.",
            "paragraph": "Examine this text paragraph and capture its visual layout and content semantics.",
            "table": "Study this table structure and understand both its visual organization and data relationships.",
            "figure": "Analyze this figure/image and understand its visual content and contextual meaning.",
            "caption": "Examine this caption and understand how it relates to nearby visual elements.",
            "list": "Analyze this list structure and understand its visual hierarchy and content organization.",
            "formula": "Study this mathematical formula and understand its visual notation and semantic structure.",
        }

        base_prompt = prompts.get(
            region_type, "Analyze this visual region and understand its content and structure."
        )
        return f"{base_prompt} Focus on extracting meaningful visual-semantic representations."

    def compress_embedding(
        self, embedding: np.ndarray, method: str = "pca", target_dim: int = 256
    ) -> np.ndarray:
        """
        Compress embedding using various methods

        Args:
            embedding: Original embedding vector
            method: Compression method ('pca', 'quantize', 'sparse')
            target_dim: Target dimensionality

        Returns:
            Compressed embedding
        """
        if method == "pca":
            return self._pca_compress(embedding, target_dim)
        elif method == "quantize":
            return self._quantize_embedding(embedding)
        elif method == "sparse":
            return self._sparse_embedding(embedding, target_dim)
        else:
            return embedding[:target_dim]  # Simple truncation

    def _pca_compress(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """PCA compression (simplified version)"""
        # In practice, you'd fit PCA on a corpus of embeddings
        # For now, just return top dimensions
        return embedding[:target_dim]

    def _quantize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """8-bit quantization"""
        # Quantize to 8-bit
        min_val, max_val = embedding.min(), embedding.max()
        quantized = ((embedding - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        # Dequantize
        return (quantized.astype(np.float32) / 255.0) * (max_val - min_val) + min_val

    def _sparse_embedding(self, embedding: np.ndarray, k: int) -> np.ndarray:
        """Keep only top-k dimensions"""
        indices = np.argsort(np.abs(embedding))[-k:]
        sparse = np.zeros_like(embedding)
        sparse[indices] = embedding[indices]
        return sparse

    def batch_extract_embeddings(
        self, image: Image.Image, regions_data: List[Dict[str, Any]]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract embeddings for multiple regions efficiently

        Args:
            image: Source image
            regions_data: List of region info dicts with 'bbox' and 'region_type'

        Returns:
            List of (embedding, confidence) tuples
        """
        embeddings = []

        for region_data in regions_data:
            bbox = region_data["bbox"]
            region_type = region_data.get("region_type", "general")

            embedding, confidence = self.extract_region_embedding(image, bbox, region_type)
            embeddings.append((embedding, confidence))

        return embeddings

    def get_embedding_stats(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate statistics for a set of embeddings"""
        if not embeddings:
            return {}

        embeddings_array = np.stack(embeddings)

        return {
            "count": len(embeddings),
            "dimensions": embeddings_array.shape[1],
            "mean_norm": np.linalg.norm(embeddings_array, axis=1).mean(),
            "std_norm": np.linalg.norm(embeddings_array, axis=1).std(),
            "sparsity": (embeddings_array == 0).mean(),
            "memory_kb": embeddings_array.nbytes / 1024,
        }
