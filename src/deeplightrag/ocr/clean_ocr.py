"""
Clean OCR Implementation - No Mock Tests
Simplified, maintainable OCR pipeline
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    confidence_threshold: float = 0.6
    quality_threshold: float = 0.5
    device: str = "auto"


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    clarity: float = 0.5
    contrast: float = 0.5
    noise: float = 0.5
    overall: float = 0.5


@dataclass
class VisualToken:
    """Visual token with metadata"""
    token_id: int
    embedding: np.ndarray
    confidence: float
    region_type: str
    spatial_position: Tuple[float, float]
    quality_metrics: QualityMetrics = field(default_factory=lambda: QualityMetrics())


@dataclass
class BoundingBox:
    """Bounding box"""
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class OCRRegion:
    """OCR region with extracted content"""
    region_id: str
    bbox: BoundingBox
    text_content: str
    confidence: float
    visual_tokens: List[VisualToken] = field(default_factory=list)
    quality_metrics: QualityMetrics = field(default_factory=lambda: QualityMetrics())

    def has_visual_tokens(self) -> bool:
        return len(self.visual_tokens) > 0


class CleanOCRProcessor:
    """Clean, maintainable OCR processor"""

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.device = self._setup_device()
        logger.info(f"Initialized Clean OCR with device: {self.device}")

    def _setup_device(self) -> str:
        """Setup device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device

    def process_page(self, image: Image.Image, page_num: int = 0) -> List[OCRRegion]:
        """Process a single page"""
        logger.info(f"Processing page {page_num}")
        regions = []

        try:
            # Detect regions
            bboxes = self._detect_regions(image)

            # Process each region
            for bbox in bboxes:
                region = self._process_region(image, bbox, page_num)
                if region and region.confidence >= self.config.confidence_threshold:
                    regions.append(region)

            logger.info(f"Extracted {len(regions)} regions")

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")

        return regions

    def _detect_regions(self, image: Image.Image) -> List[BoundingBox]:
        """Detect regions in image"""
        gray = image.convert("L")
        pixels = np.array(gray)
        return self._detect_by_projection(pixels, image.size)

    def _detect_by_projection(
        self, pixels: np.ndarray, image_size: Tuple[int, int]
    ) -> List[BoundingBox]:
        """Detect regions using projection"""
        height, width = pixels.shape
        regions = []

        # Horizontal projection
        h_proj = np.sum(255 - pixels, axis=1)
        threshold = np.max(h_proj) * 0.05 if np.max(h_proj) > 0 else 0

        # Find text lines
        in_line = False
        start_y = 0

        for y, val in enumerate(h_proj):
            if val > threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= threshold and in_line:
                in_line = False
                if y - start_y > 10:  # Minimum height
                    x1, x2 = self._find_vertical_bounds(pixels, start_y, y)
                    if x1 is not None and x2 is not None:
                        bbox = BoundingBox(
                            x1 / width, start_y / height,
                            x2 / width, y / height
                        )
                        regions.append(bbox)

        return regions

    def _find_vertical_bounds(
        self, pixels: np.ndarray, y_start: int, y_end: int
    ) -> Optional[Tuple[int, int]]:
        """Find vertical bounds for segment"""
        if y_start >= y_end or y_end > pixels.shape[0]:
            return None

        segment = pixels[y_start:y_end, :]
        v_proj = np.sum(255 - segment, axis=0)

        if len(v_proj) == 0:
            return None

        threshold = np.max(v_proj) * 0.05 if np.max(v_proj) > 0 else 0

        # Find bounds
        x1 = 0
        for i, val in enumerate(v_proj):
            if val > threshold:
                x1 = i
                break

        x2 = len(v_proj) - 1
        for i in range(len(v_proj) - 1, -1, -1):
            if v_proj[i] > threshold:
                x2 = i
                break

        return (x1, x2 + 1) if x2 > x1 else None

    def _process_region(
        self, image: Image.Image, bbox: BoundingBox, page_num: int
    ) -> Optional[OCRRegion]:
        """Process a region"""
        try:
            region_id = f"page_{page_num}_region_{hash((bbox.x1, bbox.y1, bbox.x2, bbox.y2)) % 10000}"

            # Crop region
            region_img = self._crop_region(image, bbox)

            # Extract text
            text = self._extract_text(region_img)

            # Assess quality
            quality = self._assess_quality(region_img)

            # Create visual token if quality is good
            tokens = []
            if quality.overall >= self.config.quality_threshold:
                embedding = self._create_embedding(region_img)
                if embedding is not None:
                    token = VisualToken(
                        token_id=hash((bbox.x1, bbox.y1, bbox.x2, bbox.y2)) % 10000,
                        embedding=embedding,
                        confidence=quality.overall,
                        region_type="text",
                        spatial_position=bbox.center(),
                        quality_metrics=quality
                    )
                    tokens.append(token)

            # Calculate confidence
            confidence = min(1.0, quality.overall * 1.2)

            return OCRRegion(
                region_id=region_id,
                bbox=bbox,
                text_content=text,
                confidence=confidence,
                visual_tokens=tokens,
                quality_metrics=quality
            )

        except Exception as e:
            logger.error(f"Error processing region: {e}")
            return None

    def _crop_region(self, image: Image.Image, bbox: BoundingBox) -> Image.Image:
        """Crop region from image"""
        w, h = image.size
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)

        # Ensure bounds are valid
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))

        return image.crop((left, top, right, bottom))

    def _extract_text(self, image: Image.Image) -> str:
        """Extract text from image (placeholder)"""
        # This would integrate with actual OCR model
        return f"Text from {image.size[0]}x{image.size[1]} region"

    def _assess_quality(self, image: Image.Image) -> QualityMetrics:
        """Assess image quality"""
        gray = image.convert("L")
        pixels = np.array(gray)

        # Basic metrics
        clarity = self._calculate_clarity(pixels)
        contrast = np.std(pixels) / 255 if pixels.size > 0 else 0
        noise = self._estimate_noise(pixels)

        # Overall quality
        overall = (clarity + (1 - noise) + min(1, contrast * 2)) / 3

        return QualityMetrics(
            clarity=clarity,
            contrast=contrast,
            noise=noise,
            overall=min(1.0, overall)
        )

    def _calculate_clarity(self, pixels: np.ndarray) -> float:
        """Calculate clarity using variance"""
        if pixels.size < 9:
            return 0.5

        laplacian = (
            pixels[1:-1, 1:-1] * 4 -
            pixels[:-2, 1:-1] - pixels[2:, 1:-1] -
            pixels[1:-1, :-2] - pixels[1:-1, 2:]
        )

        variance = np.var(laplacian)
        return min(1.0, variance / 1000)

    def _estimate_noise(self, pixels: np.ndarray) -> float:
        """Estimate noise level"""
        if pixels.size < 9:
            return 0.5

        h_diff = np.abs(np.diff(pixels, axis=0))
        v_diff = np.abs(np.diff(pixels, axis=1))
        noise = (np.mean(h_diff) + np.mean(v_diff)) / 2

        return min(1.0, noise / 50)

    def _create_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Create embedding for image"""
        try:
            img_array = np.array(image)
            features = []

            # Basic features
            if len(img_array.shape) == 3:
                for channel in range(3):
                    channel = img_array[:, :, channel].astype(float)
                    features.extend([
                        np.mean(channel) / 255,
                        np.std(channel) / 255,
                    ])
            else:
                gray = img_array.astype(float)
                features.extend([
                    np.mean(gray) / 255,
                    np.std(gray) / 255,
                ])

            # Texture features
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            features.extend([
                np.std(gray) / 255,
                np.mean(np.abs(np.diff(gray, axis=0))) / 255,
                np.mean(np.abs(np.diff(gray, axis=1))) / 255,
            ])

            # Convert and ensure correct size
            embedding = np.array(features, dtype=np.float32)
            embedding = np.pad(embedding, (0, max(0, 512 - len(embedding))))[:512]

            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def cleanup_memory(self):
        """Clean GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        memory_info = {
            "device": self.device,
            "torch_allocated": 0,
            "torch_cached": 0,
        }

        if torch.cuda.is_available():
            memory_info.update({
                "torch_allocated": torch.cuda.memory_allocated() / (1024 ** 3),  # GB
                "torch_cached": torch.cuda.memory_reserved() / (1024 ** 3),  # GB
            })

        return memory_info

    def batch_process(
        self, images: List[Image.Image], start_page: int = 0
    ) -> List[List[OCRRegion]]:
        """Process multiple images"""
        return [self.process_page(img, start_page + i) for i, img in enumerate(images)]