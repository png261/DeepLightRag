"""
Enhanced OCR Pipeline for DeepLightRAG
Clean, maintainable, and generic implementation without hardcoded rules
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available. Some features will be limited.")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available. Some statistical features will be limited.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available. Memory monitoring will be limited.")


@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    confidence_threshold: float = 0.6
    quality_threshold: float = 0.5
    max_region_size: int = 1000
    min_region_size: int = 5
    embedding_dim: int = 768
    device: str = "auto"
    batch_size: int = 1
    cleanup_frequency: int = 10  # Clean up every N regions


@dataclass
class QualityMetrics:
    """Quality assessment metrics for regions"""
    clarity: float = 0.5
    contrast: float = 0.5
    noise: float = 0.5
    text_density: float = 0.5
    edge_density: float = 0.5
    overall: float = 0.5


@dataclass
class VisualToken:
    """Visual token with enhanced metadata"""
    token_id: int
    embedding: np.ndarray
    confidence: float
    region_type: str
    spatial_position: Tuple[float, float]
    quality_metrics: Optional[QualityMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize quality metrics if not provided"""
        if self.quality_metrics is None:
            self.quality_metrics = QualityMetrics()

    def is_high_quality(self) -> bool:
        """Check if token meets quality standards"""
        return (
            self.confidence >= 0.6 and
            self.quality_metrics.overall >= 0.5
        )


@dataclass
class BoundingBox:
    """Bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0, self.y2 - self.y1)

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class OCRRegion:
    """Generic OCR region without document type assumptions"""
    region_id: str
    bbox: BoundingBox
    text_content: str
    confidence: float
    visual_tokens: List[VisualToken] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize quality metrics if not provided"""
        if self.quality_metrics is None:
            self.quality_metrics = QualityMetrics()

    def has_visual_tokens(self) -> bool:
        """Check if region has visual tokens"""
        return len(self.visual_tokens) > 0

    def get_primary_token(self) -> Optional[VisualToken]:
        """Get the highest confidence visual token"""
        if not self.visual_tokens:
            return None
        return max(self.visual_tokens, key=lambda t: t.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "region_id": self.region_id,
            "bbox": self.bbox.to_dict(),
            "text_content": self.text_content,
            "confidence": self.confidence,
            "has_visual_tokens": self.has_visual_tokens(),
            "token_count": len(self.visual_tokens),
            "quality_score": self.quality_metrics.overall,
        }


class EnhancedOCRProcessor:
    """
    Enhanced OCR processor with clean, generic implementation
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.device = self._setup_device()
        self._region_counter = 0
        self._cleanup_counter = 0

        logger.info(f"Initialized Enhanced OCR with device: {self.device}")

    def _setup_device(self) -> str:
        """Setup processing device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device

    def process_page(self, image: Image.Image, page_num: int = 0) -> List[OCRRegion]:
        """
        Process a single page and extract OCR regions

        Args:
            image: PIL Image of the page
            page_num: Page number for identification

        Returns:
            List of OCR regions
        """
        logger.info(f"Processing page {page_num}")
        regions = []

        try:
            # Detect regions using adaptive method
            bboxes = self._detect_regions(image)

            # Process each detected region
            for bbox in bboxes:
                region = self._process_region(image, bbox, page_num)
                if region and region.confidence >= self.config.confidence_threshold:
                    regions.append(region)

                # Periodic cleanup
                self._maybe_cleanup()

            logger.info(f"Extracted {len(regions)} regions from page {page_num}")

        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")

        return regions

    def _detect_regions(self, image: Image.Image) -> List[BoundingBox]:
        """
        Detect text regions in the image

        Args:
            image: PIL Image to analyze

        Returns:
            List of bounding boxes for detected regions
        """
        # Convert to grayscale for analysis
        gray = image.convert("L")
        pixels = np.array(gray)

        # Simple projection-based region detection
        regions = self._detect_regions_by_projection(pixels, image.size)

        # Validate and merge overlapping regions
        regions = self._validate_and_merge_regions(regions, image.size)

        return regions

    def _detect_regions_by_projection(
        self, pixels: np.ndarray, image_size: Tuple[int, int]
    ) -> List[BoundingBox]:
        """Detect regions using horizontal and vertical projection"""
        height, width = pixels.shape
        regions = []

        # Horizontal projection to find text lines
        h_proj = np.sum(255 - pixels, axis=1)
        h_threshold = np.max(h_proj) * 0.1

        # Find horizontal segments
        in_segment = False
        start_y = 0

        for y, value in enumerate(h_proj):
            if value > h_threshold and not in_segment:
                in_segment = True
                start_y = y
            elif value <= h_threshold and in_segment:
                in_segment = False
                if y - start_y > 10:  # Minimum height
                    # Find vertical bounds for this segment
                    v_bounds = self._find_vertical_bounds(pixels, start_y, y)
                    if v_bounds:
                        x1, x2 = v_bounds
                        # Normalize coordinates
                        bbox = BoundingBox(
                            x1 / width,
                            start_y / height,
                            x2 / width,
                            y / height
                        )
                        regions.append(bbox)

        return regions

    def _find_vertical_bounds(
        self, pixels: np.ndarray, y_start: int, y_end: int
    ) -> Optional[Tuple[int, int]]:
        """Find vertical bounds for a horizontal segment"""
        if y_start >= y_end or y_end > pixels.shape[0]:
            return None

        segment_pixels = pixels[y_start:y_end, :]
        v_proj = np.sum(255 - segment_pixels, axis=0)

        if len(v_proj) == 0:
            return None

        v_threshold = np.max(v_proj) * 0.1

        # Find left bound
        x1 = 0
        for i, val in enumerate(v_proj):
            if val > v_threshold:
                x1 = i
                break

        # Find right bound
        x2 = len(v_proj) - 1
        for i in range(len(v_proj) - 1, -1, -1):
            if v_proj[i] > v_threshold:
                x2 = i
                break

        if x2 > x1:
            return (x1, x2 + 1)  # +1 for inclusive bounds
        return None

    def _validate_and_merge_regions(
        self, regions: List[BoundingBox], image_size: Tuple[int, int]
    ) -> List[BoundingBox]:
        """Validate and merge overlapping regions"""
        if not regions:
            return regions

        # Filter out regions that are too small or too large
        width, height = image_size
        valid_regions = []

        for region in regions:
            area = region.area() * width * height  # Convert to pixel area
            if (self.config.min_region_size <= area <= self.config.max_region_size and
                region.width() > 0 and region.height() > 0):
                valid_regions.append(region)

        # Merge overlapping regions
        merged = self._merge_overlapping_regions(valid_regions)

        return merged

    def _merge_overlapping_regions(self, regions: List[BoundingBox]) -> List[BoundingBox]:
        """Merge regions that overlap significantly"""
        if len(regions) <= 1:
            return regions

        merged = []
        used = set()

        for i, region1 in enumerate(regions):
            if i in used:
                continue

            current_region = region1
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                if self._regions_overlap(current_region, region2, threshold=0.5):
                    # Merge regions
                    current_region = self._merge_two_regions(current_region, region2)
                    used.add(j)

            merged.append(current_region)
            used.add(i)

        return merged

    def _regions_overlap(
        self, region1: BoundingBox, region2: BoundingBox, threshold: float = 0.5
    ) -> bool:
        """Check if two regions overlap beyond threshold"""
        # Calculate intersection
        x_overlap = max(0, min(region1.x2, region2.x2) - max(region1.x1, region2.x1))
        y_overlap = max(0, min(region1.y2, region2.y2) - max(region1.y1, region2.y1))

        # Calculate overlap ratios
        area1 = region1.area()
        area2 = region2.area()
        overlap_area = x_overlap * y_overlap

        if area1 == 0 or area2 == 0:
            return False

        overlap_ratio1 = overlap_area / area1
        overlap_ratio2 = overlap_area / area2

        return max(overlap_ratio1, overlap_ratio2) > threshold

    def _merge_two_regions(self, region1: BoundingBox, region2: BoundingBox) -> BoundingBox:
        """Merge two bounding boxes"""
        return BoundingBox(
            min(region1.x1, region2.x1),
            min(region1.y1, region2.y1),
            max(region1.x2, region2.x2),
            max(region1.y2, region2.y2)
        )

    def _process_region(
        self, image: Image.Image, bbox: BoundingBox, page_num: int
    ) -> Optional[OCRRegion]:
        """Process a single region"""
        self._region_counter += 1
        region_id = f"page_{page_num}_region_{self._region_counter}"

        try:
            # Crop region
            region_img = self._crop_region(image, bbox)

            # Extract text (placeholder for actual OCR implementation)
            text = self._extract_text(region_img)

            # Assess quality
            quality = self._assess_region_quality(region_img, text)

            # Create visual tokens if quality is sufficient
            tokens = []
            if quality.overall >= self.config.quality_threshold:
                embedding = self._create_embedding(region_img, quality)
                if embedding is not None:
                    token = VisualToken(
                        token_id=self._region_counter,
                        embedding=embedding,
                        confidence=quality.overall,
                        region_type="text",
                        spatial_position=bbox.center(),
                        quality_metrics=quality
                    )
                    tokens.append(token)

            # Calculate overall confidence
            confidence = self._calculate_region_confidence(text, quality, tokens)

            return OCRRegion(
                region_id=region_id,
                bbox=bbox,
                text_content=text,
                confidence=confidence,
                visual_tokens=tokens,
                quality_metrics=quality,
                metadata={"page_num": page_num}
            )

        except Exception as e:
            logger.error(f"Error processing region {region_id}: {e}")
            return None

    def _crop_region(self, image: Image.Image, bbox: BoundingBox) -> Image.Image:
        """Crop region from image"""
        w, h = image.size
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)

        # Ensure bounds are within image
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))

        return image.crop((left, top, right, bottom))

    def _extract_text(self, image: Image.Image) -> str:
        """
        Extract text from region image
        Placeholder for actual OCR implementation
        """
        # This would integrate with actual OCR model
        # For now, return placeholder
        return f"Extracted text from {image.size[0]}x{image.size[1]} region"

    def _assess_region_quality(self, image: Image.Image, text: str) -> QualityMetrics:
        """Assess quality of region"""
        gray = image.convert("L")
        pixels = np.array(gray)

        # Basic quality metrics
        clarity = self._calculate_clarity(pixels)
        contrast = np.std(pixels) / 255 if pixels.size > 0 else 0
        noise = self._estimate_noise(pixels)
        text_density = np.mean(255 - pixels) / 255 if pixels.size > 0 else 0
        edge_density = self._calculate_edge_density(pixels)

        # Calculate overall quality
        overall = (clarity * 0.3 + (1 - noise) * 0.2 + contrast * 0.2 + text_density * 0.15 + edge_density * 0.15)

        return QualityMetrics(
            clarity=clarity,
            contrast=contrast,
            noise=noise,
            text_density=text_density,
            edge_density=edge_density,
            overall=np.clip(overall, 0, 1)
        )

    def _calculate_clarity(self, pixels: np.ndarray) -> float:
        """Calculate image clarity using variance of Laplacian"""
        if pixels.size < 9:
            return 0.5

        # Simple Laplacian approximation
        laplacian = (
            pixels[1:-1, 1:-1] * 4 -
            pixels[:-2, 1:-1] - pixels[2:, 1:-1] -
            pixels[1:-1, :-2] - pixels[1:-1, 2:]
        )

        variance = np.var(laplacian)
        return np.clip(variance / 1000, 0, 1)

    def _estimate_noise(self, pixels: np.ndarray) -> float:
        """Estimate noise level"""
        if pixels.size < 9:
            return 0.5

        # Use high-frequency components as noise indicator
        h_diff = np.abs(np.diff(pixels, axis=0))
        v_diff = np.abs(np.diff(pixels, axis=1))
        noise = np.mean(h_diff) + np.mean(v_diff)

        return np.clip(noise / 50, 0, 1)

    def _calculate_edge_density(self, pixels: np.ndarray) -> float:
        """Calculate edge density"""
        if pixels.size < 9:
            return 0.5

        # Simple edge detection
        h_edges = np.abs(np.diff(pixels, axis=0))
        v_edges = np.abs(np.diff(pixels, axis=1))

        edge_pixels = np.sum(h_edges) + np.sum(v_edges)
        return np.clip(edge_pixels / (pixels.size * 255), 0, 1)

    def _create_embedding(
        self, image: Image.Image, quality: QualityMetrics
    ) -> Optional[np.ndarray]:
        """Create embedding for region"""
        try:
            # Convert to array
            img_array = np.array(image)

            # Create features
            features = []

            # Basic statistics
            if len(img_array.shape) == 3:
                # Color image
                for channel in range(3):
                    channel_data = img_array[:, :, channel].astype(float)
                    features.extend([
                        np.mean(channel_data) / 255,
                        np.std(channel_data) / 255,
                        np.percentile(channel_data, 25) / 255,
                        np.percentile(channel_data, 75) / 255,
                    ])
            else:
                # Grayscale
                gray = img_array.astype(float)
                features.extend([
                    np.mean(gray) / 255,
                    np.std(gray) / 255,
                    np.percentile(gray, 25) / 255,
                    np.percentile(gray, 75) / 255,
                ])

            # Texture features
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            features.extend([
                np.std(gray) / 255,
                np.mean(np.abs(np.diff(gray, axis=0))) / 255,
                np.mean(np.abs(np.diff(gray, axis=1))) / 255,
            ])

            # Quality features
            features.extend([
                quality.clarity,
                quality.contrast,
                quality.noise,
                quality.text_density,
                quality.edge_density,
            ])

            # Convert to numpy array
            embedding = np.array(features, dtype=np.float32)

            # Pad or truncate to target dimension
            target_dim = self.config.embedding_dim
            if len(embedding) < target_dim:
                # Pad with zeros
                embedding = np.pad(embedding, (0, target_dim - len(embedding)), 'constant')
            else:
                # Truncate
                embedding = embedding[:target_dim]

            return embedding

        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def _calculate_region_confidence(
        self, text: str, quality: QualityMetrics, tokens: List[VisualToken]
    ) -> float:
        """Calculate overall confidence for region"""
        # Base confidence from quality
        confidence = quality.overall * 0.6

        # Boost from text presence
        if text and len(text.strip()) > 5:
            confidence += 0.2

        # Boost from visual tokens
        if tokens:
            token_confidence = np.mean([t.confidence for t in tokens])
            confidence += token_confidence * 0.2

        return np.clip(confidence, 0, 1)

    def _maybe_cleanup(self):
        """Perform periodic cleanup"""
        self._cleanup_counter += 1

        if self._cleanup_counter >= self.config.cleanup_frequency:
            self.cleanup_memory()
            self._cleanup_counter = 0

    def cleanup_memory(self):
        """Clean up GPU memory if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleaned GPU memory")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        usage = {"cpu_mb": 0}

        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                usage["cpu_mb"] = process.memory_info().rss / 1024 / 1024
            except:
                pass

        if torch.cuda.is_available():
            usage["gpu_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return usage

    def batch_process(
        self, images: List[Image.Image], start_page: int = 0
    ) -> List[List[OCRRegion]]:
        """Process multiple images in batch"""
        results = []

        for i, image in enumerate(images):
            page_num = start_page + i
            regions = self.process_page(image, page_num)
            results.append(regions)

        return results

    def get_statistics(self, results: List[List[OCRRegion]]) -> Dict[str, Any]:
        """Get processing statistics"""
        total_regions = sum(len(page) for page in results)
        total_tokens = sum(
            len(region.visual_tokens)
            for page in results
            for region in page
        )

        # Calculate quality distribution
        all_qualities = []
        all_confidences = []

        for page in results:
            for region in page:
                all_qualities.append(region.quality_metrics.overall)
                all_confidences.append(region.confidence)
                for token in region.visual_tokens:
                    all_qualities.append(token.quality_metrics.overall)
                    all_confidences.append(token.confidence)

        stats = {
            "total_pages": len(results),
            "total_regions": total_regions,
            "total_tokens": total_tokens,
            "avg_regions_per_page": total_regions / len(results) if results else 0,
            "avg_tokens_per_region": total_tokens / total_regions if total_regions > 0 else 0,
            "avg_quality": np.mean(all_qualities) if all_qualities else 0,
            "avg_confidence": np.mean(all_confidences) if all_confidences else 0,
            "high_quality_regions": sum(1 for q in all_qualities if q > 0.7),
            "high_confidence_tokens": sum(1 for c in all_confidences if c > 0.7),
        }

        return stats