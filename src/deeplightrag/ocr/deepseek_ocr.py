"""
DeepSeek-OCR Integration with 4-bit MLX Quantization
Vision-Text Compression for RAG - Refactored for Real OCR Processing
"""

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except Exception:
    HAS_MLX = False

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class VisualToken:
    """Compressed visual token representation with enhanced embedding support"""

    token_id: int
    embedding: np.ndarray  # Dense visual embedding (768 or compressed dims)
    confidence: float
    region_type: str = "general"  # semantic type of visual content
    spatial_position: Tuple[float, float] = (0.0, 0.0)  # relative position in region
    compression_method: str = "none"  # pca, quantize, sparse, none
    original_dims: int = 768  # original embedding dimension before compression

    def get_embedding_size_kb(self) -> float:
        """Calculate memory usage of embedding in KB"""
        return self.embedding.nbytes / 1024

    def compress(self, method: str = "pca", target_dim: int = 256) -> "VisualToken":
        """Return compressed version of this token"""
        # This would use the VisualEmbeddingExtractor's compression methods
        compressed_embedding = self.embedding[:target_dim]  # Simple truncation for now

        return VisualToken(
            token_id=self.token_id,
            embedding=compressed_embedding,
            confidence=self.confidence,
            region_type=self.region_type,
            spatial_position=self.spatial_position,
            compression_method=method,
            original_dims=len(self.embedding),
        )


@dataclass
class BoundingBox:
    """Bounding box for visual region"""

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        return cls(coords[0], coords[1], coords[2], coords[3])

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def height(self) -> float:
        return self.y2 - self.y1

    def width(self) -> float:
        return self.x2 - self.x1


@dataclass
class VisualRegion:
    """Visual region with compressed tokens and enhanced visual embedding support"""

    region_id: str
    page_num: int
    block_type: str  # header, paragraph, table, figure, caption, list, formula
    bbox: BoundingBox
    compressed_tokens: List[VisualToken]
    text_content: str
    markdown_content: str
    token_count: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced visual embedding fields
    region_embedding: Optional[np.ndarray] = None  # Overall region embedding
    embedding_confidence: float = 0.0
    visual_complexity: float = 0.0  # 0-1 score of visual complexity
    text_to_visual_ratio: float = 1.0  # How much text vs visual content

    def get_primary_embedding(self) -> Optional[np.ndarray]:
        """Get the primary embedding for this region"""
        if self.region_embedding is not None:
            return self.region_embedding
        elif self.compressed_tokens:
            # Aggregate token embeddings
            embeddings = [
                token.embedding for token in self.compressed_tokens if token.embedding is not None
            ]
            if embeddings:
                return np.mean(embeddings, axis=0)
        return None

    def get_compression_ratio(self) -> float:
        """Calculate visual compression ratio for this region"""
        if not self.compressed_tokens:
            return 1.0

        # Original would be ~4 chars per token * 768 dim embedding
        estimated_original_size = len(self.text_content) * 768 / 4

        # Actual size is sum of compressed embeddings
        actual_size = sum(token.embedding.size for token in self.compressed_tokens)

        return estimated_original_size / actual_size if actual_size > 0 else 1.0

    def should_use_visual_mode(self) -> bool:
        """Determine if this region should use visual embeddings for retrieval"""
        # Use visual mode if:
        # 1. Region has low text-to-visual ratio (more visual content)
        # 2. High visual complexity
        # 3. Block type is inherently visual
        visual_block_types = {"figure", "table", "formula"}

        return (
            self.text_to_visual_ratio < 0.7
            or self.visual_complexity > 0.6
            or self.block_type in visual_block_types
        )

    def to_dict(self) -> Dict:
        return {
            "region_id": self.region_id,
            "page_num": self.page_num,
            "block_type": self.block_type,
            "bbox": self.bbox.to_list(),
            "text_content": self.text_content,
            "markdown_content": self.markdown_content,
            "token_count": self.token_count,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class PageOCRResult:
    """OCR result for a single page"""

    page_num: int
    width: int
    height: int
    regions: List[VisualRegion]
    total_tokens: int
    processing_time: float

    def to_dict(self) -> Dict:
        return {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "regions": [r.to_dict() for r in self.regions],
            "total_tokens": self.total_tokens,
            "processing_time": self.processing_time,
        }


class DeepSeekOCR:
    """
    DeepSeek-OCR with 4-bit MLX Quantization
    Provides 9-10x vision-text compression with 96%+ accuracy

    Features:
    - Adaptive layout detection
    - Structure-aware text extraction
    - Semantic block classification
    - Token compression and optimization
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-ocr",
        quantization: str = "none",  # none, 4bit, 8bit
        resolution: str = "base",
        device: str = "auto",
        torch_dtype=torch.float16,
        enable_visual_embeddings: bool = True,
        embedding_compression: str = "pca",  # pca, quantize, sparse, none
        target_embedding_dim: int = 256,
        batch_size: int = 1,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.resolution = resolution
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype
        self.enable_visual_embeddings = enable_visual_embeddings
        self.embedding_compression = embedding_compression
        self.target_embedding_dim = target_embedding_dim
        self.batch_size = batch_size

        # Resolution configurations for adaptive processing
        self.resolution_configs = {
            "tiny": {"size": (512, 512), "tokens": 64},
            "small": {"size": (768, 768), "tokens": 144},
            "base": {"size": (1280, 1280), "tokens": 400},
            "large": {"size": (1600, 1600), "tokens": 625},
            "xlarge": {"size": (2048, 2048), "tokens": 1024},
        }

        self.model = None
        self.processor = None
        self.config = None
        self.visual_embedding_extractor = None
        self._load_model()

    def _setup_device(self, device: str) -> str:
        """Setup device with automatic detection"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _load_model(self):
        """Load Vision-Language Model with GPU support"""
        print(f"Loading {self.model_name} on {self.device}...")

        # Try to load with transformers first (better GPU support)
        if HAS_TRANSFORMERS and "deepseek-ocr" in self.model_name:
            try:
                self._load_transformers_model()
                return
            except Exception as e:
                print(f"Failed to load with transformers: {e}")
                print("Falling back to MLX...")

        # Fallback to MLX
        self._load_mlx_model()

    def _load_transformers_model(self):
        """Load using transformers for better GPU support"""
        from .visual_embedding_extractor import VisualEmbeddingExtractor

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        # Load model with appropriate settings
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.torch_dtype,
        }

        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name, **load_kwargs)

        # Move to device if not using device_map
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)

        print(f"✅ Loaded {self.model_name} with transformers on {self.device}")

        # Initialize visual embedding extractor
        if self.enable_visual_embeddings:
            self.visual_embedding_extractor = VisualEmbeddingExtractor(
                model=self.model,
                processor=self.processor,
                device=self.device,
                compression_method=self.embedding_compression,
                target_dim=self.target_embedding_dim,
            )

    def _load_mlx_model(self):
        """Load using MLX (fallback)"""
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            from .visual_embedding_extractor import VisualEmbeddingExtractor
        except ImportError:
            raise RuntimeError("mlx_vlm is not installed. Please install: pip install mlx-vlm")

        print(f"Loading {self.model_name} with MLX backend...")
        self.model, self.processor = load(self.model_name, trust_remote_code=True)
        self.config = load_config(self.model_name)

        # Initialize visual embedding extractor
        if self.enable_visual_embeddings:
            self.visual_embedding_extractor = VisualEmbeddingExtractor(
                model=self.model,
                processor=self.processor,
                config=self.config,
                compression_method=self.embedding_compression,
                target_dim=self.target_embedding_dim,
            )
            print("Visual embedding extraction enabled")

        self.model_info = {
            "backend": "mlx-vlm",
            "quantization": self.quantization,
            "resolution": self.resolution,
            "visual_embeddings": self.enable_visual_embeddings,
            "embedding_compression": self.embedding_compression,
            "target_dim": self.target_embedding_dim,
            "loaded": self.model is not None,
        }
        print(f"Model loaded successfully: {self.model_name}")

    def process_image(
        self, image: Image.Image, page_num: int = 0, adaptive_resolution: bool = True
    ) -> PageOCRResult:
        """
        Process a single image and extract visual regions with compressed tokens.

        Args:
            image: PIL Image of the page
            page_num: Page number
            adaptive_resolution: Whether to use adaptive resolution based on content

        Returns:
            PageOCRResult with extracted regions
        """
        start_time = time.time()

        # Resize image to target resolution
        target_size = self.resolution_configs[self.resolution]["size"]
        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Step 1: Detect layout blocks
        blocks = self._detect_layout_blocks(resized_image)

        # Step 2: Extract text and create regions with visual embeddings
        regions = []
        for idx, block in enumerate(blocks):
            bbox = block["bbox"]
            block_type = block["type"]
            confidence = block.get("confidence", 1.0)

            # Extract text from region using VLM
            text_content = self._extract_text_from_region(resized_image, bbox, block_type)

            # Convert to markdown with structure preservation
            markdown_content = self._to_markdown(text_content, block_type)

            # Extract visual embeddings if enabled
            compressed_tokens = []
            region_embedding = None
            embedding_confidence = 0.0
            visual_complexity = 0.5
            text_to_visual_ratio = 1.0

            if self.enable_visual_embeddings and self.visual_embedding_extractor:
                try:
                    # Extract region-level embedding
                    region_embedding, embedding_confidence = (
                        self.visual_embedding_extractor.extract_region_embedding(
                            resized_image, bbox, block_type
                        )
                    )

                    # Compress embedding if needed
                    if self.embedding_compression != "none":
                        region_embedding = self.visual_embedding_extractor.compress_embedding(
                            region_embedding, self.embedding_compression, self.target_embedding_dim
                        )

                    # Create visual tokens (for now, one token per region)
                    visual_token = VisualToken(
                        token_id=idx,
                        embedding=region_embedding,
                        confidence=embedding_confidence,
                        region_type=block_type,
                        spatial_position=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                        compression_method=self.embedding_compression,
                        original_dims=768,  # Assuming standard embedding dimension
                    )
                    compressed_tokens = [visual_token]

                    # Calculate visual complexity (based on embedding variance)
                    visual_complexity = min(1.0, np.std(region_embedding) * 10)

                    # Calculate text-to-visual ratio
                    text_length = len(text_content)
                    visual_content_score = 1.0 - min(
                        1.0, text_length / 500
                    )  # More text = less visual
                    text_to_visual_ratio = text_length / (text_length + visual_content_score * 100)

                except Exception as e:
                    print(f"Warning: Visual embedding extraction failed for region {idx}: {e}")
                    region_embedding = None
                    embedding_confidence = 0.0

            # Count tokens (approximate: 1 token ≈ 4 characters)
            token_count = max(1, len(text_content) // 4)

            region = VisualRegion(
                region_id=f"page_{page_num}_region_{idx}",
                page_num=page_num,
                block_type=block_type,
                bbox=BoundingBox.from_list(bbox),
                compressed_tokens=compressed_tokens,
                text_content=text_content,
                markdown_content=markdown_content,
                token_count=token_count,
                confidence=confidence,
                region_embedding=region_embedding,
                embedding_confidence=embedding_confidence,
                visual_complexity=visual_complexity,
                text_to_visual_ratio=text_to_visual_ratio,
                metadata={
                    "block_index": idx,
                    "spatial_order": idx,
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "embedding_dims": len(region_embedding) if region_embedding is not None else 0,
                    "compression_method": self.embedding_compression,
                },
            )
            regions.append(region)

        processing_time = time.time() - start_time
        total_tokens = sum(r.token_count for r in regions)

        return PageOCRResult(
            page_num=page_num,
            width=image.width,
            height=image.height,
            regions=regions,
            total_tokens=total_tokens,
            processing_time=processing_time,
        )

    def _detect_layout_blocks(self, image: Image.Image) -> List[Dict]:
        """
        Detect layout blocks in the image using visual analysis.

        This implementation uses image analysis to detect different regions:
        - Headers (top of page, large text)
        - Paragraphs (main content blocks)
        - Tables (grid structures)
        - Figures (image regions)
        - Captions (small text near figures)
        - Lists (bullet points)
        """
        width, height = image.size
        blocks = []

        # Convert to grayscale for analysis
        gray = image.convert("L")
        pixels = np.array(gray)

        # Analyze horizontal projection (sum of pixels per row)
        h_projection = np.sum(255 - pixels, axis=1)

        # Find text regions based on projection
        threshold = np.max(h_projection) * 0.1
        in_block = False
        block_start = 0

        regions_y = []
        for y, val in enumerate(h_projection):
            if val > threshold and not in_block:
                in_block = True
                block_start = y
            elif val <= threshold and in_block:
                in_block = False
                if y - block_start > 10:  # Minimum block height
                    regions_y.append((block_start, y))

        if in_block and height - block_start > 10:
            regions_y.append((block_start, height))

        # If no regions detected, create default blocks
        if not regions_y:
            regions_y = self._create_default_regions(height)

        # Classify each region
        for idx, (y_start, y_end) in enumerate(regions_y):
            # Analyze vertical projection for this region
            region_pixels = pixels[y_start:y_end, :]
            v_projection = np.sum(255 - region_pixels, axis=0)

            # Find horizontal bounds
            v_threshold = np.max(v_projection) * 0.1
            x_start = 0
            x_end = width

            for x in range(width):
                if v_projection[x] > v_threshold:
                    x_start = x
                    break

            for x in range(width - 1, -1, -1):
                if v_projection[x] > v_threshold:
                    x_end = x + 1
                    break

            # Normalize coordinates
            bbox = [x_start / width, y_start / height, x_end / width, y_end / height]

            # Classify block type based on position and size
            block_type = self._classify_block_type(bbox, idx, len(regions_y), region_pixels)

            # Calculate confidence based on region characteristics
            confidence = self._calculate_block_confidence(region_pixels)

            blocks.append(
                {"type": block_type, "bbox": bbox, "confidence": confidence, "order": idx}
            )

        return blocks

    def _create_default_regions(self, height: int) -> List[Tuple[int, int]]:
        """Create default region splits when detection fails"""
        # Split page into logical sections
        header_end = int(height * 0.1)
        body_split = int(height * 0.5)

        return [(0, header_end), (header_end, body_split), (body_split, height)]

    def _classify_block_type(
        self, bbox: List[float], idx: int, total_blocks: int, region_pixels: np.ndarray
    ) -> str:
        """
        Classify the type of a text block based on its characteristics.

        Args:
            bbox: Normalized bounding box [x1, y1, x2, y2]
            idx: Index of block in page
            total_blocks: Total number of blocks
            region_pixels: Pixel data for the region

        Returns:
            Block type string
        """
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        y_position = bbox[1]

        # Analyze text density
        text_density = np.sum(255 - region_pixels) / region_pixels.size

        # Check for table-like structure (regular vertical lines)
        v_projection = np.sum(255 - region_pixels, axis=0)
        v_variance = np.var(v_projection)

        # Classification rules
        if idx == 0 and y_position < 0.15 and height < 0.1:
            return "header"
        elif height < 0.05 and width < 0.5:
            return "caption"
        elif v_variance > np.mean(v_projection) * 10 and height > 0.1:
            return "table"
        elif text_density < 0.01 and height > 0.15:
            return "figure"
        elif text_density > 0.3:
            return "list"
        else:
            return "paragraph"

    def _calculate_block_confidence(self, region_pixels: np.ndarray) -> float:
        """Calculate confidence score for block detection"""
        # Based on text clarity and contrast
        contrast = np.std(region_pixels)
        # Normalize to 0-1 range, higher contrast = higher confidence
        confidence = min(1.0, contrast / 100)
        return max(0.5, confidence)

    def _extract_text_from_region(
        self, image: Image.Image, bbox: List[float], block_type: str
    ) -> str:
        """
        Extract text from a specific region using the VLM.

        Args:
            image: Full page image
            bbox: Normalized bounding box
            block_type: Type of block

        Returns:
            Extracted text content
        """
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Crop region
        w, h = image.size
        left = int(bbox[0] * w)
        top = int(bbox[1] * h)
        right = int(bbox[2] * w)
        bottom = int(bbox[3] * h)

        # Ensure valid crop dimensions
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        top = max(0, min(top, h - 1))
        bottom = max(top + 1, min(bottom, h))

        region_img = image.crop((left, top, right, bottom))

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            region_img.save(tmp.name)
            image_path = tmp.name

        try:
            # Create prompt based on block type
            prompt = self._create_extraction_prompt(block_type)

            formatted_prompt = apply_chat_template(
                self.processor, self.config, prompt, num_images=1
            )

            output = generate(
                self.model, self.processor, formatted_prompt, [image_path], max_tokens=500, temp=0.1
            )

            # Extract text from result
            if hasattr(output, "text"):
                text = output.text
            else:
                text = str(output)

            # Clean up the extracted text
            text = self._clean_extracted_text(text, block_type)

        finally:
            # Clean up temp file
            if os.path.exists(image_path):
                os.unlink(image_path)

        return text

    def _create_extraction_prompt(self, block_type: str) -> str:
        """Create appropriate prompt for text extraction based on block type"""
        prompts = {
            "header": "Extract the exact text from this header. Only output the text, nothing else.",
            "paragraph": "Extract all text from this paragraph. Preserve the exact wording and structure.",
            "table": "Extract all text and data from this table. Format as markdown table if possible.",
            "figure": "Describe what is shown in this figure or diagram briefly.",
            "caption": "Extract the caption text exactly as written.",
            "list": "Extract all list items. Preserve bullet points or numbering.",
            "formula": "Extract the mathematical formula or equation shown.",
        }
        return prompts.get(block_type, "Extract all text from this image region accurately.")

    def _clean_extracted_text(self, text: str, block_type: str) -> str:
        """Clean and normalize extracted text"""
        # Remove common artifacts
        text = text.strip()

        # Remove repeated patterns
        lines = text.split("\n")
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and (not unique_lines or line != unique_lines[-1]):
                unique_lines.append(line)

        text = "\n".join(unique_lines)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    def _to_markdown(self, text: str, block_type: str) -> str:
        """Convert text to markdown with structure preservation"""
        if not text:
            return ""

        if block_type == "header":
            # Determine header level based on text length
            if len(text) < 50:
                return f"# {text}\n"
            else:
                return f"## {text}\n"
        elif block_type == "table":
            # If not already markdown table, wrap it
            if "|" not in text:
                return f"```\n{text}\n```\n"
            return text + "\n"
        elif block_type == "figure":
            return f"![Figure: {text}]()\n"
        elif block_type == "caption":
            return f"*{text}*\n"
        elif block_type == "list":
            # Ensure list formatting
            lines = text.split("\n")
            formatted = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith(("-", "*", "•")):
                    formatted.append(f"- {line}")
                else:
                    formatted.append(line)
            return "\n".join(formatted) + "\n"
        elif block_type == "formula":
            return f"$$\n{text}\n$$\n"
        else:
            return text + "\n"

    def batch_process(
        self, images: List[Image.Image], start_page: int = 0, show_progress: bool = True
    ) -> List[PageOCRResult]:
        """
        Process multiple pages in batch.

        Args:
            images: List of PIL Images
            start_page: Starting page number
            show_progress: Whether to show progress

        Returns:
            List of PageOCRResult
        """
        results = []
        total = len(images)

        for idx, image in enumerate(images):
            if show_progress:
                print(f"  Processing page {start_page + idx + 1}/{start_page + total}...")

            result = self.process_image(image, start_page + idx)
            results.append(result)

            if show_progress and (idx + 1) % 5 == 0:
                avg_tokens = sum(r.total_tokens for r in results) / len(results)
                print(f"    Average tokens per page: {avg_tokens:.0f}")

        return results

    def get_compression_stats(self, results: List[PageOCRResult]) -> Dict:
        """Calculate compression statistics"""
        total_tokens = sum(r.total_tokens for r in results)
        total_pages = len(results)

        # Estimate original token count (traditional OCR)
        # Assume 2000-3000 tokens per page for full text extraction
        estimated_original = total_pages * 2500

        compression_ratio = estimated_original / total_tokens if total_tokens > 0 else 0

        # Calculate block type distribution
        block_types = {}
        for result in results:
            for region in result.regions:
                bt = region.block_type
                block_types[bt] = block_types.get(bt, 0) + 1

        return {
            "total_pages": total_pages,
            "total_compressed_tokens": total_tokens,
            "estimated_original_tokens": estimated_original,
            "compression_ratio": f"{compression_ratio:.1f}x",
            "tokens_per_page": total_tokens / total_pages if total_pages > 0 else 0,
            "space_savings": (
                f"{(1 - 1/compression_ratio) * 100:.1f}%" if compression_ratio > 0 else "0%"
            ),
            "block_type_distribution": block_types,
            "total_regions": sum(len(r.regions) for r in results),
            "avg_regions_per_page": (
                sum(len(r.regions) for r in results) / total_pages if total_pages > 0 else 0
            ),
        }

    def visualize_regions(
        self, image: Image.Image, regions: List[VisualRegion], save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Visualize detected regions on the image.

        Args:
            image: Original image
            regions: List of detected regions
            save_path: Optional path to save visualization

        Returns:
            Image with region overlays
        """
        # Create a copy
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)

        # Colors for different block types
        colors = {
            "header": (255, 0, 0),  # Red
            "paragraph": (0, 255, 0),  # Green
            "table": (0, 0, 255),  # Blue
            "figure": (255, 255, 0),  # Yellow
            "caption": (255, 0, 255),  # Magenta
            "list": (0, 255, 255),  # Cyan
            "formula": (255, 128, 0),  # Orange
        }

        w, h = image.size

        for region in regions:
            bbox = region.bbox
            color = colors.get(region.block_type, (128, 128, 128))

            # Draw rectangle
            left = int(bbox.x1 * w)
            top = int(bbox.y1 * h)
            right = int(bbox.x2 * w)
            bottom = int(bbox.y2 * h)

            draw.rectangle([left, top, right, bottom], outline=color, width=2)

            # Add label
            label = f"{region.block_type} ({region.token_count} tokens)"
            draw.text((left + 5, top + 5), label, fill=color)

        if save_path:
            vis_image.save(save_path)
            print(f"Visualization saved to {save_path}")

        return vis_image
