"""
Comprehensive Visual Feature Extractor for DeepSeek-OCR
Extracts ALL visual features and embeddings for enhanced document understanding
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from dataclasses import dataclass

from .deepseek_ocr import VisualRegion


class ComprehensiveVisualExtractor:
    """
    Extracts comprehensive visual features from DeepSeek-OCR regions
    Utilizes ALL available visual information for enhanced processing
    """

    def __init__(self):
        self.feature_extractors = {
            "spatial": SpatialFeatureExtractor(),
            "quality": QualityAssessmentExtractor(),
            "readability": ReadabilityExtractor(),
            "content": ContentSpecificExtractor(),
            "structure": StructuralFeatureExtractor(),
            "layout": LayoutAnalysisExtractor(),
        }

    def extract_all_features(
        self, image: Image.Image, regions: List[VisualRegion]
    ) -> Dict[str, Any]:
        """
        Extract ALL visual features from the image and regions

        Args:
            image: Full page image
            regions: List of VisualRegion objects

        Returns:
            Comprehensive feature dictionary
        """
        # Page-level features
        page_features = self._extract_page_level_features(image)

        # Region-level features
        region_features = {}
        spatial_relationships = self._build_spatial_relationships(regions)

        for region in regions:
            # Extract all feature types for each region
            region_feats = {}

            for feat_name, extractor in self.feature_extractors.items():
                region_feats[feat_name] = extractor.extract(image, region)

            # Add embeddings
            region_feats["embeddings"] = self._extract_all_embeddings(image, region)

            region_features[region.region_id] = region_feats

        # Global analysis
        layout_structure = self._analyze_layout_structure(regions)
        content_patterns = self._analyze_content_patterns(regions)

        return {
            "page_level": page_features,
            "region_features": region_features,
            "spatial_relationships": spatial_relationships,
            "layout_structure": layout_structure,
            "content_analysis": content_patterns,
            "extracted_at": np.datetime64('now').astype(str)
        }

    def _extract_page_level_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract page-level visual features"""
        img_array = np.array(image)
        gray = image.convert("L")
        pixels = np.array(gray)

        # Global page embedding
        page_embedding = self._create_global_embedding(image)

        # Enhanced quality metrics
        quality_metrics = self._calculate_page_quality_metrics(img_array, gray)

        features = {
            "global_embedding": page_embedding,
            "dimensions": (image.width, image.height),
            "aspect_ratio": image.width / image.height,
            "text_density": np.mean(255 - pixels) / 255,
            "contrast": np.std(pixels) / 255,
            "brightness": np.mean(pixels) / 255,
            "edge_density": self._calculate_edge_density(pixels),
            "reading_direction": self._detect_reading_direction(pixels),
            "column_count": self._estimate_columns_with_confidence(pixels),
            "margins": self._detect_margins_with_validation(pixels),
            "visual_complexity": self._calculate_page_complexity(img_array),
            "layout_balance": self._calculate_layout_balance(pixels),
            "content_density": self._calculate_content_density_map(pixels),
            **quality_metrics
        }

        return features

    def _calculate_page_quality_metrics(self, img_array: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive page quality metrics"""
        # Basic quality metrics
        blur_score = self._detect_blur(gray)
        noise_level = self._estimate_noise_level(gray)
        skew_angle = self._detect_skew(gray)
        illumination = self._assess_illumination(gray)

        # Advanced quality metrics
        compression_artifacts = self._detect_compression_artifacts(img_array)
        scan_quality = self._assess_scan_quality(img_array)
        text_clarity = self._assess_text_clarity(gray)

        return {
            "blur_score": blur_score,
            "noise_level": noise_level,
            "skew_angle": skew_angle,
            "illumination_uniformity": illumination,
            "compression_artifacts": compression_artifacts,
            "scan_quality": scan_quality,
            "text_clarity": text_clarity,
            "overall_quality": (blur_score + (1 - noise_level) + illumination + text_clarity) / 4
        }

    def _detect_blur(self, gray: np.ndarray) -> float:
        """Detect blur using Laplacian variance"""
        if gray.size < 9:
            return 0.5

        # Calculate Laplacian variance
        laplacian_var = np.var(gray[1:-1, 1:-1] - (gray[:-2, 1:-1] + gray[2:, 1:-1] +
                                gray[1:-1, :-2] + gray[1:-1, 2:]) / 4)

        # Normalize to 0-1 (higher is sharper)
        return min(1.0, laplacian_var / 1000)

    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using high-frequency components"""
        if gray.size < 9:
            return 0.5

        # Use Laplacian to capture high-frequency noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F) if hasattr(cv2, 'Laplacian') else None

        if laplacian is not None:
            noise = np.std(laplacian)
        else:
            # Fallback: simple difference method
            h_diff = np.abs(np.diff(gray, axis=0))
            v_diff = np.abs(np.diff(gray, axis=1))
            noise = np.mean(h_diff) + np.mean(v_diff)

        return min(1.0, noise / 50)

    def _detect_skew(self, gray: np.ndarray) -> float:
        """Detect skew angle in degrees"""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150) if hasattr(cv2, 'Canny') else None

        if edges is not None:
            # Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100) if hasattr(cv2, 'HoughLines') else None

            if lines is not None:
                angles = []
                for rho, theta in lines[:100]:  # Limit to first 100 lines
                    angle = np.degrees(theta - np.pi/2)
                    if abs(angle) < 45:  # Only consider reasonable angles
                        angles.append(angle)

                if angles:
                    # Return median angle
                    return float(np.median(angles))

        # Fallback: simple heuristic based on text direction
        return 0.0

    def _assess_illumination(self, gray: np.ndarray) -> float:
        """Assess illumination uniformity"""
        # Divide image into grid and calculate local brightness
        h, w = gray.shape
        grid_size = min(h, w) // 10
        grid_size = max(1, grid_size)

        brightness_values = []
        for i in range(0, h, grid_size):
            for j in range(0, w, grid_size):
                region = gray[i:i+grid_size, j:j+grid_size]
                if region.size > 0:
                    brightness_values.append(np.mean(region))

        if brightness_values:
            # Uniformity is inverse of standard deviation
            uniformity = 1 - (np.std(brightness_values) / 255)
            return max(0, min(1, uniformity))

        return 0.5

    def _detect_compression_artifacts(self, img_array: np.ndarray) -> float:
        """Detect JPEG compression artifacts"""
        if len(img_array.shape) == 3:
            # Check for 8x8 block artifacts (JPEG signature)
            h, w = img_array.shape[:2]

            # Sample blocks at regular intervals
            block_scores = []
            for i in range(8, h-8, 16):
                for j in range(8, w-8, 16):
                    # Extract 8x8 block
                    block = img_array[i:i+8, j:j+8]

                    # Calculate block boundary differences
                    if block.shape == (8, 8, 3):
                        h_diff = np.mean(np.abs(block[4:, :] - block[:4, :]))
                        v_diff = np.mean(np.abs(block[:, 4:] - block[:, :4]))

                        # High differences indicate block artifacts
                        block_score = (h_diff + v_diff) / 2
                        block_scores.append(block_score)

            if block_scores:
                artifact_score = np.mean(block_scores) / 255
                return min(1.0, artifact_score * 5)  # Scale up

        return 0.0

    def _assess_scan_quality(self, img_array: np.ndarray) -> float:
        """Assess overall scan quality"""
        # Factors affecting scan quality
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Check for common scan issues
        # 1. Dynamic range
        dynamic_range = (np.max(gray) - np.min(gray)) / 255
        range_score = min(1.0, dynamic_range)

        # 2. Histogram distribution
        hist, _ = np.histogram(gray, bins=256)
        hist_score = 1 - (np.sum(hist[:10]) + np.sum(hist[-10:])) / gray.size  # Less pure black/white

        # 3. Edge density (indicates content)
        edge_density = self._calculate_edge_density(gray)

        return (range_score * 0.4 + hist_score * 0.3 + min(1.0, edge_density * 10) * 0.3)

    def _assess_text_clarity(self, gray: np.ndarray) -> float:
        """Assess text clarity and readability"""
        # Use morphological operations to detect text patterns
        if hasattr(cv2, 'morphologyEx'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            # Top-hat transform to find bright regions on dark background
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

            # Bottom-hat transform to find dark regions on bright background
            bothat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

            # Text regions have strong responses in both transforms
            text_response = np.mean(tophat) + np.mean(bothat)

            return min(1.0, text_response / 100)

        # Fallback: simple gradient magnitude
        if gray.size > 9:
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            grad_mag = np.mean(grad_x) + np.mean(grad_y)
            return min(1.0, grad_mag / 50)

        return 0.5

    def _detect_reading_direction(self, pixels: np.ndarray) -> str:
        """Detect primary reading direction"""
        # Analyze text flow patterns
        h_projection = np.sum(pixels, axis=0)
        v_projection = np.sum(pixels, axis=1)

        # Check for vertical text patterns
        v_text_score = np.sum(np.diff(v_projection)**2) / len(v_projection)
        h_text_score = np.sum(np.diff(h_projection)**2) / len(h_projection)

        # If vertical has stronger patterns, it might be vertical text
        if v_text_score > h_text_score * 1.5:
            return "ttb"  # Top-to-bottom
        else:
            return "ltr"  # Left-to-right (default)

    def _estimate_columns_with_confidence(self, pixels: np.ndarray) -> Dict[str, Any]:
        """Estimate number of columns with confidence score"""
        v_projection = np.sum(pixels, axis=0)

        # Find valleys (column separators)
        valleys = []
        for i in range(1, len(v_projection) - 1):
            if v_projection[i] < v_projection[i-1] and v_projection[i] < v_projection[i+1]:
                # Check if it's a significant valley
                local_min = min(v_projection[max(0, i-10):i+10])
                if v_projection[i] < local_min * 0.7:
                    valleys.append(i)

        # Filter valleys based on spacing
        if len(valleys) > 1:
            # Filter out valleys that are too close
            filtered_valleys = [valleys[0]]
            for valley in valleys[1:]:
                if valley - filtered_valleys[-1] > len(v_projection) * 0.05:  # 5% of width
                    filtered_valleys.append(valley)
            valleys = filtered_valleys

        num_columns = max(1, len(valleys) + 1)

        # Calculate confidence based on valley depth and regularity
        if len(valleys) > 0:
            valley_depths = [v_projection[v] / np.max(v_projection) for v in valleys]
            confidence = 1 - np.mean(valley_depths)

            # Check regularity of spacing
            if len(valleys) > 1:
                spacings = [valleys[i+1] - valleys[i] for i in range(len(valleys)-1)]
                regularity = 1 - (np.std(spacings) / max(1, np.mean(spacings)))
                confidence = (confidence + regularity) / 2
        else:
            confidence = 0.5

        return {
            "count": num_columns,
            "confidence": max(0, min(1, confidence)),
            "valley_positions": valleys
        }

    def _detect_margins_with_validation(self, pixels: np.ndarray) -> Dict[str, Any]:
        """Detect page margins with validation"""
        h_projection = np.sum(pixels, axis=1)
        v_projection = np.sum(pixels, axis=0)

        h_threshold = np.max(h_projection) * 0.05  # Lower threshold for better detection
        v_threshold = np.max(v_projection) * 0.05

        # Find content boundaries
        top = next((i for i, v in enumerate(h_projection) if v > h_threshold), 0)
        bottom = len(h_projection) - next((i for i, v in enumerate(reversed(h_projection)) if v > h_threshold), len(h_projection))

        left = next((i for i, v in enumerate(v_projection) if v > v_threshold), 0)
        right = len(v_projection) - next((i for i, v in enumerate(reversed(v_projection)) if v > h_threshold), len(v_projection))

        margins = {
            "top": top / len(h_projection),
            "bottom": (len(h_projection) - bottom) / len(h_projection),
            "left": left / len(v_projection),
            "right": (len(v_projection) - right) / len(v_projection),
        }

        # Validate margins
        validation = {
            "reasonable_margins": all(0.01 <= m <= 0.5 for m in margins.values()),
            "asymmetric": abs(margins["left"] - margins["right"]) > 0.1 or abs(margins["top"] - margins["bottom"]) > 0.1,
            "tight_margins": any(m < 0.02 for m in margins.values()),
            "wide_margins": any(m > 0.3 for m in margins.values()),
        }

        return {
            **margins,
            "validation": validation,
            "confidence": 0.8 if validation["reasonable_margins"] else 0.5
        }

    def _calculate_layout_balance(self, pixels: np.ndarray) -> float:
        """Calculate layout balance score"""
        h, w = pixels.shape

        # Divide page into quadrants
        mid_h, mid_w = h // 2, w // 2

        quadrants = {
            "top_left": np.sum(pixels[:mid_h, :mid_w]),
            "top_right": np.sum(pixels[:mid_h, mid_w:]),
            "bottom_left": np.sum(pixels[mid_h:, :mid_w]),
            "bottom_right": np.sum(pixels[mid_h:, mid_w:]),
        }

        # Calculate balance (lower variance = more balanced)
        quadrant_values = list(quadrants.values())
        total = sum(quadrant_values)

        if total > 0:
            quadrant_ratios = [v / total for v in quadrant_values]
            balance = 1 - np.std(quadrant_ratios)
            return max(0, min(1, balance))

        return 0.5

    def _calculate_content_density_map(self, pixels: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed content density map"""
        h, w = pixels.shape

        # Create density grid
        grid_size = min(h, w) // 20
        grid_size = max(1, grid_size)

        density_map = []
        for i in range(0, h, grid_size):
            row = []
            for j in range(0, w, grid_size):
                region = pixels[i:i+grid_size, j:j+grid_size]
                if region.size > 0:
                    # Normalize by region size
                    density = np.sum(255 - region) / (region.size * 255)
                    row.append(min(1.0, density))
                else:
                    row.append(0.0)
            density_map.append(row)

        # Calculate statistics
        all_densities = [d for row in density_map for d in row]

        return {
            "map": density_map,
            "grid_size": grid_size,
            "statistics": {
                "mean_density": np.mean(all_densities),
                "std_density": np.std(all_densities),
                "min_density": np.min(all_densities),
                "max_density": np.max(all_densities),
                "density_distribution": np.histogram(all_densities, bins=10)[0].tolist()
            }
        }

    def _extract_all_embeddings(
        self, image: Image.Image, region: VisualRegion
    ) -> Dict[str, np.ndarray]:
        """Extract all types of embeddings for a region with quality awareness"""
        w, h = image.size
        bbox = region.bbox

        # Crop region
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)
        region_img = image.crop((left, top, right, bottom))

        # Assess region quality first
        region_quality = self._assess_region_quality(region_img, region)

        embeddings = {}

        # Multi-scale embeddings with quality weighting
        scales = {
            "tiny": (64, 64),
            "small": (128, 128),
            "medium": (224, 224),
            "large": (384, 384),
            "original": region_img.size
        }

        # Adaptive scale selection based on region size and quality
        selected_scales = self._select_optimal_scales(region_img, region_quality)

        for scale_name in selected_scales:
            if scale_name == "original":
                resized = region_img
            else:
                size = scales[scale_name]
                # Use high-quality resampling
                resized = region_img.resize(size, Image.Resampling.LANCZOS)

            # Create quality-aware embedding
            embedding = self._create_quality_aware_embedding(resized, region_quality)
            embeddings[scale_name] = embedding

        # Specialized embeddings
        embeddings.update({
            "semantic": self._create_semantic_embedding(region),
            "structural": self._create_structural_embedding(region),
            "contextual": self._create_contextual_embedding(image, region),
            "enhanced": self._create_enhanced_embedding(region_img, region, region_quality),
            "quality_weighted": self._create_quality_weighted_embedding(embeddings, region_quality),
        })

        return embeddings

    def _assess_region_quality(self, region_img: Image.Image, region: VisualRegion) -> Dict[str, float]:
        """Assess quality of a specific region"""
        gray = region_img.convert("L")
        img_array = np.array(gray)

        quality_metrics = {
            "clarity": self._detect_blur(img_array),
            "contrast": np.std(img_array) / 255,
            "noise": self._estimate_noise_level(img_array),
            "text_density": np.mean(255 - img_array) / 255 if img_array.size > 0 else 0,
            "edge_density": self._calculate_edge_density(img_array),
        }

        # Calculate overall quality
        quality_metrics["overall"] = (
            quality_metrics["clarity"] * 0.3 +
            (1 - quality_metrics["noise"]) * 0.2 +
            min(1.0, quality_metrics["contrast"] * 2) * 0.2 +
            quality_metrics["edge_density"] * 0.3
        )

        return quality_metrics

    def _select_optimal_scales(self, region_img: Image.Image, quality: Dict[str, float]) -> List[str]:
        """Select optimal scales based on region size and quality"""
        w, h = region_img.size
        area = w * h
        quality_score = quality["overall"]

        # Scale selection logic
        scales = []

        # Always include medium scale as baseline
        scales.append("medium")

        # Add large scale for large, high-quality regions
        if area > 50000 and quality_score > 0.6:
            scales.append("large")

        # Add original if very high quality
        if area > 100000 and quality_score > 0.8:
            scales.append("original")

        # Add small scale for better detail preservation
        if quality_score > 0.5 or area < 10000:
            scales.append("small")

        # Add tiny only for very small regions
        if area < 5000:
            scales.append("tiny")

        # Remove duplicates while preserving order
        seen = set()
        scales = [s for s in scales if not (s in seen or seen.add(s))]

        return scales

    def _create_quality_aware_embedding(self, image: Image.Image, quality: Dict[str, float]) -> np.ndarray:
        """Create embedding that considers region quality"""
        # Base embedding
        base_embedding = self._create_embedding(image)

        # Adjust embedding based on quality
        quality_factor = quality["overall"]

        # Reduce dimensionality for low-quality regions
        if quality_factor < 0.5:
            # Apply more aggressive dimensionality reduction
            effective_dims = int(len(base_embedding) * quality_factor)
            base_embedding = base_embedding[:effective_dims]

        # Apply quality weighting
        quality_weighted = base_embedding * quality_factor

        # Add quality indicators as additional features
        quality_features = np.array([
            quality["clarity"],
            quality["contrast"],
            quality["noise"],
            quality["text_density"],
            quality["edge_density"]
        ])

        # Combine features
        if len(quality_weighted) + len(quality_features) <= 768:
            combined = np.concatenate([quality_weighted, quality_features])
        else:
            # Truncate base embedding and add quality features
            combined = np.concatenate([quality_weighted[:768-len(quality_features)], quality_features])

        return combined

    def _create_enhanced_embedding(
        self, region_img: Image.Image, region: VisualRegion, quality: Dict[str, float]
    ) -> np.ndarray:
        """Create enhanced embedding with multiple feature types"""
        img_array = np.array(region_img)

        # Collect different types of features
        features = []

        # 1. Basic statistical features
        if len(img_array.shape) == 3:
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                features.extend([
                    np.mean(channel_data) / 255,
                    np.std(channel_data) / 255,
                    np.percentile(channel_data, 25) / 255,
                    np.percentile(channel_data, 75) / 255,
                ])
        else:
            features.extend([
                np.mean(img_array) / 255,
                np.std(img_array) / 255,
                np.percentile(img_array, 25) / 255,
                np.percentile(img_array, 75) / 255,
            ])

        # 2. Texture features
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        features.extend([
            np.std(gray) / 255,
            np.mean(np.abs(np.diff(gray, axis=0))) / 255,
            np.mean(np.abs(np.diff(gray, axis=1))) / 255,
        ])

        # 3. Edge features
        edges = self._detect_edges(gray)
        features.append(np.mean(edges))

        # 4. Structural features
        bbox = region.bbox
        features.extend([
            bbox.x1, bbox.y1, bbox.x2, bbox.y2,
            bbox.width(), bbox.height(),
            bbox.area(),
            bbox.width() / max(0.001, bbox.height()),
        ])

        # 5. Quality features
        features.extend([
            quality["clarity"],
            quality["contrast"],
            quality["noise"],
            quality["edge_density"],
        ])

        # 6. Text features (simplified)
        text_features = self._extract_text_features(region.text_content)
        features.extend(text_features)

        # Convert to numpy array and normalize
        embedding = np.array(features, dtype=np.float32)

        # Normalize features to [0, 1]
        if len(embedding) > 0:
            embedding = np.clip(embedding, 0, 1)

        # Pad or truncate to 512 dimensions
        target_dim = 512
        if len(embedding) < target_dim:
            # Pad with zeros
            embedding = np.pad(embedding, (0, target_dim - len(embedding)), 'constant')
        else:
            # Truncate
            embedding = embedding[:target_dim]

        return embedding

    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Detect edges in grayscale image"""
        if gray.size < 9:
            return np.zeros_like(gray)

        # Simple edge detection
        h_edges = np.abs(np.diff(gray, axis=0))
        v_edges = np.abs(np.diff(gray, axis=1))

        # Combine edges
        edges = np.zeros_like(gray)
        edges[1:, :] += h_edges
        edges[:, 1:] += v_edges

        return edges / 255  # Normalize

    def _extract_text_features(self, text: str) -> List[float]:
        """Extract simple text features"""
        if not text:
            return [0] * 5

        # Basic text statistics
        words = text.split()
        chars = list(text)

        features = [
            len(text) / 1000,  # Normalized length
            len(words) / 100,  # Normalized word count
            sum(1 for c in chars if c.isupper()) / max(1, len(chars)),  # Uppercase ratio
            sum(1 for c in chars if c.isdigit()) / max(1, len(chars)),  # Digit ratio
            sum(1 for c in chars if c in '.,!?;:') / max(1, len(chars)),  # Punctuation ratio
        ]

        return features

    def _create_quality_weighted_embedding(
        self, embeddings: Dict[str, np.ndarray], quality: Dict[str, float]
    ) -> np.ndarray:
        """Create quality-weighted combination of embeddings"""
        if not embeddings:
            return np.zeros(256, dtype=np.float32)

        # Use medium scale as base if available
        if "medium" in embeddings:
            base_embedding = embeddings["medium"]
        else:
            # Use first available embedding
            base_embedding = list(embeddings.values())[0]

        # Apply quality weighting
        quality_weight = min(1.0, quality["overall"] * 1.2)  # Slight amplification
        weighted_embedding = base_embedding * quality_weight

        return weighted_embedding

    def _build_spatial_relationships(self, regions: List[VisualRegion]) -> Dict[str, Any]:
        """Build comprehensive spatial relationship graph"""
        relationships = {
            "adjacent_pairs": [],
            "vertical_flow": [],
            "horizontal_flow": [],
            "spatial_clusters": [],
            "reading_order": [],
            "distance_matrix": {},
            "neighbors": {},
        }

        # Sort regions by reading order
        sorted_regions = sorted(regions, key=lambda r: (r.page_num, r.bbox.y1, r.bbox.x1))

        # Build reading order
        relationships["reading_order"] = [r.region_id for r in sorted_regions]

        # Calculate distances and find relationships
        for i, region1 in enumerate(regions):
            relationships["neighbors"][region1.region_id] = []

            for j, region2 in enumerate(regions):
                if i != j:
                    distance = self._calculate_region_distance(region1, region2)
                    relationships["distance_matrix"][f"{region1.region_id}-{region2.region_id}"] = distance

                    # Find adjacent regions
                    if distance < 0.1:  # Threshold for adjacency
                        relationships["neighbors"][region1.region_id].append({
                            "region_id": region2.region_id,
                            "distance": distance,
                            "relationship": self._determine_spatial_relationship(region1, region2)
                        })

                        if i < j:  # Avoid duplicates
                            relationships["adjacent_pairs"].append({
                                "region1": region1.region_id,
                                "region2": region2.region_id,
                                "distance": distance,
                                "relationship": self._determine_spatial_relationship(region1, region2)
                            })

        return relationships

    def _analyze_layout_structure(self, regions: List[VisualRegion]) -> Dict[str, Any]:
        """Analyze overall page layout structure"""
        # Count and analyze block types
        block_counts = {}
        block_positions = {"header": [], "body": [], "footer": []}

        for region in regions:
            block_counts[region.block_type] = block_counts.get(region.block_type, 0) + 1

            # Categorize by vertical position
            if region.bbox.y1 < 0.2:
                block_positions["header"].append(region.region_id)
            elif region.bbox.y2 > 0.8:
                block_positions["footer"].append(region.region_id)
            else:
                block_positions["body"].append(region.region_id)

        structure = {
            "block_distribution": block_counts,
            "column_count": self._estimate_page_columns(regions),
            "has_header": len(block_positions["header"]) > 0,
            "has_footer": len(block_positions["footer"]) > 0,
            "dominant_pattern": self._detect_dominant_pattern(regions),
            "balance_score": self._calculate_balance_score(regions),
            "complexity_score": self._calculate_layout_complexity(regions),
            "positions": block_positions,
            "region_count": len(regions),
            "visual_regions": sum(1 for r in regions if r.block_type in ["figure", "table", "formula"]),
            "text_regions": sum(1 for r in regions if r.block_type in ["paragraph", "header", "list"]),
        }

        return structure

    def _analyze_content_patterns(self, regions: List[VisualRegion]) -> Dict[str, Any]:
        """Analyze content patterns across regions"""
        patterns = {
            "content_density": self._calculate_content_density(regions),
            "visual_to_text_ratio": self._calculate_visual_text_ratio(regions),
            "content_flow": self._analyze_content_flow(regions),
            "topic_distribution": self._analyze_topic_distribution(regions),
            "readability_stats": self._calculate_readability_stats(regions),
            "complexity_distribution": self._analyze_complexity_distribution(regions),
        }

        return patterns

    # Helper methods
    def _create_global_embedding(self, image: Image.Image) -> np.ndarray:
        """Create global page embedding"""
        # Use a larger size for global context
        resized = image.resize((384, 384))
        return self._create_embedding(resized)

    def _create_embedding(self, image: Image.Image) -> np.ndarray:
        """Create embedding from image"""
        # Convert to numpy
        img_array = np.array(image)

        # Create feature vector
        if len(img_array.shape) == 3:
            # Color image
            features = []
            # Color statistics
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                ])

            # Texture features (simplified)
            gray = np.mean(img_array, axis=2)
            features.extend([
                np.std(gray),
                np.mean(np.abs(np.diff(gray, axis=0))),
                np.mean(np.abs(np.diff(gray, axis=1))),
            ])
        else:
            # Grayscale image
            features = [
                np.mean(img_array),
                np.std(img_array),
                np.percentile(img_array, 25),
                np.percentile(img_array, 75),
                np.mean(np.abs(np.diff(img_array, axis=0))),
                np.mean(np.abs(np.diff(img_array, axis=1))),
            ]

        # Pad to 768 dimensions
        target_dim = 768
        while len(features) < target_dim:
            features.extend([0.0] * min(64, target_dim - len(features)))

        return np.array(features[:target_dim], dtype=np.float32)

    def _create_semantic_embedding(self, region: VisualRegion) -> np.ndarray:
        """Create semantic embedding based on text content"""
        text = region.text_content
        if not text:
            return np.zeros(256, dtype=np.float32)

        # Simple text features
        features = [
            len(text) / 1000,  # Normalized length
            text.count('.') / 10,  # Sentence count proxy
            sum(c.isupper() for c in text) / len(text),  # Caps ratio
            len(set(text.lower())) / len(text),  # Vocabulary diversity
            region.confidence,
            hash(region.block_type) % 100 / 100,  # Block type
        ]

        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)

        return np.array(features[:256], dtype=np.float32)

    def _create_structural_embedding(self, region: VisualRegion) -> np.ndarray:
        """Create structural embedding based on layout"""
        features = [
            region.bbox.x1,
            region.bbox.y1,
            region.bbox.x2,
            region.bbox.y2,
            region.bbox.width(),
            region.bbox.height(),
            region.bbox.area(),
            region.bbox.width() / max(0.001, region.bbox.height()),
            region.page_num / 10,  # Normalized page number
            hash(region.block_type) % 100 / 100,
        ]

        # Pad to 256 dimensions
        while len(features) < 256:
            features.append(0.0)

        return np.array(features[:256], dtype=np.float32)

    def _create_contextual_embedding(self, image: Image.Image, region: VisualRegion) -> np.ndarray:
        """Create contextual embedding considering neighborhood"""
        # Get page-level features
        page_embedding = self._create_global_embedding(image)

        # Get region embedding
        w, h = image.size
        bbox = region.bbox
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)
        region_img = image.crop((left, top, right, bottom))
        region_embedding = self._create_embedding(region_img)

        # Combine with spatial context
        context_features = [
            region.bbox.x1, region.bbox.y1, region.bbox.x2, region.bbox.y2,
            region.bbox.width(), region.bbox.height(),
        ]

        # Concatenate and truncate
        combined = np.concatenate([region_embedding[:200], page_embedding[:200], context_features])

        # Pad to 512 dimensions
        if len(combined) < 512:
            combined = np.pad(combined, (0, 512 - len(combined)), 'constant')

        return combined[:512].astype(np.float32)

    def _calculate_region_distance(self, region1: VisualRegion, region2: VisualRegion) -> float:
        """Calculate distance between two regions"""
        # Use center-to-center distance
        center1 = ((region1.bbox.x1 + region1.bbox.x2) / 2, (region1.bbox.y1 + region1.bbox.y2) / 2)
        center2 = ((region2.bbox.x1 + region2.bbox.x2) / 2, (region2.bbox.y1 + region2.bbox.y2) / 2)

        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _determine_spatial_relationship(self, region1: VisualRegion, region2: VisualRegion) -> str:
        """Determine spatial relationship between regions"""
        # Check vertical relationship
        if region1.bbox.y2 < region2.bbox.y1 - 0.02:
            return "above"
        elif region2.bbox.y2 < region1.bbox.y1 - 0.02:
            return "below"

        # Check horizontal relationship
        if region1.bbox.x2 < region2.bbox.x1 - 0.02:
            return "left_of"
        elif region2.bbox.x2 < region1.bbox.x1 - 0.02:
            return "right_of"

        # Check overlap
        if self._regions_overlap(region1, region2):
            return "overlapping"

        return "nearby"

    def _regions_overlap(self, region1: VisualRegion, region2: VisualRegion) -> bool:
        """Check if two regions overlap"""
        x_overlap = max(0, min(region1.bbox.x2, region2.bbox.x2) - max(region1.bbox.x1, region2.bbox.x1))
        y_overlap = max(0, min(region1.bbox.y2, region2.bbox.y2) - max(region1.bbox.y1, region2.bbox.y1))

        return x_overlap > 0.01 and y_overlap > 0.01

    def _calculate_edge_density(self, pixels: np.ndarray) -> float:
        """Calculate edge density"""
        h_diff = np.abs(np.diff(pixels, axis=0))
        v_diff = np.abs(np.diff(pixels, axis=1))

        edge_pixels = np.sum(h_diff) + np.sum(v_diff)
        total_pixels = pixels.size

        return edge_pixels / (total_pixels * 255)

    def _estimate_columns(self, pixels: np.ndarray) -> int:
        """Estimate number of text columns"""
        v_projection = np.sum(pixels, axis=0)

        # Find valleys (column separators)
        valleys = []
        for i in range(1, len(v_projection) - 1):
            if v_projection[i] < v_projection[i-1] and v_projection[i] < v_projection[i+1]:
                valleys.append(i)

        return max(1, len(valleys))

    def _detect_margins(self, pixels: np.ndarray) -> Dict[str, float]:
        """Detect page margins"""
        h_projection = np.sum(pixels, axis=1)
        v_projection = np.sum(pixels, axis=0)

        h_threshold = np.max(h_projection) * 0.1
        v_threshold = np.max(v_projection) * 0.1

        # Find content boundaries
        top = next((i for i, v in enumerate(h_projection) if v > h_threshold), 0)
        bottom = len(h_projection) - next((i for i, v in enumerate(reversed(h_projection)) if v > h_threshold), len(h_projection))

        left = next((i for i, v in enumerate(v_projection) if v > v_threshold), 0)
        right = len(v_projection) - next((i for i, v in enumerate(reversed(v_projection)) if v > v_threshold), len(v_projection))

        return {
            "top": top / len(h_projection),
            "bottom": (len(h_projection) - bottom) / len(h_projection),
            "left": left / len(v_projection),
            "right": (len(v_projection) - right) / len(v_projection),
        }

    def _calculate_page_complexity(self, img_array: np.ndarray) -> float:
        """Calculate overall page complexity"""
        # Simplified complexity score
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Factors affecting complexity
        edge_density = self._calculate_edge_density(gray)
        std_dev = np.std(gray) / 255

        complexity = (edge_density + std_dev) / 2
        return min(1.0, complexity)

    def _estimate_page_columns(self, regions: List[VisualRegion]) -> int:
        """Estimate number of columns from regions"""
        x_positions = [(r.bbox.x1 + r.bbox.x2) / 2 for r in regions
                      if r.block_type in ["paragraph", "list"]]

        if not x_positions:
            return 1

        # Simple clustering
        x_positions.sort()
        clusters = [[x_positions[0]]]

        for pos in x_positions[1:]:
            if pos - clusters[-1][-1] < 0.1:  # Threshold for same column
                clusters[-1].append(pos)
            else:
                clusters.append([pos])

        return len(clusters)

    def _detect_dominant_pattern(self, regions: List[VisualRegion]) -> str:
        """Detect dominant layout pattern"""
        if not regions:
            return "empty"

        # Check for common patterns
        has_header = any(r.block_type == "header" for r in regions)
        has_tables = any(r.block_type == "table" for r in regions)
        has_figures = any(r.block_type == "figure" for r in regions)

        if has_tables and has_figures:
            return "mixed_visual"
        elif has_tables:
            return "table_heavy"
        elif has_figures:
            return "figure_heavy"
        elif has_header:
            return "standard_article"
        else:
            return "simple_text"

    def _calculate_balance_score(self, regions: List[VisualRegion]) -> float:
        """Calculate layout balance score"""
        if not regions:
            return 0

        # Check left-right balance
        left_weight = sum(r.bbox.width() for r in regions
                         if (r.bbox.x1 + r.bbox.x2) / 2 < 0.5)
        right_weight = sum(r.bbox.width() for r in regions
                          if (r.bbox.x1 + r.bbox.x2) / 2 >= 0.5)

        total_weight = left_weight + right_weight
        if total_weight == 0:
            return 0

        return 1 - abs(left_weight - right_weight) / total_weight

    def _calculate_layout_complexity(self, regions: List[VisualRegion]) -> float:
        """Calculate layout complexity"""
        if not regions:
            return 0

        # Factors: number of regions, variety of block types, spatial distribution
        num_regions = len(regions)
        block_types = set(r.block_type for r in regions)

        # Normalize
        region_score = min(1.0, num_regions / 20)  # Assuming 20 is complex
        type_score = min(1.0, len(block_types) / 7)  # 7 possible types

        return (region_score + type_score) / 2

    def _calculate_content_density(self, regions: List[VisualRegion]) -> float:
        """Calculate average content density"""
        if not regions:
            return 0

        total_text = sum(len(r.text_content) for r in regions)
        total_area = sum(r.bbox.area() for r in regions)

        return total_text / max(1, total_area)

    def _calculate_visual_text_ratio(self, regions: List[VisualRegion]) -> float:
        """Calculate visual-to-text ratio"""
        visual_regions = sum(1 for r in regions if r.block_type in ["figure", "table", "formula"])
        text_regions = sum(1 for r in regions if r.block_type in ["paragraph", "header", "list"])

        total = visual_regions + text_regions
        return visual_regions / max(1, total)

    def _analyze_content_flow(self, regions: List[VisualRegion]) -> List[str]:
        """Analyze content flow between regions"""
        # Return reading order with flow indicators
        sorted_regions = sorted(regions, key=lambda r: (r.page_num, r.bbox.y1, r.bbox.x1))

        flow = []
        for i, region in enumerate(sorted_regions):
            flow_type = "sequential"

            if i > 0:
                prev_region = sorted_regions[i-1]
                if region.block_type != prev_region.block_type:
                    flow_type = "transition"
                elif region.bbox.x1 < prev_region.bbox.x1 - 0.05:
                    flow_type = "column_change"

            flow.append(f"{region.region_id}:{flow_type}")

        return flow

    def _analyze_topic_distribution(self, regions: List[VisualRegion]) -> Dict[str, float]:
        """Analyze topic distribution (simplified)"""
        # Placeholder for topic analysis
        return {"topic_diversity": 0.5, "topic_coherence": 0.7}

    def _calculate_readability_stats(self, regions: List[VisualRegion]) -> Dict[str, float]:
        """Calculate readability statistics"""
        total_words = 0
        total_sentences = 0
        avg_readability = 0

        for region in regions:
            text = region.text_content
            if text:
                words = text.split()
                sentences = text.split('.')
                total_words += len(words)
                total_sentences += len(sentences)

        return {
            "avg_words_per_region": total_words / max(1, len(regions)),
            "avg_sentences_per_region": total_sentences / max(1, len(regions)),
            "avg_readability_score": avg_readability,
        }

    def _analyze_complexity_distribution(self, regions: List[VisualRegion]) -> Dict[str, float]:
        """Analyze complexity distribution across regions"""
        complexities = [getattr(r, 'visual_complexity', 0.5) for r in regions]

        return {
            "mean_complexity": np.mean(complexities) if complexities else 0,
            "std_complexity": np.std(complexities) if complexities else 0,
            "min_complexity": min(complexities) if complexities else 0,
            "max_complexity": max(complexities) if complexities else 0,
        }


# Specialized feature extractors
class SpatialFeatureExtractor:
    """Extract spatial relationship features"""

    def extract(self, image: Image.Image, region: VisualRegion) -> Dict[str, float]:
        bbox = region.bbox

        return {
            "x_center": (bbox.x1 + bbox.x2) / 2,
            "y_center": (bbox.y1 + bbox.y2) / 2,
            "width": bbox.x2 - bbox.x1,
            "height": bbox.y2 - bbox.y1,
            "area": bbox.area(),
            "aspect_ratio": bbox.width() / max(0.001, bbox.height()),
            "position_score": self._calculate_position_score(region),
            "is_header": bbox.y1 < 0.15,
            "is_footer": bbox.y2 > 0.85,
            "is_left_aligned": bbox.x1 < 0.1,
            "is_right_aligned": bbox.x2 > 0.9,
            "is_centered": abs(0.5 - (bbox.x1 + bbox.x2) / 2) < 0.1,
        }

    def _calculate_position_score(self, region: VisualRegion) -> float:
        y_pos = region.bbox.y1
        x_center = (region.bbox.x1 + region.bbox.x2) / 2

        position_score = (1 - y_pos) * 0.6  # Top content gets higher score
        center_score = (1 - abs(0.5 - x_center)) * 0.4  # Centered content gets bonus

        return position_score + center_score


class QualityAssessmentExtractor:
    """Assess visual quality metrics"""

    def extract(self, image: Image.Image, region: VisualRegion) -> Dict[str, float]:
        # Get region image
        w, h = image.size
        bbox = region.bbox
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)
        region_img = image.crop((left, top, right, bottom))

        gray = region_img.convert("L")
        gray_array = np.array(gray)

        return {
            "overall": self._calculate_overall_quality(gray_array),
            "clarity": self._calculate_clarity(gray_array),
            "contrast": np.std(gray_array) / 255,
            "brightness": np.mean(gray_array) / 255,
            "sharpness": self._calculate_sharpness(gray_array),
            "noise_level": self._estimate_noise(gray_array),
            "text_density": np.mean(255 - gray_array) / 255,
            "uniformity": self._calculate_uniformity(gray_array),
        }

    def _calculate_overall_quality(self, gray_array: np.ndarray) -> float:
        clarity = self._calculate_clarity(gray_array)
        contrast = np.std(gray_array) / 255
        sharpness = self._calculate_sharpness(gray_array)
        noise = self._estimate_noise(gray_array)

        return (clarity * 0.3 + contrast * 0.3 + sharpness * 0.2 + (1 - noise) * 0.2)

    def _calculate_clarity(self, gray_array: np.ndarray) -> float:
        return min(1.0, np.var(gray_array) / 10000)

    def _calculate_sharpness(self, gray_array: np.ndarray) -> float:
        if gray_array.size < 9:
            return 0

        # Laplacian approximation
        laplacian = np.var(gray_array[1:-1, 1:-1] - gray_array[:-2, :-2])
        return min(1.0, laplacian / 1000)

    def _estimate_noise(self, gray_array: np.ndarray) -> float:
        if gray_array.size < 9:
            return 0

        noise = np.std(gray_array[1:-1, 1:-1] - gray_array[:-2, :-2])
        return min(1.0, noise / 100)

    def _calculate_uniformity(self, gray_array: np.ndarray) -> float:
        # Lower variance = more uniform
        return 1 - min(1.0, np.var(gray_array) / 10000)


class ReadabilityExtractor:
    """Extract readability metrics"""

    def extract(self, image: Image.Image, region: VisualRegion) -> Dict[str, float]:
        text = region.text_content

        if not text:
            return {
                "readability_score": 0.0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
                "avg_sentence_length": 0,
                "caps_ratio": 0,
                "punctuation_ratio": 0,
                "flesch_score": 0,
            }

        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        return {
            "readability_score": self._flesch_score(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "avg_sentence_length": len(words) / max(1, len(sentences)),
            "caps_ratio": sum(1 for c in text if c.isupper()) / max(1, len(text)),
            "punctuation_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, len(text)),
            "flesch_score": self._flesch_score(text),
        }

    def _flesch_score(self, text: str) -> float:
        if not text:
            return 0

        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        avg_sentence_length = len(words) / max(1, len(sentences))
        avg_syllables = np.mean([self._count_syllables(w) for w in words]) if words else 1

        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
        return max(0, min(100, score / 100))

    def _count_syllables(self, word: str) -> int:
        vowels = "aeiouy"
        word = word.lower()
        syllable_count = 0

        prev_char_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel

        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)


class ContentSpecificExtractor:
    """Extract content-specific features based on block type"""

    def extract(self, image: Image.Image, region: VisualRegion) -> Dict[str, Any]:
        if region.block_type == "table":
            return self._extract_table_features(image, region)
        elif region.block_type == "figure":
            return self._extract_figure_features(image, region)
        elif region.block_type == "formula":
            return self._extract_formula_features(image, region)
        else:
            return {"type": "text", "features": {}}

    def _extract_table_features(self, image: Image.Image, region: VisualRegion) -> Dict[str, Any]:
        # Get region image
        w, h = image.size
        bbox = region.bbox
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)
        region_img = image.crop((left, top, right, bottom))

        gray = region_img.convert("L")
        img_array = np.array(gray)

        # Detect grid structure
        h_projection = np.sum(img_array, axis=1)
        v_projection = np.sum(img_array, axis=0)

        h_lines = self._find_lines(h_projection, threshold=0.3)
        v_lines = self._find_lines(v_projection, threshold=0.3)

        return {
            "type": "table",
            "rows": max(0, len(h_lines) - 1),
            "cols": max(0, len(v_lines) - 1),
            "has_headers": True,  # Simplified
            "grid_density": len(h_lines) + len(v_lines),
            "regularity": self._calculate_grid_regularity(h_lines, v_lines),
            "cell_count": max(1, (len(h_lines) - 1) * (len(v_lines) - 1)),
        }

    def _extract_figure_features(self, image: Image.Image, region: VisualRegion) -> Dict[str, Any]:
        # Get region image
        w, h = image.size
        bbox = region.bbox
        left = int(bbox.x1 * w)
        top = int(bbox.y1 * h)
        right = int(bbox.x2 * w)
        bottom = int(bbox.y2 * h)
        region_img = image.crop((left, top, right, bottom))

        return {
            "type": "figure",
            "colorfulness": self._calculate_colorfulness(region_img),
            "edge_density": self._calculate_edge_density(region_img),
            "dominant_colors": self._get_dominant_colors(region_img),
            "has_text": len(region.text_content.strip()) > 0,
            "complexity": self._estimate_figure_complexity(region_img),
        }

    def _extract_formula_features(self, image: Image.Image, region: VisualRegion) -> Dict[str, Any]:
        return {
            "type": "formula",
            "has_symbols": True,
            "complexity": 0.5,  # Would analyze formula structure
            "is_inline": region.bbox.width() < region.bbox.height() * 3,
            "symbol_density": 0.5,
        }

    def _find_lines(self, projection: np.ndarray, threshold: float) -> List[int]:
        max_val = np.max(projection) if projection.size > 0 else 1
        line_threshold = max_val * threshold

        lines = []
        in_line = False

        for i, val in enumerate(projection):
            if val > line_threshold and not in_line:
                in_line = True
                lines.append(i)
            elif val <= line_threshold and in_line:
                in_line = False

        return lines

    def _calculate_grid_regularity(self, h_lines: List[int], v_lines: List[int]) -> float:
        if len(h_lines) < 2 or len(v_lines) < 2:
            return 0

        h_spacings = [h_lines[i+1] - h_lines[i] for i in range(len(h_lines)-1)]
        v_spacings = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines)-1)]

        h_reg = 1 - (np.std(h_spacings) / max(1, np.mean(h_spacings))) if h_spacings else 0
        v_reg = 1 - (np.std(v_spacings) / max(1, np.mean(v_spacings))) if v_spacings else 0

        return (h_reg + v_reg) / 2

    def _calculate_colorfulness(self, image: Image.Image) -> float:
        img_array = np.array(image)

        if len(img_array.shape) < 3:
            return 0

        rg = np.abs(img_array[:, :, 0].astype(float) - img_array[:, :, 1].astype(float))
        yb = np.abs(0.5 * (img_array[:, :, 0].astype(float) + img_array[:, :, 1].astype(float)) - img_array[:, :, 2].astype(float))

        rg_mean, rg_std = np.mean(rg), np.std(rg)
        yb_mean, yb_std = np.mean(yb), np.std(yb)

        std_root = np.sqrt(rg_std**2 + yb_std**2)
        mean_root = np.sqrt(rg_mean**2 + yb_mean**2)

        colorfulness = std_root + 0.3 * mean_root
        return min(1.0, colorfulness / 255)

    def _calculate_edge_density(self, image: Image.Image) -> float:
        gray = image.convert("L")
        pixels = np.array(gray)

        h_diff = np.abs(np.diff(pixels, axis=0))
        v_diff = np.abs(np.diff(pixels, axis=1))

        edge_pixels = np.sum(h_diff) + np.sum(v_diff)
        total_pixels = pixels.size

        return edge_pixels / (total_pixels * 255)

    def _get_dominant_colors(self, image: Image.Image, n_colors: int = 3) -> List[List[int]]:
        """Get dominant colors from image"""
        # Simplified - return average color
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            avg_color = np.mean(img_array, axis=(0, 1))
            return [avg_color.astype(int).tolist()]
        else:
            return [[int(np.mean(img_array))] * 3]

    def _estimate_figure_complexity(self, image: Image.Image) -> float:
        """Estimate figure complexity"""
        edge_density = self._calculate_edge_density(image)
        colorfulness = self._calculate_colorfulness(image)

        return min(1.0, (edge_density + colorfulness) / 2)


class StructuralFeatureExtractor:
    """Extract structural features"""

    def extract(self, image: Image.Image, region: VisualRegion) -> Dict[str, Any]:
        return {
            "layout_pattern": self._detect_layout_pattern(region),
            "alignment": self._detect_alignment(region),
            "spacing": self._analyze_spacing(region),
            "hierarchy_level": self._determine_hierarchy(region),
            "reading_order": self._estimate_reading_order(region),
        }

    def _detect_layout_pattern(self, region: VisualRegion) -> str:
        if region.block_type == "header":
            return "title"
        elif region.block_type == "list":
            return "bullet_points"
        elif region.bbox.width() > 0.8:
            return "full_width"
        else:
            return "column"

    def _detect_alignment(self, region: VisualRegion) -> str:
        if region.bbox.x1 < 0.1:
            return "left"
        elif region.bbox.x2 > 0.9:
            return "right"
        else:
            return "center"

    def _analyze_spacing(self, region: VisualRegion) -> Dict[str, float]:
        return {
            "line_spacing": 1.5,  # Placeholder
            "paragraph_spacing": 2.0,  # Placeholder
            "margin_before": 0.5,  # Placeholder
            "margin_after": 0.5,  # Placeholder
        }

    def _determine_hierarchy(self, region: VisualRegion) -> int:
        if region.block_type == "header":
            return 1
        elif region.block_type == "paragraph":
            return 2
        else:
            return 3

    def _estimate_reading_order(self, region: VisualRegion) -> float:
        # Higher score for earlier in reading order
        return (1 - region.bbox.y1) * 0.7 + (1 - region.bbox.x1) * 0.3


class LayoutAnalysisExtractor:
    """Extract layout analysis features"""

    def extract(self, image: Image.Image, region: VisualRegion) -> Dict[str, Any]:
        return {
            "page_position": self._get_page_position(region),
            "column_assignment": self._assign_column(region),
            "region_relationships": self._find_related_regions(region),
            "visual_weight": self._calculate_visual_weight(region),
            "content_category": self._categorize_content(region),
        }

    def _get_page_position(self, region: VisualRegion) -> Dict[str, Any]:
        y_pos = region.bbox.y1
        x_pos = (region.bbox.x1 + region.bbox.x2) / 2

        position = "middle"
        if y_pos < 0.2:
            position = "top"
        elif y_pos > 0.8:
            position = "bottom"

        return {
            "vertical": position,
            "horizontal": "left" if x_pos < 0.33 else "center" if x_pos < 0.66 else "right",
            "quadrant": self._get_quadrant(region),
        }

    def _get_quadrant(self, region: VisualRegion) -> str:
        x_center = (region.bbox.x1 + region.bbox.x2) / 2
        y_center = (region.bbox.y1 + region.bbox.y2) / 2

        if x_center < 0.5 and y_center < 0.5:
            return "top-left"
        elif x_center >= 0.5 and y_center < 0.5:
            return "top-right"
        elif x_center < 0.5 and y_center >= 0.5:
            return "bottom-left"
        else:
            return "bottom-right"

    def _assign_column(self, region: VisualRegion) -> int:
        # Simple column assignment based on x-position
        x_center = (region.bbox.x1 + region.bbox.x2) / 2

        if x_center < 0.33:
            return 1
        elif x_center < 0.66:
            return 2
        else:
            return 3

    def _find_related_regions(self, region: VisualRegion) -> List[str]:
        # Placeholder - would find related regions
        return []

    def _calculate_visual_weight(self, region: VisualRegion) -> float:
        # Calculate visual importance based on size, position, and type
        size_weight = region.bbox.area()
        position_weight = (1 - region.bbox.y1) * 0.5  # Top gets more weight

        # Type weights
        type_weights = {
            "header": 1.5,
            "paragraph": 1.0,
            "table": 1.3,
            "figure": 1.4,
            "formula": 1.2,
            "list": 1.0,
            "caption": 0.9,
        }

        type_weight = type_weights.get(region.block_type, 1.0)

        return (size_weight + position_weight + type_weight) / 3

    def _categorize_content(self, region: VisualRegion) -> str:
        """Categorize content type"""
        if region.block_type == "header":
            return "title"
        elif region.block_type == "paragraph":
            if len(region.text_content) > 500:
                return "long_text"
            else:
                return "short_text"
        elif region.block_type == "table":
            return "data"
        elif region.block_type == "figure":
            return "visual"
        elif region.block_type == "formula":
            return "mathematical"
        else:
            return "other"