"""
Visual Similarity and Search Metrics
Advanced similarity calculations using DeepSeek-OCR's visual features
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import numpy.typing as npt

from ..ocr.deepseek_ocr import VisualRegion


@dataclass
class VisualSimilarityResult:
    """Result of visual similarity comparison"""

    overall_similarity: float
    component_scores: Dict[str, float]
    detailed_analysis: Dict[str, Any]
    confidence: float


class VisualSimilarityCalculator:
    """
    Advanced visual similarity calculator using multiple metrics
    """

    def __init__(self):
        self.feature_weights = {
            "embedding": 0.4,      # Visual embedding similarity
            "spatial": 0.2,        # Spatial position similarity
            "structural": 0.2,    # Structure similarity
            "content": 0.15,        # Content similarity
            "quality": 0.05,        # Visual quality similarity
        }

    def calculate_similarity(
        self,
        region1: VisualRegion,
        region2: VisualRegion,
        include_analysis: bool = True
    ) -> VisualSimilarityResult:
        """
        Calculate comprehensive visual similarity between two regions

        Args:
            region1: First visual region
            region2: Second visual region
            include_analysis: Whether to include detailed analysis

        Returns:
            VisualSimilarityResult with comprehensive similarity metrics
        """
        # Calculate component similarities
        component_scores = {}

        # 1. Embedding similarity
        component_scores["embedding"] = self._calculate_embedding_similarity(region1, region2)

        # 2. Spatial similarity
        component_scores["spatial"] = self._calculate_spatial_similarity(region1, region2)

        # 3. Structural similarity
        component_scores["structural"] = self._calculate_structural_similarity(region1, region2)

        # 4. Content similarity
        component_scores["content"] = self._calculate_content_similarity(region1, region2)

        # 5. Quality similarity
        component_scores["quality"] = self._calculate_quality_similarity(region1, region2)

        # Calculate weighted overall similarity
        overall_similarity = sum(
            score * self.feature_weights[feature]
            for feature, score in component_scores.items()
        )

        # Calculate confidence based on feature availability
        confidence = self._calculate_similarity_confidence(region1, region2, component_scores)

        # Generate detailed analysis
        detailed_analysis = {}
        if include_analysis:
            detailed_analysis = self._generate_similarity_analysis(
                region1, region2, component_scores
            )

        return VisualSimilarityResult(
            overall_similarity=overall_similarity,
            component_scores=component_scores,
            detailed_analysis=detailed_analysis,
            confidence=confidence
        )

    def _calculate_embedding_similarity(self, region1: VisualRegion, region2: VisualRegion) -> float:
        """Calculate similarity between region embeddings"""
        # Get embeddings
        emb1 = region1.get_primary_embedding()
        emb2 = region2.get_primary_embedding()

        if emb1 is None or emb2 is None:
            return 0.0

        # Ensure same dimensions
        min_dim = min(len(emb1), len(emb2))
        emb1 = emb1[:min_dim]
        emb2 = emb2[:min_dim]

        # Calculate cosine similarity
        return cosine(emb1, emb2)

    def _calculate_spatial_similarity(self, region1: VisualRegion, region2: VisualRegion) -> float:
        """Calculate spatial position similarity"""
        bbox1 = region1.bbox
        bbox2 = region2.bbox

        # Calculate spatial features
        features1 = [
            bbox1.x1, bbox1.y1, bbox1.x2, bbox1.y2,
            bbox1.width(), bbox1.height(), bbox1.area(),
            (bbox1.x1 + bbox1.x2) / 2, (bbox1.y1 + bbox1.y2) / 2
        ]

        features2 = [
            bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2,
            bbox2.width(), bbox2.height(), bbox2.area(),
            (bbox2.x1 + bbox2.x2) / 2, (bbox2.y1 + bbox2.y2) / 2
        ]

        # Calculate similarity for each feature
        similarities = []
        for f1, f2 in zip(features1, features2):
            # Handle different scales
            if abs(f1) < 0.001 and abs(f2) < 0.001:
                similarities.append(1.0)
            else:
                similarity = 1 - abs(f1 - f2) / max(abs(f1), abs(f2))
                similarities.append(max(0, similarity))

        return np.mean(similarities)

    def _calculate_structural_similarity(self, region1: VisualRegion, region2: VisualRegion) -> float:
        """Calculate structural similarity"""
        # Compare structural features
        features1 = self._extract_structural_features(region1)
        features2 = self._extract_structural_features(region2)

        if not features1 or not features2:
            return 0.0

        # Compare features
        similarities = []
        for key in features1:
            if key in features2:
                if isinstance(features1[key], (int, float)):
                    sim = 1 - abs(features1[key] - features2[key]) / max(abs(features1[key]), abs(features2[key]), 1)
                    similarities.append(sim)
                else:
                    similarities.append(1.0 if features1[key] == features2[key] else 0.0)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_content_similarity(self, region1: VisualRegion, region2: VisualRegion) -> float:
        """Calculate content similarity"""
        text1 = region1.text_content.lower().strip()
        text2 = region2.text_content.lower().strip()

        if not text1 or not text2:
            return 0.0

        # Calculate Jaccard similarity for words
        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_quality_similarity(self, region1: VisualRegion, region2: VisualRegion) -> float:
        """Calculate visual quality similarity"""
        # Compare quality metrics
        quality1 = getattr(region1, 'quality_metrics', {})
        quality2 = getattr(region2, 'quality_metrics', {})

        if not quality1 or not quality2:
            return 0.5  # Default

        metrics = ["overall", "clarity", "contrast", "sharpness"]
        similarities = []

        for metric in metrics:
            if metric in quality1 and metric in quality2:
                val1 = quality1[metric]
                val2 = quality2[metric]
                similarity = 1 - abs(val1 - val2) / max(val1, val2, 0.1)
                similarities.append(max(0, similarity))

        return np.mean(similarities) if similarities else 0.5

    def _extract_structural_features(self, region: VisualRegion) -> Dict[str, Any]:
        """Extract structural features from region"""
        return {
            "block_type": region.block_type,
            "area": region.bbox.area(),
            "aspect_ratio": region.bbox.width() / max(0.001, region.bbox.height()),
            "token_count": region.token_count,
            "visual_complexity": getattr(region, 'visual_complexity', 0.5),
            "text_to_visual_ratio": getattr(region, 'text_to_visual_ratio', 1.0),
        }

    def _calculate_similarity_confidence(
        self,
        region1: VisualRegion,
        region2: VisualRegion,
        component_scores: Dict[str, float]
    ) -> float:
        """Calculate confidence in similarity score"""
        available_features = sum(1 for score in component_scores.values() if score > 0)
        total_features = len(self.feature_weights)

        return available_features / total_features

    def _generate_similarity_analysis(
        self,
        region1: VisualRegion,
        region2: VisualRegion,
        component_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate detailed analysis of similarity calculation"""
        analysis = {
            "region1_info": {
                "id": region1.region_id,
                "type": region1.block_type,
                "page": region1.page_num,
                "size": f"{region1.bbox.width():.2f}x{region1.bbox.height():.2f}",
            },
            "region2_info": {
                "id": region2.region_id,
                "type": region2.block_type,
                "page": region2.page_num,
                "size": f"{region2.bbox.width():.2f}x{region2.bbox.height():.2f}",
            },
            "component_breakdown": {
                feature: {
                    "score": score,
                    "weight": self.feature_weights[feature],
                    "contribution": score * self.feature_weights[feature]
                }
                for feature, score in component_scores.items()
            },
            "recommendations": self._generate_similarity_recommendations(
                region1, region2, component_scores
            ),
        }

        return analysis

    def _generate_similarity_recommendations(
        self,
        region1: VisualRegion,
        region2: VisualRegion,
        component_scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on similarity analysis"""
        recommendations = []

        # Find strongest and weakest components
        if component_scores:
            max_component = max(component_scores.items(), key=lambda x: x[1])
            min_component = min(component_scores.items(), key=lambda x: x[1])

            recommendations.append(
                f"Strongest match: {max_component[0]} ({max_component[1]:.3f})"
            )
            recommendations.append(
                f"Weakest match: {min_component[0]} ({min_component[1]:.3f})"
            )

        # Block type recommendations
        if region1.block_type == region2.block_type:
            recommendations.append("Same block type - consider visual features for distinction")
        elif region1.block_type in ["figure", "table"] and region2.block_type in ["figure", "table"]:
            recommendations.append("Both visual content types - check for captions")
        else:
            recommendations.append("Different content types - consider relationship")

        return recommendations


class VisualSearchEngine:
    """
    Search engine optimized for visual features
    """

    def __init__(self, similarity_calculator: Optional[VisualSimilarityCalculator] = None):
        self.similarity_calculator = similarity_calculator or VisualSimilarityCalculator()
        self.indexed_regions = []
        self.regions_by_type = defaultdict(list)
        self.regions_by_page = defaultdict(list)

    def index_regions(self, regions: List[VisualRegion]) -> None:
        """Index regions for fast visual search"""
        self.indexed_regions = regions
        self.regions_by_type.clear()
        self.regions_by_page.clear()

        for region in regions:
            self.regions_by_type[region.block_type].append(region)
            self.regions_by_page[region.page_num].append(region)

    def search_by_visual(
        self,
        query_region: VisualRegion,
        top_k: int = 10,
        min_similarity: float = 0.3,
        block_type_filter: Optional[str] = None
    ) -> List[Tuple[VisualRegion, VisualSimilarityResult]]:
        """Search for visually similar regions"""
        candidates = self.indexed_regions

        # Apply block type filter if specified
        if block_type_filter:
            candidates = [r for r in candidates if r.block_type == block_type_filter]

        results = []
        for candidate in candidates:
            if candidate.region_id == query_region.region_id:
                continue  # Skip self

            similarity_result = self.similarity_calculator.calculate_similarity(
                query_region, candidate
            )

            if similarity_result.overall_similarity >= min_similarity:
                results.append((candidate, similarity_result))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1].overall_similarity, reverse=True)

        return results[:top_k]

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.3,
        block_type_filter: Optional[str] = None
    ) -> List[Tuple[VisualRegion, float]]:
        """Search by embedding similarity"""
        candidates = self.indexed_regions

        # Apply block type filter if specified
        if block_type_filter:
            candidates = [r for r in candidates if r.block_type == block_type_filter]

        results = []
        for region in candidates:
            if region.get_primary_embedding() is not None:
                embedding = region.get_primary_embedding()
                similarity = cosine(query_embedding, embedding[:min(len(query_embedding), len(embedding))])

                if similarity >= min_similarity:
                    results.append((region, similarity))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def search_by_spatial_properties(
        self,
        target_bbox,
        top_k: int = 10,
        max_distance: float = 0.2,
        block_type_filter: Optional[str] = None
    ) -> List[Tuple[VisualRegion, float]]:
        """Search by spatial properties"""
        candidates = self.indexed_regions

        # Apply block type filter if specified
        if block_type_filter:
            candidates = [r for r in candidates if r.block_type == block_type_filter]

        results = []
        for region in candidates:
            bbox = region.bbox
            target_center = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)
            region_center = ((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2)

            # Calculate Euclidean distance
            distance = euclidean([target_center[0], target_center[1]], [region_center[0], region_center[1]])

            if distance <= max_distance:
                results.append((region, distance))

        # Sort by distance and return closest
        results.sort(key=lambda x: x[1])

        return results[:top_k]

    def search_by_content(
        self,
        query_text: str,
        top_k: int = 10,
        min_jaccard: float = 0.2,
        block_type_filter: Optional[str] = None
    ) -> List[Tuple[VisualRegion, float]]:
        """Search by content similarity (Jaccard)"""
        candidates = self.indexed_regions

        # Apply block type filter if specified
        if block_type_filter:
            candidates = [r for r in candidates if r.block_type == block_type_filter]

        results = []
        query_words = set(query_text.lower().split())

        for region in candidates:
            region_words = set(region.text_content.lower().split())

            # Calculate Jaccard similarity
            if region_words:
                intersection = len(query_words & region_words)
                union = len(query_words | region_words)
                jaccard = intersection / union

                if jaccard >= min_jaccard:
                    results.append((region, jaccard))

        # Sort by Jaccard similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def hybrid_search(
        self,
        query_region: VisualRegion,
        query_text: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        min_score: float = 0.3,
        block_type_filter: Optional[str] = None
    ) -> List[Tuple[VisualRegion, float, Dict[str, float]]]:
        """
        Hybrid search combining multiple similarity metrics
        """
        default_weights = {
            "visual": 0.4,
            "spatial": 0.2,
            "content": 0.2,
            "structural": 0.2,
        }

        weights = weights or default_weights

        candidates = self.indexed_regions

        # Apply block type filter if specified
        if block_type_filter:
            candidates = [r for r in candidates if r.block_type == block_type_filter]

        results = []
        for candidate in candidates:
            if candidate.region_id == query_region.region_id:
                continue

            # Calculate different similarity scores
            scores = {}

            # Visual similarity
            similarity_result = self.similarity_calculator.calculate_similarity(
                query_region, candidate
            )
            scores["visual"] = similarity_result.overall_similarity

            # Spatial similarity
            target_center = ((query_region.bbox.x1 + query_region.bbox.x2) / 2,
                              (query_region.bbox.y1 + query_region.bbox.y2) / 2)
            candidate_center = ((candidate.bbox.x1 + candidate.bbox.x2) / 2,
                                (candidate.bbox.y1 + candidate.bbox.y2) / 2)
            spatial_dist = euclidean([target_center[0], target_center[1]], [candidate_center[0], candidate_center[1]])
            scores["spatial"] = max(0, 1 - spatial_dist)

            # Content similarity
            if query_text:
                query_words = set(query_text.lower().split())
                region_words = set(candidate.text_content.lower().split())
                if region_words:
                    intersection = len(query_words & region_words)
                    union = len(query_words | region_words)
                    scores["content"] = intersection / union if union > 0 else 0
                else:
                    scores["content"] = 0
            else:
                scores["content"] = 0

            # Structural similarity
            if query_region.block_type == candidate.block_type:
                scores["structural"] = 0.8
            else:
                scores["structural"] = 0.2

            # Calculate weighted score
            weighted_score = sum(
                scores.get(metric, 0) * weight
                for metric, weight in weights.items()
            )

            if weighted_score >= min_score:
                results.append((candidate, weighted_score, scores))

        # Sort by weighted score
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        return {
            "total_regions": len(self.indexed_regions),
            "regions_by_type": {k: len(v) for k, v in self.regions_by_type.items()},
            "regions_by_page": {k: len(v) for k, v in self.regions_by_page.items()},
            "block_types": list(self.regions_by_type.keys()),
            "page_range": (min(self.regions_by_page.keys()) if self.regions_by_page else 0,
                           max(self.regions_by_page.keys()) if self.regions_by_page else 0),
        }


class VisualSimilarityIndex:
    """
    Advanced indexing system for fast visual similarity search
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.regions = []
        self.embeddings = []
        self.spatial_index = {}  # Simple spatial hash
        self.type_index = defaultdict(list)
        self.page_index = defaultdict(list)

    def add_region(self, region: VisualRegion) -> None:
        """Add region to index"""
        self.regions.append(region)

        # Add to embeddings index
        embedding = region.get_primary_embedding()
        if embedding is not None:
            self.embeddings.append(embedding)
            self.regions[-1]._embedding_index = len(self.embeddings) - 1

        # Add to spatial index
        spatial_key = self._get_spatial_key(region)
        self.spatial_index.setdefault(spatial_key, []).append(len(self.regions) - 1)

        # Add to type index
        self.type_index[region.block_type].append(len(self.regions) - 1)

        # Add to page index
        self.page_index[region.page_num].append(len(self.regions) - 1)

    def _get_spatial_key(self, region: VisualRegion) -> str:
        """Get spatial hash key for region"""
        # Simple spatial hashing - divide page into grid
        grid_size = 0.1  # 10% of page
        x_key = int(region.bbox.x1 / grid_size)
        y_key = int(region.block_type, region.block_type)  # Include block type

        return f"{x_key},{y_key}"

    def find_similar_regions(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.7,
        max_results: int = 100,
        block_type_filter: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """Find regions with similar embeddings"""
        if not self.embeddings:
            return []

        # Filter by block type if specified
        indices = range(len(self.regions))
        if block_type_filter:
            indices = [i for i in indices if self.regions[i].block_type == block_type_filter]

        # Calculate similarities
        similarities = []
        for idx in indices:
            embedding = self.embeddings[idx]
            if embedding is not None:
                # Ensure same dimensions
                min_dim = min(len(query_embedding), len(embedding))
                query_trimmed = query_embedding[:min_dim]
                emb_trimmed = embedding[:min_dim]

                similarity = cosine(query_trimmed, emb_trimmed)
                if similarity >= threshold:
                    similarities.append((idx, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:max_results]

    def find_regions_in_area(
        self,
        bbox: Tuple[float, float, float, float],
        block_type_filter: Optional[str] = None
    ) -> List[int]:
        """Find regions in specified area"""
        results = []

        for i, region in enumerate(self.regions):
            # Check if region intersects with bbox
            if self._regions_intersect(region.bbox, bbox):
                if block_type_filter is None or region.block_type == block_type_filter:
                    results.append(i)

        return results

    def _regions_intersect(
        self, bbox1: List[float], bbox2: List[float]
    ) -> bool:
        """Check if two bounding boxes intersect"""
        return not (
            bbox1[2] < bbox2[0] or  # bbox1 right < bbox2 left
            bbox1[0] > bbox2[2] or  # bbox1 left > bbox2 right
            bbox1[3] < bbox2[1] or  # bbox1 bottom < bbox2 top
            bbox1[1] > bbox2[3]      # bbox1 top > bbox2 bottom
        )

    def get_region_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed regions"""
        return {
            "total_regions": len(self.regions),
            "indexed_embeddings": len(self.embeddings),
            "type_distribution": {k: len(v) for k, v in self.type_index.items()},
            "page_distribution": {k: len(v) for k, v in self.page_index.items()},
            "spatial_buckets": len(self.spatial_index),
        }