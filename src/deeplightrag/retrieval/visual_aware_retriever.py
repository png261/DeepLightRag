"""
Visual-Aware Adaptive Retriever
Enhanced retriever that uses visual embeddings for similarity search
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .adaptive_retriever import AdaptiveRetriever, RetrievedRegion
from .query_classifier import QueryClassifier
from ..graph.dual_layer import DualLayerGraph


@dataclass
class VisualRetrievalResult:
    """Enhanced retrieval result with visual similarity scores"""

    context: str
    visual_context: List[np.ndarray]  # Visual embeddings for multimodal LLM
    nodes_retrieved: int
    token_count: int
    visual_similarity_scores: List[float]
    text_similarity_scores: List[float]
    hybrid_scores: List[float]
    regions: List[RetrievedRegion]
    entities: List[Dict]
    compression_ratio: float
    visual_mode_used: bool


class VisualAwareRetriever(AdaptiveRetriever):
    """
    Enhanced retriever that leverages visual embeddings for better context retrieval
    """

    def __init__(
        self,
        graph: DualLayerGraph,
        query_classifier: QueryClassifier,
        visual_weight: float = 0.3,  # Weight for visual similarity vs text similarity
        enable_visual_fallback: bool = True,
    ):
        super().__init__(graph, query_classifier)
        self.visual_weight = visual_weight
        self.text_weight = 1.0 - visual_weight
        self.enable_visual_fallback = enable_visual_fallback

        # Statistics
        self.visual_retrieval_stats = {
            "visual_mode_queries": 0,
            "text_mode_queries": 0,
            "hybrid_mode_queries": 0,
            "total_visual_embeddings_used": 0,
        }

    def retrieve(
        self, query: str, override_level: Optional[int] = None, force_visual_mode: bool = False
    ) -> VisualRetrievalResult:
        """
        Enhanced retrieval with visual embedding support

        Args:
            query: User query
            override_level: Override automatic level detection
            force_visual_mode: Force use of visual embeddings

        Returns:
            VisualRetrievalResult with enhanced context
        """
        # Classify query
        classification = self.classifier.analyze_query(query)
        if override_level is not None:
            classification["level"] = override_level

        # Determine if visual mode should be used
        should_use_visual = self._should_use_visual_mode(query, classification, force_visual_mode)

        if should_use_visual:
            return self._visual_enhanced_retrieve(query, classification)
        else:
            # Fallback to text-based retrieval
            traditional_result = super().retrieve(query, override_level)
            return self._convert_to_visual_result(traditional_result, visual_mode_used=False)

    def _should_use_visual_mode(self, query: str, classification: Dict, force_visual: bool) -> bool:
        """Determine if visual embeddings should be used for this query"""
        if force_visual:
            return True

        # Check if visual content is available
        has_visual_content = any(
            node.region.region_embedding is not None
            for node in self.graph.visual_spatial.nodes.values()
        )

        if not has_visual_content:
            return False

        # Use visual mode for:
        # 1. Visual query keywords
        visual_keywords = {
            "figure",
            "chart",
            "graph",
            "table",
            "diagram",
            "image",
            "picture",
            "layout",
            "structure",
            "visual",
            "appearance",
            "format",
            "design",
        }

        query_lower = query.lower()
        has_visual_keywords = any(keyword in query_lower for keyword in visual_keywords)

        # 2. Complex queries that might benefit from visual understanding
        is_complex_query = classification.get("level", 1) >= 3

        return has_visual_keywords or is_complex_query

    def _visual_enhanced_retrieve(self, query: str, classification: Dict) -> VisualRetrievalResult:
        """Retrieve using visual embeddings"""
        self.visual_retrieval_stats["visual_mode_queries"] += 1

        # Get query embedding (simulate for now)
        query_embedding = self._encode_query_to_visual_space(query)

        # Find visually similar regions
        visual_candidates = self._find_visual_candidates(
            query_embedding, classification.get("max_tokens", 2000)
        )

        # Combine with text-based retrieval
        text_candidates = self._get_text_candidates(query, classification)

        # Merge and rank candidates
        final_regions = self._merge_and_rank_candidates(
            visual_candidates, text_candidates, query_embedding
        )

        # Build context
        context_parts = []
        visual_embeddings = []
        token_count = 0

        for region_info in final_regions:
            region = region_info["region"]

            # Add text context
            if region.should_use_visual_mode():
                # For visual regions, add minimal text + embedding reference
                text_part = f"### Page {region.page_num} [{region.block_type}] (Visual)\n"
                text_part += f"**Visual Content**: {region.text_content[:100]}...\n"
                text_part += f"**Embedding ID**: visual_emb_{len(visual_embeddings)}\n\n"
            else:
                # For text regions, use full markdown
                text_part = f"### Page {region.page_num} [{region.block_type}]\n"
                text_part += f"{region.markdown_content}\n\n"

            context_parts.append(text_part)
            token_count += len(text_part) // 4

            # Add visual embedding
            if region.region_embedding is not None:
                visual_embeddings.append(region.region_embedding)
                self.visual_retrieval_stats["total_visual_embeddings_used"] += 1

        # Calculate compression ratio
        traditional_tokens = len(" ".join(context_parts)) // 4
        visual_tokens = len(visual_embeddings) * 2  # Assume 2 tokens per embedding
        compression_ratio = (
            traditional_tokens / (token_count + visual_tokens)
            if (token_count + visual_tokens) > 0
            else 1.0
        )

        return VisualRetrievalResult(
            context=" ".join(context_parts),
            visual_context=visual_embeddings,
            nodes_retrieved=len(final_regions),
            token_count=token_count,
            visual_similarity_scores=[r["visual_score"] for r in final_regions],
            text_similarity_scores=[r["text_score"] for r in final_regions],
            hybrid_scores=[r["hybrid_score"] for r in final_regions],
            regions=[self._region_to_retrieved_region(r["region"]) for r in final_regions],
            entities=[],  # Could add entity extraction here
            compression_ratio=compression_ratio,
            visual_mode_used=True,
        )

    def _encode_query_to_visual_space(self, query: str) -> np.ndarray:
        """Encode query to match visual embedding space"""
        # This is a placeholder - in practice, you'd use the VLM to encode the query
        # For now, create a simple encoding based on query terms

        # Simple TF-IDF-like encoding
        words = query.lower().split()

        # Create a pseudo-embedding based on word characteristics
        features = []

        # Visual content words
        visual_words = {"figure", "chart", "table", "graph", "image", "diagram"}
        visual_score = sum(1 for word in words if word in visual_words) / len(words)
        features.append(visual_score)

        # Content type words
        content_words = {"data", "results", "analysis", "method", "conclusion"}
        content_score = sum(1 for word in words if word in content_words) / len(words)
        features.append(content_score)

        # Question type
        question_words = {"what", "how", "why", "where", "when"}
        question_score = sum(1 for word in words if word in question_words) / len(words)
        features.append(question_score)

        # Pad to match embedding dimension (256 for compressed embeddings)
        np.random.seed(hash(" ".join(words)) % 2**32)
        padding = np.random.randn(253) * 0.1

        query_embedding = np.concatenate([features, padding]).astype(np.float32)
        return query_embedding

    def _find_visual_candidates(self, query_embedding: np.ndarray, max_tokens: int) -> List[Dict]:
        """Find regions using visual similarity"""
        candidates = []

        for node in self.graph.visual_spatial.nodes.values():
            region = node.region

            if region.region_embedding is not None:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, region.region_embedding)

                candidates.append(
                    {
                        "region": region,
                        "visual_score": similarity,
                        "text_score": 0.0,  # Will be filled later
                        "hybrid_score": similarity * self.visual_weight,
                    }
                )

        # Sort by visual similarity
        candidates.sort(key=lambda x: x["visual_score"], reverse=True)

        # Limit by token budget
        selected = []
        token_count = 0

        for candidate in candidates:
            estimated_tokens = candidate["region"].token_count
            if token_count + estimated_tokens <= max_tokens:
                selected.append(candidate)
                token_count += estimated_tokens
            else:
                break

        return selected

    def _get_text_candidates(self, query: str, classification: Dict) -> List[Dict]:
        """Get candidates using traditional text-based retrieval"""
        # Use parent class method to get text-based results
        traditional_result = super().retrieve(query)

        text_candidates = []
        for region in traditional_result.get("regions", []):
            # Find the corresponding visual spatial node
            for node in self.graph.visual_spatial.nodes.values():
                if node.region.region_id == region.region_id:
                    text_candidates.append(
                        {
                            "region": node.region,
                            "visual_score": 0.0,  # Will be calculated if needed
                            "text_score": region.relevance_score,
                            "hybrid_score": region.relevance_score * self.text_weight,
                        }
                    )
                    break

        return text_candidates

    def _merge_and_rank_candidates(
        self,
        visual_candidates: List[Dict],
        text_candidates: List[Dict],
        query_embedding: np.ndarray,
    ) -> List[Dict]:
        """Merge visual and text candidates with hybrid ranking"""

        # Create a map of region_id to candidate info
        candidate_map = {}

        # Add visual candidates
        for candidate in visual_candidates:
            region_id = candidate["region"].region_id
            candidate_map[region_id] = candidate

        # Merge with text candidates
        for candidate in text_candidates:
            region_id = candidate["region"].region_id

            if region_id in candidate_map:
                # Update existing candidate with text score
                existing = candidate_map[region_id]
                existing["text_score"] = candidate["text_score"]
                existing["hybrid_score"] = (
                    existing["visual_score"] * self.visual_weight
                    + candidate["text_score"] * self.text_weight
                )
            else:
                # Add new text-only candidate
                # Calculate visual score if embedding exists
                region = candidate["region"]
                if region.region_embedding is not None:
                    visual_score = self._cosine_similarity(query_embedding, region.region_embedding)
                    candidate["visual_score"] = visual_score
                    candidate["hybrid_score"] = (
                        visual_score * self.visual_weight
                        + candidate["text_score"] * self.text_weight
                    )

                candidate_map[region_id] = candidate

        # Sort by hybrid score
        final_candidates = list(candidate_map.values())
        final_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return final_candidates

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _region_to_retrieved_region(self, region) -> RetrievedRegion:
        """Convert VisualRegion to RetrievedRegion for compatibility"""
        return RetrievedRegion(
            region_id=region.region_id,
            page_num=region.page_num,
            block_type=region.block_type,
            text_content=region.text_content,
            markdown_content=region.markdown_content,
            relevance_score=0.8,  # Placeholder
            bbox=region.bbox.to_list(),
        )

    def _convert_to_visual_result(
        self, traditional_result: Dict, visual_mode_used: bool = False
    ) -> VisualRetrievalResult:
        """Convert traditional retrieval result to VisualRetrievalResult"""
        return VisualRetrievalResult(
            context=traditional_result["context"],
            visual_context=[],  # No visual embeddings in traditional mode
            nodes_retrieved=traditional_result["nodes_retrieved"],
            token_count=traditional_result["token_count"],
            visual_similarity_scores=[],
            text_similarity_scores=[],
            hybrid_scores=[],
            regions=traditional_result.get("regions", []),
            entities=traditional_result.get("entities", []),
            compression_ratio=1.0,  # No additional compression
            visual_mode_used=visual_mode_used,
        )

    def get_visual_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about visual retrieval usage"""
        total_queries = sum(self.visual_retrieval_stats.values())

        stats = self.visual_retrieval_stats.copy()
        if total_queries > 0:
            stats.update(
                {
                    "visual_mode_percentage": (stats["visual_mode_queries"] / total_queries) * 100,
                    "avg_embeddings_per_query": (
                        stats["total_visual_embeddings_used"] / max(1, stats["visual_mode_queries"])
                    ),
                }
            )

        return stats
