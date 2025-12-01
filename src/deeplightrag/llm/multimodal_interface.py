"""
Multimodal LLM Interface
Enables LLMs to work with both text and visual embeddings
"""

import numpy as np
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from .base import BaseLLM


class MultimodalLLMInterface(ABC):
    """
    Abstract interface for multimodal LLMs that can process visual embeddings
    """

    @abstractmethod
    def generate_with_visual_context(
        self,
        text_context: str,
        visual_embeddings: List[np.ndarray],
        query: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response using both text and visual context

        Args:
            text_context: Traditional text context
            visual_embeddings: List of visual embedding vectors
            query: User query
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        pass

    @abstractmethod
    def encode_visual_embeddings(self, embeddings: List[np.ndarray]) -> str:
        """
        Convert visual embeddings to a format the LLM can understand

        Args:
            embeddings: List of visual embeddings

        Returns:
            Text representation that encodes visual information
        """
        pass


class VisualEmbeddingEncoder:
    """
    Utility class to encode visual embeddings for LLM consumption
    """

    def __init__(self, encoding_method: str = "token_ids"):
        """
        Initialize encoder

        Args:
            encoding_method: Method to encode embeddings
                - "token_ids": Map embeddings to special token IDs
                - "descriptive": Convert to descriptive text
                - "compressed": Use compressed representation
                - "hybrid": Combine multiple methods
        """
        self.encoding_method = encoding_method
        self.embedding_vocabulary = {}  # Map embeddings to tokens
        self.next_token_id = 50000  # Start visual tokens at high ID

    def encode_embeddings_to_text(
        self, embeddings: List[np.ndarray], region_types: Optional[List[str]] = None
    ) -> str:
        """
        Convert visual embeddings to text representation

        Args:
            embeddings: List of embedding vectors
            region_types: Optional list of region types for context

        Returns:
            Text representation of visual content
        """
        if self.encoding_method == "token_ids":
            return self._encode_as_token_ids(embeddings)
        elif self.encoding_method == "descriptive":
            return self._encode_as_descriptive_text(embeddings, region_types)
        elif self.encoding_method == "compressed":
            return self._encode_as_compressed_features(embeddings)
        elif self.encoding_method == "hybrid":
            return self._encode_as_hybrid(embeddings, region_types)
        else:
            return self._encode_as_token_ids(embeddings)

    def _encode_as_token_ids(self, embeddings: List[np.ndarray]) -> str:
        """Encode embeddings as special token IDs"""
        visual_tokens = []

        for i, embedding in enumerate(embeddings):
            # Create a hash-based token ID for this embedding
            embedding_hash = hash(embedding.tobytes()) % 10000
            token_id = f"<VIS_{embedding_hash}>"
            visual_tokens.append(token_id)

        return " ".join(visual_tokens)

    def _encode_as_descriptive_text(
        self, embeddings: List[np.ndarray], region_types: Optional[List[str]] = None
    ) -> str:
        """Encode embeddings as descriptive text features"""
        descriptions = []

        for i, embedding in enumerate(embeddings):
            region_type = region_types[i] if region_types else "content"

            # Analyze embedding characteristics
            mean_val = np.mean(embedding)
            std_val = np.std(embedding)
            max_val = np.max(embedding)
            min_val = np.min(embedding)

            # Convert to descriptive features
            if mean_val > 0.5:
                intensity = "high-intensity"
            elif mean_val > 0:
                intensity = "medium-intensity"
            else:
                intensity = "low-intensity"

            if std_val > 0.3:
                complexity = "complex"
            elif std_val > 0.1:
                complexity = "moderate"
            else:
                complexity = "simple"

            # Determine content type hints from embedding patterns
            content_hints = []
            if embedding[0:10].sum() > embedding[-10:].sum():
                content_hints.append("structured")
            if abs(max_val) > 2:
                content_hints.append("prominent-features")
            if std_val < 0.05:
                content_hints.append("uniform")

            description = f"[VISUAL_{i+1}: {region_type} {intensity} {complexity}"
            if content_hints:
                description += f" {'-'.join(content_hints)}"
            description += "]"

            descriptions.append(description)

        return " ".join(descriptions)

    def _encode_as_compressed_features(self, embeddings: List[np.ndarray]) -> str:
        """Encode embeddings as compressed feature vectors"""
        compressed_features = []

        for i, embedding in enumerate(embeddings):
            # Extract key statistical features
            features = {
                "mean": np.mean(embedding),
                "std": np.std(embedding),
                "max": np.max(embedding),
                "min": np.min(embedding),
                "l2_norm": np.linalg.norm(embedding),
                "sparsity": (embedding == 0).mean(),
                "positive_ratio": (embedding > 0).mean(),
            }

            # Convert to compact text representation
            feature_str = f"<VIS_{i+1}:"
            for key, value in features.items():
                feature_str += f"{key}={value:.3f},"
            feature_str = feature_str.rstrip(",") + ">"

            compressed_features.append(feature_str)

        return " ".join(compressed_features)

    def _encode_as_hybrid(
        self, embeddings: List[np.ndarray], region_types: Optional[List[str]] = None
    ) -> str:
        """Combine multiple encoding methods"""
        # Use descriptive for semantic understanding
        descriptive = self._encode_as_descriptive_text(embeddings, region_types)

        # Use compressed for precise features
        compressed = self._encode_as_compressed_features(embeddings)

        return f"{descriptive}\n{compressed}"


class VisualAwareLLMWrapper:
    """
    Wrapper that adds visual embedding support to existing LLMs
    """

    def __init__(
        self,
        base_llm: BaseLLM,
        embedding_encoder: VisualEmbeddingEncoder = None,
        visual_context_template: str = None,
    ):
        """
        Initialize wrapper

        Args:
            base_llm: The base LLM to wrap
            embedding_encoder: Encoder for visual embeddings
            visual_context_template: Template for integrating visual context
        """
        self.base_llm = base_llm
        self.embedding_encoder = embedding_encoder or VisualEmbeddingEncoder("hybrid")
        self.visual_context_template = visual_context_template or self._default_visual_template()

        # Statistics
        self.visual_generation_stats = {
            "total_visual_generations": 0,
            "total_embeddings_processed": 0,
            "avg_embeddings_per_query": 0.0,
        }

    def generate_with_visual_context(
        self,
        text_context: str,
        visual_embeddings: List[np.ndarray],
        query: str,
        region_types: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate response with visual embedding support

        Args:
            text_context: Traditional text context
            visual_embeddings: List of visual embeddings
            query: User query
            region_types: Optional region types for context
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        self.visual_generation_stats["total_visual_generations"] += 1
        self.visual_generation_stats["total_embeddings_processed"] += len(visual_embeddings)

        if visual_embeddings:
            # Encode visual embeddings to text
            visual_context = self.embedding_encoder.encode_embeddings_to_text(
                visual_embeddings, region_types
            )

            # Combine with text context using template
            enhanced_context = self.visual_context_template.format(
                text_context=text_context,
                visual_context=visual_context,
                num_visual_elements=len(visual_embeddings),
            )
        else:
            # No visual context, use text only
            enhanced_context = text_context

        # Generate using base LLM
        result = self.base_llm.generate(enhanced_context, query, system_prompt)

        # Update statistics
        if self.visual_generation_stats["total_visual_generations"] > 0:
            self.visual_generation_stats["avg_embeddings_per_query"] = (
                self.visual_generation_stats["total_embeddings_processed"]
                / self.visual_generation_stats["total_visual_generations"]
            )

        return result

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Compatibility method for regular generation without visual embeddings
        """
        return self.generate_with_visual_context(
            text_context=context, visual_embeddings=[], query=query, system_prompt=system_prompt
        )

    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compatibility method for reasoning without visual embeddings
        """
        return self.generate_with_reasoning_and_visual(
            text_context=context, visual_embeddings=[], query=query, system_prompt=system_prompt
        )

    def generate_with_reasoning_and_visual(
        self,
        text_context: str,
        visual_embeddings: List[np.ndarray],
        query: str,
        region_types: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate with both reasoning and visual context
        """
        # Enhanced system prompt for visual reasoning
        visual_reasoning_prompt = self._get_visual_reasoning_prompt()
        final_system_prompt = system_prompt or visual_reasoning_prompt

        if visual_embeddings:
            visual_context = self.embedding_encoder.encode_embeddings_to_text(
                visual_embeddings, region_types
            )
            enhanced_context = self.visual_context_template.format(
                text_context=text_context,
                visual_context=visual_context,
                num_visual_elements=len(visual_embeddings),
            )
        else:
            enhanced_context = text_context

        # Use base LLM's reasoning capability
        if hasattr(self.base_llm, "generate_with_reasoning"):
            return self.base_llm.generate_with_reasoning(
                enhanced_context, query, final_system_prompt
            )
        else:
            # Fallback to regular generation
            answer = self.generate_with_visual_context(
                text_context, visual_embeddings, query, region_types, final_system_prompt
            )
            return {
                "answer": answer,
                "reasoning": "Visual context processing applied",
                "visual_embeddings_used": len(visual_embeddings),
            }

    def _default_visual_template(self) -> str:
        """Default template for combining text and visual context"""
        return """## Context Information

### Text Content:
{text_context}

### Visual Content Analysis:
The following visual elements have been analyzed and encoded:
{visual_context}

Total visual elements: {num_visual_elements}

**Note**: Visual elements contain spatial, structural, and semantic information that complements the text content. Consider both textual and visual aspects when formulating your response.

---
"""

    def _get_visual_reasoning_prompt(self) -> str:
        """System prompt for visual reasoning"""
        return """You are an advanced AI assistant capable of understanding both textual and visual content. 

When analyzing the provided context, consider:

1. **Text Analysis**: Extract key information from textual content
2. **Visual Analysis**: Interpret visual elements marked with [VISUAL_X] tags or <VIS_X> tokens
3. **Cross-Modal Integration**: Connect textual and visual information to form complete understanding
4. **Spatial Reasoning**: Consider layout, structure, and positional relationships in visual content
5. **Semantic Coherence**: Ensure your response integrates insights from both modalities

Visual elements may contain:
- Structural information (layouts, hierarchies)
- Data representations (charts, tables, diagrams)
- Contextual clues (captions, labels, relationships)
- Spatial arrangements and visual organization

Provide comprehensive answers that demonstrate understanding of both textual and visual aspects of the content."""

    def get_visual_stats(self) -> Dict[str, Any]:
        """Get visual generation statistics"""
        return self.visual_generation_stats.copy()

    def estimate_visual_token_savings(
        self, text_context: str, visual_embeddings: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Estimate token savings from using visual embeddings vs full text

        Returns:
            Dictionary with savings statistics
        """
        # Estimate traditional text tokens
        traditional_tokens = len(text_context.split()) * 1.3  # Rough token estimate

        # Estimate visual representation tokens
        visual_tokens = len(visual_embeddings) * 2  # Assume 2 tokens per embedding

        # Calculate savings
        total_visual_tokens = visual_tokens + len(text_context.split()) * 0.3  # Reduced text
        token_savings = traditional_tokens - total_visual_tokens
        savings_percentage = (
            (token_savings / traditional_tokens) * 100 if traditional_tokens > 0 else 0
        )

        return {
            "traditional_tokens": traditional_tokens,
            "visual_mode_tokens": total_visual_tokens,
            "token_savings": token_savings,
            "savings_percentage": savings_percentage,
            "compression_ratio": (
                traditional_tokens / total_visual_tokens if total_visual_tokens > 0 else 1.0
            ),
        }
