"""
Gemini API Provider for DeepLightRAG
Supports multimodal generation with visual embeddings
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
import base64
import io
from PIL import Image

from .base import BaseLLM, LLMConfig
from .multimodal_interface import MultimodalLLMInterface, VisualEmbeddingEncoder
from ..utils.helpers import compress_text
import numpy as np

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLM, MultimodalLLMInterface):
    """Gemini API provider with multimodal support"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.model_name = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Use the model name as provided - Gemini API will handle it
        model_name = self.model_name

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        self.model = genai.GenerativeModel(
            model_name=model_name, generation_config=generation_config
        )

        # Initialize visual embedding encoder
        self.visual_encoder = VisualEmbeddingEncoder("hybrid")

        self.model_info.update({"initialized": True, "multimodal": True, "supports_vision": True})
        logger.info(f"Initialized Gemini provider with model: {self.model_name}")

    def generate(self, context: str, query: str, system_prompt: Optional[str] = None) -> str:
        """Generate answer from context and query (BaseLLM interface)"""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_rag_prompt(system_prompt, context, query)
        return self._generate_response(prompt)

    def generate_with_reasoning(
        self, context: str, query: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate answer with reasoning (BaseLLM interface)"""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        prompt = self._build_reasoning_prompt(system_prompt, context, query)
        response = self._generate_response(prompt)

        # Parse reasoning from response
        parts = response.split("[ANSWER]")
        if len(parts) == 2:
            reasoning = parts[0].replace("[REASONING]", "").strip()
            answer = parts[1].strip()
        else:
            reasoning = "No explicit reasoning provided"
            answer = response

        return {"answer": answer, "reasoning": reasoning, "confidence": 0.8}  # Default confidence

    def generate_multimodal(
        self,
        prompt: str,
        context: Optional[str] = None,
        visual_embeddings: Optional[List[Dict]] = None,
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> str:
        """
        Generate response using Gemini with optional multimodal inputs

        Args:
            prompt: The text prompt
            context: Optional context text
            visual_embeddings: Optional visual embedding representations
            images: Optional PIL Images or base64 encoded images
            **kwargs: Additional generation parameters
        """
        try:
            # Build the complete prompt
            full_prompt = self._build_prompt(prompt, context, visual_embeddings)

            # Prepare multimodal inputs
            inputs = [full_prompt]

            # Add images if provided
            if images:
                for img in images:
                    if isinstance(img, str):  # base64 encoded
                        img_data = base64.b64decode(img)
                        img_pil = Image.open(io.BytesIO(img_data))
                        inputs.append(img_pil)
                    elif isinstance(img, Image.Image):
                        inputs.append(img)
                    else:
                        logger.warning(f"Unsupported image type: {type(img)}")

            return self._execute_generation(inputs)

        except Exception as e:
            logger.error(f"Error in Gemini multimodal generation: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _generate_response(self, prompt: str) -> str:
        """Generate simple text response"""
        try:
            return self._execute_generation([prompt])
        except Exception as e:
            logger.error(f"Error in Gemini generation: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _execute_generation(self, inputs: List[Any]) -> str:
        """Execute generation with Gemini API"""
        try:
            response = self.model.generate_content(inputs)

            # Check for safety blocks or empty response
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]

                # Check finish reason
                if hasattr(candidate, "finish_reason"):
                    if candidate.finish_reason == 1:  # STOP
                        if response.text:
                            return response.text.strip()
                        else:
                            return "I provided a response, but it may have been filtered."
                    elif candidate.finish_reason == 2:  # MAX_TOKENS
                        return "The response was cut off due to length limits. Please try a shorter query."
                    elif candidate.finish_reason == 3:  # SAFETY
                        return "I cannot provide a response due to safety filters. Please rephrase your query."
                    elif candidate.finish_reason == 4:  # RECITATION
                        return "I cannot provide a response due to recitation concerns. Please try a different approach."

            # Fallback - try to get text
            if hasattr(response, "text") and response.text:
                return response.text.strip()

            logger.error("Gemini returned empty or blocked response")
            return "I apologize, but I couldn't generate a response to your query. Please try rephrasing."

        except Exception as e:
            logger.error(f"Error in Gemini API call: {e}")
            return f"Error: {str(e)}"

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        visual_embeddings: Optional[List[Dict]] = None,
    ) -> str:
        """Build the complete prompt with context and visual embeddings"""

        parts = []

        # Add context if provided
        if context:
            # Compress context to fit within token limits
            compressed_context = compress_text(context, max_length=4000)
            parts.append(f"Context:\n{compressed_context}\n")

        # Add visual embedding information if available
        if visual_embeddings:
            visual_context = self._format_visual_embeddings(visual_embeddings)
            if visual_context:
                parts.append(f"Visual Context:\n{visual_context}\n")

        # Add the main prompt
        parts.append(f"Query: {prompt}")

        return "\n".join(parts)

    def _format_visual_embeddings(self, visual_embeddings: List[Dict]) -> str:
        """Format visual embeddings for text-based representation"""
        if not visual_embeddings:
            return ""

        formatted_parts = []
        for i, embedding in enumerate(visual_embeddings):
            page_num = embedding.get("page", i + 1)

            # Extract visual features summary
            features = embedding.get("features", {})
            layout = embedding.get("layout", {})

            visual_summary = f"Page {page_num} Visual Summary:\n"

            if layout:
                visual_summary += f"- Layout: {layout.get('type', 'unknown')} structure\n"
                visual_summary += f"- Regions: {len(layout.get('regions', []))} identified\n"

            if features:
                visual_summary += (
                    f"- Visual features: {features.get('dominant_colors', 'varied')} colors\n"
                )
                visual_summary += f"- Text density: {features.get('text_density', 'medium')}\n"

            # Add compressed embedding representation if available
            if "compressed_embedding" in embedding:
                visual_summary += (
                    f"- Semantic visual encoding: [compressed representation available]\n"
                )

            formatted_parts.append(visual_summary)

        return "\n".join(formatted_parts)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat interface for conversational generation"""
        try:
            # Convert messages to Gemini format
            conversation_parts = []

            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "system":
                    conversation_parts.append(f"System: {content}")
                elif role == "user":
                    conversation_parts.append(f"User: {content}")
                elif role == "assistant":
                    conversation_parts.append(f"Assistant: {content}")

            # Join conversation and generate response
            conversation_text = "\n".join(conversation_parts)
            conversation_text += "\nAssistant:"

            response = self.model.generate_content(conversation_text)

            if response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a response."

        except Exception as e:
            logger.error(f"Error in Gemini chat: {str(e)}")
            return f"Error in conversation: {str(e)}"

    def get_embedding(self, text: str) -> List[float]:
        """Get text embeddings using Gemini (if available)"""
        try:
            # Note: Gemini embedding API might be different
            # This is a placeholder - check actual Gemini embedding API
            logger.warning("Gemini embeddings not implemented - using fallback")
            return []
        except Exception as e:
            logger.error(f"Error getting Gemini embeddings: {str(e)}")
            return []

    def is_multimodal(self) -> bool:
        """Check if provider supports multimodal inputs"""
        return True

    def generate_with_visual_context(
        self,
        text_context: str,
        visual_embeddings: List[np.ndarray],
        query: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response using both text and visual context (MultimodalLLMInterface)"""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Encode visual embeddings to text representation
        visual_context = self.encode_visual_embeddings(visual_embeddings)

        # Build enhanced context
        enhanced_context = f"{text_context}\n\nVisual Context:\n{visual_context}"

        # Generate response
        prompt = self._build_rag_prompt(system_prompt, enhanced_context, query)
        return self._generate_response(prompt)

    def encode_visual_embeddings(self, embeddings: List[np.ndarray]) -> str:
        """Convert visual embeddings to text representation (MultimodalLLMInterface)"""
        if not embeddings:
            return ""

        return self.visual_encoder.encode_embeddings_to_text(embeddings)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": "gemini",
            "model_name": self.model_name,
            "multimodal": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "supports_vision": True,
            "supports_embeddings": False,
        }
